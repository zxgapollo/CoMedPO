import argparse
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from PIL import Image


@dataclass
class Sample:
    image: str
    conversations: List[Dict[str, Any]]
    rejected_conversations: List[Dict[str, Any]]
    weighted_score: float


class DpoVisualTextDataset(Dataset):
    def __init__(self, data_path: str, image_folder: str):
        with open(data_path, 'r') as f:
            self.items = json.load(f)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return Sample(
            image=item['image'],
            conversations=item['conversations'],
            rejected_conversations=item.get('rejected_conversations', []),
            weighted_score=float(item.get('weighted_score', 1.0))
        )


def build_inputs(processor, messages, image: Image.Image):
    chat_text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return processor(text=[chat_text], images=[image], return_tensors="pt")


def dpo_loss(prefer_logprobs: torch.Tensor, disp_logprobs: torch.Tensor, beta: float, weights: torch.Tensor):
    diff = prefer_logprobs - disp_logprobs
    return ( - torch.nn.functional.logsigmoid(beta * diff) * weights ).mean()


def answer_logprob_mean(model, processor, question: str, answer: str, image: Image.Image) -> torch.Tensor:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    chat_text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc_prompt = processor(text=[chat_text], images=[image], return_tensors="pt")
    enc_full = processor(text=[chat_text + answer], images=[image], return_tensors="pt")

    # build labels: -100 for prompt part, answer token ids for answer part
    labels = enc_full["input_ids"].clone()
    prompt_len = enc_prompt["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    # move to device
    for k, v in list(enc_full.items()):
        if isinstance(v, torch.Tensor):
            enc_full[k] = v.to(model.device)
    labels = labels.to(model.device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**enc_full, labels=labels)
    loss = outputs.loss
    del outputs, enc_prompt, enc_full, labels
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return (-loss)


def str2bool(x):
    return str(x).lower() in ("1", "true", "t", "yes", "y")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--image_folder', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--device_map', type=str, default='')
    parser.add_argument('--convert_bin_path', type=str, default='')
    parser.add_argument('--convert_save_dir', type=str, default='')
    parser.add_argument('--convert_base_model_path', type=str, default='')
    parser.add_argument('--safe_serialization', type=str2bool, default=False)

    args = parser.parse_args()

    # Optional conversion only mode (run before any training-only setup)
    if args.convert_bin_path and args.convert_save_dir and args.convert_base_model_path:
        base = AutoModelForImageTextToText.from_pretrained(args.convert_base_model_path, torch_dtype=torch.bfloat16)
        sd = torch.load(os.path.expanduser(args.convert_bin_path), map_location='cpu')
        try:
            base.load_state_dict(sd, strict=False)
        except Exception:
            base.load_state_dict(sd, strict=True)
        os.makedirs(os.path.expanduser(args.convert_save_dir), exist_ok=True)
        base.save_pretrained(os.path.expanduser(args.convert_save_dir), safe_serialization=args.safe_serialization)
        proc = AutoProcessor.from_pretrained(args.convert_base_model_path)
        proc.save_pretrained(os.path.expanduser(args.convert_save_dir))
        print(f"Saved HF checkpoint to: {args.convert_save_dir}")
        return

    # Training mode requires full arguments
    if not (args.model_path and args.data_path and args.image_folder and args.output_dir):
        raise ValueError("Training mode requires --model_path, --data_path, --image_folder, --output_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mp_kwargs = {"torch_dtype": torch.bfloat16}
    if args.device_map and args.device_map.strip():
        mp_kwargs["device_map"] = args.device_map
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **mp_kwargs
    )
    if args.device_map is None or args.device_map == '':
        model.to(device)
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception:
        pass
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model_path)

    dataset = DpoVisualTextDataset(args.data_path, args.image_folder)
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=lambda batch: batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, math.ceil(len(dataloader) * args.num_train_epochs))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    global_step = 0
    model.train()
    for epoch in range(args.num_train_epochs):
        for items in tqdm(dataloader, desc=f"epoch {epoch+1}/{args.num_train_epochs}", unit="step"):
            batch_losses = []
            for it in items:
                q = it.conversations[0]['value']
                pref_ans = it.conversations[1]['value']
                if it.rejected_conversations:
                    disp_ans = it.rejected_conversations[1]['value']
                else:
                    disp_ans = ""
                img_path = os.path.join(args.image_folder, it.image)
                try:
                    image = Image.open(img_path).convert('RGB')
                except FileNotFoundError:
                    continue
                try:
                    lp_pref = answer_logprob_mean(model, processor, q, pref_ans, image)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                try:
                    lp_disp = answer_logprob_mean(model, processor, q, disp_ans, image) if disp_ans else (lp_pref - 1.0)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    lp_disp = lp_pref - 1.0
                w = torch.tensor(it.weighted_score, dtype=torch.float32, device=device)
                loss_i = dpo_loss(lp_pref, lp_disp, args.beta, w)
                batch_losses.append(loss_i)
            loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):
                continue
            try:
                loss.backward()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            try:
                optimizer.step()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.logging_steps == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"pytorch_model_epoch_{epoch+1}.bin"))
        hf_dir = os.path.join(args.output_dir, "hf_latest")
        if os.path.isdir(hf_dir):
            try:
                import shutil
                shutil.rmtree(hf_dir)
            except Exception:
                pass
        os.makedirs(hf_dir, exist_ok=True)
        model.save_pretrained(hf_dir, safe_serialization=args.safe_serialization)
        processor.save_pretrained(hf_dir)


if __name__ == '__main__':
    main()
