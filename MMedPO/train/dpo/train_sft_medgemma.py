import argparse
import os
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


class SFTVisualTextDataset(Dataset):
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
        )


def sft_loss(model, processor, question: str, answer: str, image: Image.Image) -> torch.Tensor:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    chat_text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc_prompt = processor(text=[chat_text], images=[image], return_tensors="pt")
    enc_full = processor(text=[chat_text + answer], images=[image], return_tensors="pt")

    labels = enc_full["input_ids"].clone()
    prompt_len = enc_prompt["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    for k, v in list(enc_full.items()):
        if isinstance(v, torch.Tensor):
            enc_full[k] = v.to(model.device)
    labels = labels.to(model.device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**enc_full, labels=labels)
    return outputs.loss


def str2bool(x):
    return str(x).lower() in ("1", "true", "t", "yes", "y")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--safe_serialization', type=str2bool, default=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
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

    dataset = SFTVisualTextDataset(args.data_path, args.image_folder)
    def collate(batch):
        return batch
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate)

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
                ans = it.conversations[1]['value']
                img_rel = it.image
                img_path = os.path.join(args.image_folder, img_rel) if img_rel else None
                if not img_path or not os.path.exists(img_path):
                    # 尝试常见命名：若是 SLAKE 结构，支持 xmlabX/source.jpg
                    alt = os.path.join(args.image_folder, os.path.basename(img_rel) if img_rel else '')
                    if alt and os.path.exists(alt):
                        img_path = alt
                    else:
                        # 跳过缺失图像样本，避免崩溃
                        continue
                image = Image.open(img_path).convert('RGB')
                loss_i = sft_loss(model, processor, q, ans, image)
                batch_losses.append(loss_i)
            loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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
