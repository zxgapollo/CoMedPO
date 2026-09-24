import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from torch.utils.data import DataLoader
import torch.distributed as dist
import warnings

# --- Original Python Path Setup ---
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)
import llava.model.language_model.llava_mistral
# --- End Path Setup ---

warnings.simplefilter(action="ignore", category=FutureWarning)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
)
from PIL import Image
from transformers import set_seed, logging
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable

logging.set_verbosity_error()

def stitch_images_side_by_side(image1, image2):
    if image1.height != image2.height:
        new_height = max(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))
    width1, height1 = image1.size
    width2, height2 = image2.size
    combined_image = Image.new('RGB', (width1 + width2, height1))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))
    return combined_image

def eval_model(args):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} started")

    set_seed(0)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=f"cuda:{rank}"
    )

    if image_processor is None:
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        if hasattr(model, 'get_vision_tower'):
             model.get_vision_tower().image_processor = image_processor

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=16, pin_memory=True)
    local_results = []

    for line in tqdm(dataloader, position=0, file=sys.stdout):
        # Handle slake_dpo_weighted.json format
        idx = line.get("id", [0])[0]
        image_rel_path = line.get("image", [""])[0]
        conversations = line.get("conversations", [])
        gt_answer = None
        if not isinstance(conversations, list) or len(conversations) < 2:
            tqdm.write(f"Warning: Invalid conversations format for item {idx}. Skipping.")
            continue
        answer_obj = conversations[1]
        if not isinstance(answer_obj, dict) or "value" not in answer_obj:
            tqdm.write(f"Warning: Invalid answer_obj for item {idx}. Skipping.")
            continue
        gt_answer_value = answer_obj["value"]
        if not isinstance(gt_answer_value, str):
            if isinstance(gt_answer_value, list):
                gt_answer_value = '\n'.join(gt_answer_value)
            else:
                gt_answer_value = str(gt_answer_value)
        gt_answer = gt_answer_value.strip()
        answer_type = "unknown"
        if image_rel_path:
            full_image_path = os.path.join(args.full_image_folder, image_rel_path)
            image_dir, image_file = os.path.split(image_rel_path)
            masked_image_path = os.path.join(args.masked_image_folder, image_rel_path)
        else:
            tqdm.write(f"Warning: No image path for item {idx}. Skipping.")
            continue
        try:
            full_image = Image.open(full_image_path).convert('RGB')
            masked_image = Image.open(masked_image_path).convert('RGB')
        except FileNotFoundError as e:
            tqdm.write(f"Warning: missing image(s). Skip. Reason: {e}")
            continue

        composite_image = stitch_images_side_by_side(full_image, masked_image)

        # Optionally save the composite image for verification (no-text context)
        if getattr(args, "output_image_folder", None) and rank == 0:
            try:
                case_id = os.path.basename(os.path.dirname(full_image_path))
                base_name = os.path.splitext(os.path.basename(full_image_path))[0]
                output_dir = os.path.join(args.output_image_folder, case_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{base_name}_without_context_composite.jpg")
                composite_image.save(output_path)
            except Exception as e:
                tqdm.write(f"Warning: Could not save composite image for {case_id}. Reason: {e}")

        # Prompt WITHOUT textual context (T_null)
        qs = (
            "You are provided with an image. Answer based only on visual information.\n"
            f"{DEFAULT_IMAGE_TOKEN}"
        )
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(rank)
        image_tensor = process_images([composite_image], image_processor, model.config)[0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(rank),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=64,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        case_id = os.path.basename(os.path.dirname(full_image_path))
        result = {
            "id": idx, "case_id": case_id, "prompt": cur_prompt, "answer": outputs,
            "gt_answer": gt_answer, "answer_id": ans_id, "model_id": model_name,
            "answer_type": answer_type, "metadata": {},
        }
        serializable_result = tensor_to_serializable(result)
        local_results.append(serializable_result)

    dist.barrier()
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        all_results = [item for sublist in gathered_results for item in sublist]
        unique_results = []
        seen_ids = set()
        for res in all_results:
            q_id = res["id"]
            if isinstance(q_id, list): q_id = q_id[0]
            if q_id not in seen_ids:
                unique_results.append(res)
                seen_ids.add(q_id)

        unique_results.sort(key=lambda x: x["id"][0] if isinstance(x["id"], list) else x["id"])

        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        with open(answers_file, "w") as ans_file:
            for res in unique_results:
                ans_file.write(json.dumps(res) + "\n")
        print(f"Finished writing to file {args.answers_file}")

    dist.barrier()
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--output-image-folder", type=str, default=None, help="Optional: Path to save composed images for without-context runs.")
    parser.add_argument("--full-image-folder", type=str, default="/workspace/MMedPO/datasets/SLAKE/imgs", help="Base folder for full images")
    parser.add_argument("--masked-image-folder", type=str, default="/workspace/MMedPO/datasets/SLAKE/processed_imgs", help="Base folder for masked images")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()