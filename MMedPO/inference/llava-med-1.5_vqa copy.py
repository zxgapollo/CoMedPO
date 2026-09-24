import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import numpy as np

# --- MODIFICATION 1: Robust Python Path Setup ---
# This is the primary fix for the 'llava_mistral' error.
# We explicitly add the directory containing the custom 'llava' code
# to Python's path, ensuring the interpreter can find it.
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    # Use insert(0) to give this path priority
    sys.path.insert(0, llava_code_path)

# By importing the model class directly, we ensure it's registered
# with the transformers library before we try to load the model.
import llava.model.language_model.llava_mistral
# --- END MODIFICATION 1 ---

warnings.simplefilter(action="ignore", category=FutureWarning)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
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
import math
from transformers import set_seed, logging

# This is now relative to the new path, so it should be correct
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable
import debugpy


logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def add_gaussian_noise(image, mean=0.0, stddev=0.1):
    image_np = np.array(image).astype(np.float32) / 255.0  # 标准化到 [0, 1]
    noise = np.random.normal(mean, stddev, image_np.shape)
    noisy_image_np = image_np + noise
    noisy_image_np = np.clip(noisy_image_np, 0.0, 1.0)
    noisy_image = Image.fromarray((noisy_image_np * 255).astype(np.uint8))
    return noisy_image


def generate_gaussian_noise(image, mean=0.0, stddev=0.1):
    image_shape = np.array(image).shape
    noise = np.random.normal(mean, stddev, image_shape).astype(np.float32)
    noise_clipped = np.clip(noise, 0.0, 1.0)
    noisy_image = Image.fromarray((noise_clipped * 255).astype(np.uint8))
    return noisy_image


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

    # --- MODIFICATION 2: Image Processor Fallback ---
    # This adds a check to prevent crashes if the image processor isn't
    # loaded correctly from the model checkpoint. It loads the default
    # one used by LLaVA-1.5 as a fallback.
    if image_processor is None:
        print(f"[Rank {rank}] Warning: Image processor not found. Loading default from CLIP.")
        from transformers import CLIPImageProcessor
        vision_tower_path = 'openai/clip-vit-large-patch14-336'
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)
        # Manually attach it to the model so 'process_images' can find it
        if hasattr(model, 'get_vision_tower'):
             model.get_vision_tower().image_processor = image_processor
    # --- END MODIFICATION 2 ---


    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=1, num_workers=16, pin_memory=True
    )
    local_results = []

    for line in tqdm(dataloader, position=0, file=sys.stdout):
        idx = line["id"][0]
        image_file = line["image_path"][0]
        gt_answer = line["answer"][0]
        answer_type = line["answer_type"][0]
        qs = line["question"][0].replace(DEFAULT_IMAGE_TOKEN, "").strip()

        cur_prompt = qs
        
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(rank)
        )

        image = Image.open(os.path.join(args.image_folder, image_file))
        if args.noised_image:
            image = generate_gaussian_noise(image, 0, 1)
            tqdm.write(f"Generate noise from image {image_file}")

        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != Separator.TWO else conv.sep2
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

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        ans_id = shortuuid.uuid()
        
        # This structure seems fine, keeping it as is.
        result = {
            "id": idx,
            "image_path": image_file,
            "prompt": cur_prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "answer_id": ans_id,
            "model_id": model_name,
            "noised_image": 1 if args.noised_image else 0,
            "answer_type": answer_type,
            "metadata": {},
        }
        
        serializable_result = tensor_to_serializable(result)
        local_results.append(serializable_result)

    dist.barrier()
    print(f"Rank {rank} reached barrier")
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)
    print(f"Rank {rank} finished all_gather_object")

    if rank == 0:
        all_results = [item for sublist in gathered_results for item in sublist]
        unique_results = []
        seen_ids = set()
        for result in all_results:
            # --- MODIFICATION 3: Robust Uniqueness Check ---
            # This handles cases where the question ID might be loaded
            # incorrectly as a list (e.g., [123]) instead of a value (123).
            q_id = result["id"]
            if isinstance(q_id, list):
                q_id = q_id[0]
            # --- END MODIFICATION 3 ---
            
            if q_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(q_id)

        # Sort results by the original question ID for consistent output
        unique_results.sort(key=lambda x: x["id"][0] if isinstance(x["id"], list) else x["id"])

        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        with open(answers_file, "w") as ans_file:
            for res in unique_results:
                ans_file.write(json.dumps(res) + "\n")
        print(f"Rank {rank} finished writing to file {args.answers_file}")
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--noised_image", action="store_true", help="If set, use noised images."
    )

    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()