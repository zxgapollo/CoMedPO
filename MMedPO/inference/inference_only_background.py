import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import warnings
import numpy as np

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
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable

logging.set_verbosity_error()

# stitch_images_side_by_side is not needed for this version.

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
        print(f"Rank {rank}: Image processor was not loaded...")
        from transformers import CLIPImageProcessor
        processor_path = 'openai/clip-vit-large-patch14-336'
        image_processor = CLIPImageProcessor.from_pretrained(processor_path)
        if hasattr(model, 'get_vision_tower'):
             model.get_vision_tower().image_processor = image_processor

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    
    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=16, pin_memory=True)
    local_results = []

    for line in tqdm(dataloader, position=0, file=sys.stdout):
        # Handle both standard format and slake_dpo_weighted.json format
        if "qid" in line:
            idx = line.get("qid")[0]
            gt_answer = line.get("answer", [None])[0]
            answer_type = line.get("answer_type", ["unknown"])[0]
            original_question = line.get("question", [""])[0].strip()
            full_image_path = line.get("full_image_path", [None])[0] # Still read for case_id
            masked_image_path = line.get("masked_image_path", [None])[0]
        else:
            # Handle slake_dpo_weighted.json format
            idx = line.get("id", [0])[0]
            image_rel_path = line.get("image", [""])[0]
            
            # Extract question and answer from conversations
            conversations = line.get("conversations", [])
            original_question = ""
            gt_answer = None
            
            if len(conversations) >= 2:
                question_obj = conversations[0]
                if isinstance(question_obj, dict) and "value" in question_obj:
                    question_text = question_obj["value"]
                    if not isinstance(question_text, str):
                        question_text = str(question_text)
                    original_question = question_text.replace("<image>", "").strip()
                
                answer_obj = conversations[1]
                if isinstance(answer_obj, dict) and "value" in answer_obj:
                    gt_answer = answer_obj["value"]
                    if not isinstance(gt_answer, str):
                        gt_answer = str(gt_answer)
                    gt_answer = gt_answer.strip()
            
            answer_type = "unknown"
            
            # Construct full paths using base folders
            if image_rel_path:
                full_image_path = os.path.join(args.full_image_folder, image_rel_path) # Still read for case_id
                
                image_dir, image_file = os.path.split(image_rel_path)
                masked_image_path = os.path.join(args.masked_image_folder, image_rel_path)
            else:
                tqdm.write(f"Warning: No image path for item {idx}. Skipping.")
                continue

        try:
            # --- MODIFICATION: Load the masked background image as the only input ---
            input_image = Image.open(masked_image_path).convert('RGB')
            # --- END OF MODIFICATION ---
        except FileNotFoundError as e:
            tqdm.write(f"Warning: Could not find masked background image at '{masked_image_path}'. Skipping. Reason: {e}")
            continue

        # Optionally save the input image for verification
        if args.output_image_folder and rank == 0:
            try:
                case_id = os.path.basename(os.path.dirname(full_image_path))
                base_name = os.path.splitext(os.path.basename(masked_image_path))[0]
                output_filename = f"{base_name}_input.jpg"
                output_dir = os.path.join(args.output_image_folder, case_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
                input_image.save(output_path)
            except Exception as e:
                tqdm.write(f"Warning: Could not save input image for {case_id}. Reason: {e}")
        
        # --- MODIFICATION: Update prompt to reflect the new input type ---
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{original_question}"
        cur_prompt =  f"The following image displays a full medical scan on the left and a version with a masked background on the right. " \
                      f"Based on this composite image, answer the following question.\n" \
                      f"{DEFAULT_IMAGE_TOKEN}\n" \
                      f"Question: {original_question}"
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(rank)
        
        # We process the single masked background image.
        image_tensor = process_images([input_image], image_processor, model.config)[0]
        
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
    print(f"Rank {rank} reached gathering barrier")
    
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)
    
    print(f"Rank {rank} finished all_gather_object")

    if rank == 0:
        print(f"Rank {rank} starting to process and write results...")
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
        print(f"Rank {rank} finished writing to file {args.answers_file}")

    dist.barrier()
    print(f"Rank {rank} passed final barrier. Preparing to clean up.")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True, help="Path to the self-contained JSON file with question data.")
    parser.add_argument("--answers-file", type=str, required=True)
    
    # Optional argument to save the input images for verification
    parser.add_argument("--output-image-folder", type=str, default=None, help="Optional: Path to save the masked background input images.")
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