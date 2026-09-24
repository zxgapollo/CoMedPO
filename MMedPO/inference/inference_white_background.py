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

def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
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
        print(f"Rank {rank}: Image processor was not loaded...")
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
        idx = line.get("qid")[0]
        gt_answer = line.get("answer", [None])[0] 
        answer_type = line.get("answer_type", ["unknown"])[0]
        original_question = line.get("question", [""])[0].strip()
        full_image_path = line.get("full_image_path")[0]
        masked_image_path = line.get("masked_image_path")[0] # Read but not used

        try:
            full_image = Image.open(full_image_path).convert('RGB')
            blank_image = Image.new('RGB', full_image.size, color='white')
        except FileNotFoundError as e:
            tqdm.write(f"Warning: Could not find full image. Skipping. Reason: {e}")
            continue

        composite_image = stitch_images_side_by_side(full_image, blank_image)

        # --- MODIFICATION: Save the composite image if requested ---
        # This logic will only run if the --output-image-folder argument is provided
        # and only on the main process (rank 0) to prevent duplicate writes.
        if args.output_image_folder and rank == 0:
            try:
                # Get the subfolder name (e.g., "xmlab0")
                case_id = os.path.basename(os.path.dirname(full_image_path))
                # Get the original filename without extension (e.g., "source")
                base_name = os.path.splitext(os.path.basename(full_image_path))[0]
                
                # Create a new, descriptive filename
                output_filename = f"{base_name}_stitched_with_blank.jpg"
                
                # Define the full output path
                output_dir = os.path.join(args.output_image_folder, case_id)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the image
                composite_image.save(output_path)
            except Exception as e:
                tqdm.write(f"Warning: Could not save composite image for {case_id}. Reason: {e}")
        # --- END OF MODIFICATION ---
        
        qs = f"The following image displays a full medical scan on the left and a version with a masked background on the right. " \
             f"Based on this composite image, answer the following question.\n" \
             f"{DEFAULT_IMAGE_TOKEN}\n" \
             f"Question: {original_question}"
        
        cur_prompt = f"The following image displays a full medical scan on the left and a version with a masked background on the right. " \
                     f"Based on this composite image, answer the following question.\n" \
                     f"{DEFAULT_IMAGE_TOKEN}\n" \
                     f"Question: {original_question}"
        

        # --- The rest of the inference logic is completely unchanged ---
        # ... (code for prompting, tokenizing, generating, and result gathering) ...

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
        serial_result = tensor_to_serializable(result)
        local_results.append(serial_result)


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
    parser.add_argument("--question-file", type=str, required=True, help="Path to the self-contained JSON file with full image paths.")
    parser.add_argument("--answers-file", type=str, required=True)
    
    # --- MODIFICATION: Add the new optional argument ---
    parser.add_argument("--output-image-folder", type=str, default=None, help="Optional: Path to save the stitched composite images.")
    # --- END OF MODIFICATION ---
    
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()