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
import logging

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
from transformers import set_seed, logging as transformers_logging, AutoImageProcessor, CLIPImageProcessor
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable

transformers_logging.set_verbosity_error()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_image_processor(model, image_processor):
    """
    Ensure a valid image_processor is available.
    Priority:
    1) Use provided image_processor
    2) Use model's vision tower image_processor
    3) Use model.config.mm_vision_tower (the correct vision tower for this model)
    4) Fallback to AutoImageProcessor/CLIPImageProcessor from model.config.vision_tower
    5) Fallback to common CLIP processors
    """
    if image_processor is not None:
        return image_processor

    # Try model's vision tower
    try:
        vt = model.get_vision_tower()
        if isinstance(vt, (list, tuple)):
            vt = vt[0]
        if hasattr(vt, 'image_processor') and vt.image_processor is not None:
            return vt.image_processor
    except Exception:
        pass

    # Try from config - prioritize mm_vision_tower which is the correct one for this model
    vision_tower_name = None
    try:
        # First try mm_vision_tower which should be the correct one
        vision_tower_name = getattr(model.config, 'mm_vision_tower', None)
        if not vision_tower_name:
            vision_tower_name = getattr(model.config, 'vision_tower', None)
        if isinstance(vision_tower_name, (list, tuple)):
            vision_tower_name = vision_tower_name[0]
    except Exception:
        pass

    # Prioritize the correct vision tower from model config
    candidates = [vision_tower_name, 'openai/clip-vit-large-patch14-336', '/workspace/CLIP/clip-vit-l14', 'openai/clip-vit-base-patch32']
    for name in candidates:
        if not name:
            continue
        # 优先本地加载，避免网络请求导致的长时间等待
        try:
            return AutoImageProcessor.from_pretrained(name, local_files_only=True)
        except Exception:
            try:
                return CLIPImageProcessor.from_pretrained(name, local_files_only=True)
            except Exception:
                continue

    # 兜底：构造默认处理器（不访问网络）
    try:
        return CLIPImageProcessor()
    except Exception:
        pass

    raise RuntimeError('Failed to initialize image_processor. Please provide a valid vision_tower or processor.')

def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
    if image1.height != image2.height:
        new_height = max(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))
    
    new_width = image1.width + image2.width
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    return new_image

def calculate_log_likelihood_image_only(model, tokenizer, image_processor, answer, image_path, conv_mode="llava_v1"):
    """
    Calculate the token-normalized log likelihood of an answer given only an image (no question text).
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Prepare conversation with only image token, no question text
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)  # Only image token
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Tokenize answer
        answer_ids = tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        
        # Ensure answer_ids has the same batch dimension as input_ids
        if answer_ids.dim() == 2 and answer_ids.shape[0] == 1:
            # answer_ids is already [1, seq_len]
            pass
        elif answer_ids.dim() == 1:
            # answer_ids is [seq_len], need to add batch dimension
            answer_ids = answer_ids.unsqueeze(0)
        else:
            # Handle other cases by taking the first batch if multiple batches
            answer_ids = answer_ids[0:1]
        
        # Prepare full sequence (input + answer)
        full_input_ids = torch.cat([input_ids, answer_ids], dim=1)
        
        with torch.inference_mode():
            outputs = model(
                input_ids=full_input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=False
            )

            logits = outputs.logits[0]

            # 向量化答案令牌的对数似然计算
            answer_start_pos = input_ids.shape[1]
            answer_tokens = answer_ids[0] if answer_ids.dim() == 2 else answer_ids
            max_len = min(answer_tokens.shape[0], logits.shape[0] - (answer_start_pos - 1))
            if max_len <= 0:
                return float('-inf')
            positions = torch.arange(max_len, device=logits.device) + (answer_start_pos - 1)
            selected_logits = logits.index_select(0, positions)
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
            gathered = log_probs.gather(1, answer_tokens[:max_len].unsqueeze(1)).squeeze(1)
            mask = torch.isfinite(gathered)
            if mask.sum() == 0:
                return float('-inf')
            norm_ll = gathered[mask].mean().item()
            if not torch.isfinite(torch.tensor(norm_ll)):
                return float('-inf')
            return norm_ll
            
    except Exception as e:
        logger.error(f"Error calculating log likelihood (image only): {e}")
        return float('-inf')

def calculate_log_likelihood(model, tokenizer, image_processor, question, answer, image_path, conv_mode="llava_v1"):
    """
    Calculate the token-normalized log likelihood of an answer given an image and question.
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Prepare conversation
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Tokenize answer
        answer_ids = tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        
        # Prepare full sequence (input + answer)
        full_input_ids = torch.cat([input_ids, answer_ids], dim=1)
        
        with torch.inference_mode():
            outputs = model(
                input_ids=full_input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=False
            )

            logits = outputs.logits[0]

            # 向量化答案令牌的对数似然计算
            answer_start_pos = input_ids.shape[1]
            answer_tokens = answer_ids[0] if answer_ids.dim() == 2 else answer_ids
            max_len = min(answer_tokens.shape[0], logits.shape[0] - (answer_start_pos - 1))
            if max_len <= 0:
                return float('-inf')
            positions = torch.arange(max_len, device=logits.device) + (answer_start_pos - 1)
            selected_logits = logits.index_select(0, positions)
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
            gathered = log_probs.gather(1, answer_tokens[:max_len].unsqueeze(1)).squeeze(1)
            mask = torch.isfinite(gathered)
            if mask.sum() == 0:
                return float('-inf')
            norm_ll = gathered[mask].mean().item()
            if not torch.isfinite(torch.tensor(norm_ll)):
                return float('-inf')
            return norm_ll
            
    except Exception as e:
        logger.error(f"Error calculating log likelihood: {e}")
        return float('-inf')

def extract_context_from_question(question):
    """
    Extract context from question. This is a simplified version.
    In practice, you might need more sophisticated context extraction.
    """
    # Look for common context patterns
    context_markers = ["Based on the", "According to", "Given that", "The image shows", "In the context of"]
    
    for marker in context_markers:
        if marker in question:
            # Extract everything after the marker as context
            parts = question.split(marker, 1)
            if len(parts) > 1:
                context = marker + parts[1]
                question_without_context = parts[0].strip()
                if question_without_context.endswith(',') or question_without_context.endswith('.'):
                    question_without_context = question_without_context[:-1]
                return context.strip(), question_without_context.strip()
    
    # If no clear context marker, try to identify descriptive parts
    sentences = question.split('.')
    if len(sentences) > 1:
        # Assume first sentence might be context, rest is question
        context = sentences[0].strip() + '.'
        question_without_context = '.'.join(sentences[1:]).strip()
        if question_without_context:
            return context, question_without_context
    
    # Fallback: return original question as context, empty as question without context
    return question, ""

def calculate_textual_tie_for_sample(model, tokenizer, image_processor, sample, full_image_folder, masked_image_folder):
    """
    Calculate Textual TIE for a single sample: LL_text - LL_null
    """
    try:
        # Handle different data formats
        qid = sample.get("qid", sample.get("id", "unknown"))
        
        # Extract question and answer from conversations format
        if "conversations" in sample:
            conversations = sample["conversations"]
            original_question = None
            gt_answer = None
            
            for conv in conversations:
                if conv["from"] == "human":
                    # Remove <image> token from question for processing
                    original_question = conv["value"].replace("<image>", "").strip()
                elif conv["from"] == "gpt":
                    gt_answer = conv["value"].strip()
            
            if not original_question or not gt_answer:
                logger.warning(f"Could not extract question/answer from conversations for sample {qid}")
                return None
        else:
            # Fallback to direct fields
            original_question = sample.get("question", "")
            gt_answer = sample.get("answer", "")
        
        # Extract context and create question without context
        context, question_without_context = extract_context_from_question(original_question)
        
        # If we couldn't extract meaningful context, use the original question as context
        # and create a minimal question without context
        if not question_without_context:
            question_with_context = original_question
            question_without_context = "What do you see?"  # Minimal question
        else:
            question_with_context = original_question
        
        # Construct image paths (use full image for both conditions)
        image_filename = sample["image"]
        full_image_path = os.path.join(full_image_folder, image_filename)
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            logger.warning(f"Image not found: {full_image_path}")
            return None
        
        # Calculate log likelihoods
        # LL_text: with full context (question + image)
        ll_text = calculate_log_likelihood(model, tokenizer, image_processor, question_with_context, gt_answer, full_image_path)
        
        # LL_null: image only (no question text)
        ll_null = calculate_log_likelihood_image_only(model, tokenizer, image_processor, gt_answer, full_image_path)
        
        # Calculate Textual TIE
        textual_tie = ll_text - ll_null
        
        result = {
            "qid": qid,
            "question_with_context": question_with_context,
            "question_without_context": "[IMAGE_ONLY]",  # Indicates image-only input
            "extracted_context": context,
            "gt_answer": gt_answer,
            "image": image_filename,
            "image_path": full_image_path,
            "ll_text": ll_text,
            "ll_null": ll_null,
            "textual_tie": textual_tie,
            "tie_positive": textual_tie if textual_tie > 0 else 0.0,
            "tie_negative": abs(textual_tie) if textual_tie < 0 else 0.0,
            "tie_difference": textual_tie
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing sample {sample.get('qid', 'unknown')}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None, help="Base model path for LoRA adapter")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--output-image-folder", type=str, required=True)
    parser.add_argument("--full-image-folder", type=str, required=True)
    parser.add_argument("--masked-image-folder", type=str, required=True)
    parser.add_argument("--calculate-textual-tie", action="store_true", help="Calculate Textual TIE")
    parser.add_argument("--conv-mode", type=str, default="llava_med_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Setup distributed training
    setup()
    
    # Disable torch init
    disable_torch_init()
    
    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Ensure image processor is valid
    image_processor = ensure_image_processor(model, image_processor)
    logger.info(f"Using image processor: {type(image_processor).__name__}")
    
    # Load questions
    with open(args.question_file, 'r') as f:
        questions = json.load(f)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:  # Only create directory if there is a directory path
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.output_image_folder, exist_ok=True)
    
    # Process samples and calculate Textual TIE
    tie_results = []
    
    for sample in tqdm(questions, desc="Calculating Textual TIE"):
        # Check if we're in distributed mode
        if dist.is_initialized():
            if dist.get_rank() == 0 or len(questions) % dist.get_world_size() == 0:
                result = calculate_textual_tie_for_sample(
                    model, tokenizer, image_processor, sample, 
                    args.full_image_folder, args.masked_image_folder
                )
                
                if result is not None:
                    tie_results.append(result)
        else:
            # Single GPU mode
            result = calculate_textual_tie_for_sample(
                model, tokenizer, image_processor, sample, 
                args.full_image_folder, args.masked_image_folder
            )
            
            if result is not None:
                tie_results.append(result)
    
    # Save TIE results
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(args.output_file, 'w') as f:
            for result in tie_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Textual TIE calculation completed. Results saved to {args.output_file}")
        logger.info(f"Processed {len(tie_results)} samples")
    
    cleanup()

if __name__ == "__main__":
    main()