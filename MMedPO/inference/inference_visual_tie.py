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

# --- Python Path Setup (robust) ---
llava_code_path = '/share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo'
if os.path.isdir(llava_code_path) and llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)
# Avoid importing internal modules directly; builder handles model imports.
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

    # 最后兜底：使用默认构造的 CLIPImageProcessor（不访问网络）
    try:
        return CLIPImageProcessor()
    except Exception:
        pass

    raise RuntimeError('Failed to initialize image_processor. Please provide a valid vision_tower or processor.')

def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
    new_height = max(image1.height, image2.height)
    if image1.height != image2.height:
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))
    
    new_width = image1.width + image2.width
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    return new_image

# Added: safe resolver for conversation templates
def resolve_conv_template(conv_mode: str):
    """Return a conversation template for the given conv_mode with safe fallbacks.
    - If conv_mode exists, use it.
    - If alias 'llava_med_v1' is requested but unavailable, fall back to 'llava_v1' if present.
    - Otherwise, fall back to the first available template and log a warning.
    """
    if conv_mode in conv_templates:
        return conv_templates[conv_mode].copy()
    # alias mapping
    alias_map = {
        'llava_med_v1': 'llava_v1',
    }
    alias = alias_map.get(conv_mode)
    if alias and alias in conv_templates:
        logger.warning(f"conv_mode '{conv_mode}' unavailable; falling back to '{alias}'. Available: {list(conv_templates.keys())}")
        return conv_templates[alias].copy()
    # final fallback: first key
    keys = list(conv_templates.keys())
    fallback = 'llava_v1' if 'llava_v1' in conv_templates else (keys[0] if keys else None)
    if fallback:
        logger.warning(f"conv_mode '{conv_mode}' unavailable; falling back to '{fallback}'. Available: {keys}")
        return conv_templates[fallback].copy()
    raise KeyError(f"No conversation templates available; requested '{conv_mode}'")

def calculate_log_likelihood(model, tokenizer, image_processor, question, answer, image_input, conv_mode="llava_med_v1"):
    """
    使用 y_gt（ground-truth answer）在给定图像与问题条件下计算 token 归一化的对数似然。
    对应 method1 公式中的 (y_gt | ·) 评分目标。

    参数 image_input 可以是图像路径（str）或 PIL.Image.Image 实例。
    """
    try:
        # Load and process image (支持路径或 PIL 图像)
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif hasattr(image_input, 'convert'):
            image = image_input.convert('RGB')
        else:
            raise ValueError('image_input must be a file path or PIL.Image.Image')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Prepare conversation
        conv = resolve_conv_template(conv_mode)
        # Check if question already contains image token
        if DEFAULT_IMAGE_TOKEN in question:
            conv.append_message(conv.roles[0], question)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
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
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(
                input_ids=full_input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=False
            )
            
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Vectorized log-likelihood over answer tokens
            answer_start_pos = input_ids.shape[1]
            if answer_ids.dim() == 2:
                answer_tokens = answer_ids[0]
            else:
                answer_tokens = answer_ids

            max_len = min(answer_tokens.shape[0], logits.shape[0] - (answer_start_pos - 1))
            if max_len <= 0:
                return float('-inf')

            positions = torch.arange(max_len, device=logits.device) + (answer_start_pos - 1)
            selected_logits = logits.index_select(0, positions)
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
            gathered = log_probs.gather(1, answer_tokens[:max_len].unsqueeze(1)).squeeze(1)

            # Filter out non-finite values
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

def calculate_visual_tie_for_sample(model, tokenizer, image_processor, sample, full_image_folder, masked_image_folder, conv_mode):
    """
    Calculate Visual TIE for a single sample using method1:
    visual_tie = (y_gt | X ⊕ X_bg) - (y_gt | X_null ⊕ X_bg)
    其中 X_null 为与原图尺寸一致的纯色空图。
    """
    try:
        # Handle SLAKE format: use "id" instead of "qid"
        qid = sample["id"]
        
        # Extract question and answer from conversations
        if "conversations" in sample and len(sample["conversations"]) >= 2:
            question = sample["conversations"][0]["value"]
            gt_answer = sample["conversations"][1]["value"]
        elif "question" in sample and "answer" in sample:
            # Fallback for other formats
            question = sample["question"]
            gt_answer = sample["answer"]
        else:
            logger.error(f"Cannot extract question/answer from sample {qid}")
            return None
        
        # Construct image paths
        image_filename = sample["image"]
        full_image_path = os.path.join(full_image_folder, image_filename)
        masked_image_path = os.path.join(masked_image_folder, image_filename)
        
        # Check if files exist
        if not os.path.exists(full_image_path):
            logger.warning(f"Full image not found: {full_image_path}")
            return None
            
        if not os.path.exists(masked_image_path):
            logger.warning(f"Masked image not found: {masked_image_path}")
            return None
        
        # 载入原图与背景图
        full_img = Image.open(full_image_path).convert('RGB')
        bg_img = Image.open(masked_image_path).convert('RGB')

        # 构造空图 X_null（中性灰）并与背景拼接
        null_img = Image.new('RGB', full_img.size, color=(128, 128, 128))

        # 组合图像：pref = full ⊕ bg；disp = null ⊕ bg
        combined_pref = stitch_images_side_by_side(full_img, bg_img)
        combined_disp = stitch_images_side_by_side(null_img, bg_img)

        # 计算 method1 的对数似然
        ll_pref = calculate_log_likelihood(model, tokenizer, image_processor, question, gt_answer, combined_pref, conv_mode=conv_mode)
        ll_disp = calculate_log_likelihood(model, tokenizer, image_processor, question, gt_answer, combined_disp, conv_mode=conv_mode)

        # （可选）保留旧度量，便于对比分析
        try:
            ll_full = calculate_log_likelihood(model, tokenizer, image_processor, question, gt_answer, full_image_path, conv_mode=conv_mode)
            ll_bg = calculate_log_likelihood(model, tokenizer, image_processor, question, gt_answer, masked_image_path, conv_mode=conv_mode)
        except Exception:
            ll_full, ll_bg = None, None

        # 计算 Visual TIE（method1）
        visual_tie = ll_pref - ll_disp
        
        result = {
            "qid": qid,
            "question": question,
            "gt_answer": gt_answer,
            "image": image_filename,
            "full_image_path": full_image_path,
            "masked_image_path": masked_image_path,
            # method1 指标
            "ll_pref": ll_pref,
            "ll_disp": ll_disp,
            # 兼容旧指标（可能为 None）
            "ll_full": ll_full,
            "ll_bg": ll_bg,
            "visual_tie": visual_tie,
            "tie_positive": visual_tie if visual_tie > 0 else 0.0,
            "tie_negative": abs(visual_tie) if visual_tie < 0 else 0.0,
            "tie_difference": visual_tie,
            # 标记与公式（method1，y_gt）
            "method": "visual_indirect_method1",
            "scoring_target": "y_gt",
            "tie_formula": "(y_gt|X \u2295 X_bg) > (y_gt|X_null \u2295 X_bg)",
            "pref_condition": "full_image",
            "disp_condition": "background_only",
            "conv_mode_used": conv_mode
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None, help="Base model path for LoRA adapter")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--output-image-folder", type=str, required=True)
    parser.add_argument("--save-images", action="store_true", help="Save stitched images for visualization")
    parser.add_argument("--full-image-folder", type=str, required=True)
    parser.add_argument("--masked-image-folder", type=str, required=True)
    parser.add_argument("--calculate-visual-tie", action="store_true", help="Calculate Visual TIE")
    parser.add_argument("--conv-mode", type=str, default="llava_med_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of samples for smoke testing")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size for inference")
    args = parser.parse_args()

    # Setup distributed training
    setup()
    
    # Disable torch init
    disable_torch_init()
    
    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device_map="auto"
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
    
    # Process samples and calculate Visual TIE
    tie_results = []

    # Optional smoke test: limit number of samples
    if args.limit is not None and args.limit > 0:
        questions = questions[:args.limit]
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    task_mode = os.environ.get('TASK_MODE', 'model_shard')
    if world_size > 1 and task_mode == 'data_shard':
        questions = [q for i, q in enumerate(questions) if i % world_size == rank]
    # Batched processing per rank
    bs = max(1, args.batch_size)
    for i in tqdm(range(0, len(questions), bs), desc="Calculating Visual TIE"):
        batch = questions[i:i+bs]
        for sample in batch:
            result = calculate_visual_tie_for_sample(
                model, tokenizer, image_processor, sample,
                args.full_image_folder, args.masked_image_folder, args.conv_mode
            )
            if result is not None:
                tie_results.append(result)
                try:
                    full_img = Image.open(result["full_image_path"]).convert('RGB')
                    masked_img = Image.open(result["masked_image_path"]).convert('RGB')
                    if args.save_images:
                        pref_img = stitch_images_side_by_side(full_img, masked_img)
                        disp_img = stitch_images_side_by_side(Image.new('RGB', full_img.size, color=(128, 128, 128)), masked_img)
                        qid = result.get('qid', sample.get('id', 'unknown'))
                        pref_out = os.path.join(args.output_image_folder, f"{qid}_pref.jpg")
                        disp_out = os.path.join(args.output_image_folder, f"{qid}_disp.jpg")
                        pref_img.save(pref_out)
                        disp_img.save(disp_out)
                except Exception:
                    pass
    
    out_file = args.output_file if (world_size == 1 or os.environ.get('TASK_MODE', 'model_shard') != 'data_shard') else args.output_file.replace('.jsonl', f'.rank{rank}.jsonl')
    with open(out_file, 'w') as f:
        for result in tie_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Visual TIE calculation completed. Results saved to {out_file}")
    logger.info(f"Processed {len(tie_results)} samples")
    
    cleanup()

if __name__ == "__main__":
    main()