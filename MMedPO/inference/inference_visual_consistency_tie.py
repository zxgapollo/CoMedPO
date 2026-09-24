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
import logging
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

def get_conv_template_safe(conv_mode: str):
    """Return a conversation template safely. If conv_mode is missing, fallback to a known template."""
    try:
        if conv_mode in conv_templates:
            return conv_templates[conv_mode].copy()
        # Fallback mapping for medical variants to base llava template
        alias_map = {
            'llava_med_v1': 'llava_v1',
            'med_v1': 'llava_v1',
            'llava_med': 'llava_v1',
        }
        if conv_mode in alias_map:
            # 安静回退，不再输出 WARNING，以避免日志噪声
            return conv_templates[alias_map[conv_mode]].copy()
        fallback = 'llava_v1'
        logging.warning(f"conv-mode '{conv_mode}' not found; falling back to '{fallback}'.")
        return conv_templates[fallback].copy()
    except Exception as e:
        logging.warning(f"Failed to get conv template for '{conv_mode}' ({e}); using 'llava_v1'.")
        return conv_templates['llava_v1'].copy()
def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
    width1, height1 = image1.size
    width2, height2 = image2.size
    
    # Create new image with combined width and max height
    new_width = width1 + width2
    new_height = max(height1, height2)
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # Paste images
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    
    return new_image

def calculate_log_likelihood(model, tokenizer, image_processor, question, answer, image_path, conv_mode="llava_v1"):
    """
    Calculate the token-normalized log likelihood of an answer given an image and question.
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Prepare conversation
        conv = get_conv_template_safe(conv_mode)
        # 与 inference_visual_tie.py 保持一致：如果问题中已包含图像标记则不重复添加
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
            
            # 向量化计算答案令牌的对数似然，减少 Python 循环开销
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

def generate_model_answer(model, tokenizer, image_processor, question, image_path, conv_mode="llava_v1", max_new_tokens=512, disable_hf_generate=False):
    """
    Generate model's answer for a given question and image.
    """

    # Patch model.forward to ignore new HF 'cache_position' kwarg for older Llava models
    def _patch_forward_cache_position(m):
        try:
            # Avoid re-patching on every sample; patch only once
            if getattr(m, "_forward_cache_position_patched", False):
                return
            orig_forward = m.forward
            def forward_patched(*args, **kwargs):
                if 'cache_position' in kwargs:
                    kwargs.pop('cache_position', None)
                return orig_forward(*args, **kwargs)
            m.forward = forward_patched
            setattr(m, "_forward_cache_position_patched", True)
            logging.info("Patched model.forward (once) to drop 'cache_position'.")
        except Exception as e:
            logging.warning(f"Failed to patch model.forward: {e}")

    def greedy_generate_with_cache(model, input_ids, images, eos_token_id, max_new_tokens):
        """贪心生成：启用缓存以加速，每步仅前向一个新令牌。"""
        model.eval()
        past_key_values = None
        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    images=images,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values
                )
            logits = outputs.logits
            past_key_values = getattr(outputs, 'past_key_values', None)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids

    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Prepare conversation
        conv = get_conv_template_safe(conv_mode)
        # 与 inference_visual_tie.py 保持一致：如果问题中已包含图像标记则不重复添加
        if DEFAULT_IMAGE_TOKEN in question:
            conv.append_message(conv.roles[0], question)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # Ensure pad token id exists to avoid generation warnings/errors
        if getattr(tokenizer, 'pad_token_id', None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        with torch.inference_mode():
            # Generate answer with robust fallback
            # Avoid passing unsupported 'cache_position' to model.forward by disabling cache
            try:
                if hasattr(model, 'generation_config'):
                    model.generation_config.use_cache = False
            except Exception:
                pass
            # Additionally monkey-patch forward to drop 'cache_position'
            _patch_forward_cache_position(model)
            images_batched = image_tensor.unsqueeze(0).half().cuda()
            # 如果 HF generate 在首次尝试中失败，则在模型对象上设置一次性的强制贪心标记，避免后续样本重复报错
            force_greedy = getattr(model, "_force_greedy", False) or disable_hf_generate
            if force_greedy:
                logging.info("Greedy generation is enabled (HF generate disabled or previously failed).")
                output_ids = greedy_generate_with_cache(
                    model,
                    input_ids=input_ids,
                    images=images_batched,
                    eos_token_id=getattr(tokenizer, 'eos_token_id', None),
                    max_new_tokens=max_new_tokens
                )
            else:
                try:
                    output_ids = model.generate(
                        input_ids,
                        images=images_batched,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=False,
                        pad_token_id=getattr(tokenizer, 'pad_token_id', None) or tokenizer.eos_token_id,
                        eos_token_id=getattr(tokenizer, 'eos_token_id', None)
                    )
                except Exception as e:
                    logging.warning(f"HF generate failed ({e}); falling back to greedy generation.")
                    try:
                        setattr(model, "_force_greedy", True)
                    except Exception:
                        pass
                    output_ids = greedy_generate_with_cache(
                        model,
                        input_ids=input_ids,
                        images=images_batched,
                        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
                        max_new_tokens=max_new_tokens
                    )
            
            # Decode generated tokens (excluding input)
            generated_tokens = output_ids[0][len(input_ids[0]):]
            generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return generated_answer
            
    except Exception as e:
        logger.error(f"Error generating model answer: {e}")
        return ""

def calculate_visual_consistency_tie_for_sample(model, tokenizer, image_processor, sample, full_image_folder, masked_image_folder, output_image_folder, conv_mode, max_new_tokens, disable_hf_generate=False, save_images=False):
    """
    Calculate Visual Consistency TIE for a single sample.
    Method 2: Compare ground-truth vs. generated answers on same visual input.
    
    TIE Metrics:
    - Δ+ = LL(y_gt|x,I_full) - LL(y_gt|x,I_bg)  # Foreground contribution to correct answer
    - Δ- = LL(y_gen|x,I_full) - LL(y_gen|x,I_bg) # Foreground contribution to generated answer  
    - γ = Δ+ - Δ-                                 # Net visual support effect
    - m_v = LL(y_gt|x,I_full) - LL(y_gen|x,I_full) # Full visual discrimination
    - m_n = LL(y_gt|x,I_bg) - LL(y_gen|x,I_bg)     # Background bias
    """
    try:
        # Extract sample information
        if "qid" in sample:
            qid = sample["qid"]
        elif "id" in sample:
            qid = sample["id"]
        else:
            raise KeyError("Sample must have 'qid' or 'id' field")
        
        # Extract question and ground truth answer
        if "question" in sample:
            original_question = sample["question"]
            gt_answer = sample.get("answer", "")
        elif "conversations" in sample:
            conversations = sample["conversations"]
            if conversations and len(conversations) > 0:
                # 不再移除图像标记，交由下游函数根据是否包含 DEFAULT_IMAGE_TOKEN 判定
                original_question = conversations[0].get("value", "").strip()
                if len(conversations) > 1:
                    gt_answer = conversations[1].get("value", "").strip()
                else:
                    gt_answer = ""
            else:
                original_question = ""
                gt_answer = ""
        else:
            original_question = ""
            gt_answer = ""
        
        # Get image paths
        image_name = sample.get("image", "")
        if not image_name:
            logger.warning(f"No image found for sample {qid}")
            return None
            
        full_image_path = os.path.join(full_image_folder, image_name)
        masked_image_path = os.path.join(masked_image_folder, image_name)
        
        if not os.path.exists(full_image_path):
            logger.warning(f"Full image not found: {full_image_path}")
            return None
        if not os.path.exists(masked_image_path):
            logger.warning(f"Masked image not found: {masked_image_path}")
            return None
        
        # Step 1: Generate model answer on full image
        generated_answer = generate_model_answer(
            model, tokenizer, image_processor,
            original_question, full_image_path,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            disable_hf_generate=disable_hf_generate
        )
        
        if not generated_answer:
            logger.warning(f"Failed to generate answer for sample {qid}")
            return None
        
        # Step 2: Calculate log likelihoods
        # LL(y_gt|x,I_full)
        ll_gt_full = calculate_log_likelihood(model, tokenizer, image_processor, original_question, gt_answer, full_image_path, conv_mode=conv_mode)
        
        # LL(y_gt|x,I_bg)  
        ll_gt_bg = calculate_log_likelihood(model, tokenizer, image_processor, original_question, gt_answer, masked_image_path, conv_mode=conv_mode)
        
        # LL(y_gen|x,I_full)
        ll_gen_full = calculate_log_likelihood(model, tokenizer, image_processor, original_question, generated_answer, full_image_path, conv_mode=conv_mode)
        
        # LL(y_gen|x,I_bg)
        ll_gen_bg = calculate_log_likelihood(model, tokenizer, image_processor, original_question, generated_answer, masked_image_path, conv_mode=conv_mode)
        
        # Step 3: Calculate TIE metrics
        delta_pos = ll_gt_full - ll_gt_bg      # Δ+: Foreground contribution to correct answer
        delta_neg = ll_gen_full - ll_gen_bg    # Δ-: Foreground contribution to generated answer
        gamma = delta_pos - delta_neg          # γ: Net visual support effect
        m_v = ll_gt_full - ll_gen_full         # m_v: Full visual discrimination
        m_n = ll_gt_bg - ll_gen_bg             # m_n: Background bias
        
        # Step 4: (Optional) Create stitched image for visualization
        if save_images and output_image_folder:
            os.makedirs(output_image_folder, exist_ok=True)
            try:
                full_image = Image.open(full_image_path).convert('RGB')
                masked_image = Image.open(masked_image_path).convert('RGB')
                stitched_image = stitch_images_side_by_side(full_image, masked_image)
                
                output_image_path = os.path.join(output_image_folder, f"{qid}_visual_consistency.jpg")
                stitched_image.save(output_image_path)
            except Exception as e:
                logger.warning(f"Failed to create stitched image for {qid}: {e}")
        
        # Return results
        result = {
            "qid": qid,
            "question": original_question,
            "gt_answer": gt_answer,
            "generated_answer": generated_answer,
            "image": image_name,
            "ll_gt_full": ll_gt_full,
            "ll_gt_bg": ll_gt_bg,
            "ll_gen_full": ll_gen_full,
            "ll_gen_bg": ll_gen_bg,
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "gamma": gamma,
            "m_v": m_v,
            "m_n": m_n,
            "method": "visual_consistency",
            "comparison_type": "gt_vs_generated"
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
    parser.add_argument("--conv-mode", type=str, default="llava_med_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--disable-hf-generate", action="store_true")
    parser.add_argument("--save-images", action="store_true", help="保存拼接可视化图 (默认关闭以加速)")
    args = parser.parse_args()
    # 允许通过环境变量控制贪心回退，以便 shell 脚本无需改动即可启用
    env_disable = os.environ.get("DISABLE_HF_GENERATE", "").strip().lower()
    if env_disable in ("1", "true", "yes"):  # 与脚本环境变量约定保持一致
        args.disable_hf_generate = True

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
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.output_image_folder, exist_ok=True)
    
    # Process samples
    results = []
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Split data across processes
    samples_per_rank = len(questions) // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank < world_size - 1 else len(questions)
    
    local_questions = questions[start_idx:end_idx]
    
    logger.info(f"Rank {rank}: Processing {len(local_questions)} samples ({start_idx}-{end_idx-1})")
    
    for sample in tqdm(local_questions, desc=f"Computing Visual Consistency TIE (Rank {rank})"):
        result = calculate_visual_consistency_tie_for_sample(
            model, tokenizer, image_processor, sample,
            args.full_image_folder, args.masked_image_folder, args.output_image_folder,
            conv_mode=args.conv_mode, max_new_tokens=args.max_new_tokens, disable_hf_generate=args.disable_hf_generate,
            save_images=args.save_images
        )
        if result:
            results.append(result)
    
    # Save results
    output_file = args.output_file
    if world_size > 1:
        output_file = f"{args.output_file}.rank{rank}"
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(tensor_to_serializable(result)) + '\n')
    
    logger.info(f"Rank {rank}: Saved {len(results)} results to {output_file}")
    
    # Cleanup
    cleanup()
    
    # Merge results from all ranks
    if rank == 0 and world_size > 1:
        logger.info("Merging results from all ranks...")
        all_results = []
        for r in range(world_size):
            rank_file = f"{args.output_file}.rank{r}"
            if os.path.exists(rank_file):
                with open(rank_file, 'r') as f:
                    for line in f:
                        all_results.append(json.loads(line.strip()))
                os.remove(rank_file)
        
        # Save merged results
        with open(args.output_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Merged {len(all_results)} total results to {args.output_file}")

if __name__ == "__main__":
    main()