import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import logging
import traceback

from utils import setup, cleanup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stitch_images_side_by_side(image1, image2):
    new_height = max(image1.height, image2.height)
    if image1.height != image2.height:
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))
    new_width = image1.width + image2.width
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    return new_image


def build_inputs(processor, messages, image):
    # 先用聊天模板生成字符串，再将 <start_of_image> 替换为处理器识别的 boi_token
    chat_text = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # 将模板中的占位符替换为 Gemma3Processor 识别的图像起始 token
    boi = getattr(processor, "boi_token", None)
    if boi is None:
        # 兼容性：从 tokenizer 读取
        boi = getattr(processor.tokenizer, "boi_token", None)
    if boi is None:
        raise RuntimeError("Processor/tokenizer missing boi_token; cannot insert image placeholder")
    chat_text = chat_text.replace("<start_of_image>", boi)
    # 兜底：若字符串中没有任何 boi，占位，则在用户轮次起始后插入一个
    if boi not in chat_text:
        insert_key = "<start_of_turn>user\n"
        if insert_key in chat_text:
            pos = chat_text.find(insert_key) + len(insert_key)
            chat_text = chat_text[:pos] + (boi + "\n") + chat_text[pos:]
        else:
            # 如果未找到用户起始标记，则直接在开头插入
            chat_text = boi + "\n" + chat_text

    # 使用 Gemma3Processor 同时编码文本与图像（它会将 boi_token 展开为完整图像序列）
    encoded = processor(text=[chat_text], images=[image], return_tensors="pt")
    # BatchFeature -> dict 提取
    if hasattr(encoded, "data"):
        data = encoded.data
    elif isinstance(encoded, dict):
        data = encoded
    else:
        # 兜底：尝试属性访问
        data = {k: getattr(encoded, k) for k in getattr(encoded, "keys", lambda: [])()}

    # 规范化输出键
    input_ids = data["input_ids"]
    attention_mask = data.get("attention_mask", None)
    out = {"input_ids": input_ids}
    if attention_mask is not None:
        out["attention_mask"] = attention_mask
    # 图像特征键可能为 pixel_values
    if "pixel_values" in data:
        out["pixel_values"] = data["pixel_values"]
    elif "images" in data:
        out["pixel_values"] = data["images"]
    else:
        raise RuntimeError("Processor output missing image features (pixel_values/images)")
    # 传递 token_type_ids（多模态标记类型）以提升稳健性
    if "token_type_ids" in data:
        out["token_type_ids"] = data["token_type_ids"]
    return out


def calculate_log_likelihood(model, processor, question, answer, image):
    # 通过聊天模板构建消息：图像占位符由模板插入，真实图像通过 images 传入
    qtext = question or ""
    if not isinstance(qtext, str):
        qtext = str(qtext)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qtext}]},
    ]
    try:
        inputs = build_inputs(processor, messages, image)
    except Exception as e:
        logger.warning(f"build_inputs failed: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # Encode the ground-truth answer using the processor's tokenizer
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids

    # Clamp answer tokens to vocab range
    vocab_size = None
    try:
        vocab_size = getattr(getattr(model, 'config', None), 'text_config', None).vocab_size
    except Exception:
        vocab_size = None
    if vocab_size is None and hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    if vocab_size is not None:
        answer_ids = torch.clamp(answer_ids, min=0, max=vocab_size - 1)
    else:
        answer_ids = torch.maximum(answer_ids, torch.zeros_like(answer_ids))

    # Move tensors to device
    device = model.device
    moved_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values", "images"):
                moved_inputs[k] = v.to(device=device)
            elif v.is_floating_point():
                moved_inputs[k] = v.to(device=device, dtype=model.dtype)
            else:
                moved_inputs[k] = v.to(device=device)
        else:
            moved_inputs[k] = v
    inputs = moved_inputs
    input_ids = input_ids.to(device=device)
    answer_ids = answer_ids.to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    # Incremental scoring using past_key_values to avoid training path
    with torch.no_grad():
        # 首次前向：传入图像特征与完整上下文
        kwargs_first = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **kwargs_first)
        past_key_values = outputs.past_key_values
        logits = outputs.logits  # [b, seq, vocab]
        # Start from last context token
        log_likelihood = 0.0
        bsz = input_ids.shape[0]
        # First token prob from last logits
        last_logits = logits[:, -1, :]
        if vocab_size is not None and last_logits.shape[-1] < vocab_size:
            # In case logits dim is smaller (should not), skip
            pass
        probs = torch.log_softmax(last_logits.to(dtype=torch.float32), dim=-1)
        log_likelihood += probs.gather(-1, answer_ids[:, 0:1]).squeeze(-1).mean().item()
        # Process remaining answer tokens
        cur_ids = answer_ids[:, 0:1]
        for t in range(1, answer_ids.shape[1]):
            next_id = answer_ids[:, t:t+1]
            # 增量步骤：不再传入图像特征，避免 tokens/features 不匹配
            out = model(input_ids=cur_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            step_logits = out.logits[:, -1, :]
            step_probs = torch.log_softmax(step_logits.to(dtype=torch.float32), dim=-1)
            log_likelihood += step_probs.gather(-1, next_id).squeeze(-1).mean().item()
            cur_ids = next_id

    total_tokens = max(1, int(answer_ids.shape[1]))
    return float(log_likelihood) / float(total_tokens)


def process_sample(model, processor, sample, full_image_folder, masked_image_folder):
    qid = sample.get("id") or sample.get("qid") or sample.get("question_id")

    if "conversations" in sample and len(sample["conversations"]) >= 2:
        question = sample["conversations"][0].get("value")
        gt_answer = sample["conversations"][1].get("value")
    else:
        question = sample.get("question")
        gt_answer = sample.get("answer")

    # 强制字符串化以避免 tokenizer/re 模块报错
    question = "" if question is None else (question if isinstance(question, str) else str(question))
    gt_answer = "" if gt_answer is None else (gt_answer if isinstance(gt_answer, str) else str(gt_answer))

    image_filename = sample.get("image")
    if isinstance(image_filename, (list, tuple)) and image_filename:
        image_filename = image_filename[0]
    # full image uses dataset-provided filename
    full_image_path = os.path.join(full_image_folder, image_filename)
    # masked/background-only image follows processed naming: source_reversed_mask.jpg
    dir_name = os.path.dirname(image_filename) if isinstance(image_filename, str) else ""
    masked_image_path = os.path.join(masked_image_folder, dir_name, "source_reversed_mask.jpg")
    # fallback: if reversed mask not found, try processed source.jpg; else try original mask_plus_full
    if not os.path.exists(masked_image_path):
        alt1 = os.path.join(masked_image_folder, image_filename) if isinstance(image_filename, str) else None
        alt2 = os.path.join(full_image_folder, dir_name, "source_mask_plus_full.jpg") if isinstance(image_filename, str) else None
        if alt1 and os.path.exists(alt1):
            masked_image_path = alt1
        elif alt2 and os.path.exists(alt2):
            masked_image_path = alt2
    if not os.path.exists(full_image_path) or not os.path.exists(masked_image_path):
        logger.warning(f"Missing image paths for sample {qid}: full={full_image_path}, masked={masked_image_path}")
        return None

    full_img = Image.open(full_image_path).convert('RGB')
    bg_img = Image.open(masked_image_path).convert('RGB')
    null_img = Image.new('RGB', full_img.size, color=(128, 128, 128))

    combined_pref = stitch_images_side_by_side(full_img, bg_img)
    combined_disp = stitch_images_side_by_side(null_img, bg_img)

    ll_pref = calculate_log_likelihood(model, processor, question, gt_answer, combined_pref)
    ll_disp = calculate_log_likelihood(model, processor, question, gt_answer, combined_disp)

    visual_tie = ll_pref - ll_disp

    ll_full = calculate_log_likelihood(model, processor, question, gt_answer, full_img)
    ll_bg = calculate_log_likelihood(model, processor, question, gt_answer, bg_img)
    return {
        "qid": qid,
        "question": question,
        "gt_answer": gt_answer,
        "image": image_filename,
        "full_image_path": full_image_path,
        "masked_image_path": masked_image_path,
        "ll_pref": ll_pref,
        "ll_disp": ll_disp,
        "ll_full": ll_full,
        "ll_bg": ll_bg,
        "visual_tie": visual_tie,
        "tie_positive": visual_tie if visual_tie > 0 else 0.0,
        "tie_negative": abs(visual_tie) if visual_tie < 0 else 0.0,
        "tie_difference": visual_tie,
        "method": "visual_indirect_medgemma",
        "scoring_target": "y_gt",
        "tie_formula": "(y_gt|X \u2295 X_bg) > (y_gt|X_null \u2295 X_bg)",
        "pref_condition": "full_image",
        "disp_condition": "background_only",
        "conv_mode_used": "llava_v1",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--output-image-folder", type=str, required=True)
    parser.add_argument("--full-image-folder", type=str, required=True)
    parser.add_argument("--masked-image-folder", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    setup()

    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except Exception as e:
        raise RuntimeError("Transformers is required for MedGemma. Please install 'transformers'.") from e

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.question_file, 'r') as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    task_mode = os.environ.get('TASK_MODE', 'model_shard')
    if world_size > 1 and task_mode == 'data_shard':
        questions = [q for i, q in enumerate(questions) if i % world_size == rank]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(args.output_image_folder, exist_ok=True)

    results = []
    bs = max(1, args.batch_size)
    for i in tqdm(range(0, len(questions), bs), desc="MedGemma Visual TIE"):
        batch = questions[i:i+bs]
        for sample in batch:
            try:
                r = process_sample(model, processor, sample, args.full_image_folder, args.masked_image_folder)
                if r is not None:
                    results.append(r)
            except Exception as e:
                sid = sample.get('id','?')
                logger.warning(f"Failed sample {sid}: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())

    out_file = args.output_file if (world_size == 1 or os.environ.get('TASK_MODE', 'model_shard') != 'data_shard') else args.output_file.replace('.jsonl', f'.rank{rank}.jsonl')
    with open(out_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cleanup()


if __name__ == "__main__":
    main()