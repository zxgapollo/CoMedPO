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

# 保证能 import llava_mistral
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)
import llava.model.language_model.llava_mistral  # noqa

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


def generate_gaussian_noise(image, mean=0.0, stddev=0.1):
    """给图片加高斯噪声（测试鲁棒性用）"""
    image_shape = np.array(image).shape
    noise = np.random.normal(mean, stddev, image_shape).astype(np.float32)
    noise_clipped = np.clip(noise, 0.0, 1.0)
    noisy_image = Image.fromarray((noise_clipped * 255).astype(np.uint8))
    return noisy_image


def eval_model(args):
    # 检测是否处于分布式环境
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        set_seed(42 + rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        torch.cuda.set_device(rank)
        set_seed(42)
        distributed = False
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    gpu_id = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{gpu_id}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        torch_dtype=torch.float16,
        load_8bit=False,
        load_4bit=False,
        device_map=None,  # 手动控制设备分配
    )
    model.to(device)
    
    # 启用梯度检查点以节省内存
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # 清理缓存
    torch.cuda.empty_cache()
    
    print(f"[RANK {rank}] using GPU {gpu_id}")
    print(f"[RANK {rank}] GPU memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")
    print(f"[RANK {rank}] GPU memory reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**3:.2f} GB")

    if image_processor is None:
        if rank == 0:
            print("Image processor not loaded; using default CLIP...")
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        if hasattr(model, "get_vision_tower"):
            model.get_vision_tower().image_processor = image_processor

    # 每个进程读同一份 question 文件
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    total = len(questions)
    if distributed:
        per_rank = (total + world_size - 1) // world_size
        start = rank * per_rank
        end   = min((rank + 1) * per_rank, total)
        sub_questions = questions[start:end]
        print(f"[RANK {rank}] samples {start}-{end-1}/{total}  (local {len(sub_questions)})")
    else:
        sub_questions = questions

    dataset = QuestionDataset(sub_questions)
    sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=2,  # 减少worker数量以节省内存
        pin_memory=False,  # 关闭pin_memory以节省内存
    )

    local_results = []
    for line in tqdm(dataloader, desc=f"GPU{rank}", position=rank):
        idx = line["qid"][0]
        # 兼容数据集中没有 full_image_path 的情况：从 img_name 和 --image-root 构建
        img_name = line.get("img_name", [None])[0]
        full_image_path = line.get("full_image_path", [None])[0]
        if full_image_path is None:
            if img_name is None:
                tqdm.write(f"GPU{rank}: missing img_name for sample {idx}")
                continue
            full_image_path = os.path.join(args.image_root, img_name)
        gt_answer = line.get("positive_answer", [None])[0]
        answer_type = line.get("answer_type", ["unknown"])[0]
        original_question = line["question"][0].strip()

        qs = original_question
        cur_prompt = (
            "The following image displays a full medical scan on the left and "
            "a version with a masked background on the right. "
            "Based on this composite image, answer the following question.\n"
            f"{DEFAULT_IMAGE_TOKEN}\nQuestion: {original_question}"
        )

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
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(rank)
        )
        try:
            image = Image.open(full_image_path).convert("RGB")
        except FileNotFoundError:
            tqdm.write(f"GPU{rank}: image not found {full_image_path}")
            continue

        if args.noised_image:
            image = generate_gaussian_noise(image, 0, 1)

        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(torch.device(f"cuda:{rank}")),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
            )
        print(f"DEBUG: input_ids shape: {input_ids.shape}")
        print(f"DEBUG: output_ids shape: {output_ids.shape}")
        print(f"DEBUG: Decoded output_ids: {tokenizer.decode(output_ids[0])}")

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()

        # 清理中间变量以释放内存
        del input_ids, image_tensor, output_ids
        torch.cuda.empty_cache()

        result = {
            "id": idx,
            "image_path": full_image_path,
            "prompt": cur_prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "answer_id": ans_id,
            "model_id": model_name,
            "noised_image": 1 if args.noised_image else 0,
            "answer_type": answer_type,
            "metadata": {},
        }
        local_results.append(tensor_to_serializable(result))

    if distributed:
        # 汇总结果（仅 rank0 写文件）
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        if rank == 0:
            all_results = [item for sublist in gathered_results for item in sublist]
        else:
            all_results = []
    else:
        all_results = local_results

    if rank == 0:
        unique_results = {res["id"]: res for res in all_results}.values()
        unique_results = sorted(unique_results, key=lambda x: x["id"])
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        with open(answers_file, "w") as f:
            for res in unique_results:
                f.write(json.dumps(res) + "\n")
        print(f"Finished writing {len(unique_results)} results to {answers_file}")

    if distributed:
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--noised_image", action="store_true")
    parser.add_argument("--image-root", type=str, default="/workspace/MMedPO/datasets/SLAKE/imgs")
    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()