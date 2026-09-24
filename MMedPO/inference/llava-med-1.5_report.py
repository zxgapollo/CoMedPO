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
# --- 这是最终的、正确的解决方案 ---
# `llava` 模块的源代码位于 MMedPO 项目的一个子目录中。
# 我们需要将这个子目录的父目录添加到 Python 路径。
correct_path = '/workspace/MMedPO/train/dpo'

# 检查一下，避免重复添加
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)
# --- 代码添加结束 ---
import llava.model.language_model.llava_mistral
warnings.simplefilter(action='ignore', category=FutureWarning)
# sys.path.append("/workspace/llava-med-v1.5-mistral-7b")
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
# sys.path.append("/workspace/MMedPO")
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable
from itertools import islice
import numpy as np
import debugpy
# rank = int(os.getenv('RANK', '0'))
# port = 5678 + rank  # 基础端口 + 进程ID

# debugpy.listen(port)
# print(f"Process {rank} waiting for debugger to attach on port {port}...")
# debugpy.wait_for_client()

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
# --- 这是最终的、正确的修复代码 ---
# 检查图像处理器是否加载失败
    if image_processor is None:
        print("[!] Warning: Image processor was not loaded. "
            "Loading it from the original vision tower path...")
        from transformers import CLIPImageProcessor
        
        # LLaVA 1.5 使用的视觉编码器是 'openai/clip-vit-large-patch14-336'
        # 我们直接从这个官方路径加载正确的图像处理器
        vision_tower_path = 'openai/clip-vit-large-patch14-336'
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)
        
        # 将其连接到主模型上，以确保后续代码能正确找到它
        # 注意：这里的 vision_tower 结构可能需要根据您的模型进行调整
        # 但通常是这样的
        if hasattr(model, 'vision_tower'):
            # 假设 vision_tower 是一个列表或模块列表
            # 我们需要找到正确的位置来附加它
            # 对于 LLaVA，通常是 vision_tower 的第一个元素
            model.vision_tower[0].image_processor = image_processor
        elif hasattr(model.config, 'mm_vision_tower'):
            # 兼容另一种配置方式
            model.config.mm_vision_tower_processor = image_processor
    # --- 修复代码结束 ---

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    # drop_last=False
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    local_results = []

    prompt_choices = [
        '<image>\nWalk through the important details of the image',
        '<image>\nClarify the contents of the displayed image with great detail',
        "<image>\nExamine the image closely and share its details",
        "<image>\nClarify the contents of the displayed image with great detail",
        "<image>\nAnalyze the image in a comprehensive and detailed manner"
        
        ]

    import random
    random.seed(42)
    # for line in tqdm(islice(dataloader, 10)):
    for line in tqdm(dataloader):
        # idx = line["id"][0]
        idx = line["id"]
        # image_file = line["image_path"][0]
        image_file = line.get("image_path", line.get("image"))[0]
        # # Get the list of image paths
        # image_paths = line.get("image_path", line.get("image"))
        # # Select the first image from the list
        # image_file = image_paths[0]
        # gt_answer = line["answer"][0]
        # gt_report=line['gt_report'][0]
        gt_report=line['report']
        # qs = line["text"][0].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        
        # qs=line["question"][0]
        qs=random.choice(prompt_choices)
        if "reference_reports" not in line:
            
            # continue
            # suffix="You are a professional ophthalmologist. Please generate a report based on the fundus image and only include the content of the report in your response."
            # cur_prompt = qs+' '+suffix
            #  The response should start with 'Generated report:', then the generated report.
            # cur_prompt = suffix
            # qs=cur_prompt
            cur_prompt = qs
            
        else:
            
            reference_report=line["reference_reports"]
            # print(reference_report)
            if not isinstance(reference_report, list):
                topk=1
                reference_report=[reference_report]
                formatted_reference_report=reference_report[0]
            else:
                topk=len(reference_report)
                formatted_reference_report=""
                for i in range(topk):
                    formatted_reference_report += f"{i + 1}. {reference_report[i][0]} "
                # print(formatted_reference_report)
            # cur_prompt = qs
            appendix_1=f"Please {qs}. You are also provided with a chest image and {topk} reference report(s): "
            # lower letter of qs
            qs=qs.lower()
            appendix_2=f"It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference and comparison.\nPlease {qs}."
            # Please only include the content of the report in your response.
            cur_prompt = appendix_1 +formatted_reference_report +"\n" + appendix_2
            # print(cur_prompt)
            qs=cur_prompt
        qs=qs.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            # tqdm.write(f"Prompt: {qs}")
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
            image = add_gaussian_noise(image, 0, 1)
            tqdm.write(f"Add noise to image {image_file}")
        # print(image)
        image_tensor = process_images([image], image_processor, model.config)[0]

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
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        tqdm.write(outputs)
        

        ans_id = shortuuid.uuid()
        if "reference_report" not in line:
            result = {
                "id": idx,
                "prompt": cur_prompt,
                "image":image_file,
                "answer": outputs,
                "gt_report":gt_report,
                # "gt_answer": gt_answer,
                "noised_image":1 if args.noised_image else 0,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {},
            }
            
        else:
            result = {
                "id": idx,
                "prompt": cur_prompt,
                "image":image_file,
                "answer": outputs,
                "gt_report":gt_report,
                "reference_report":reference_report,
                # "gt_answer": gt_answer,
                "noised_image":1 if args.noised_image else 0,
                "answer_id": ans_id,
                "model_id": model_name,
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
        # all_results.sort(key=lambda x: x["id"])
        unique_results = []
        seen_ids = set()
        for result in all_results:
            # if result["id"] not in seen_ids:
            #     unique_results.append(result)
            #     seen_ids.add(result["id"])
                # --- 开始修改 ---
            # 假设 id 被错误地包装成了列表，例如 [123]
            # 我们需要提取出列表中的第一个元素
            q_id = result["id"]
            if isinstance(q_id, list):
                q_id = q_id[0]  # 提取列表的第一个元素
            # --- 结束修改 ---

            # 使用处理后的 q_id 进行去重检查
            if q_id not in seen_ids:
                seen_ids.add(q_id)
                unique_results.append(result)
        # ... (后续写入文件的代码)
        unique_results.sort(key=lambda x: x["id"])
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
    parser.add_argument("--noised_image", action='store_true', help="If set, use noised images.")

    args = parser.parse_args()

    eval_model(args)


if __name__ == "__main__":
    main()
