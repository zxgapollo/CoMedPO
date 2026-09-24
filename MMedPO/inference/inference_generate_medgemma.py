#!/usr/bin/env python3
"""
Generate answers with MedGemma (AutoModelForImageTextToText).
Inputs: question JSON (array), full_image_folder, model_path
Outputs: answers jsonl with fields {id, prompt, answer, gt_answer, image}
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--question-file", required=True)
    p.add_argument("--answers-file", required=True)
    p.add_argument("--full-image-folder", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def build_messages(question_text: str):
    system = {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]}
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]}
    return [system, user]


def main():
    args = parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )

    with open(args.question_file, "r", encoding="utf-8") as fr:
        questions = json.load(fr)

    out_dir = os.path.dirname(args.answers_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    for sample in tqdm(questions, desc="Generating"):
        # extract question and image path
        if "conversations" in sample and isinstance(sample.get("conversations"), list) and len(sample["conversations"]) >= 2:
            question = sample["conversations"][0].get("value", "")
            gt_answer = sample["conversations"][1].get("value", "")
        else:
            question = sample.get("question", "")
            gt_answer = sample.get("answer", "")
        image_filename = sample.get("image") or sample.get("full_image_path") or sample.get("image_path")
        if isinstance(image_filename, list):
            image_filename = image_filename[0] if image_filename else None
        if image_filename is None:
            continue
        # normalize path
        full_image_path = os.path.join(args.full_image_folder, image_filename) if not os.path.isabs(image_filename) else image_filename
        if not os.path.exists(full_image_path):
            # try basename fallback
            cand = os.path.join(args.full_image_folder, os.path.basename(image_filename))
            if os.path.exists(cand):
                full_image_path = cand
            else:
                print(f"[warn] image not found: {full_image_path}")
                continue

        # build messages and construct chat text via tokenizer.apply_chat_template to match Gemma expectations
        messages = build_messages(question)
        chat_text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # replace placeholder with processor boi token
        boi = getattr(processor, "boi_token", None) or getattr(processor.tokenizer, "boi_token", None)
        if boi is None:
            raise RuntimeError("Processor/tokenizer missing boi_token; cannot insert image placeholder")
        chat_text = chat_text.replace("<start_of_image>", boi)

        from PIL import Image
        img = Image.open(full_image_path).convert("RGB")
        inputs = processor(text=[chat_text], images=[img], return_tensors="pt")
        # move tensors to model device
        device = next(model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device=device)

        # generate
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=(args.temperature > 0.0), temperature=args.temperature)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        # decode
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        result = {"id": sample.get("id") or sample.get("qid"), "image": image_filename, "prompt": chat_text, "answer": answer, "gt_answer": gt_answer}
        results.append(result)

    # write jsonl
    with open(args.answers_file, "w", encoding="utf-8") as fw:
        for r in results:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} answers to {args.answers_file}")


if __name__ == "__main__":
    main()

