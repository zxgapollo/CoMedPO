#!/bin/bash

model_base="/workspace/llava-med-v1.5-mistral-7b"
model_path=/workspace/MMedPO/reproduce_checkpoints/DPO+SFT
model_path_basename=$(basename $model_path)
image_folder=/workspace/MMedPO/datasets/Slake1.0/imgs

question_file=/workspace/MMedPO/data/slake_questions.jsonl
answer_file=/workspace/MMedPO/outputs/${model_path_basename}_dpo_weighted.json

# Create output directory if it doesn't exist
mkdir -p /workspace/MMedPO/outputs

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=$((RANDOM + 10000)) /workspace/MMedPO/MMedPO/inference/llava-med-1.5_vqa.py \
    --model-base $model_base \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answer_file \
    --temperature 0.2






