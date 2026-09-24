#!/bin/bash

# DPO (Direct Preference Optimization) Training Script
# Based on the analysis of existing DPO training scripts

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_PROJECT="MMedPO_DPO"

# Training parameters
SFT_MODEL_PATH="/workspace/llava-med-v1.5-mistral-7b"
DATA_PATH="/workspace/MMedPO/outputs/tie_dpo_dataset_improved.json"
IMAGE_FOLDER="/workspace/MMedPO/outputs/tie_results_1/composites"
OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/dpo_model"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run DPO training
deepspeed /workspace/MMedPO/MMedPO/train/dpo/train_dpo_weighted.py \
    --deepspeed /workspace/MMedPO/MMedPO/train/dpo/scripts/zero3_simple.json \
    --model_name_or_path $SFT_MODEL_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower /workspace/CLIP/clip-vit-l14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "dpo_training_$(date +%Y%m%d_%H%M%S)" \
    --remove_unused_columns False \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_weight_path /workspace/MMedPO/MMedPO/checkpoints/sft_model_lora \
    --loss_use_weight True \
    --lora_bias "none" \
    --mm_projector_lr 2e-5

echo "DPO training completed. Model saved to: $OUTPUT_DIR"