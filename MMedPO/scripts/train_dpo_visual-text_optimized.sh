#!/bin/bash

# 优化的DPO训练脚本
# 主要改进：
# 1. 使用改进后的数据集
# 2. 优化内存使用
# 3. 调整batch size和gradient accumulation

export CUDA_VISIBLE_DEVICES=1,2,3
CUDA=1,2,3

echo "=== 开始DPO训练 ==="
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "数据集: tie_dpo_dataset_improved.jsonl"
echo "========================"

# 检查数据集是否存在
if [ ! -f "/workspace/MMedPO/outputs/tie_dpo_dataset_improved.jsonl" ]; then
    echo "错误: 改进的数据集文件不存在，请先运行数据集生成脚本"
    exit 1
fi

# 检查图像文件夹是否存在
if [ ! -d "/workspace/MMedPO/outputs/tie_results_1/composites" ]; then
    echo "错误: 图像文件夹不存在"
    exit 1
fi

cd ../train/dpo || exit

# 使用更保守的内存设置
deepspeed --include localhost:$CUDA --master_port $((RANDOM + 30000)) ./train_dpo_weighted.py \
    --model_name_or_path /workspace/llava-med-v1.5-mistral-7b \
    --lora_checkpoint_path /workspace/MMedPO/MMedPO/checkpoints/sft_model_lora \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --loss_use_weight True \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --data_path /workspace/MMedPO/outputs/tie_dpo_dataset_improved.jsonl \
    --image_folder /workspace/MMedPO/outputs/tie_results_1/composites \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /workspace/MMedPO/MMedPO/checkpoints/dpo_model_improved \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-8 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to wandb \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --remove_unused_columns False

echo "=== DPO训练完成 ==="