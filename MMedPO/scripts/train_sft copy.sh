#!/bin/bash

# SFT (Supervised Fine-Tuning) Training Script with LoRA
# Based on the pipeline script configuration

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_PROJECT="MMedPO_SFT"

# Set offline mode for Hugging Face Hub to avoid network issues
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optimize CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set PYTHONPATH to ensure llava module can be imported
export PYTHONPATH="/workspace/MMedPO/MMedPO/train/dpo:$PYTHONPATH"

# Training parameters
MODEL_NAME="/workspace/llava-med-v1.5-mistral-7b"
DPO_DATA_PATH="/workspace/MMedPO/data/tie_dpo_dataset_VQA.json"
SFT_DATA_PATH="/workspace/MMedPO/data/tie_sft_dataset_VQA_fixed.json"
IMAGE_FOLDER="/workspace/MMedPO/datasets/VQA_RAD/VQA_RAD_Image_Folder"
OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/sft_model_lora_vqa-rad"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create SFT format data from DPO format data with correct image paths
echo "Converting DPO format data to SFT format with correct image paths..."
python3 -c "
import json

# Load original VQA_RAD dataset to get correct image names
with open('/workspace/MMedPO/datasets/VQA_RAD/VQA_RAD_Dataset_Public.json', 'r') as f:
    original_data = json.load(f)

# Build qid to image_name mapping
qid_to_image = {}
for item in original_data:
    qid = int(item['qid'])
    image_name = item['image_name']
    qid_to_image[qid] = image_name

# Load DPO format data
with open('$DPO_DATA_PATH', 'r') as f:
    dpo_data = json.load(f)

# Convert to SFT format with correct image paths
sft_data = []
for item in dpo_data:
    qid = item['id']
    # Use correct image name from original dataset
    correct_image_name = qid_to_image.get(qid, item['image'])
    
    sft_item = {
        'id': item['id'],
        'image': correct_image_name,
        'conversations': item['conversations']  # Only use chosen conversations for SFT
    }
    sft_data.append(sft_item)

# Save SFT format data
with open('$SFT_DATA_PATH', 'w') as f:
    json.dump(sft_data, f, indent=2)

print(f'Created SFT format data with {len(sft_data)} samples')
print(f'Saved to: $SFT_DATA_PATH')
print(f'Fixed image paths using original VQA_RAD dataset mapping')
"

# Change to the training directory
cd /workspace/MMedPO/MMedPO/train/dpo

# Run SFT training with LoRA using torchrun
echo "Starting SFT training with LoRA configuration..."
echo "Dataset: $SFT_DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "LoRA Configuration: r=128, alpha=256, dropout=0.05"

torchrun --nproc_per_node=3 --master_port=$((RANDOM + 30000)) llava/train/train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --version v1 \
    --data_path $SFT_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_weight_path "" \
    --lora_bias "none" \
    --mm_projector_lr 2e-5 \
    --bits 16 \
    --double_quant True \
    --quant_type nf4

# Check if training completed successfully
if [[ $? -eq 0 ]]; then
    echo "SFT training with LoRA completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
else
    echo "SFT training failed!"
    exit 1
fi