#!/bin/bash

# SFT (Supervised Fine-Tuning) Training Script with LoRA
# Based on the pipeline script configuration

# Set environment variables
export CUDA_VISIBLE_DEVICES=2,3
PY="/share_docker/conda_envs/MMedPO/bin/python"
TORCHRUN="/share_docker/conda_envs/MMedPO/bin/torchrun"
export WANDB_PROJECT="MMedPO_SFT"
set -e

# Set offline mode for Hugging Face Hub to avoid network issues
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optimize CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set PYTHONPATH to ensure llava module can be imported
export PYTHONPATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo:$PYTHONPATH"

# Training parameters
MODEL_NAME="/share_docker/medgemma-4b-it"
export DPO_DATA_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/tie_dpo_dataset_method1_slake_med_gemma.json"
export SFT_DATA_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/tie_sft_dataset_method1_slake_med_gemma.json"
IMAGE_FOLDER="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs"
OUTPUT_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/checkpoint/sft_model_lora_medgemma_slake"
TRAIN_BS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    --data_path)
      SFT_DATA_PATH="$2"; shift 2;;
    --image_folder)
      IMAGE_FOLDER="$2"; shift 2;;
    --model_path|--base_model_path)
      MODEL_NAME="$2"; shift 2;;
    --CUDA|--cuda|--gpus)
      export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --train_batch_size)
      TRAIN_BS="$2"; shift 2;;
    *)
      echo "Ignoring unknown parameter: $1"; shift;;
  esac
done

echo "=== Starting SFT Training ==="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $SFT_DATA_PATH"
echo "Image Directory: $IMAGE_FOLDER"
echo "Output Directory: $OUTPUT_DIR"
echo "Base Model Path: $MODEL_NAME"
echo "================================"

if [ ! -f "$SFT_DATA_PATH" ]; then
  echo "Error: Dataset file not found: $SFT_DATA_PATH"; exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "Error: Image directory not found: $IMAGE_FOLDER"; exit 1
fi
if [ ! -d "$MODEL_NAME" ]; then
  echo "Error: Base model path not found: $MODEL_NAME"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Validate paths and optionally convert DPO -> SFT
if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "Error: Image directory not found: $IMAGE_FOLDER"; exit 1
fi
if [ -f "$DPO_DATA_PATH" ]; then
  echo "Converting DPO format data to SFT format with correct image paths..."
  $PY - <<PY
import json, os
dpo_path = "${DPO_DATA_PATH}"
sft_path = "${SFT_DATA_PATH}"
map_path = "/share_docker/workspace/Med/Med-main/Med-main/MMedPO/datasets/VQA_RAD/VQA_RAD_Dataset_Public.json"

with open(dpo_path, 'r') as f:
    dpo_data = json.load(f)

qid_to_image = {}
if os.path.exists(map_path):
    try:
        with open(map_path, 'r') as f:
            original_data = json.load(f)
        qid_to_image = {int(item.get('qid', -1)): item.get('image_name', item.get('image')) for item in original_data}
    except Exception:
        qid_to_image = {}

def map_image(item):
    return qid_to_image.get(item.get('id'), item.get('image'))

sft_data = []
for item in dpo_data:
    sft_data.append({
        'id': item.get('id'),
        'image': map_image(item),
        'conversations': item.get('conversations', [])
    })

with open(sft_path, 'w') as f:
    json.dump(sft_data, f, indent=2)
print(f'Created SFT format data with {len(sft_data)} samples')
print(f'Saved to: {sft_path}')
PY
else
  echo "Warning: DPO data not found: $DPO_DATA_PATH; skipping conversion"
fi

# Change to the training directory
cd /share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo

echo "Starting SFT training..."
echo "Dataset: $SFT_DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"

NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$NPROC" -lt 1 ]]; then
  NPROC=1
fi

if [[ "${MODEL_NAME,,}" == *"medgemma"* ]]; then
  echo "Detected MedGemma base: $MODEL_NAME"
  "$PY" \
    /share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo/train_sft_medgemma.py \
    --model_path "$MODEL_NAME" \
    --data_path "$SFT_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --safe_serialization false \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$TRAIN_BS" \
    --learning_rate 2e-4 \
    --logging_steps 10
else
  echo "Detected LLaVA-style base: $MODEL_NAME"
  echo "LoRA Configuration: r=128, alpha=256, dropout=0.05"
  "$TORCHRUN" --nproc_per_node=$NPROC --master_port=$((RANDOM + 30000)) llava/train/train_sft.py \
      --model_name_or_path $MODEL_NAME \
      --version v1 \
      --data_path $SFT_DATA_PATH \
      --image_folder $IMAGE_FOLDER \
      --vision_tower openai/clip-vit-large-patch14-336 \
      --mm_projector_type mlp2x_gelu \
      --mm_vision_select_layer -2 \
      --mm_use_im_start_end False \
      --image_aspect_ratio pad \
      --group_by_modality_length True \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --num_train_epochs 3 \
      --per_device_train_batch_size "$TRAIN_BS" \
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
fi

# Check if training completed successfully
if [[ $? -eq 0 ]]; then
    echo "SFT training with LoRA completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
else
    echo "SFT training failed!"
    exit 1
fi
