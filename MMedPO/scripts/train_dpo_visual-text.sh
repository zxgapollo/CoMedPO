#!/bin/bash

# Default GPUs and key parameters (can be overridden by command line)
export CUDA_VISIBLE_DEVICES=2,3
CUDA="0,1"
PY="/share_docker/conda_envs/MMedPO/bin/python"
TORCHRUN="/share_docker/conda_envs/MMedPO/bin/torchrun"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e
OUTPUT_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/checkpoint/SFT_method1_Slake_medgemma"
DATA_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/tie_dpo_dataset_method1_slake_med_gemma.json"
IMAGE_FOLDER="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs"
BASE_MODEL_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/checkpoint/sft_model_lora_medgemma_slake/hf_latest" # Add base model path as a variable
CONVERT_BIN_PATH=""      # Optional: path to an existing pytorch_model_epoch_*.bin to convert
CONVERT_SAVE_DIR=""      # Optional: target HF directory to save converted weights
CONVERT_BASE_MODEL_PATH=""  # Optional: base model path for conversion
TRAIN_BS=1


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    --data_path)
      DATA_PATH="$2"; shift 2;;
    --image_folder)
      IMAGE_FOLDER="$2"; shift 2;;
    --base_model_path)
      BASE_MODEL_PATH="$2"; shift 2;;
    --merged_model_path)
      MERGED_MODEL_PATH="$2"; shift 2;;
    --convert_bin_path)
      CONVERT_BIN_PATH="$2"; shift 2;;
    --convert_save_dir)
      CONVERT_SAVE_DIR="$2"; shift 2;;
    --convert_base_model_path)
      CONVERT_BASE_MODEL_PATH="$2"; shift 2;;
    --CUDA|--cuda|--gpus)
      CUDA="$2"; export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --train_batch_size)
      TRAIN_BS="$2"; shift 2;;
    *)
      echo "Ignoring unknown parameter: $1"; shift;;
  esac
done

echo "=== Starting DPO Training (Visual-Text) ==="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATA_PATH"
echo "Image Directory: $IMAGE_FOLDER"
echo "Output Directory: $OUTPUT_DIR"
echo "Base Model Path: $BASE_MODEL_PATH"
echo "Merged Model Path: $MERGED_MODEL_PATH"
echo "Convert Bin Path: $CONVERT_BIN_PATH"
echo "Convert Save Dir: $CONVERT_SAVE_DIR"
echo "Convert Base Model: $CONVERT_BASE_MODEL_PATH"
echo "=========================================="

# Path consistency check
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Dataset file not found: $DATA_PATH"; exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "Error: Image directory not found: $IMAGE_FOLDER"; exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
  echo "Error: Base model path not found: $BASE_MODEL_PATH"; exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Determine number of processes from CUDA_VISIBLE_DEVICES
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$NPROC" -lt 1 ]]; then
  NPROC=1
fi

cd /share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo || exit
if [[ "${BASE_MODEL_PATH,,}" == *"medgemma"* ]]; then
  # If conversion args are provided, run conversion-only mode
  if [[ -n "$CONVERT_BIN_PATH" && -n "$CONVERT_SAVE_DIR" && -n "$CONVERT_BASE_MODEL_PATH" ]]; then
    "$PY" /share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo/train_dpo_medgemma.py \
      --model_path "$BASE_MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --image_folder "$IMAGE_FOLDER" \
      --output_dir "$OUTPUT_DIR" \
      --convert_bin_path "$CONVERT_BIN_PATH" \
      --convert_save_dir "$CONVERT_SAVE_DIR" \
      --convert_base_model_path "$CONVERT_BASE_MODEL_PATH"
  else
    "$PY" \
      /share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo/train_dpo_medgemma.py \
      --model_path "$BASE_MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --image_folder "$IMAGE_FOLDER" \
      --output_dir "$OUTPUT_DIR" \
      --device_map "auto" \
      --num_train_epochs 3 \
      --per_device_train_batch_size "$TRAIN_BS" \
      --learning_rate 1e-6 \
      --beta 2.0 \
      --logging_steps 10 \
      --safe_serialization false
    echo "HF latest checkpoint saved at: $OUTPUT_DIR/hf_latest"
  fi
else
  "$TORCHRUN" --nproc_per_node=$NPROC --master_port $((RANDOM + 30000)) llava/train/train_dpo.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --version v1 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_weight_path "" \
    --lora_bias "none" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$TRAIN_BS" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to wandb \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False
fi
