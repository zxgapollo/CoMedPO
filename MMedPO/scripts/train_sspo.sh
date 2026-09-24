#!/bin/bash

# SPPO (Shaped Preference Policy Optimization) Training Script
# 使用加权 DPO 入口，严格对齐公式版 SPPO：
# 对称平方损失 [logratio_w·η(Pθ(w≻l)−0.5)]^2 + [logratio_l·η(Pθ(l≻w)−0.5)]^2，
# 其中 Pθ(w≻l)=sigmoid(β·(logratio_w−logratio_l))，由 --beta 与 --sppo_eta 控制。

# 默认参数（可通过命令行覆盖
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=1,2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
CUDA="1,2"
SFT_MODEL_PATH="/workspace/llava-med-v1.5-mistral-7b"
REFERENCE_MODEL_PATH="/workspace/MMedPO/Models/SFT_Slake"
DATA_PATH="/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_combined.json"
IMAGE_FOLDER="/workspace/MMedPO/datasets/SLAKE/imgs"
# 默认改为本地 CLIP（可通过 --vision_tower 覆盖）
VISION_TOWER="openai/clip-vit-large-patch14-336"
OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/sppo_model"

SPPO_ETA=0.5
BETA=0.5
LR=1e-7
NUM_EPOCHS=3
DATALOADER_NUM_WORKERS=0
POLICY_GPU=0
REFERENCE_GPU=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --image_folder) IMAGE_FOLDER="$2"; shift 2;;
    --model_path|--sft_model_path) SFT_MODEL_PATH="$2"; shift 2;;
    --reference_model_path) REFERENCE_MODEL_PATH="$2"; shift 2;;
    --CUDA|--cuda|--gpus) CUDA="$2"; export CUDA_VISIBLE_DEVICES="$2"; shift 2;;
    --vision_tower) VISION_TOWER="$2"; shift 2;;
    --beta) BETA="$2"; shift 2;;
    --num_epochs|--epochs) NUM_EPOCHS="$2"; shift 2;;
    --eta|--sppo_eta) SPPO_ETA="$2"; shift 2;;
    --lr|--learning_rate) LR="$2"; shift 2;;
    --dataloader_num_workers) DATALOADER_NUM_WORKERS="$2"; shift 2;;
    --policy_gpu) POLICY_GPU="$2"; shift 2;;
    --reference_gpu) REFERENCE_GPU="$2"; shift 2;;
    *) echo "Ignoring unknown parameter: $1"; shift;;
  esac
done

echo "=== Starting SPPO Training (sppo) ==="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATA_PATH"
echo "Image Directory: $IMAGE_FOLDER"
echo "SFT Model Path (Policy): $SFT_MODEL_PATH"
echo "Reference Model Path: $REFERENCE_MODEL_PATH"
echo "Vision Tower: $VISION_TOWER"
echo "Output Directory: $OUTPUT_DIR"
echo "beta=$BETA"
echo "====================================="
echo "[SPPO] 严格对齐公式：对称平方损失 + η·(Pθ−0.5) 塑形，Pθ=σ[β·(logratio_w−logratio_l)]"
echo "loss_variant=sppo, sppo_eta=$SPPO_ETA"

# 路径检查
if [ ! -f "$DATA_PATH" ]; then echo "Error: Dataset not found: $DATA_PATH"; exit 1; fi
if [ ! -d "$IMAGE_FOLDER" ]; then echo "Error: Image folder not found: $IMAGE_FOLDER"; exit 1; fi
if [ ! -d "$SFT_MODEL_PATH" ]; then echo "Error: SFT model path not found: $SFT_MODEL_PATH"; exit 1; fi
if [ ! -d "$REFERENCE_MODEL_PATH" ]; then echo "Error: Reference model path not found: $REFERENCE_MODEL_PATH"; exit 1; fi

mkdir -p "$OUTPUT_DIR"

cd /workspace/MMedPO/MMedPO/train/dpo || exit
python3 train_dpo_dual_gpu.py \
    --model_name_or_path $SFT_MODEL_PATH \
    --reference_model_path $REFERENCE_MODEL_PATH \
    --policy_gpu $POLICY_GPU \
    --reference_gpu $REFERENCE_GPU \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --bits 4 \
    --double_quant True \
    --quant_type nf4 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to none \
    --loss_use_weight True \
    --beta $BETA \
    --loss_type sigmoid \
    --loss_variant sppo \
    --sppo_eta $SPPO_ETA


echo "SPPO training (sppo) completed. Model saved to: $OUTPUT_DIR"