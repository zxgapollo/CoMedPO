#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
# 方案三：Context/Text Ablation with Textual TIE calculation
# 计算 Textual TIE (LL_text - LL_null) 并构建 DPO pairs（type=text_contrast）

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC="${NPROC:-1}"
PORT="${PORT:-29501}"

MODEL_PATH="${MODEL_PATH:-/workspace/MMedPO/Models/SFT_DPO_combined}"
QUESTION_FILE="${QUESTION_FILE:-/workspace/MMedPO/data/slake_dpo_weighted.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/MMedPO/MMedPO/outputs/combined/text_contrast}"
PAIRS_OUT="${PAIRS_OUT:-/workspace/MMedPO/MMedPO/outputs/combined/dpo_pairs_text_contrast.jsonl}"

# Image folder paths
FULL_IMAGE_FOLDER="${FULL_IMAGE_FOLDER:-/workspace/MMedPO/datasets/SLAKE/imgs}"
MASKED_IMAGE_FOLDER="${MASKED_IMAGE_FOLDER:-/workspace/MMedPO/datasets/SLAKE/processed_imgs}"
OUTPUT_IMAGE_FOLDER="${OUTPUT_IMAGE_FOLDER:-$OUTPUT_DIR/images}"

TIE_SCRIPT="/workspace/MMedPO/inference/inference_textual_tie.py"
BUILDER="/workspace/MMedPO/inference/build_dpo_pairs_text_contrast.py"

mkdir -p "$OUTPUT_DIR"

TIE_RESULTS="$OUTPUT_DIR/textual_tie_results.jsonl"

echo "[text_contrast] model=$MODEL_PATH questions=$QUESTION_FILE out=$OUTPUT_DIR"
echo "[text_contrast] Computing Textual TIE (LL_text - LL_null)..."

# Step 1: Calculate Textual TIE
"${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${PORT}" "$TIE_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --output-file "$TIE_RESULTS" \
  --output-image-folder "$OUTPUT_IMAGE_FOLDER" \
  --full-image-folder "$FULL_IMAGE_FOLDER" \
  --masked-image-folder "$MASKED_IMAGE_FOLDER"

echo "[text_contrast] Building DPO pairs from TIE results..."

# Step 2: Build DPO pairs from TIE results
python "$BUILDER" \
  --tie-results-file "$TIE_RESULTS" \
  --output-pairs-file "$PAIRS_OUT" \
  --tie-threshold 0.0 \
  --w-min 0.05 \
  --beta 2.0

echo "[text_contrast] Done. TIE results: $TIE_RESULTS"
echo "[text_contrast] Done. DPO pairs: $PAIRS_OUT"