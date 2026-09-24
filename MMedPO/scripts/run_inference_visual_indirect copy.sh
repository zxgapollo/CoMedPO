#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
# 方案一（method1）：Visual Evidence vs. No Foreground with TIE calculation
# 公式：1. (y_{gt}|X \oplus X_{X_bg})  > (y_{gt}|X_{null} \oplus X_{X_bg})
# 注：将原来的 y_{gen} 修改为 y_{gt}
# 运行两类推理，计算Visual TIE，并导出 DPO pairs（type=visual_indirect）

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC="${NPROC:-1}"
PORT="${PORT:-29500}"

MODEL_PATH="${MODEL_PATH:-/share_docker/medgemma-4b-it}"
QUESTION_FILE="${QUESTION_FILE:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/slake_dpo_weighted.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/visual_indirect}"
PAIRS_OUT="${PAIRS_OUT:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/dpo_pairs_visual_indirect.jsonl}"

# 使用新的TIE计算推理脚本
SCRIPT_VISUAL_TIE="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/inference_visual_tie.py"
BUILDER="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/build_dpo_pairs_visual_indirect.py"

mkdir -p "$OUTPUT_DIR"

TIE_RESULTS="$OUTPUT_DIR/visual_tie_results.jsonl"
STITCHED_DIR="$OUTPUT_DIR/stitched_visual_indirect"
mkdir -p "$STITCHED_DIR"

FULL_IMAGE_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs"
MASKED_IMAGE_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/processed_imgs"

echo "[visual_indirect] model=$MODEL_PATH questions=$QUESTION_FILE out=$OUTPUT_DIR"
echo "[visual_indirect] Computing Visual TIE (LL_full - LL_bg)..."

# Step 1: Calculate Visual TIE
"${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${PORT}" "$SCRIPT_VISUAL_TIE" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --output-file "$TIE_RESULTS" \
  --output-image-folder "$STITCHED_DIR" \
  --full-image-folder "$FULL_IMAGE_DIR" \
  --masked-image-folder "$MASKED_IMAGE_DIR"

echo "[visual_indirect] Building DPO pairs from TIE results..."

# Step 2: Build DPO pairs from TIE results
python "$BUILDER" \
  --tie-results-file "$TIE_RESULTS" \
  --output-pairs-file "$PAIRS_OUT" \
  --tie-threshold 0.0 \
  --w-min 0.05 \
  --beta 2.0

echo "[visual_indirect] Done. TIE results: $TIE_RESULTS, DPO pairs: $PAIRS_OUT"