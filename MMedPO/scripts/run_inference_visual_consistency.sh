#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
# 方案二：Visual Consistency Preference (Ground-truth vs. Generated) with TIE calculation
# 在相同视觉输入下比较真实答案与模型生成答案，计算 Visual Consistency TIE，并构建 DPO pairs（type=visual_consistency）

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC="${NPROC:-1}"
PORT="${PORT:-29502}"

MODEL_PATH="${MODEL_PATH:-/workspace/MMedPO/Models/SFT_DPO_combined}"
QUESTION_FILE="${QUESTION_FILE:-/workspace/MMedPO/data/slake_dpo_weighted.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/MMedPO/MMedPO/outputs/combined/visual_consistency}"
PAIRS_OUT="${PAIRS_OUT:-/workspace/MMedPO/MMedPO/outputs/combined/dpo_pairs_visual_consistency.jsonl}"

# Image folder paths
FULL_IMAGE_FOLDER="${FULL_IMAGE_FOLDER:-/workspace/MMedPO/datasets/SLAKE/imgs}"
MASKED_IMAGE_FOLDER="${MASKED_IMAGE_FOLDER:-/workspace/MMedPO/datasets/SLAKE/processed_imgs}"
OUTPUT_IMAGE_FOLDER="${OUTPUT_IMAGE_FOLDER:-$OUTPUT_DIR/images}"

TIE_SCRIPT="/workspace/MMedPO/inference/inference_visual_consistency_tie.py"
BUILDER="/workspace/MMedPO/inference/build_dpo_pairs_visual_consistency.py"

mkdir -p "$OUTPUT_DIR"

TIE_RESULTS="$OUTPUT_DIR/visual_consistency_tie_results.jsonl"

echo "[visual_consistency] model=$MODEL_PATH questions=$QUESTION_FILE out=$OUTPUT_DIR"
echo "[visual_consistency] Computing Visual Consistency TIE (γ = Δ+ - Δ-, m_v, m_n)..."

# Step 1: Calculate Visual Consistency TIE
"${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${PORT}" "$TIE_SCRIPT" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --output-file "$TIE_RESULTS" \
  --output-image-folder "$OUTPUT_IMAGE_FOLDER" \
  --full-image-folder "$FULL_IMAGE_FOLDER" \
  --masked-image-folder "$MASKED_IMAGE_FOLDER"

echo "[visual_consistency] Building DPO pairs from TIE results..."

# Step 2: Build DPO pairs from TIE results
python "$BUILDER" \
  --tie-results-file "$TIE_RESULTS" \
  --output-pairs-file "$PAIRS_OUT" \
  --tie-threshold 0.0 \
  --w-min 0.05 \
  --beta 2.0 \
  --alpha 0.8 \
  --beta-penalty 0.6 \
  --tau-n 0.3

echo "[visual_consistency] Done. TIE results: $TIE_RESULTS"
echo "[visual_consistency] Done. DPO pairs: $PAIRS_OUT"
echo "[visual_consistency] Method 2: Visual Consistency Preference completed successfully!"
echo ""
echo "=== Method 2 Summary ==="
echo "Input: Same visual input (I_full ⊕ I_bg)"
echo "Comparison: Ground-truth answer vs. Generated answer"
echo "TIE Metrics: γ (net visual support), m_v (full discrimination), m_n (background bias)"
echo "Pair Construction: Preferred=y_gt(I_full), Dispreferred=y_gen(I_full)"
echo "Weight Formula: S_raw = (Δ+ - Δ-) + α*m_v - β*max(0, m_n - τ_n)"