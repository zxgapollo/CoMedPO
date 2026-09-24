#!/usr/bin/env bash
set -euo pipefail

# run_compute_iou_slake.sh
# Wrapper to create venv (optional), install deps, and run compute_iou_slake.py
#
# Usage examples:
#  ./run_compute_iou_slake.sh --gt-mask-dir /abs/path/to/gt_masks \
#      --sam-pred /abs/path/to/sam_pred_masks \
#      --medklip-pred /abs/path/to/medklip_pred_masks \
#      --out /abs/path/to/outputs/slake_iou_per_sample.csv
#
#  ./run_compute_iou_slake.sh --gt-csv /path/to/gt.csv \
#      --sam-pred /path/to/sam_preds.csv \
#      --medklip-pred /path/to/medklip_preds.jsonl \
#      --out outputs/slake_iou_per_sample.csv \
#      --skip-install
#
# Example with new options:
#  ./run_compute_iou_slake.sh --gt-mask-dir /abs/path/to/gt_masks \
#      --medklip-pred /abs/path/to/medklip_preds.jsonl \
#      --point-prompt --topk 3 --threshold-method top_percent --threshold-value 0.05 \
#      --multi-hypothesis --area-filter --clip-rerank --llm-path /share_docker/workspace/DeepSeek-R1-0528-Qwen3-8B \
#      --out /abs/path/to/outputs/slake_iou_per_sample.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PY_SCRIPT_DIR="${SCRIPT_DIR}"

# New pipeline scripts
INFER_SCRIPT="${SCRIPT_DIR}/eval_slake_inference.py"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_slake_iou.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_slake_results.py"
VENV_DIR="${SCRIPT_DIR}/.venv"

show_help() {
   cat <<EOF
Usage: $0 [options]

Options:
  --gt-csv PATH           Ground-truth CSV (must include sample_id and mask_path or bbox)
  --gt-mask-dir PATH      Ground-truth mask directory (files named <sample_id>.png)
  --sam-pred PATH         SAM prediction path (dir/csv/jsonl)
  --medklip-pred PATH     MedKLIP prediction path (dir/csv/jsonl)
  --export-zero-iou       Export IoU=0 cases and visualizations (zero_iou_cases.jsonl and visuals/)
  --point-prompt          Use Point Prompt mode (extract Top-K points from heatmap instead of box)
  --topk N                Number of top points/boxes to generate (default: 1)
  --threshold-method STR  Heatmap thresholding method: fixed | otsu | top_percent (default: fixed)
  --threshold-value VAL   Threshold value when using fixed method (default: 0.5)
  --only-sam              Only run SAM inference (do not run MedKLIP)
  --only-medklip          Only run MedKLIP inference (do not run SAM)
  --run-medklip-fallback  Run MedKLIP fallback inference when no checkpoint provided
  --multi-hypothesis      Generate multiple hypotheses (Top-3) and select best
  --area-filter           Enable area-based filtering of SAM masks (reject too large/small masks)
  --clip-rerank           Enable CLIP/MedKLIP re-ranking of SAM masks
  --llm-path PATH         Path to local LLM model directory for prompt cleaning (default set to provided path)
  --run-combined          Run combined SAM+text-guided box evaluation pipeline and exit
  --out PATH              Output CSV path (default: outputs/slake_iou_per_sample.csv)
  --repo-root PATH        Repo root to resolve relative mask paths (default: .)
  --skip-install          Skip creating venv / installing dependencies
  -h, --help              Show this help
EOF
}

# default values
GT_CSV=""
GT_MASK_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs"
SAM_PRED="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/models/sam/sam_vit_h_4b8939.pth"
MEDKLIP_PRED=""
OUT_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/slake_iou_per_sample.csv"
REPO_ROOT="."
SKIP_INSTALL=0
RUN_COMBINED=1
EXPORT_ZERO_IOU=0
POINT_PROMPT=1
TOPK=3
THRESH_METHOD="top_percent" # options: fixed, otsu, top_percent
THRESH_VALUE="0.05"    # used when THRESH_METHOD=fixed or top_percent as fraction
MULTI_HYP=1
AREA_FILTER=1
CLIP_RERANK=1
LLM_PATH="/share_docker/workspace/DeepSeek-R1-0528-Qwen3-8B"
CLIP_MODEL_PATH=""
CLIPSEG_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/models/clipseg-rd64-refined"
CLIP_BATCH_SIZE=8
LLM_ENSEMBLE=1
ONLY_SAM=0
ONLY_MEDKLIP=0
RUN_MEDKLIP_FALLBACK=0

# GPU / memory tuning defaults (can be overridden by CLI)
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
# default: enable fp16 where supported; use --no-fp16 to disable
DEFAULT_FP16=1
NO_FP16=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt-csv) GT_CSV="$2"; shift 2;;
        --gt-mask-dir) GT_MASK_DIR="$2"; shift 2;;
        --cuda-devices) CUDA_DEVICES="$2"; shift 2;;
        --sam-pred) SAM_PRED="$2"; shift 2;;
        --export-zero-iou) EXPORT_ZERO_IOU=1; shift 1;;
        --point-prompt) POINT_PROMPT=1; shift 1;;
        --topk) TOPK="$2"; shift 2;;
        --threshold-method) THRESH_METHOD="$2"; shift 2;;
        --threshold-value) THRESH_VALUE="$2"; shift 2;;
        --multi-hypothesis) MULTI_HYP=1; shift 1;;
        --area-filter) AREA_FILTER=1; shift 1;;
        --clip-rerank) CLIP_RERANK=1; shift 1;;
        --llm-path) LLM_PATH="$2"; shift 2;;
        --llm-ensemble) LLM_ENSEMBLE=1; shift 1;;
        --clip-batch-size) CLIP_BATCH_SIZE="$2"; shift 2;;
        --clip-model-path) CLIP_MODEL_PATH="$2"; shift 2;;
        --clipseg-path) CLIPSEG_PATH="$2"; shift 2;;
        --run-combined) RUN_COMBINED=1; shift 1;;
        --only-sam) ONLY_SAM=1; shift 1;;
        --only-medklip) ONLY_MEDKLIP=1; shift 1;;
        --run-medklip-fallback) RUN_MEDKLIP_FALLBACK=1; shift 1;;
        --medklip-pred) MEDKLIP_PRED="$2"; shift 2;;
        --out) OUT_PATH="$2"; shift 2;;
        --repo-root) REPO_ROOT="$2"; shift 2;;
        --skip-install) SKIP_INSTALL=1; shift 1;;
        --no-fp16) NO_FP16=1; shift 1;;
        -h|--help) show_help; exit 0;;
        *) echo "Unknown arg: $1"; show_help; exit 1;;
    esac
done

if [[ -z "$GT_CSV" && -z "$GT_MASK_DIR" ]]; then
    echo "Error: one of --gt-csv or --gt-mask-dir must be provided"
    show_help
    exit 1
fi

if [[ -z "$SAM_PRED" && -z "$MEDKLIP_PRED" ]]; then
    echo "Error: at least one of --sam-pred or --medklip-pred must be provided"
    show_help
    exit 1
fi

# prepare python command
PY_CMD="python3"

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtualenv at ${VENV_DIR}..."
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    echo "Installing dependencies into venv..."
    pip install --upgrade pip
    pip install -r "${SCRIPT_DIR}/requirements.txt" || pip install pandas numpy pillow matplotlib scikit-image opencv-python torch torchvision
    PY_CMD="${VENV_DIR}/bin/python"
else
    echo "Skipping venv creation / dependency install (--skip-install)"
fi

#
# New pipeline orchestration:
# 1) Run inference (SAM and/or MedKLIP) to generate prediction masks under output dir
# 2) Run evaluation to compute IoU and results.json/csv
# 3) Run plotting to create distribution plots
#
OUT_DIR="$(dirname "$OUT_PATH")"
mkdir -p "$OUT_DIR"

# export memory / CUDA tuning env vars
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# decide fp16 flag
FP16_FLAG=""
if [[ "$NO_FP16" -eq 0 && "$DEFAULT_FP16" -eq 1 ]]; then
    FP16_FLAG="--fp16"
fi

INFER_CMD=( "$PY_CMD" "$INFER_SCRIPT" --data-root "${GT_MASK_DIR:-$GT_CSV}" --output-dir "${OUT_DIR}" )
if [[ -n "$SAM_PRED" ]]; then
    INFER_CMD+=( --sam-checkpoint "$SAM_PRED" )
fi
if [[ -n "$MEDKLIP_PRED" ]]; then
    INFER_CMD+=( --medklip-checkpoint "$MEDKLIP_PRED" )
fi
if [[ "$EXPORT_ZERO_IOU" -eq 1 ]]; then
    INFER_CMD+=( --export-zero-iou )
fi
if [[ "$POINT_PROMPT" -eq 1 ]]; then
    INFER_CMD+=( --point-prompt --topk "$TOPK" )
fi
if [[ -n "$THRESH_METHOD" ]]; then
    INFER_CMD+=( --threshold-method "$THRESH_METHOD" )
fi
if [[ -n "$THRESH_VALUE" ]]; then
    INFER_CMD+=( --threshold-value "$THRESH_VALUE" )
fi
if [[ "$MULTI_HYP" -eq 1 ]]; then
    INFER_CMD+=( --multi-hypothesis )
fi
if [[ "$AREA_FILTER" -eq 1 ]]; then
    INFER_CMD+=( --area-filter )
fi
if [[ "$CLIP_RERANK" -eq 1 ]]; then
    INFER_CMD+=( --clip-rerank )
fi
if [[ -n "$LLM_PATH" ]]; then
    INFER_CMD+=( --llm-path "$LLM_PATH" )
fi
if [[ "$LLM_ENSEMBLE" -eq 1 ]]; then
    INFER_CMD+=( --llm-ensemble )
fi
if [[ -n "$CLIP_BATCH_SIZE" ]]; then
    INFER_CMD+=( --clip-batch-size "$CLIP_BATCH_SIZE" )
fi
if [[ -n "$CLIP_MODEL_PATH" ]]; then
    INFER_CMD+=( --clip-model-path "$CLIP_MODEL_PATH" )
fi
if [[ "$ONLY_SAM" -eq 1 ]]; then
    INFER_CMD+=( --only-sam )
fi
if [[ "$ONLY_MEDKLIP" -eq 1 ]]; then
    INFER_CMD+=( --only-medklip )
fi
if [[ "$RUN_MEDKLIP_FALLBACK" -eq 1 ]]; then
    INFER_CMD+=( --run-medklip-fallback )
fi

# append fp16 flag if enabled
if [[ -n "$FP16_FLAG" ]]; then
    INFER_CMD+=( $FP16_FLAG )
fi


echo "Running inference pipeline..."
echo "Command: ${INFER_CMD[*]}"
"${INFER_CMD[@]}"

# Combined evaluation mode: run combined SAM+text-guided box evaluation and exit
COMBINED_SCRIPT="${SCRIPT_DIR}/eval_slake_combined.py"
if [[ "$RUN_COMBINED" -eq 1 ]]; then
    if [[ -n "$GT_MASK_DIR" ]]; then
        COMB_CMD=( "$PY_CMD" "$COMBINED_SCRIPT" --data-root "$GT_MASK_DIR" --output-dir "${OUT_DIR}" )
        if [[ -n "$SAM_PRED" ]]; then
            COMB_CMD+=( --sam-checkpoint "$SAM_PRED" )
        fi
        if [[ "$EXPORT_ZERO_IOU" -eq 1 ]]; then
            COMB_CMD+=( --export-zero-iou )
        fi
        if [[ "$POINT_PROMPT" -eq 1 ]]; then
            COMB_CMD+=( --point-prompt --topk "$TOPK" )
        fi
        if [[ -n "$THRESH_METHOD" ]]; then
            COMB_CMD+=( --threshold-method "$THRESH_METHOD" )
        fi
        if [[ -n "$THRESH_VALUE" ]]; then
            COMB_CMD+=( --threshold-value "$THRESH_VALUE" )
        fi
        if [[ "$MULTI_HYP" -eq 1 ]]; then
            COMB_CMD+=( --multi-hypothesis )
        fi
        if [[ "$AREA_FILTER" -eq 1 ]]; then
            COMB_CMD+=( --area-filter )
        fi
        if [[ "$CLIP_RERANK" -eq 1 ]]; then
            COMB_CMD+=( --clip-rerank )
        fi
        if [[ -n "$LLM_PATH" ]]; then
            COMB_CMD+=( --llm-path "$LLM_PATH" )
        fi
        if [[ "$LLM_ENSEMBLE" -eq 1 ]]; then
            COMB_CMD+=( --llm-ensemble )
        fi
        if [[ -n "$CLIP_BATCH_SIZE" ]]; then
            COMB_CMD+=( --clip-batch-size "$CLIP_BATCH_SIZE" )
        fi
        if [[ -n "$CLIP_MODEL_PATH" ]]; then
            COMB_CMD+=( --clip-model-path "$CLIP_MODEL_PATH" )
        fi
        if [[ -n "$CLIPSEG_PATH" ]]; then
            COMB_CMD+=( --clipseg-path "$CLIPSEG_PATH" )
        fi
        echo "Running combined SAM+text-guided evaluation..."
        echo "Command: ${COMB_CMD[*]}"
        # append fp16 flag if enabled for combined step
        if [[ -n "$FP16_FLAG" ]]; then
            COMB_CMD+=( $FP16_FLAG )
        fi
        "${COMB_CMD[@]}"
        echo "Combined evaluation complete. Outputs under ${OUT_DIR}"
        exit 0
    else
        echo "Error: --run-combined requires --gt-mask-dir to point to SLAKE imgs root"
        exit 1
    fi
fi

echo "Running evaluation..."
EVAL_CMD=( "$PY_CMD" "$EVAL_SCRIPT" --gt-mask-dir "$GT_MASK_DIR" --pred-dir "${OUT_DIR}" --out-json "${OUT_DIR}/results.json" )
echo "Command: ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

echo "Generating plots..."
PLOT_CMD=( "$PY_CMD" "$PLOT_SCRIPT" --results-json "${OUT_DIR}/results.json" --out-dir "${OUT_DIR}/plots" )
echo "Command: ${PLOT_CMD[*]}"
"${PLOT_CMD[@]}"

echo "Done. Outputs saved to ${OUT_DIR}"

