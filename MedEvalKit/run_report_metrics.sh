#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,2,3
# Usage:
#   ./run_report_metrics.sh <MODEL_NAME_DIR> [DATASETS]
# Example:
#   ./run_report_metrics.sh /workspace/MMedPO/MedEvalKit/Eval_Results/SFT_iu_xray "IU_XRAY"
#   ./run_report_metrics.sh /workspace/MMedPO/MedEvalKit/Eval_Results/SFT_iu_xray "IU_XRAY,CheXpert_Plus,MIMIC_CXR"

MODEL_NAME_DIR=${1:-"/workspace/MMedPO/MedEvalKit/Eval_Results/MMedPO"}
DATASETS_CSV=${2:-"IU_XRAY"}

if [[ -z "$MODEL_NAME_DIR" ]]; then
  echo "[ERROR] MODEL_NAME_DIR is required."
  echo "Usage: $0 <MODEL_NAME_DIR> [DATASETS]"
  exit 1
fi

# Optional LLM judge envs (disabled by default)
export USE_LLM_JUDGE=${USE_LLM_JUDGE:-False}
export JUDGE_MODEL_TYPE=${JUDGE_MODEL_TYPE:-"deepseek"}
export GPT_MODEL=${GPT_MODEL:-"deepseek-chat"}
export API_KEY="${API_KEY:-}"
export BASE_URL=${BASE_URL:-"https://api.deepseek.com"}
export LLM_JUDGE_SAMPLE_LIMIT=${LLM_JUDGE_SAMPLE_LIMIT:-64}

# Core envs for metrics script
export MODEL_NAME="$MODEL_NAME_DIR"
export EVAL_DATASETS="$DATASETS_CSV"

# Run the metrics script
python3 /workspace/MMedPO/MedEvalKit/utils/Metrics_Compute/cal_report_metrics.py
