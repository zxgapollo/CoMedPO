#!/bin/bash
set -euo pipefail

# Wrapper to run evaluate_and_make_latex.py with optional DeepSeek LLM judge.
# Defaults are set for the current repo layout; override via environment variables.
#
# Usage examples:
# 1) Default run (uses env or eval.sh API key if available):
#    ./run_evaluate_llmjudge.sh
#
# 2) Explicitly disable LLM judge:
#    JUDGE=none ./run_evaluate_llmjudge.sh
#
# 3) Provide your own DeepSeek key:
#    JUDGE=deepseek JUDGE_API_KEY=sk-xxx ./run_evaluate_llmjudge.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BASELINE="${BASELINE:-${ROOT_DIR}/rebuttal/baseline_slake_500.jsonl}"
COND_ZERO="${COND_ZERO:-${ROOT_DIR}/rebuttal/fill_remove_zero_scale_1.0_slake_500_preds.jsonl}"
COND_NOISE02="${COND_NOISE02:-${ROOT_DIR}/rebuttal/fill_remove_noise_s0.2_scale_1.0_slake_500_preds.jsonl}"
COND_NOISE05="${COND_NOISE05:-${ROOT_DIR}/rebuttal/fill_remove_noise_s0.5_scale_1.5_slake_500_preds.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/rebuttal}"

# Judge settings (default to deepseek)
JUDGE="${JUDGE:-deepseek}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

# If no explicit JUDGE_API_KEY, try to read from MedEvalKit/eval.sh (if present)
EVAL_SH="${ROOT_DIR%/MMedPO}/MedEvalKit/eval.sh"
if [ -z "${JUDGE_API_KEY}" ] && [ -f "${EVAL_SH}" ]; then
  candidate=$(grep -E '^API_KEY=' "${EVAL_SH}" | head -n1 | cut -d'=' -f2- | tr -d '"' | tr -d ' ')
  if [ -n "${candidate}" ]; then
    JUDGE_API_KEY="${candidate}"
  fi
fi

echo "[info] Running evaluation with JUDGE=${JUDGE}, OUT_DIR=${OUT_DIR}"
echo "[info] Baseline: ${BASELINE}"
echo "[info] Conditions:"
echo "  zero:  ${COND_ZERO}"
echo "  noise02: ${COND_NOISE02}"
echo "  noise05: ${COND_NOISE05}"

PY_CMD=(python3 "${SCRIPT_DIR}/evaluate_and_make_latex.py"
  --baseline "${BASELINE}"
  --conds "zero:${COND_ZERO}" "noise02:${COND_NOISE02}" "noise05:${COND_NOISE05}"
  --out_dir "${OUT_DIR}"
)

if [ "${JUDGE}" != "none" ]; then
  PY_CMD+=(--judge "${JUDGE}" --judge_api_key "${JUDGE_API_KEY}" --judge_base_url "${JUDGE_BASE_URL}" --judge_model "${JUDGE_MODEL}")
fi

echo "[info] Command: ${PY_CMD[*]}"
exec "${PY_CMD[@]}"

