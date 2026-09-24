#!/bin/bash
set -euo pipefail

# Run MedGemma inference for all conditions in background and then evaluate.

export PYTHONPATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo:${PYTHONPATH:-}"
MODEL_PATH="/share_docker/medgemma-4b-it"
BASE_QJSON="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/experiments/lesion_subset/slake_subset_500.json"
REBUTTAL_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/rebuttal"
CONDS=(fill_remove_zero_scale_1.0 fill_remove_noise_s0.2_scale_1.0 fill_remove_noise_s0.5_scale_1.5)

mkdir -p "$REBUTTAL_DIR"
PIDS=()
ERROR_PIDS=()

# Control whether each inference runs in background (1) or foreground (0).
# Default to foreground so user can run the script directly in tmux.
BACKGROUND="${BACKGROUND:-0}"

# On exit or interrupt, kill any spawned children
trap 'rc=$?; echo "[$(date -Iseconds)] Caught EXIT/INT/TERM (rc=$rc), killing children ${PIDS[*]:-}" >> "${REBUTTAL_DIR}/run_full_inference_background.trap.log"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; exit $rc' EXIT INT TERM

for c in "${CONDS[@]}"; do
  OUTDIR_IMG="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/background_randomized/conds/${c}/imgs"
  QF="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/experiments/lesion_subset/slake_subset_500_${c}.json"
  PREDS="${REBUTTAL_DIR}/${c}_slake_500_preds.jsonl"
  LOG="${REBUTTAL_DIR}/${c}_inference.log"

  if [ ! -s "$QF" ]; then
    echo "[$(date -Iseconds)] Skipping $c: question file empty or missing: $QF" >> "$LOG"
    continue
  fi

  echo "[$(date -Iseconds)] Starting medgemma inference for $c (questions: $QF)"
  echo "[$(date -Iseconds)] Starting medgemma inference for $c (questions: $QF)" >> "$LOG"
  if [ "${BACKGROUND}" -eq 1 ]; then
  python3 /share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/inference_generate_medgemma.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QF" \
    --answers-file "$PREDS" \
    --full-image-folder "$OUTDIR_IMG" \
      --max-new-tokens 64 --temperature 0.0 >> "$LOG" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    echo "[$(date -Iseconds)] Launched PID $pid for $c (questions: $QF), logging to $LOG" >> "${REBUTTAL_DIR}/run_full_inference_background.log"
  else
    if python3 /share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/inference_generate_medgemma.py \
      --model-path "$MODEL_PATH" \
      --question-file "$QF" \
      --answers-file "$PREDS" \
      --full-image-folder "$OUTDIR_IMG" \
      --max-new-tokens 64 --temperature 0.0 2>&1 | tee -a "$LOG"; then
      echo "[$(date -Iseconds)] Finished $c, outputs $PREDS"
  echo "[$(date -Iseconds)] Finished $c, outputs $PREDS" >> "$LOG"
    else
      echo "[$(date -Iseconds)] Failed $c"
      echo "[$(date -Iseconds)] Failed $c" >> "$LOG"
      ERROR_PIDS+=("$c")
    fi
  fi
done
# If running in background mode, wait for all background inference jobs to finish
EVAL_LOG="${REBUTTAL_DIR}/evaluate_and_make_latex.log"
if [ "${BACKGROUND}" -eq 1 ]; then
  echo "[$(date -Iseconds)] Waiting for ${#PIDS[@]} inference jobs: ${PIDS[*]:-}" >> "${REBUTTAL_DIR}/run_full_inference_background.log"
  for pid in "${PIDS[@]:-}"; do
    if wait "$pid"; then
      echo "[$(date -Iseconds)] PID $pid finished successfully" >> "${REBUTTAL_DIR}/run_full_inference_background.log"
    else
      echo "[$(date -Iseconds)] PID $pid failed" >> "${REBUTTAL_DIR}/run_full_inference_background.log"
      ERROR_PIDS+=("$pid")
    fi
  done

  if [ "${#ERROR_PIDS[@]}" -ne 0 ]; then
    echo "[$(date -Iseconds)] Some inference jobs failed: ${ERROR_PIDS[*]}" >> "${REBUTTAL_DIR}/run_full_inference_background.log"
  fi
fi

# Run evaluation and produce LaTeX table
EVAL_LOG="${REBUTTAL_DIR}/evaluate_and_make_latex.log"
echo "[$(date -Iseconds)] Starting evaluation" >> "$EVAL_LOG"
python3 /share_docker/workspace/Med/Med-main/Med-main/MMedPO/scripts/evaluate_and_make_latex.py \
  --baseline "${REBUTTAL_DIR}/baseline_slake_500.jsonl" \
  --conds "zero:${REBUTTAL_DIR}/fill_remove_zero_scale_1.0_slake_500_preds.jsonl" \
          "noise02:${REBUTTAL_DIR}/fill_remove_noise_s0.2_scale_1.0_slake_500_preds.jsonl" \
          "noise05:${REBUTTAL_DIR}/fill_remove_noise_s0.5_scale_1.5_slake_500_preds.jsonl" \
  --out_dir "${REBUTTAL_DIR}" >> "${EVAL_LOG}" 2>&1 || echo "[$(date -Iseconds)] Evaluation failed" >> "${EVAL_LOG}"
echo "[$(date -Iseconds)] ALL_DONE" >> "${EVAL_LOG}"

exit 0

