#!/bin/bash
# 视觉间接证据（Visual-Indirect TIE）推理脚本
# 修复：增加依赖检查（tqdm 等），并适配 medgemma-4b-it 模型路径
set -euo pipefail

# 可配置 GPU（默认 1）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 确保 Python 能找到本仓库的 llava 代码
export PYTHONPATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/train/dpo:${PYTHONPATH:-}"

# 可选：提供本地 Transformers 源码路径以离线加载（例如克隆的仓库）
# 用法：导出 TRANSFORMERS_SRC=/path/to/transformers_repo
if [[ -n "${TRANSFORMERS_SRC:-}" && -d "${TRANSFORMERS_SRC}/src/transformers" ]]; then
  export PYTHONPATH="${TRANSFORMERS_SRC}/src:${PYTHONPATH}"
  echo "[setup] Using local transformers from: ${TRANSFORMERS_SRC}/src"
elif [[ -n "${TRANSFORMERS_SRC:-}" && -d "${TRANSFORMERS_SRC}/transformers" ]]; then
  export PYTHONPATH="${TRANSFORMERS_SRC}:${PYTHONPATH}"
  echo "[setup] Using local transformers from: ${TRANSFORMERS_SRC}"
fi

# 选择合适的 Python 解释器（优先能加载 transformers 的）
has_module() {
  local pybin="$1"; shift
  local mod="$1"; shift || true
  "$pybin" - <<PY >/dev/null 2>&1 || return 1
import importlib, sys
mod = sys.argv[1]
importlib.import_module(mod)
PY
  return 0
}

pick_python_bin() {
  local candidates=(
    "${PYTHON_BIN:-}"
    "/share_docker/conda_envs/MMedPO/bin/python3"
    "/usr/bin/python3"
    "/usr/bin/python3.8"
    "/usr/bin/python3.9"
    "/usr/bin/python3.10"
  )
  for p in "${candidates[@]}"; do
    [[ -x "$p" ]] || continue
    if has_module "$p" transformers; then
      echo "$p"; return 0
    fi
  done
  # 如果没有找到带 transformers 的 Python，回退到当前环境的 python3
  command -v python3 2>/dev/null || true
}

PYTHON_BIN="$(pick_python_bin)"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
echo "[setup] Using Python: $PYTHON_BIN"

# 依赖检查与自动安装（避免 ModuleNotFoundError: tqdm）
ensure_py_module() {
  local mod="$1"; shift || true
  if ! "$PYTHON_BIN" -c "import importlib; import sys; import importlib.util as u; importlib.import_module('$mod')" >/dev/null 2>&1; then
    echo "[setup] Installing Python module: ${mod}"
    # 仅安装轻量模块，避免在无网环境下长时间等待
    if [[ "$mod" == "transformers" ]]; then
      echo "[setup] Skip auto-install transformers; require preinstalled."
      return 0
    fi
    "$PYTHON_BIN" -m pip install -q "$mod"
  fi
}

# 最小必需包
ensure_py_module tqdm
ensure_py_module shortuuid
ensure_py_module pillow
ensure_py_module numpy
ensure_py_module transformers
# 方案一（method1）：Visual Evidence vs. No Foreground with TIE calculation
# 公式：1. (y_{gt}|X \oplus X_{X_bg})  > (y_{gt}|X_{null} \oplus X_{X_bg})
# 注：将原来的 y_{gen} 修改为 y_{gt}
# 运行两类推理，计算Visual TIE，并导出 DPO pairs（type=visual_indirect）

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC="${NPROC:-1}"
PORT="${PORT:-29500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TASK_MODE="${TASK_MODE:-model_shard}"

MODEL_PATH="${MODEL_PATH:-/share_docker/medgemma-4b-it}"
# 可选：基础模型（LoRA 基座），默认不传
MODEL_BASE="${MODEL_BASE:-}"
QUESTION_FILE="${QUESTION_FILE:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/slake_dpo_weighted.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/visual_indirect}"
PAIRS_OUT="${PAIRS_OUT:-/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/dpo_pairs_visual_indirect.jsonl}"

SCRIPT_VISUAL_TIE="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/inference_visual_tie.py"
if [[ "${MODEL_PATH,,}" == *"medgemma"* ]]; then
  SCRIPT_VISUAL_TIE="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/inference_visual_tie_medgemma.py"
fi
BUILDER="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/inference/build_dpo_pairs_visual_indirect.py"

mkdir -p "$OUTPUT_DIR"

TIE_RESULTS="$OUTPUT_DIR/visual_tie_results.jsonl"
STITCHED_DIR="$OUTPUT_DIR/stitched_visual_indirect"
mkdir -p "$STITCHED_DIR"

FULL_IMAGE_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs"
MASKED_IMAGE_DIR="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/processed_imgs"

echo "[visual_indirect] model=$MODEL_PATH questions=$QUESTION_FILE out=$OUTPUT_DIR"
echo "[visual_indirect] Computing Visual TIE (LL_full - LL_bg)..."

CONV_MODE="${CONV_MODE:-llava_med_v1}"
if [[ "${MODEL_PATH,,}" == *"medgemma"* ]]; then
  CONV_MODE="llava_v1"
  echo "[visual_indirect] Detected MedGEMMA path; using conv-mode=${CONV_MODE}"
fi

# 最小导入校验（提前失败而不是在分布式启动内部）
"$PYTHON_BIN" - <<'PY'
import sys
print("[precheck] Python:", sys.executable, sys.version)
try:
    import tqdm, shortuuid, numpy, PIL
    print("[precheck] Base imports OK")
except Exception as e:
    print("[precheck] Base import failed:", e)
    raise
PY

# 若为 MedGemma 任务，要求 transformers 可用，否则给出明确指引
if [[ "${MODEL_PATH,,}" == *"medgemma"* ]]; then
  if ! "$PYTHON_BIN" -c "import transformers" >/dev/null 2>&1; then
    echo "[warn] transformers 不可用，继续运行前请按 README 安装。当前将直接退出以避免无意义的错误。"
    exit 2
  fi
fi

HF_ACCELERATE="${HF_ACCELERATE:-false}"
if [[ "$HF_ACCELERATE" == "true" ]]; then
  export ACCELERATE_USE_DEVICE_MAP=1
fi

if [[ "$TASK_MODE" == "model_shard" ]]; then
  LAUNCH_NPROC=1
  echo "[visual_indirect] Task mode=model_shard; using device_map=auto with GPUs: ${CUDA_VISIBLE_DEVICES}"
else
  LAUNCH_NPROC="$NPROC"
  echo "[visual_indirect] Task mode=data_shard; ranks compute disjoint subsets (nproc=${LAUNCH_NPROC})"
fi

${TORCHRUN} --nproc_per_node="${LAUNCH_NPROC}" --master_port="${PORT}" "$SCRIPT_VISUAL_TIE" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --output-file "$TIE_RESULTS" \
  --output-image-folder "$STITCHED_DIR" \
  --full-image-folder "$FULL_IMAGE_DIR" \
  --masked-image-folder "$MASKED_IMAGE_DIR" \
  --batch-size "$BATCH_SIZE"

echo "[visual_indirect] Building DPO pairs from TIE results..."
if [[ "$LAUNCH_NPROC" -gt 1 ]]; then
  echo "[visual_indirect] Merging distributed outputs..."
  export TIE_BASE="$OUTPUT_DIR/visual_tie_results"
  python - <<PY
import glob, os, sys
base = os.environ.get('TIE_BASE')
if not base:
    print('[merge] TIE_BASE not set; skip merge')
    sys.exit(0)
out = base + '.jsonl'
parts = sorted(glob.glob(base + '.rank*.jsonl'))
if not parts:
    print('[merge] No rank shards found; skip merge')
    sys.exit(0)
with open(out, 'w', encoding='utf-8') as fw:
    for p in parts:
        with open(p, 'r', encoding='utf-8') as fr:
            for line in fr:
                fw.write(line)
print(f"Merged {len(parts)} shards -> {out}")
PY
fi

python "$BUILDER" \
  --tie-results-file "$TIE_RESULTS" \
  --output-pairs-file "$PAIRS_OUT" \
  --tie-threshold 0.0 \
  --w-min 0.05 \
  --beta 2.0

echo "[visual_indirect] Done. TIE results: $TIE_RESULTS, DPO pairs: $PAIRS_OUT"