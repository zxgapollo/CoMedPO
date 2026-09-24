#!/bin/bash
set -euo pipefail

# Shell script to run RadGraph F1 evaluation using the local radgraph repository.
# Expects CSV files in /share_docker/workspace/Med/CheXbert/output:
#   - baseline_gt.csv
#   - sft_preds.csv
#   - sft_dpo_preds.csv
#   - dpo_preds.csv
# Writes:
#   - radgraph_summary.csv
#   - radgraph_table.tex
# Requires: a conda env named "radgraph_env" (or active env with radgraph deps),
# and the radgraph model files placed under the cache dir:
#   /root/.cache/radgraph/0.1.18/radgraph-xl/  (contains config.json, weights.th, vocabulary/, etc.)

LOG_DIR="/share_docker/workspace/Med/CheXbert/output"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_radgraph_repo.log"
exec >"$LOG_FILE" 2>&1

echo "Started RadGraph run at $(date)"

# Activate conda env if available
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        if conda env list | grep -q "^\\s*radgraph_env\\s"; then
            echo "Activating conda env radgraph_env"
            conda activate radgraph_env
        else
            echo "Conda env 'radgraph_env' not found; please create it and install dependencies."
        fi
    fi
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "python not found on PATH; aborting"
    exit 1
fi

echo "Using python: $PYTHON_BIN"

# Configure HuggingFace/transformers to work offline and point to local cache.
# Set HF cache root where we'll link the local model so AutoTokenizer.from_pretrained()
# can load it without network access.
export HF_HOME="/root/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOCAL_MODEL_DIR="/share_docker/workspace/Med/CheXbert/BiomedVLP-CXR-BERT-general"
HF_CACHE="$HF_HOME/transformers/microsoft"
mkdir -p "$HF_CACHE"
if [ -d "$LOCAL_MODEL_DIR" ]; then
    if [ ! -e "$HF_CACHE/BiomedVLP-CXR-BERT-general" ]; then
        echo "Linking local HF model into cache: $LOCAL_MODEL_DIR -> $HF_CACHE/BiomedVLP-CXR-BERT-general"
        # try symlink, fallback to copy
        if ! ln -s "$LOCAL_MODEL_DIR" "$HF_CACHE/BiomedVLP-CXR-BERT-general" 2>/dev/null; then
            echo "symlink failed, copying instead"
            cp -r "$LOCAL_MODEL_DIR" "$HF_CACHE/" || true
        fi
    else
        echo "HF cache entry already exists: $HF_CACHE/BiomedVLP-CXR-BERT-general"
    fi
else
    echo "Local HF model directory not found at $LOCAL_MODEL_DIR"
fi

# Allow optional first argument to set GPUs (e.g. ./run_radgraph_repo.sh 2,3)
if [ -n "${1-}" ]; then
    echo "Setting CUDA_VISIBLE_DEVICES to '$1'"
    export CUDA_VISIBLE_DEVICES="$1"
fi

# Ensure radgraph model cache exists and contains required files
RAD_CACHE_ROOT="/root/.cache/radgraph/0.1.18"
RAD_MODEL_DIR="$RAD_CACHE_ROOT/radgraph-xl"
MISSING_FILES=0
if [ ! -d "$RAD_MODEL_DIR" ]; then
    echo "RadGraph model dir not found at $RAD_MODEL_DIR"
    MISSING_FILES=1
else
    if [ ! -f "$RAD_MODEL_DIR/weights.th" ]; then
        echo "Missing weights.th in $RAD_MODEL_DIR"
        MISSING_FILES=1
    fi
    if [ ! -d "$RAD_MODEL_DIR/vocabulary" ]; then
        echo "Missing vocabulary/ in $RAD_MODEL_DIR"
        MISSING_FILES=1
    fi
fi

if [ "$MISSING_FILES" -eq 1 ]; then
    # Try to find local radgraph archive in workspace and extract
    LOCAL_ARCHIVE="/share_docker/workspace/Med/CheXbert/radgraph-xl.tar.gz"
    LOCAL_DIR="/share_docker/workspace/Med/CheXbert/radgraph-xl"
    if [ -f "$LOCAL_ARCHIVE" ]; then
        echo "Found local radgraph archive at $LOCAL_ARCHIVE; extracting to $RAD_MODEL_DIR"
        mkdir -p "$RAD_MODEL_DIR"
        tar -xzf "$LOCAL_ARCHIVE" -C "$RAD_MODEL_DIR" || echo "Extraction failed"
    elif [ -d "$LOCAL_DIR" ]; then
        echo "Copying local radgraph dir $LOCAL_DIR -> $RAD_MODEL_DIR"
        mkdir -p "$RAD_MODEL_DIR"
        cp -a "$LOCAL_DIR/." "$RAD_MODEL_DIR/"
    else
        echo "RadGraph model not found locally. Please place radgraph-xl.tar.gz at $LOCAL_ARCHIVE or unpacked dir at $LOCAL_DIR"
        echo "Aborting."
        exit 1
    fi
fi

# Run evaluation using radgraph repo code
 "$PYTHON_BIN" - <<'PY'
import os, csv, sys, json

repo_path = "/share_docker/workspace/Med/radgraph-master"
if not os.path.isdir(repo_path):
    print("radgraph repo not found at", repo_path)
    sys.exit(1)

sys.path.insert(0, repo_path)
try:
    from radgraph import F1RadGraph
except Exception as e:
    print("Failed to import radgraph from repo:", e)
    sys.exit(1)

outdir = "/share_docker/workspace/Med/CheXbert/output"
baseline_gt = os.path.join(outdir, "baseline_gt.csv")
files = {
    "SFT": os.path.join(outdir, "sft_preds.csv"),
    "SFT_DPO": os.path.join(outdir, "sft_dpo_preds.csv"),
    "DPO": os.path.join(outdir, "dpo_preds.csv"),
}

if not os.path.exists(baseline_gt):
    print("MISSING_BASELINE_GT", baseline_gt)
    sys.exit(1)

baseline_list = []
has_image = False
with open(baseline_gt, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        baseline_list.append(row.get("Report Impression", ""))
        if row.get("image", "").strip():
            has_image = True

baseline_map = {}
if has_image:
    # build mapping from image -> impression (last one wins if duplicates)
    with open(baseline_gt, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            baseline_map[row.get("image", "")] = row.get("Report Impression", "")
else:
    print("No image IDs found in baseline_gt.csv; will align preds by row order.")

results = []
model_type = "radgraph-xl"
print("Initializing F1RadGraph with model_type=", model_type)
f1rad = F1RadGraph(reward_level="all", model_type=model_type)

for name, path in files.items():
    if not os.path.exists(path):
        print("MISSING", path)
        continue
    hyps = []
    refs = []
    if has_image:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                img = row.get("image", "")
                pred = row.get("Report Impression", "")
                if img in baseline_map:
                    refs.append(baseline_map[img] or "")
                    hyps.append(pred or "")
    else:
        # align by row order
        preds_list = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                preds_list.append(row.get("Report Impression", ""))
        n = min(len(baseline_list), len(preds_list))
        if len(baseline_list) != len(preds_list):
            print(f\"Warning: baseline rows={len(baseline_list)} preds rows={len(preds_list)}; aligning first {n} entries.\")
        refs = baseline_list[:n]
        hyps = preds_list[:n]
    print("Evaluating", name, "samples=", len(hyps))
    mean_reward, reward_list, hyp_ann, ref_ann = f1rad(refs, hyps)

    # save RadGraph annotations for debugging (one file per model)
    try:
        ann_out_h = os.path.join(outdir, f"radgraph_hyp_annotations_{name}.json")
        ann_out_r = os.path.join(outdir, f"radgraph_ref_annotations_{name}.json")
        with open(ann_out_h, "w", encoding="utf-8") as fh:
            json.dump(hyp_ann, fh, ensure_ascii=False, indent=2)
        with open(ann_out_r, "w", encoding="utf-8") as fr:
            json.dump(ref_ann, fr, ensure_ascii=False, indent=2)
        print("WROTE annotations:", ann_out_h, ann_out_r)
    except Exception as e:
        print("Failed to write annotations:", e)

    # quick diagnostic: warn if many refs or hyps are very short (likely no entities)
    try:
        short_refs = sum(1 for r in refs if (not r) or len(r.split()) < 3)
        short_hyps = sum(1 for h in hyps if (not h) or len(h.split()) < 3)
        if short_refs > 0 or short_hyps > 0:
            print(f\"Diagnostic: short refs={short_refs}/{len(refs)}, short hyps={short_hyps}/{len(hyps)}\") 
    except Exception:
        pass
    results.append(
        {
            "name": name,
            "samples": len(hyps),
            "simple": float(mean_reward[0]),
            "partial": float(mean_reward[1]),
            "complete": float(mean_reward[2]),
        }
    )

# write CSV summary
csv_path = os.path.join(outdir, "radgraph_summary.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "samples", "simple_f1", "partial_f1", "complete_f1"])
    for r in results:
        w.writerow([r["name"], r["samples"], f"{r['simple']:.4f}", f"{r['partial']:.4f}", f"{r['complete']:.4f}"])
print("WROTE", csv_path)

# write LaTeX table
tex_path = os.path.join(outdir, "radgraph_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(
        "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lrrrr}\n\\hline\nModel & Samples & Simple F1 & Partial F1 & Complete F1 \\\\\n\\hline\n"
    )
    for r in results:
        line = "%s & %d & %.3f & %.3f & %.3f \\\\" % (r["name"], r["samples"], r["simple"], r["partial"], r["complete"])
        f.write(line + "\n")
    f.write("\\hline\n\\end{tabular}\n\\caption{RadGraph F1 scores for models on SLAKE subset}\n\\end{table}\n")
print("WROTE", tex_path)
PY

echo "Finished at $(date)"

exit 0

