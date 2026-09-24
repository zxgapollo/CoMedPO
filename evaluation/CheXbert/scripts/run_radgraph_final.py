#!/usr/bin/env python3
import os
import csv
import sys
import json

sys.path.insert(0, "/share_docker/workspace/Med/radgraph-master")
from radgraph import F1RadGraph

OUT_DIR = "/share_docker/workspace/Med/CheXbert/output"

MODELS = {
    "SFT": os.path.join(OUT_DIR, "sft_preds.csv"),
    "SFT_DPO": os.path.join(OUT_DIR, "sft_dpo_preds.csv"),
    "DPO": os.path.join(OUT_DIR, "dpo_preds.csv"),
    "DPO_SFT": os.path.join(OUT_DIR, "dpo_sft_preds.csv"),
    "MMedPO": os.path.join(OUT_DIR, "mmedpo_preds.csv"),
    "New_Pairs": os.path.join(OUT_DIR, "new_pairs_preds.csv"),
}


def load_column_list(path, column_name="Report Impression"):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            arr.append(row.get(column_name, ""))
    return arr


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    baseline_path = os.path.join(OUT_DIR, "baseline_preds.csv")
    if not os.path.exists(baseline_path):
        print("Baseline CSV missing:", baseline_path)
        return
    baseline = load_column_list(baseline_path)

    results = []

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print("Missing preds for", name, path)
            continue
        preds = load_column_list(path)
        n = min(len(baseline), len(preds))
        refs = baseline[:n]
        hyps = preds[:n]
        # try GPU, fallback to CPU
        try:
            f1rad = F1RadGraph(reward_level="all", model_type="radgraph-xl")
            mean_reward, _, hyp_ann, ref_ann = f1rad(refs, hyps)
        except Exception as e:
            print("GPU eval failed for", name, ":", e)
            f1rad = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=-1)
            mean_reward, _, hyp_ann, ref_ann = f1rad(refs, hyps)

        # save annotations
        try:
            write_json(os.path.join(OUT_DIR, f"radgraph_hyp_annotations_{name}.json"), hyp_ann)
            write_json(os.path.join(OUT_DIR, f"radgraph_ref_annotations_{name}.json"), ref_ann)
        except Exception as e:
            print("Failed to write annotations for", name, e)

        if isinstance(mean_reward, (list, tuple)):
            simple, partial, complete = mean_reward
        else:
            simple = mean_reward
            partial = complete = 0.0

        results.append(
            {
                "model": name,
                "samples": n,
                "simple": float(simple),
                "partial": float(partial),
                "complete": float(complete),
            }
        )

    # write combined CSV
    csv_path = os.path.join(OUT_DIR, "radgraph_summary_all.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "samples", "simple_f1", "partial_f1", "complete_f1"])
        for r in results:
            w.writerow([r["model"], r["samples"], f"{r['simple']:.4f}", f"{r['partial']:.4f}", f"{r['complete']:.4f}"])
    print("WROTE", csv_path)

    # write LaTeX table
    tex_path = os.path.join(OUT_DIR, "radgraph_table_all.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lrrrr}\n\\hline\nModel & Samples & Simple F1 & Partial F1 & Complete F1 \\\\\n\\hline\n")
        for r in results:
            f.write(f"{r['model']} & {r['samples']} & {r['simple']:.3f} & {r['partial']:.3f} & {r['complete']:.3f} \\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\caption{RadGraph F1 scores for all models}\n\\end{table}\n")
    print("WROTE", tex_path)


if __name__ == "__main__":
    main()

