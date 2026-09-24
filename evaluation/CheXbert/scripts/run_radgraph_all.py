#!/usr/bin/env python3
import os
import csv
import json
import sys

sys.path.insert(0, "/share_docker/workspace/Med/radgraph-master")
from radgraph import F1RadGraph

def load_csv_list(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append(row.get("Report Impression", ""))
    return items

def has_image_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("image", "").strip():
                return True
    return False

def build_image_map(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m[row.get("image", "")] = row.get("Report Impression", "")
    return m

def main():
    outdir = "/share_docker/workspace/Med/CheXbert/output"
    baseline = os.path.join(outdir, "baseline_preds.csv")
    conds = {
        "Base_line": os.path.join(outdir, "baseline_preds.csv"),
        "DPO_SFT": os.path.join(outdir, "dpo_sft_preds.csv"),
        "MMedPO": os.path.join(outdir, "mmedpo_preds.csv"),
        "New_Pairs": os.path.join(outdir, "new_pairs_preds.csv"),
        "SFT": os.path.join(outdir, "sft_preds.csv"),
        "SFT_DPO_R2": os.path.join(outdir, "sft_dpo_round2_preds.csv"),
    }

    if not os.path.exists(baseline):
        print("Missing baseline:", baseline)
        return

    use_image = has_image_ids(baseline)
    print("baseline_has_image =", use_image)

    if use_image:
        base_map = build_image_map(baseline)
    else:
        base_list = load_csv_list(baseline)

    # Force CPU execution to avoid GPU OOM on large batches
    f1rad = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=-1)
    results = []

    for name, path in conds.items():
        if not os.path.exists(path):
            print("MISSING", path)
            continue
        refs = []
        hyps = []
        if use_image:
            with open(path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    img = row.get("image", "")
                    pred = row.get("Report Impression", "")
                    if img in base_map:
                        refs.append(base_map.get(img, "") or "")
                        hyps.append(pred or "")
        else:
            pred_list = load_csv_list(path)
            n = min(len(base_list), len(pred_list))
            if len(base_list) != len(pred_list):
                print(f"Warning: baseline rows={len(base_list)} preds rows={len(pred_list)}; aligning first {n} entries.")
            refs = base_list[:n]
            hyps = pred_list[:n]

        print(f"Evaluating {name}, samples= {len(hyps)}")
        mean_reward, reward_list, hyp_ann, ref_ann = f1rad(refs, hyps)

        # write annotations for inspection
        try:
            ah = os.path.join(outdir, f"radgraph_hyp_annotations_{name}.json")
            ar = os.path.join(outdir, f"radgraph_ref_annotations_{name}.json")
            with open(ah, "w", encoding="utf-8") as fh:
                json.dump(hyp_ann, fh, ensure_ascii=False, indent=2)
            with open(ar, "w", encoding="utf-8") as fr:
                json.dump(ref_ann, fr, ensure_ascii=False, indent=2)
            print("WROTE annotations:", ah, ar)
        except Exception as e:
            print("Failed writing annotations:", e)

        results.append((name, len(hyps), mean_reward))

    # write combined CSV and tex
    csv_path = os.path.join(outdir, "radgraph_summary_all.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "samples", "simple_f1", "partial_f1", "complete_f1"])
        for name, samples, mean in results:
            if isinstance(mean, (list, tuple)):
                s1, s2, s3 = mean
            else:
                s1 = mean; s2 = s3 = 0.0
            w.writerow([name, samples, f"{s1:.4f}", f"{s2:.4f}", f"{s3:.4f}"])
    print("WROTE", csv_path)

    tex_path = os.path.join(outdir, "radgraph_table_all.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\\\begin{table}[ht]\n\\\\centering\n\\\\begin{tabular}{lrrrr}\n\\\\hline\nModel & Samples & Simple F1 & Partial F1 & Complete F1 \\\\\n\\\\hline\n")
        for name, samples, mean in results:
            if isinstance(mean, (list, tuple)):
                s1, s2, s3 = mean
            else:
                s1 = mean; s2 = s3 = 0.0
            f.write(f"{name} & {samples} & {s1:.3f} & {s2:.3f} & {s3:.3f} \\\\\\\\n")
        f.write("\\\\hline\n\\\\end{tabular}\n\\\\caption{RadGraph F1 scores for all evaluated models}\n\\\\end{table}\n")
    print("WROTE", tex_path)

if __name__ == '__main__':
    main()

