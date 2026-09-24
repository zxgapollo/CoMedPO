#!/usr/bin/env python3
import json
import os
import re
import argparse
import random
from collections import defaultdict
import math
from statistics import mean

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def normalize_text(s: str):
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # remove surrounding artifacts like 'user', 'model' if present
    s = re.sub(r"<.*?>", " ", s)
    # remove punctuation except when inside words
    s = re.sub(r"[^\w\s]", "", s)
    s = s.strip()
    return s


def read_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            idx = j.get("id")
            answer = j.get("answer") or j.get("pred") or ""
            gt = j.get("gt_answer") or j.get("gt") or j.get("ground_truth") or ""
            image = j.get("image") or j.get("image_path") or ""
            data[idx] = {"answer_raw": answer, "gt_raw": gt, "image": image}
    return data


def merge_three(a, b, c):
    ids = sorted(set(list(a.keys()) + list(b.keys()) + list(c.keys())))
    rows = []
    for i in ids:
        row = {
            "id": i,
            "a_raw": a.get(i, {}).get("answer_raw", ""),
            "b_raw": b.get(i, {}).get("answer_raw", ""),
            "c_raw": c.get(i, {}).get("answer_raw", ""),
            "gt_raw": (a.get(i) or b.get(i) or c.get(i) or {}).get("gt_raw", ""),
            "image_raw": (a.get(i) or b.get(i) or c.get(i) or {}).get("image", ""),
        }
        row["a"] = normalize_text(row["a_raw"])
        row["b"] = normalize_text(row["b_raw"])
        row["c"] = normalize_text(row["c_raw"])
        row["gt"] = normalize_text(row["gt_raw"])
        rows.append(row)
    return rows


def ngram_counts(tokens, n):
    counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def modified_precision(reference_tokens, hypothesis_tokens, n):
    ref_counts = ngram_counts(reference_tokens, n)
    hyp_counts = ngram_counts(hypothesis_tokens, n)
    if not hyp_counts:
        return 0.0
    clipped = 0
    total = 0
    for gram, cnt in hyp_counts.items():
        total += cnt
        clipped += min(cnt, ref_counts.get(gram, 0))
    return clipped / total if total > 0 else 0.0


def bleu_sentence(reference, hypothesis, max_n=4, weights=None):
    # simple cumulative BLEU implementation
    if weights is None:
        weights = [0.25] * max_n
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    precisions = []
    for n in range(1, max_n + 1):
        p_n = modified_precision(ref_tokens, hyp_tokens, n)
        precisions.append(p_n)
    # geometric mean of precisions with weights, avoid log(0)
    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
        geo_mean = math.exp(log_sum)
    # brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean


def compute_bleu_for_rows(rows, key):
    scores = []
    for r in rows:
        ref = r["gt"] or ""
        hyp = r.get(key, "") or ""
        scores.append(bleu_sentence(ref, hyp))
    return scores


def bootstrap_mean(scores, n_boot=1000, seed=42):
    rng = random.Random(seed)
    n = len(scores)
    samples = []
    for _ in range(n_boot):
        res = [scores[rng.randrange(0, n)] for __ in range(n)]
        samples.append(mean(res))
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot) - 1]
    return mean(samples), (lo, hi)


def exact_match(gt, pred):
    return gt == pred


def mcnemar_pval(pair1, pair2):
    # pair1 and pair2 are lists of booleans (True correct, False wrong)
    b = 0  # pair1 correct, pair2 wrong
    c = 0  # pair1 wrong, pair2 correct
    for x, y in zip(pair1, pair2):
        if x and not y:
            b += 1
        elif not x and y:
            c += 1
    n = b + c
    if n == 0:
        return 1.0, b, c
    # normal approximation
    z = (b - c) / math.sqrt(n)
    # two-sided p-value
    # Phi using erf
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    p = 2 * (1 - phi(abs(z)))
    return p, b, c


def save_outputs(rows, out_dir, scores_a, scores_b, scores_c, acc_a, acc_b, acc_c, boot_a, boot_b, boot_c, p_ab, p_ac, p_bc):
    os.makedirs(out_dir, exist_ok=True)
    import csv

    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "gt",
                "a",
                "b",
                "c",
                "bleu_a",
                "bleu_b",
                "bleu_c",
            ]
        )
        for r, sa, sb, sc in zip(rows, scores_a, scores_b, scores_c):
            writer.writerow([r["id"], r["gt"], r["a_raw"], r["b_raw"], r["c_raw"], f"{sa:.6f}", f"{sb:.6f}", f"{sc:.6f}"])

    report = os.path.join(out_dir, "summary_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# 分析汇总\n\n")
        f.write(f"- 样本数: {len(rows)}\n")
        f.write(f"- 平均 BLEU (A): {mean(scores_a):.6f}, bootstrap mean {boot_a[0]:.6f}, 95% CI {boot_a[1]}\n")
        f.write(f"- 平均 BLEU (B): {mean(scores_b):.6f}, bootstrap mean {boot_b[0]:.6f}, 95% CI {boot_b[1]}\n")
        f.write(f"- 平均 BLEU (C): {mean(scores_c):.6f}, bootstrap mean {boot_c[0]:.6f}, 95% CI {boot_c[1]}\n")
        f.write("\n## 精确匹配(辅助)\n")
        f.write(f"- Accuracy A: {acc_a:.4f}\n")
        f.write(f"- Accuracy B: {acc_b:.4f}\n")
        f.write(f"- Accuracy C: {acc_c:.4f}\n")
        f.write("\n## McNemar 检验(p-values, normal approx):\n")
        f.write(f"- A vs B: p={p_ab[0]:.6f}, b={p_ab[1]}, c={p_ab[2]}\n")
        f.write(f"- A vs C: p={p_ac[0]:.6f}, b={p_ac[1]}, c={p_ac[2]}\n")
        f.write(f"- B vs C: p={p_bc[0]:.6f}, b={p_bc[1]}, c={p_bc[2]}\n")
        # embed plots if exist
        f.write("\n## 可视化\n")
        if os.path.exists(os.path.join(out_dir, "bleu_means.png")):
            f.write("![BLEU means](bleu_means.png)\n\n")
        if os.path.exists(os.path.join(out_dir, "accuracy.png")):
            f.write("![Accuracy](accuracy.png)\n\n")

    # write LaTeX summary table
    tex_path = os.path.join(out_dir, "summary.tex")
    with open(tex_path, "w", encoding="utf-8") as tf:
        tf.write("\\begin{table}[ht]\n")
        tf.write("\\centering\n")
        tf.write("\\begin{tabular}{lccc}\n")
        tf.write("\\toprule\n")
        tf.write("Metric & A (s0.2) & B (s0.5) & C (zero)\\\\\n")
        tf.write("\\midrule\n")
        tf.write(f"Mean BLEU & {mean(scores_a):.4f} & {mean(scores_b):.4f} & {mean(scores_c):.4f}\\\\\n")
        tf.write(f"BLEU 95\\% CI & [{boot_a[1][0]:.4f}, {boot_a[1][1]:.4f}] & [{boot_b[1][0]:.4f}, {boot_b[1][1]:.4f}] & [{boot_c[1][0]:.4f}, {boot_c[1][1]:.4f}]\\\\\n")
        tf.write(f"Accuracy & {acc_a:.4f} & {acc_b:.4f} & {acc_c:.4f}\\\\\n")
        tf.write("\\midrule\n")
        tf.write(f"A vs B p & \\multicolumn{{3}}{{l}}{{{p_ab[0]:.6f}}}\\\\\n")
        tf.write(f"A vs C p & \\multicolumn{{3}}{{l}}{{{p_ac[0]:.6f}}}\\\\\n")
        tf.write(f"B vs C p & \\multicolumn{{3}}{{l}}{{{p_bc[0]:.6f}}}\\\\\n")
        tf.write("\\bottomrule\n")
        tf.write("\\end{tabular}\n")
        tf.write("\\caption{Summary of BLEU and accuracy for three conditions}\n")
        tf.write("\\label{tab:summary}\n")
        tf.write("\\end{table}\n")

    # embed example screenshots in a LaTeX file (if images available)
    examples_tex = os.path.join(out_dir, "examples.tex")
    diffs_examples = []
    for r in rows:
        if not (r["a"] == r["b"] == r["c"]):
            if r.get("image_raw"):
                diffs_examples.append(r)
    with open(examples_tex, "w", encoding="utf-8") as ef:
        ef.write("% Example images and their predictions\n")
        ef.write("\\begin{figure}[ht]\n")
        ef.write("\\centering\n")
        max_show = min(5, len(diffs_examples))
        for i in range(max_show):
            r = diffs_examples[i]
            img_path = r.get("image_raw", "")
            # include graphics if exists relative to workspace
            if img_path and os.path.exists(os.path.join(os.path.dirname(out_dir), img_path)):
                ef.write(f"\\begin{{minipage}}{{0.19\\linewidth}}\\centering\n")
                ef.write(f"\\includegraphics[width=\\linewidth]{{{img_path}}}\n")
                ef.write(f"\\\\\n")
                ef.write(f"\\scriptsize ID: {r['id']}\\\\\n")
                ef.write(f"\\scriptsize GT: {r['gt']}\\\\\n")
                ef.write(f"\\scriptsize A: {normalize_text(r['a_raw'])[:40]}\\\\\n")
                ef.write(f"\\scriptsize B: {normalize_text(r['b_raw'])[:40]}\\\\\n")
                ef.write(f"\\scriptsize C: {normalize_text(r['c_raw'])[:40]}\\\\\n")
                ef.write("\\end{minipage}\n")
        ef.write("\\caption{示例 — 预测不一致的样例（最多显示5个）}\n")
        ef.write("\\end{figure}\n")

    # diffs: export inconsistent examples
    diffs = []
    for r, sa, sb, sc in zip(rows, scores_a, scores_b, scores_c):
        if not (r["a"] == r["b"] == r["c"]):
            diffs.append({"id": r["id"], "gt": r["gt"], "a": r["a_raw"], "b": r["b_raw"], "c": r["c_raw"]})
    diffs_file = os.path.join(out_dir, "diff_examples.jsonl")
    with open(diffs_file, "w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # plots
    if plt is not None:
        labels = ["A", "B", "C"]
        means = [mean(scores_a), mean(scores_b), mean(scores_c)]
        cis_lo = [boot_a[1][0], boot_b[1][0], boot_c[1][0]]
        cis_hi = [boot_a[1][1], boot_b[1][1], boot_c[1][1]]
        yerr = [ [m - lo for m, lo in zip(means, cis_lo)], [hi - m for hi, m in zip(cis_hi, means)] ]
        x = range(len(labels))
        plt.figure(figsize=(6,4))
        plt.bar(x, means, yerr=yerr, capsize=6)
        plt.xticks(x, labels)
        plt.ylabel("BLEU")
        plt.title("Group mean BLEU with 95% CI (bootstrap)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bleu_means.png"))

        # accuracy bar
        accs = [acc_a, acc_b, acc_c]
        plt.figure(figsize=(6,4))
        plt.bar(x, accs)
        plt.xticks(x, labels)
        plt.ylabel("Exact-match Accuracy")
        plt.title("Exact-match Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "accuracy.png"))

    print("Outputs written to", out_dir)
    print("Summary CSV:", summary_csv)
    print("Report:", report)
    print("Diffs:", diffs_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/rebuttal/fill_remove_noise_s0.2_scale_1.0_slake_500_preds.jsonl")
    parser.add_argument("--b", default="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/rebuttal/fill_remove_noise_s0.5_scale_1.5_slake_500_preds.jsonl")
    parser.add_argument("--c", default="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/rebuttal/fill_remove_zero_scale_1.0_slake_500_preds.jsonl")
    parser.add_argument("--out", default="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/rebuttal/analysis")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    a = read_jsonl(args.a)
    b = read_jsonl(args.b)
    c = read_jsonl(args.c)
    rows = merge_three(a, b, c)
    scores_a = compute_bleu_for_rows(rows, "a")
    scores_b = compute_bleu_for_rows(rows, "b")
    scores_c = compute_bleu_for_rows(rows, "c")

    boot_a = bootstrap_mean(scores_a, n_boot=args.bootstrap)
    boot_b = bootstrap_mean(scores_b, n_boot=args.bootstrap)
    boot_c = bootstrap_mean(scores_c, n_boot=args.bootstrap)

    # exact-match accuracy
    corr_a = [exact_match(r["gt"], r["a"]) for r in rows]
    corr_b = [exact_match(r["gt"], r["b"]) for r in rows]
    corr_c = [exact_match(r["gt"], r["c"]) for r in rows]
    acc_a = sum(corr_a) / len(corr_a)
    acc_b = sum(corr_b) / len(corr_b)
    acc_c = sum(corr_c) / len(corr_c)

    p_ab = mcnemar_pval(corr_a, corr_b)
    p_ac = mcnemar_pval(corr_a, corr_c)
    p_bc = mcnemar_pval(corr_b, corr_c)

    save_outputs(rows, args.out, scores_a, scores_b, scores_c, acc_a, acc_b, acc_c, boot_a, boot_b, boot_c, p_ab, p_ac, p_bc)


if __name__ == "__main__":
    main()
