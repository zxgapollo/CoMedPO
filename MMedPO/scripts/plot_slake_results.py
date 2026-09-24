#!/usr/bin/env python3
"""
Plot IoU distributions and threshold curves from evaluation JSON.
Bins are fixed: 0-20,20-40,40-60,60-80,80-100 (percent).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


def fixed_bins():
    # return bin edges in fractions
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    return edges, labels


def plot_histograms(models_ious, out_dir: Path):
    edges, labels = fixed_bins()
    fig, axes = plt.subplots(1, len(models_ious), figsize=(5 * max(1, len(models_ious)), 4), sharey=True)
    if len(models_ious) == 1:
        axes = [axes]
    for ax, (mname, ious) in zip(axes, models_ious.items()):
        arr = np.array(ious)
        # convert to percent for display
        arr_pct = arr * 100.0
        bins_pct = edges * 100.0
        ax.hist(arr_pct, bins=bins_pct, edgecolor="black")
        ax.set_title(mname)
        ax.set_xlabel("IoU (%)")
        ax.set_xticks(bins_pct)
    axes[0].set_ylabel("Count")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "iou_histograms.png", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(models_ious, out_dir: Path):
    thresholds = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mname, ious in models_ious.items():
        arr = np.array(ious)
        coverage = [(arr >= t).sum() / arr.size if arr.size > 0 else 0.0 for t in thresholds]
        ax.plot(thresholds * 100.0, coverage, label=mname)
    ax.set_xlabel("IoU Threshold (%)")
    ax.set_ylabel("Proportion of GTs with IoU >= threshold")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "iou_threshold_curve.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True, help="Evaluation JSON produced by eval_slake_iou.py")
    parser.add_argument("--out-dir", required=True, help="Directory to save plots")
    args = parser.parse_args()
    j = Path(args.results_json)
    out_dir = Path(args.out_dir)
    data = json.loads(j.read_text())
    models_ious = {}
    for m, v in data.get("models", {}).items():
        models_ious[m] = v.get("raw_ious", [])
    if len(models_ious) == 0:
        print("No models / ious found in results JSON")
        return
    plot_histograms(models_ious, out_dir)
    plot_threshold_curve(models_ious, out_dir)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()

