#!/usr/bin/env python3
"""
Generate simulated illustrative IoU plots (SIMULATED - not real results).
Saves:
 - outputs/combined/plots_combined/iou_histograms.png
 - outputs/combined/plots_combined/iou_threshold_curve.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "combined", "plots_combined")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# Simulated "improved" IoU distribution:
# Goal: peak around 0.72-0.78, low counts in 0-0.2, small long tail and modest spread.
# Large right-side peak (majority)
high_peak = np.clip(np.random.normal(loc=0.78, scale=0.07, size=900), 0.0, 1.0)
# Moderate midcases: increase to raise 40-60 bin
mid_noise = np.clip(np.random.normal(loc=0.50, scale=0.08, size=900), 0.0, 1.0)
# Small left hard-case tail concentrated below 0.2 but minimal mass
left_tail = np.clip(np.random.beta(a=0.7, b=12.0, size=80), 0.0, 1.0) * 0.18
sim_iou = np.concatenate([high_peak, mid_noise, left_tail])

# Clip and ensure length
sim_iou = np.clip(sim_iou, 0.0, 1.0)

# Histogram plot aggregated into five bins: 0-20,20-40,40-60,60-80,80-100
plt.figure(figsize=(7,4.5))
bin_edges = [0,20,40,60,80,100]
counts, _ = np.histogram(sim_iou*100, bins=bin_edges)
labels = ["0-20","20-40","40-60","60-80","80-100"]
x = np.arange(len(labels))
bar_colors = ["#d73027","#fc8d59","#fee090","#91bfdb","#4575b4"]  # gradient from low to high
plt.bar(x, counts, color=bar_colors, edgecolor="white", linewidth=0.7)
plt.xticks(x, labels)
plt.xlabel("IoU (%)")
plt.ylabel("Count")
# title removed per user request
plt.grid(axis="y", alpha=0.2)
mean_val = sim_iou.mean()
median_val = np.median(sim_iou)
plt.text(0.02, max(counts)*0.95, f"Mean {mean_val:.2f}   Median {median_val:.2f}", fontsize=9)
out_hist = os.path.join(OUT_DIR, "iou_histograms_simulated.png")
plt.tight_layout()
plt.savefig(out_hist, dpi=150)
plt.close()

# Threshold curve: percentage of samples with IoU >= threshold for thresholds 0..1
thresholds = np.linspace(0,1,101)
perc = [(sim_iou >= t).mean() for t in thresholds]

plt.figure(figsize=(7,4))
plt.plot(thresholds*100, np.array(perc)*100, color="#ff7f0e", linewidth=2)
plt.fill_between(thresholds*100, np.array(perc)*100, alpha=0.12, color="#ff7f0e")
plt.xlabel("IoU Threshold (%)")
plt.ylabel("Percent of samples ≥ threshold (%)")
plt.title("Simulated IoU Threshold Curve (SIMULATED ILLUSTRATION)")
plt.xlim(0,100)
plt.ylim(0,100)
plt.grid(alpha=0.2)
# no watermark text per user request
out_curve = os.path.join(OUT_DIR, "iou_threshold_curve_simulated.png")
plt.tight_layout()
plt.savefig(out_curve, dpi=150)
plt.close()

print("Wrote simulated plots:")
print(out_hist)
print(out_curve)

