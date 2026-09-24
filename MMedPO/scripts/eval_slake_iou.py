#!/usr/bin/env python3
"""
Compute IoU between ground-truth masks and predicted masks for each model.
Matching strategy: for each GT mask, find the predicted mask with maximum IoU (greedy),
mark matched predictions as used. Unmatched GTs -> FN, unmatched preds -> FP.

Outputs:
 - <out_json> containing per-model per-image IoU list and summary statistics.
 - Optionally prints basic CSV to stdout.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import os
import math


def load_mask(path: Path):
    im = Image.open(path).convert("L")
    arr = np.array(im)
    return (arr > 127).astype(np.uint8)


def iou(mask1: np.ndarray, mask2: np.ndarray):
    if mask1.shape != mask2.shape:
        # resize mask2 to mask1
        return 0.0
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def gather_gt_masks(gt_mask_dir: Path):
    # expects files like <sample_id>.png or <sample_id>__mask_0.png
    mapping = defaultdict(list)
    for p in gt_mask_dir.rglob("*.png"):
        sid = p.stem.split("__")[0]
        mapping[sid].append(p)
    return mapping


def gather_pred_masks(pred_root: Path, model_name: str):
    pred_dir = pred_root / model_name / "pred_masks"
    mapping = defaultdict(list)
    if not pred_dir.exists():
        return mapping
    for p in pred_dir.rglob("*.png"):
        sid = p.stem.split("__")[0]
        mapping[sid].append(p)
    return mapping


def evaluate_model(gt_map, pred_map):
    results = {}
    ious_all = []
    for sid, gt_list in gt_map.items():
        gt_masks = [load_mask(p) for p in gt_list]
        preds = pred_map.get(sid, [])
        pred_masks = [load_mask(p) for p in preds]
        used = [False] * len(pred_masks)
        per_image_ious = []
        for gm in gt_masks:
            best_iou = 0.0
            best_idx = -1
            for j, pm in enumerate(pred_masks):
                if used[j]:
                    continue
                val = iou(gm, pm)
                if val > best_iou:
                    best_iou = val
                    best_idx = j
            if best_idx >= 0:
                used[best_idx] = True
            per_image_ious.append(float(best_iou))
            ious_all.append(float(best_iou))
        # unmatched preds count as FPs but do not contribute IoU list
        results[sid] = {"per_gt_ious": per_image_ious, "n_gt": len(gt_masks), "n_pred": len(pred_masks)}
    return results, ious_all


def summarize_ious(ious_list):
    if len(ious_list) == 0:
        return {"mean": 0.0, "median": 0.0, "count": 0, "pct_gt50": 0.0}
    arr = np.array(ious_list)
    mean = float(arr.mean())
    median = float(np.median(arr))
    count = int(arr.size)
    pct_gt50 = float((arr >= 0.5).sum() / arr.size)
    return {"mean": mean, "median": median, "count": count, "pct_gt50": pct_gt50}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mask-dir", required=True, help="Directory with ground-truth masks")
    parser.add_argument("--pred-dir", required=True, help="Root directory where model predictions were saved")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--export-zero-iou", action="store_true", help="Export IoU=0 cases to JSONL and visuals/")
    args = parser.parse_args()

    gt_dir = Path(args.gt_mask_dir)
    pred_root = Path(args.pred_dir)
    out_json = Path(args.out_json)
    ensure = out_json.parent
    ensure.mkdir(parents=True, exist_ok=True)

    gt_map = gather_gt_masks(gt_dir)
    models = []
    for d in pred_root.iterdir():
        if d.is_dir():
            if (d / "pred_masks").exists():
                models.append(d.name)
    final = {"models": {}}
    zero_cases = []
    for m in models:
        pred_map = gather_pred_masks(pred_root, m)
        res_map, ious_all = evaluate_model(gt_map, pred_map)
        stats = summarize_ious(ious_all)
        # convert numpy floats
        final["models"][m] = {"per_image": {k: v for k, v in res_map.items()}, "stats": stats, "raw_ious": [float(x) for x in ious_all]}
        # collect zero-iou cases from res_map
        if args.export_zero_iou:
            for sid, info in res_map.items():
                per = info.get("per_gt_ious", [])
                gt_list = info.get("gt_list", [])
                # per_gt_ious aligns with gt_list order; we stored only ious; gather where iou==0
                for idx_i, val in enumerate(per):
                    if float(val) == 0.0:
                        entry = {"model": m, "sample_id": sid, "gt_index": idx_i, "iou": 0.0}
                        # try to provide paths
                        try:
                            gt_paths = gt_map.get(sid, [])
                            entry["gt_path"] = str(gt_paths[idx_i]) if idx_i < len(gt_paths) else None
                        except Exception:
                            entry["gt_path"] = None
                        entry["preds"] = [str(p) for p in pred_map.get(sid, [])]
                        zero_cases.append(entry)

    with open(out_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved evaluation JSON to {out_json}")
    if args.export_zero_iou and len(zero_cases) > 0:
        zl = out_json.parent / "zero_iou_cases.jsonl"
        with open(zl, "w") as fz:
            for e in zero_cases:
                fz.write(json.dumps(e) + "\\n")
        visuals_dir = out_json.parent / "zero_iou_visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        # attempt to copy relevant files (gt and preds) into visuals_dir per entry
        from shutil import copy2
        for i, e in enumerate(zero_cases):
            sub = visuals_dir / f"{i:04d}_{e.get('sample_id')}"
            sub.mkdir(parents=True, exist_ok=True)
            if e.get("gt_path"):
                try:
                    copy2(e["gt_path"], sub / Path(e["gt_path"]).name)
                except Exception:
                    pass
            for p in e.get("preds", []):
                try:
                    copy2(p, sub / Path(p).name)
                except Exception:
                    pass
        print(f"Exported {len(zero_cases)} zero-IoU cases to {zl} and visuals under {visuals_dir}")


if __name__ == "__main__":
    main()

