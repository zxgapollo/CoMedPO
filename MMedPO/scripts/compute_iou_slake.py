#!/usr/bin/env python3
\"\"\"compute_iou_slake.py

Lightweight script to compute per-sample IOU between ground-truth and model predictions
and export a CSV with columns: sample_id, model_name, iou, bin_label.

Supported GT inputs:
 - --gt-csv : CSV with column `sample_id` and either `mask_path` or bbox columns `x1,y1,x2,y2`
 - --gt-mask-dir : directory with binary mask images named `<sample_id>.(png|jpg)`

Supported prediction inputs (one or more):
 - --pred model_name:PATH where PATH can be:
     - a directory with mask images named `<sample_id>.png`
     - a CSV with `sample_id` and `mask_path` or bbox columns
     - a JSONL file where each line is a JSON object with keys `id` or `sample_id` (and either `mask_path` or `bbox`)

The script does NOT run any model inference; it only processes existing prediction files.

Dependencies: pandas, numpy, pillow
Example:
  python compute_iou_slake.py --gt-csv /path/to/gt.csv --pred SAM:/path/to/sam_preds_dir MedKLIP:/path/to/medklip_preds.csv --out outputs/slake_iou_per_sample.csv
\"\"\"

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image


def read_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    # treat any non-zero as foreground
    return (arr > 0).astype(np.uint8)


def bbox_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    # boxes as (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - interArea
    if union <= 0.0:
        return 0.0
    return float(interArea / union)


def mask_iou_from_paths(gt_path: Path, pred_path: Path) -> float:
    gt_mask = read_mask(gt_path)
    pred_mask = read_mask(pred_path)
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def parse_bbox_row(row: Dict, prefix: str = "") -> Optional[Tuple[float, float, float, float]]:
    try:
        x1 = float(row.get(prefix + "x1", row.get("x1")))
        y1 = float(row.get(prefix + "y1", row.get("y1")))
        x2 = float(row.get(prefix + "x2", row.get("x2")))
        y2 = float(row.get(prefix + "y2", row.get("y2")))
        return (x1, y1, x2, y2)
    except Exception:
        return None


def load_gt_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError("GT CSV must contain 'sample_id' column")
    return df


def build_mask_map_from_dir(mask_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(mask_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            sample_id = p.stem
            rows.append({"sample_id": sample_id, "mask_path": str(p)})
    return pd.DataFrame(rows)


def load_preds_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        # try id or image
        if "id" in df.columns:
            df = df.rename(columns={"id": "sample_id"})
        elif "image" in df.columns:
            df = df.rename(columns={"image": "sample_id"})
        else:
            raise ValueError("Pred CSV must contain 'sample_id'/'id'/'image' column")
    return df


def load_preds_from_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = obj.get("sample_id") or obj.get("id") or obj.get("image")
            if sid is None:
                continue
            row = {"sample_id": sid}
            # accept bbox or mask_path
            if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
                row.update({"bbox_x1": obj["bbox"][0], "bbox_y1": obj["bbox"][1], "bbox_x2": obj["bbox"][2], "bbox_y2": obj["bbox"][3]})
            if "mask_path" in obj:
                row["mask_path"] = obj["mask_path"]
            rows.append(row)
    return pd.DataFrame(rows)


def compute_iou_for_pair(sample_id: str, gt_row: Dict, pred_row: Dict, repo_root: Path) -> Optional[float]:
    # prefer masks if both provide mask_path
    if "mask_path" in gt_row and pd.notna(gt_row["mask_path"]):
        gt_mask_path = repo_root.joinpath(str(gt_row["mask_path"])) if not Path(gt_row["mask_path"]).is_absolute() else Path(gt_row["mask_path"])
    else:
        gt_mask_path = None

    pred_mask_path = None
    if "mask_path" in pred_row and pd.notna(pred_row["mask_path"]):
        pred_mask_path = repo_root.joinpath(str(pred_row["mask_path"])) if not Path(pred_row["mask_path"]).is_absolute() else Path(pred_row["mask_path"])

    if gt_mask_path and pred_mask_path and gt_mask_path.exists() and pred_mask_path.exists():
        try:
            return mask_iou_from_paths(gt_mask_path, pred_mask_path)
        except Exception:
            return None

    # try bbox
    gt_bbox = parse_bbox_row(gt_row)
    pred_bbox = parse_bbox_row(pred_row)
    if gt_bbox and pred_bbox:
        return bbox_iou(gt_bbox, pred_bbox)

    # nothing computable
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-sample IOU for SLAKE predictions")
    p.add_argument("--gt-csv", type=str, help="Ground-truth CSV path (must contain sample_id and mask_path or bbox columns)")
    p.add_argument("--gt-mask-dir", type=str, help="Ground-truth masks directory (files named <sample_id>.png)")
    p.add_argument("--pred", action="append", help="Prediction spec as model_name:PATH (PATH can be dir/csv/jsonl). Repeat for multiple models", required=True)
    p.add_argument("--out", type=str, default="outputs/slake_iou_per_sample.csv", help="Output CSV path")
    p.add_argument("--repo-root", type=str, default=".", help="Base path to resolve relative mask paths")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    # load GT
    if args.gt_csv:
        gt_df = load_gt_from_csv(Path(args.gt_csv))
    elif args.gt_mask_dir:
        gt_df = build_mask_map_from_dir(Path(args.gt_mask_dir))
    else:
        raise ValueError("One of --gt-csv or --gt-mask-dir must be provided")

    gt_df = gt_df.drop_duplicates(subset=["sample_id"]).set_index("sample_id", drop=False)

    results = []
    for pred_spec in args.pred:
        if ":" not in pred_spec:
            raise ValueError("Each --pred must be MODEL_NAME:PATH")
        model_name, path_str = pred_spec.split(":", 1)
        pred_path = Path(path_str)
        if pred_path.is_dir():
            pred_df = build_mask_map_from_dir(pred_path)
        elif pred_path.suffix.lower() in (".csv", ".tsv"):
            pred_df = load_preds_from_csv(pred_path)
        elif pred_path.suffix.lower() in (".jsonl", ".ndjson", ".json"):
            pred_df = load_preds_from_jsonl(pred_path)
        else:
            # unsupported - attempt to read as csv
            pred_df = load_preds_from_csv(pred_path)

        pred_df = pred_df.drop_duplicates(subset=["sample_id"]).set_index("sample_id", drop=False)

        # iterate over intersection of sample ids
        common = sorted(set(gt_df.index).intersection(set(pred_df.index)))
        if not common:
            # fallback: iterate over pred samples and try to match by filename stem
            common = sorted(set(pred_df.index))

        for sid in common:
            gt_row = gt_df.loc[sid].to_dict() if sid in gt_df.index else {}
            pred_row = pred_df.loc[sid].to_dict() if sid in pred_df.index else {}
            iou = compute_iou_for_pair(sid, gt_row, pred_row, repo_root)
            if iou is None:
                # skip non-computable
                continue
            results.append({"sample_id": sid, "model_name": model_name, "iou": float(iou)})

    if not results:
        print("No IOU results computed. Check GT and prediction formats/paths.")
        return

    out_df = pd.DataFrame(results)
    # iou expected in [0,1] - if in 0-100 convert
    if out_df["iou"].max() > 1.0:
        out_df["iou"] = out_df["iou"] / 100.0

    # quantile bins across the combined set (5 bins)
    try:
        out_df["bin_label"] = pd.qcut(out_df["iou"], q=5, labels=[f"bin_{i+1}" for i in range(5)])
    except Exception:
        # fallback to equal-width bins 0-20,... if qcut fails (e.g., too few unique values)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        out_df["bin_label"] = pd.cut(out_df["iou"], bins=bins, labels=[f"bin_{i+1}" for i in range(5)], include_lowest=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # print summary per model
    summary = out_df.groupby("model_name").agg(
        n_samples=("iou", "count"),
        mean_iou=("iou", "mean"),
        median_iou=("iou", "median"),
    )
    print(summary.round(4).to_string())
    print(f"\nSaved per-sample IOU CSV to {out_path}")


if __name__ == "__main__":
    main()

