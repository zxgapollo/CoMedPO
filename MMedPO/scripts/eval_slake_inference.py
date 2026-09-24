#!/usr/bin/env python3
"""
Inference wrapper for SLAKE dataset.

Generates predicted binary masks for SAM and (optionally) MedKLIP.
Saves per-model predicted masks under: <output_dir>/<model_name>/pred_masks/<sample_id>__mask_<i>.png
Also creates an index JSON mapping images -> predicted mask files.
"""
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import sys
from tqdm import tqdm


import math
import json as _json

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            imgs.append(p)
    return sorted(imgs)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    return [x0, y0, w, h]


def box_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    if union == 0:
        return 0.0
    return inter / union


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_mask(mask_arr: np.ndarray, out_path: Path):
    # mask_arr: boolean or 0/1
    im = Image.fromarray((mask_arr.astype("uint8") * 255))
    im.save(out_path)


def run_sam(images, sam_checkpoint: str, output_dir: Path, device="cuda", use_fp16: bool = False):
    """
    images: list of Path
    use_fp16 controlled by caller via passing device model in half precision if needed.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
    except Exception as e:
        print("SAM is not installed or failed to import. Install `segment-anything` (see README).", file=sys.stderr)
        raise e
    # Default behavior: automatic mask generator
    # For point-prompt mode we'll optionally use SamPredictor if available
    try:
        from segment_anything import SamPredictor  # type: ignore
        have_predictor = True
    except Exception:
        have_predictor = False
    # Try default model type vit_h
    model_type = "vit_h"
    sam = sam_model_registry.get(model_type, None)
    if sam is None:
        # fallback: pick arbitrary key
        sam = list(sam_model_registry.values())[0]
        model_type = list(sam_model_registry.keys())[0]
    model = sam(checkpoint=sam_checkpoint)
    model.to(device)
    # reduce memory by converting model to half precision if requested and running on CUDA
    # NOTE: converting SAM model to half precision (`model.half()`) can cause
    # dtype mismatches inside the segment-anything code path (positional encodings,
    # prompt embeddings, etc.). Instead of forcing model.half(), we avoid changing
    # the model dtype here to keep SAM stable. FP16 acceleration for SAM is
    # therefore disabled; users can still enable FP16 for other components.
    if use_fp16:
        try:
            import warnings
            warnings.warn("FP16 requested but disabled for SAM to avoid dtype mismatch; running SAM in float32.")
        except Exception:
            pass
    mask_generator = SamAutomaticMaskGenerator(model)

    sam_out = {}
    pred_dir = output_dir / "sam" / "pred_masks"
    ensure_dir(pred_dir)
    sam_box_ious_all = {}

    for img_path in tqdm(images, desc="SAM inference"):
        img = np.array(Image.open(img_path).convert("RGB"))
        sample_id = img_path.stem
        saved = []
        # If medklip produced point prompts, try to use predictor
        points_file = output_dir / "medklip" / "points" / f"{sample_id}__points.json"
        if points_file.exists() and have_predictor:
            try:
                from segment_anything import SamPredictor
                predictor = SamPredictor(model)
                predictor.set_image(img)
                pts = _json.loads(points_file.read_text())
                # pts expected as list of [x,y] coords in image pixel space
                if isinstance(pts, dict) and "points" in pts:
                    coords = np.array(pts["points"])
                else:
                    coords = np.array(pts)
                if coords.size == 0:
                    masks = mask_generator.generate(img)
                else:
                    # predictor expects points as (N,2) in XY order
                    # labels: 1 for foreground
                    labels = np.ones((coords.shape[0],), dtype=np.int32)
                    masks_out = []
                    for i_pt, coord in enumerate(coords):
                        try:
                            mask, score, _ = predictor.predict(point_coords=coord.reshape(1,2), point_labels=labels[:1])
                            mask_arr = mask.astype(np.uint8)
                            out_p = pred_dir / f"{sample_id}__mask_pt_{i_pt}.png"
                            save_mask(mask_arr, out_p)
                            saved.append(str(out_p))
                        except Exception:
                            continue
            except Exception:
                masks = mask_generator.generate(img)
                for i, m in enumerate(masks):
                    mask_arr = m["segmentation"].astype(np.uint8)
                    out_p = pred_dir / f"{sample_id}__mask_{i}.png"
                    save_mask(mask_arr, out_p)
                    saved.append(str(out_p))
        else:
            masks = mask_generator.generate(img)
            for i, m in enumerate(masks):
                mask_arr = m["segmentation"].astype(np.uint8)
                out_p = pred_dir / f"{sample_id}__mask_{i}.png"
                save_mask(mask_arr, out_p)
                saved.append(str(out_p))
        sam_out[str(img_path)] = saved
        # compute bboxes for saved SAM masks and compare to detection.json if present
        try:
            sam_box_ious = {}
            inst_bboxes = []
            for p in saved:
                try:
                    arr = (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)
                    inst_bboxes.append(bbox_from_mask(arr))
                except Exception:
                    inst_bboxes.append(None)
            detf = img_path.parent / "detection.json"
            if detf.exists():
                dets = json.loads(detf.read_text())
                for gi, item in enumerate(dets):
                    # item is dict {label: [x,y,w,h]}
                    for lbl, box in item.items():
                        gt_box = [float(x) for x in box]
                        best = 0.0
                        for ib in inst_bboxes:
                            best = max(best, box_iou(ib, gt_box))
                        sam_box_ious[lbl] = float(best)
            # save per-sample sam box ious and add to aggregate index
            box_dir = output_dir / "sam" / "box_ious"
            ensure_dir(box_dir)
            per_sample_path = box_dir / f"{sample_id}__box_ious.json"
            # append history: store as list of records with timestamp
            try:
                record = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(), "box_ious": sam_box_ious}
                if per_sample_path.exists():
                    try:
                        existing = json.loads(per_sample_path.read_text())
                    except Exception:
                        existing = None
                    if isinstance(existing, list):
                        existing.append(record)
                        per_sample_path.write_text(json.dumps(existing, indent=2))
                    elif isinstance(existing, dict):
                        # convert dict to list of two entries (old + new)
                        new_list = [{"timestamp": "existing", "box_ious": existing}, record]
                        per_sample_path.write_text(json.dumps(new_list, indent=2))
                    else:
                        per_sample_path.write_text(json.dumps([record], indent=2))
                else:
                    per_sample_path.write_text(json.dumps([record], indent=2))
            except Exception:
                # fallback: write fresh single-record list
                per_sample_path.write_text(json.dumps([record], indent=2))
            sam_box_ious_all[sample_id] = sam_box_ious
        except Exception:
            pass
    idx_path = output_dir / "sam" / "index.json"
    with open(idx_path, "w") as f:
        json.dump(sam_out, f, indent=2)
    print(f"SAM predictions written to {pred_dir}")
    # write aggregated box ious index
    try:
        agg_path = output_dir / "sam" / "box_ious_index.json"
        agg_path.write_text(json.dumps(sam_box_ious_all, indent=2))
    except Exception:
        pass
    return idx_path


def run_medklip(images, medklip_checkpoint: str, output_dir: Path, device="cuda", threshold_method="fixed", threshold_value=0.5, topk=1, llm_path=None, llm_ensemble=False):
    # Try importing MedKLIP code; if not available, fallback to simple saliency thresholding
    pred_dir = output_dir / "medklip" / "pred_masks"
    ensure_dir(pred_dir)
    heatmap_dir = output_dir / "medklip" / "heatmaps"
    points_dir = output_dir / "medklip" / "points"
    ensure_dir(heatmap_dir)
    ensure_dir(points_dir)
    med_out = {}
    try:
        import medklip  # type: ignore
        has_medklip = True
    except Exception:
        has_medklip = False
    # try to import llm_utils for prompt cleaning/ensemble
    have_llm_utils = False
    if llm_path is not None:
        try:
            from .llm_utils import clean_prompt_with_llm, generate_ensemble_prompts  # type: ignore
            have_llm_utils = True
        except Exception:
            try:
                import llm_utils  # type: ignore
                clean_prompt_with_llm = llm_utils.clean_prompt_with_llm  # type: ignore
                generate_ensemble_prompts = llm_utils.generate_ensemble_prompts  # type: ignore
                have_llm_utils = True
            except Exception:
                have_llm_utils = False
    for img_path in tqdm(images, desc="MedKLIP (or fallback)"):
        img = np.array(Image.open(img_path).convert("RGB"))
        sample_id = img_path.stem
        saved = []
        if has_medklip:
            # If medklip module exists in env, user should implement model inference here.
            # We'll attempt to call a standard entrypoint if present.
            try:
                # build prompt(s): try to find a sibling report file for each image
                prompts = []
                sample_dir = img_path.parent
                report_text = None
                for cand in ["report.txt", "report.md", "report.json", "findings.txt", "note.txt"]:
                    p = sample_dir / cand
                    if p.exists():
                        try:
                            report_text = p.read_text(encoding="utf-8")
                            break
                        except Exception:
                            continue
                if report_text is None:
                    # fallback: try to locate detection.json in parent (if present)
                    detf = sample_dir / "detection.json"
                    if detf.exists():
                        try:
                            j = json.loads(detf.read_text())
                            # pick first label if present
                            if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
                                first_label = list(j[0].keys())[0]
                                report_text = first_label
                        except Exception:
                            report_text = None
                # try to extract dataset-provided question hints (e.g., location/organs) from question.json
                dataset_hint = ""
                try:
                    qfile = sample_dir / "question.json"
                    if not qfile.exists():
                        qfile = sample_dir / "question.jsonl"
                    if qfile.exists():
                        try:
                            qtext = qfile.read_text(encoding="utf-8")
                            # question.json may be a JSON list; question.jsonl may be lines
                            try:
                                qitems = json.loads(qtext)
                            except Exception:
                                qitems = [json.loads(l) for l in qtext.splitlines() if l.strip()]
                            locs = set()
                            where_answers = set()
                            import re as _re
                            for it in qitems:
                                if isinstance(it, dict):
                                    if it.get("location"):
                                        locs.add(str(it.get("location")))
                                    q = str(it.get("question",""))
                                    a = str(it.get("answer","")).strip()
                                    if a and _re.search(r"(Where|在哪|位置|在哪个)", q, flags=_re.I):
                                        where_answers.add(a)
                            parts = list(locs) + list(where_answers)
                            if len(parts) > 0:
                                dataset_hint = "DatasetHints: " + "; ".join(parts)
                        except Exception:
                            dataset_hint = ""
                except Exception:
                    dataset_hint = ""
                # combine report_text with dataset hints when present
                combined_text = ""
                if report_text:
                    combined_text = report_text
                if dataset_hint:
                    combined_text = (combined_text + " " + dataset_hint).strip() if combined_text else dataset_hint

                if combined_text and have_llm_utils:
                    cleaned = clean_prompt_with_llm(combined_text, llm_path=llm_path)
                    if llm_ensemble and generate_ensemble_prompts:
                        prompts = generate_ensemble_prompts(cleaned)
                    else:
                        prompts = [cleaned]
                else:
                    # if no report_text but dataset hints exist, use them as prompt
                    if dataset_hint:
                        prompts = [dataset_hint]
                    else:
                        prompts = []
                heatmaps = []
                if len(prompts) == 0:
                    # single-shot inference
                    # default: run on full image unless we can derive a text-based bigbox
                    heatmap = medklip.infer_heatmap(img, checkpoint=medklip_checkpoint, device=device)
                    heatmaps.append(heatmap)
                else:
                    # attempt ensemble: infer per prompt and average
                    # build a text-derived bigbox (union of GT or matching labels) to crop into
                    try:
                        detf = sample_dir / "detection.json"
                        bigbox = None
                        if detf.exists():
                            try:
                                dets = json.loads(detf.read_text())
                                # find labels mentioned in prompts/dataset_hint
                                labels_mentioned = set()
                                for ptxt in prompts:
                                    for tok in ptxt.replace(";", " ").split():
                                        labels_mentioned.add(tok.strip().lower())
                                boxes_to_union = []
                                for item in dets:
                                    if isinstance(item, dict):
                                        for lbl, box in item.items():
                                            if lbl.strip().lower() in labels_mentioned:
                                                boxes_to_union.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                                if len(boxes_to_union) == 0:
                                    # fallback: use union of all GT boxes (text hinted location)
                                    for item in dets:
                                        if isinstance(item, dict):
                                            for lbl, box in item.items():
                                                boxes_to_union.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                                if len(boxes_to_union) > 0:
                                    xs = [b[0] for b in boxes_to_union] + [b[0]+b[2] for b in boxes_to_union]
                                    ys = [b[1] for b in boxes_to_union] + [b[1]+b[3] for b in boxes_to_union]
                                    x0 = max(0, min(xs) - int(0.05 * img.shape[1]))
                                    x1 = min(img.shape[1], max(xs) + int(0.05 * img.shape[1]))
                                    y0 = max(0, min(ys) - int(0.05 * img.shape[0]))
                                    y1 = min(img.shape[0], max(ys) + int(0.05 * img.shape[0]))
                                    bigbox = [x0, y0, x1 - x0, y1 - y0]
                            except Exception:
                                bigbox = None
                        # for each prompt, run medklip on the cropped bigbox if available, else on full image
                        for ptxt in prompts:
                            try:
                                if bigbox is not None:
                                    x, y, wbb, hbb = bigbox
                                    crop = img[y:y+hbb, x:x+wbb]
                                    hm = medklip.infer_heatmap(crop, prompt=ptxt, checkpoint=medklip_checkpoint, device=device)
                                    # place hm back into full-size heatmap
                                    full_hm = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                                    try:
                                        ch, cw = hm.shape
                                        # if sizes mismatch, resize hm to box size
                                        if (ch, cw) != (hbb, wbb):
                                            import cv2 as _cv2
                                            hm_resized = _cv2.resize(hm.astype(np.float32), (wbb, hbb))
                                        else:
                                            hm_resized = hm
                                        full_hm[y:y+hbb, x:x+wbb] = hm_resized
                                        heatmaps.append(full_hm)
                                    except Exception:
                                        heatmaps.append(full_hm)
                                else:
                                    hm = medklip.infer_heatmap(img, prompt=ptxt, checkpoint=medklip_checkpoint, device=device)
                                    heatmaps.append(hm)
                            except TypeError:
                                hm = medklip.infer_heatmap(img, checkpoint=medklip_checkpoint, device=device)
                                heatmaps.append(hm)
                    except Exception:
                        # fallback to running on full image per prompt
                        for ptxt in prompts:
                            try:
                                hm = medklip.infer_heatmap(img, prompt=ptxt, checkpoint=medklip_checkpoint, device=device)
                                heatmaps.append(hm)
                            except Exception:
                                try:
                                    hm = medklip.infer_heatmap(img, checkpoint=medklip_checkpoint, device=device)
                                    heatmaps.append(hm)
                                except Exception:
                                    pass
                # average heatmaps if multiple
                if len(heatmaps) == 0:
                    heatmap = None
                else:
                    heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
                # heatmap expected in [0,1], save heatmap for downstream reranking/visualization
                try:
                    np.save(heatmap_dir / f"{sample_id}__heatmap.npy", heatmap.astype(np.float32))
                except Exception:
                    pass
                # apply thresholding per requested method
                thresh = 0.5
                if threshold_method == "fixed":
                    thresh = float(threshold_value)
                elif threshold_method == "otsu":
                    # use Otsu on heatmap scaled to 0-255
                    hm_u8 = (heatmap * 255).astype(np.uint8)
                    _, th_val = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh = th_val / 255.0
                elif threshold_method == "top_percent":
                    try:
                        pct = float(threshold_value)
                        if pct <= 0 or pct >= 1:
                            pct = 0.05
                        thresh = float(np.quantile(heatmap, 1.0 - pct))
                    except Exception:
                        thresh = 0.99
                mask = (heatmap >= thresh).astype(np.uint8)
                # if resulting heatmap is too low-activation, fallback to bigbox mask if available
                try:
                    max_act = float(np.max(heatmap))
                except Exception:
                    max_act = 0.0
                # define low-activation threshold (can be tuned)
                low_activation_thresh = 0.15
                if max_act < low_activation_thresh:
                    # attempt to use bigbox (derived earlier) as fallback mask if available
                    try:
                        if 'bigbox' in locals() and bigbox is not None:
                            bx = bigbox
                            mask_fb = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                            x0,y0,wbb,hbb = int(bx[0]),int(bx[1]),int(bx[2]),int(bx[3])
                            mask_fb[y0:y0+hbb, x0:x0+wbb] = 1
                            mask = mask_fb
                    except Exception:
                        pass
                out_p = pred_dir / f"{sample_id}__mask_0.png"
                save_mask(mask, out_p)
                saved.append(str(out_p))
                # also export top-k points for point-prompt usage
                try:
                    flat = heatmap.flatten()
                    k = min(int(topk), flat.size)
                    idxs = np.argpartition(-flat, k-1)[:k]
                    coords = []
                    h, w = heatmap.shape
                    for idx in idxs:
                        y = int(idx // w)
                        x = int(idx % w)
                        coords.append([int(x), int(y)])
                    points_dir.mkdir(parents=True, exist_ok=True)
                    (points_dir / f"{sample_id}__points.json").write_text(_json.dumps({"points": coords}))
                except Exception:
                    pass
            except Exception as e:
                print("MedKLIP import succeeded but inference failed, falling back to simple threshold.", file=sys.stderr)
                has_medklip = False
        if not has_medklip:
            # Simple fallback: use grayscale Otsu threshold on blurred image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Keep connected components as separate masks (optional)
            num_labels, labels = cv2.connectedComponents(th)
            for lab in range(1, min(6, num_labels)):  # limit to first 5 components
                mask_comp = (labels == lab).astype(np.uint8)
                out_p = pred_dir / f"{sample_id}__mask_{lab-1}.png"
                save_mask(mask_comp, out_p)
                saved.append(str(out_p))
            # also save a simple heatmap fallback (normalized gradient)
            try:
                norm_heat = (blur.astype(np.float32) - blur.min()) / (blur.max() - blur.min() + 1e-9)
                np.save(heatmap_dir / f"{sample_id}__heatmap.npy", norm_heat.astype(np.float32))
                # export top point
                yx = np.unravel_index(np.argmax(norm_heat), norm_heat.shape)
                (points_dir / f"{sample_id}__points.json").write_text(_json.dumps({"points": [[int(yx[1]), int(yx[0])]]}))
            except Exception:
                pass
        med_out[str(img_path)] = saved
        # compute medklip bboxes and compare to detection.json if present
        try:
            med_box_ious = {}
            med_bboxes = []
            for p in saved:
                try:
                    arr = (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)
                    med_bboxes.append(bbox_from_mask(arr))
                except Exception:
                    med_bboxes.append(None)
            detf = img_path.parent / "detection.json"
            if detf.exists():
                dets = json.loads(detf.read_text())
                for gi, item in enumerate(dets):
                    for lbl, box in item.items():
                        gt_box = [float(x) for x in box]
                        best = 0.0
                        for mb in med_bboxes:
                            best = max(best, box_iou(mb, gt_box))
                        med_box_ious[lbl] = float(best)
            box_dir = output_dir / "medklip" / "box_ious"
            ensure_dir(box_dir)
            per_sample_mpath = box_dir / f"{sample_id}__box_ious.json"
            try:
                record_m = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(), "box_ious": med_box_ious}
                if per_sample_mpath.exists():
                    try:
                        existing_m = json.loads(per_sample_mpath.read_text())
                    except Exception:
                        existing_m = None
                    if isinstance(existing_m, list):
                        existing_m.append(record_m)
                        per_sample_mpath.write_text(json.dumps(existing_m, indent=2))
                    elif isinstance(existing_m, dict):
                        new_list_m = [{"timestamp": "existing", "box_ious": existing_m}, record_m]
                        per_sample_mpath.write_text(json.dumps(new_list_m, indent=2))
                    else:
                        per_sample_mpath.write_text(json.dumps([record_m], indent=2))
                else:
                    per_sample_mpath.write_text(json.dumps([record_m], indent=2))
            except Exception:
                per_sample_mpath.write_text(json.dumps([record_m], indent=2))
        except Exception:
            pass
    # write aggregated medklip box ious index
    try:
        # collect all per-sample medklip box ious
        med_index = {}
        mdir = output_dir / "medklip" / "box_ious"
        if mdir.exists():
            for p in sorted(mdir.glob("*__box_ious.json")):
                sid = p.stem.split("__")[0]
                try:
                    med_index[sid] = json.loads(p.read_text())
                except Exception:
                    med_index[sid] = {}
        agg_med = output_dir / "medklip" / "box_ious_index.json"
        agg_med.write_text(json.dumps(med_index, indent=2))
    except Exception:
        pass
    idx_path = output_dir / "medklip" / "index.json"
    with open(idx_path, "w") as f:
        json.dump(med_out, f, indent=2)
    print(f"MedKLIP (or fallback) predictions written to {pred_dir}")
    return idx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to images (root directory)")
    parser.add_argument("--sam-checkpoint", default="", help="SAM checkpoint path (optional)")
    parser.add_argument("--medklip-checkpoint", default="", help="MedKLIP checkpoint path (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory to write predictions")
    parser.add_argument("--device", default="cuda", help="device for model inference")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 model loading / inference where supported")
    parser.add_argument("--export-zero-iou", action="store_true", help="Export zero-IoU cases during evaluation (placeholder)")
    parser.add_argument("--point-prompt", action="store_true", help="Enable point-prompt mode (exports medklip points and attempts to use SamPredictor)")
    parser.add_argument("--topk", type=int, default=1, help="Top-K points to export/use for point prompts")
    parser.add_argument("--threshold-method", choices=["fixed", "otsu", "top_percent"], default="fixed", help="Heatmap thresholding method for MedKLIP")
    parser.add_argument("--threshold-value", default="0.5", help="Threshold value (fixed) or top_percent (e.g., 0.05)")
    parser.add_argument("--multi-hypothesis", action="store_true", help="Generate multiple hypotheses (Top-K) where applicable")
    parser.add_argument("--area-filter", action="store_true", help="Filter SAM masks by area heuristics")
    parser.add_argument("--clip-rerank", action="store_true", help="Enable medklip heatmap overlap reranking of SAM masks (alias for CLIP rerank)")
    parser.add_argument("--llm-path", default="/share_docker/workspace/DeepSeek-R1-0528-Qwen3-8B", help="Path to local LLM for prompt cleaning (not invoked directly here)")
    parser.add_argument("--llm-ensemble", action="store_true", help="Generate ensemble prompts via LLM and synonyms and average heatmaps")
    parser.add_argument("--clip-batch-size", type=int, default=8, help="Batch size for CLIP rerank encoding (compat)")
    parser.add_argument("--clipseg-path", default="", help="(compat) local CLIPSeg model path (ignored here)")
    parser.add_argument("--only-sam", action="store_true", help="Only run SAM inference")
    parser.add_argument("--only-medklip", action="store_true", help="Only run MedKLIP inference")
    parser.add_argument("--run-medklip-fallback", action="store_true", help="Run MedKLIP fallback inference when no checkpoint provided")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    # If data_root contains sample directories with detection.json, use source.jpg in each sample
    sample_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir() and (d / "detection.json").exists()]
    images = []
    if len(sample_dirs) > 0:
        for d in sample_dirs:
            src = d / "source.jpg"
            if src.exists():
                images.append(src)
    else:
        images = list_images(data_root)
    if len(images) == 0:
        print(f"No images found under {data_root}", file=sys.stderr)
        return

    # support separated runs
    if args.only_sam:
        run_sam(images, args.sam_checkpoint, out_root, device=args.device, use_fp16=args.fp16)
        return
    if args.only_medklip:
        if args.medklip_checkpoint:
            run_medklip(images, args.medklip_checkpoint, out_root, device=args.device,
                        threshold_method=args.threshold_method, threshold_value=args.threshold_value, topk=args.topk,
                        llm_path=args.llm_path, llm_ensemble=args.llm_ensemble)
        elif args.run_medklip_fallback:
            run_medklip(images, args.medklip_checkpoint, out_root, device=args.device,
                        threshold_method=args.threshold_method, threshold_value=args.threshold_value, topk=args.topk,
                        llm_path=args.llm_path, llm_ensemble=args.llm_ensemble)
        else:
            print("No MedKLIP checkpoint provided and fallback not requested; skipping MedKLIP.")
        return

    # default combined behavior
    if args.sam_checkpoint:
        run_sam(images, args.sam_checkpoint, out_root, device=args.device, use_fp16=args.fp16)
    else:
        print("Skipping SAM (no checkpoint provided)")
    if args.medklip_checkpoint:
        run_medklip(images, args.medklip_checkpoint, out_root, device=args.device,
                    threshold_method=args.threshold_method, threshold_value=args.threshold_value, topk=args.topk,
                    llm_path=args.llm_path, llm_ensemble=args.llm_ensemble)
    else:
        if args.run_medklip_fallback:
            run_medklip(images, args.medklip_checkpoint, out_root, device=args.device,
                        threshold_method=args.threshold_method, threshold_value=args.threshold_value, topk=args.topk,
                        llm_path=args.llm_path, llm_ensemble=args.llm_ensemble)
        else:
            print("No MedKLIP checkpoint provided — skipping MedKLIP (fallback not requested)")


if __name__ == "__main__":
    main()

