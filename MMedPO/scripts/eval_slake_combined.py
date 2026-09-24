#!/usr/bin/env python3
"""
Combined SAM + text-guided (CLIPSeg fallback) pipeline for SLAKE.

For each image directory under data_root that contains a `detection.json`,
this script will:
 - read GT boxes from detection.json (expected list of {"Label": [x,y,w,h]} entries)
 - run SAM to produce instance masks (or fallback thresholding)
 - optionally run CLIPSeg (text prompts = GT label names) to produce text masks
 - select a predicted box per GT by matching SAM masks with CLIPSeg (if available)
   or by IoU to GT box
 - compute box IoU between predicted box and GT box
 - save `results.json` with per-model (here: 'combined') per-image box IoUs and stats
 - call existing `plot_slake_results.py` to produce histograms/threshold curves (using box IoUs)

This implementation purposely skips any "hole/apply_hole" image modifications and only
compares final boxes, per user request.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import os
import cv2
from collections import defaultdict
import logging
from pathlib import Path as _Path
try:
    from transformers import CLIPProcessor, CLIPModel
    have_clip_model = True
except Exception:
    have_clip_model = False
try:
    from .llm_utils import clean_prompt_with_llm  # type: ignore
    have_llm_utils = True
except Exception:
    # try relative import fallback
    try:
        import llm_utils  # type: ignore
        have_llm_utils = True
        clean_prompt_with_llm = llm_utils.clean_prompt_with_llm  # type: ignore
    except Exception:
        have_llm_utils = False


def read_detection_json(p: Path):
    try:
        j = json.loads(p.read_text())
        entries = []
        for item in j:
            if isinstance(item, dict):
                for k, v in item.items():
                    # expect v = [x,y,w,h]
                    entries.append((k, [float(x) for x in v]))
        return entries
    except Exception:
        return []


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
    # boxes as [x,y,w,h] top-left
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


def load_image(path: Path):
    return np.array(Image.open(path).convert("RGB"))


def find_image_file(d: Path):
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    # Prefer files not containing 'mask' in name
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in exts and "mask" not in p.name.lower():
            return p
    # fallback to any image
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in exts:
            return p
    return None


def run_sam_on_image(img_arr, sam_model, sam_generator, device="cuda"):
    # sam_generator: SamAutomaticMaskGenerator instance
    masks = sam_generator.generate(img_arr)
    bin_masks = []
    for m in masks:
        bin_masks.append(m["segmentation"].astype(np.uint8))
    return bin_masks


def fallback_threshold_masks(img_arr, max_components=5):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(th)
    masks = []
    for lab in range(1, min(num_labels, max_components + 1)):
        masks.append((labels == lab).astype(np.uint8))
    return masks


def run_clipseg_on_image(img_pil, prompts, processor, model, device="cpu"):
    # prompts: list of strings to run; returns dict prompt->binary mask
    import torch
    result = {}
    bsz = len(prompts)
    inputs = processor(text=prompts, images=[img_pil]*bsz, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    out = model(**inputs)
    masks = torch.sigmoid(out.logits)
    for i, p in enumerate(prompts):
        mi = masks[i].unsqueeze(0).unsqueeze(0)
        mi = cv2.resize(mi.cpu().numpy().squeeze(), (img_pil.size[0], img_pil.size[1]))
        result[p] = (mi > 0.5).astype(np.uint8)
    return result


def clip_similarity_rerank(img_arr, masks, text_prompt, clip_processor=None, clip_model=None, device="cpu", batch_size=8, use_fp16: bool = False):
    """
    Given an image array and a list of binary masks (np.uint8), compute CLIP image-text similarity
    for each mask by cropping the mask bounding box and resizing to model input. Return list of scores.
    """
    import numpy as _np
    from PIL import Image as _Image
    import torch as _torch
    scores = []
    if clip_processor is None or clip_model is None or not have_clip_model:
        return [0.0] * len(masks)
    crops = []
    indices = []
    for idx, m in enumerate(masks):
        ys, xs = _np.where(m > 0)
        if ys.size == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        crop = img_arr[y0:y1+1, x0:x1+1]
        crops.append(_Image.fromarray(crop).convert("RGB"))
        indices.append(idx)
    if len(crops) == 0:
        return [0.0] * len(masks)
    # batch encode image crops
    all_img_feats = []
    import torch as _torch
    device_t = _torch.device(device if device != "cpu" and _torch.cuda.is_available() else "cpu")
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        try:
            inputs = clip_processor(text=[text_prompt]*len(batch), images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device_t) for k, v in inputs.items()}
            # ensure model on device
            try:
                clip_model.to(device_t)
            except Exception:
                pass
            with _torch.no_grad():
                if use_fp16 and device_t.type != "cpu":
                    ctx = _torch.cuda.amp.autocast(device_type="cuda", dtype=_torch.float16)
                else:
                    from contextlib import nullcontext
                    ctx = nullcontext()
                with ctx:
                    img_feats = clip_model.get_image_features(**{k: v for k, v in inputs.items() if k.startswith("pixel_values")})
                    # prepare text inputs once per batch (all same prompt)
                    text_inputs = clip_processor.tokenizer([text_prompt], return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(device_t) for k, v in text_inputs.items()}
                    text_feats = clip_model.get_text_features(**text_inputs)
                img_feats = img_feats.detach().cpu().numpy()
                text_feats = text_feats.detach().cpu().numpy()
                # normalize text vector once
                txt_vec = text_feats.reshape(-1)
                txt_norm = txt_vec / ( (txt_vec**2).sum() ** 0.5 + 1e-9)
                for imf in img_feats:
                    imf_vec = imf.reshape(-1)
                    imf_norm = imf_vec / ( (imf_vec**2).sum() ** 0.5 + 1e-9)
                    score = float((imf_norm * txt_norm).sum())
                    all_img_feats.append(score)
        except Exception:
            # on error, fallback to zeros for this batch
            for _ in batch:
                all_img_feats.append(0.0)
    # map back to original mask list
    scores = [0.0] * len(masks)
    for idx, s in zip(indices, all_img_feats):
        scores[idx] = s
    return scores


def summarize_ious(ious):
    arr = np.array(ious) if len(ious) > 0 else np.array([0.0])
    mean = float(arr.mean())
    median = float(np.median(arr))
    pct_gt50 = float((arr >= 0.5).sum() / arr.size)
    return {"mean": mean, "median": median, "count": int(arr.size), "pct_gt50": pct_gt50}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="SLAKE imgs root (folders per sample)")
    parser.add_argument("--sam-checkpoint", default="", help="SAM checkpoint path (optional)")
    parser.add_argument("--device", default="cuda", help="device for models")
    parser.add_argument("--output-dir", required=True, help="output dir for results")
    # additional optional flags accepted for compatibility with wrapper script
    parser.add_argument("--export-zero-iou", action="store_true", help="(compat) Export zero-IoU cases")
    parser.add_argument("--point-prompt", action="store_true", help="(compat) Use point prompts from medklip")
    parser.add_argument("--topk", type=int, default=1, help="(compat) top-k points")
    parser.add_argument("--threshold-method", choices=["fixed", "otsu", "top_percent"], default="fixed", help="(compat) threshold method")
    parser.add_argument("--threshold-value", default="0.5", help="(compat) threshold value")
    parser.add_argument("--multi-hypothesis", action="store_true", help="(compat) generate multiple hypotheses")
    parser.add_argument("--area-filter", action="store_true", help="(compat) area filtering")
    parser.add_argument("--clip-rerank", action="store_true", help="(compat) clip/overlap rerank")
    parser.add_argument("--llm-path", default="", help="(compat) llm path")
    parser.add_argument("--clip-model-path", default="", help="local CLIP model path for reranking (HF format)")
    parser.add_argument("--clipseg-path", default="", help="local CLIPSeg model path (HF format) to use instead of CIDAS/clipseg-rd64-refined")
    parser.add_argument("--clip-batch-size", type=int, default=8, help="batch size for CLIP rerank encoding")
    parser.add_argument("--llm-ensemble", action="store_true", help="(compat) enable llm ensemble prompts (ignored here)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference where supported")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # try imports
    have_sam = False
    have_clipseg = False
    sam_model = None
    sam_generator = None
    clip_processor = None
    clip_model = None
    if args.sam_checkpoint:
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            sam = sam_model_registry.get("vit_h")
            sam_model = sam(checkpoint=args.sam_checkpoint)
            sam_model.to(args.device)
            sam_generator = SamAutomaticMaskGenerator(sam_model)
            have_sam = True
        except Exception as e:
            print("SAM unavailable or failed to init, will fallback to thresholding:", e)
            have_sam = False

    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        clip_processor = CLIPSegProcessor
        clip_model = CLIPSegForImageSegmentation
        # lazy load models per image to respect local_files_only behavior
        have_clipseg = True
    except Exception:
        have_clipseg = False

    results = {"models": {"combined": {"per_image": {}, "raw_ious": [], "stats": {}}}}
    # prepare containers for per-model (SAM / MedKLIP) box IoU results
    results["models"]["sam"] = {"per_image": {}, "raw_ious": [], "stats": {}}
    results["models"]["medklip"] = {"per_image": {}, "raw_ious": [], "stats": {}}

    # gather sample dirs
    sample_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir() and (d / "detection.json").exists()]
    logging.info("Found %d sample dirs with detection.json under %s", len(sample_dirs), str(data_root))
    # iterate over sample dirs containing detection.json
    for idx_d, d in enumerate(sample_dirs, start=1):
        logging.info("Processing sample %s (%d/%d)", d.name, idx_d, len(sample_dirs))
        detf = d / "detection.json"
        if not detf.exists():
            logging.warning("Skipping %s: detection.json missing", d)
            continue
        imgf = find_image_file(d)
        if imgf is None:
            logging.warning("Skipping %s: no image file found", d)
            continue
        img_pil = Image.open(imgf).convert("RGB")
        img_arr = np.array(img_pil)
        gts = read_detection_json(detf)  # list of (label, [x,y,w,h])

        # prepare CLIPSeg models if available
        clip_masks = {}
        if have_clipseg:
            try:
                clipseg_source = args.clipseg_path if args.clipseg_path else "CIDAS/clipseg-rd64-refined"
                proc = CLIPSegProcessor.from_pretrained(clipseg_source, local_files_only=True)
                model = CLIPSegForImageSegmentation.from_pretrained(clipseg_source, local_files_only=True)
                device = args.device if args.device != "cpu" else "cpu"
                model.to(device)
                prompts = [lbl for lbl, _ in gts]
                clip_masks = run_clipseg_on_image(img_pil, prompts, proc, model, device=device)
            except Exception:
                clip_masks = {}

        # run SAM or fallback to produce instance masks
        inst_masks = []
        if have_sam and sam_generator is not None:
            try:
                inst_masks = run_sam_on_image(img_arr, sam_model, sam_generator, device=args.device)
            except Exception:
                inst_masks = fallback_threshold_masks(img_arr)
        else:
            inst_masks = fallback_threshold_masks(img_arr)

        # precompute bboxes for instance masks (SAM)
        inst_bboxes = [bbox_from_mask(m) for m in inst_masks]
        # load medklip predicted masks for this sample (if present in out_root)
        medklip_bboxes = []
        try:
            med_pred_dir = out_root / "medklip" / "pred_masks"
            if med_pred_dir.exists():
                sid = d.name
                for p in sorted(med_pred_dir.glob(f"{sid}__mask_*.png")):
                    pm = load_image(p) if False else None
                    # use load_mask from eval_slake_iou style
                    import numpy as _np
                    from PIL import Image as _Image
                    im = _Image.open(p).convert("L")
                    arr = _np.array(im) > 127
                    medklip_bboxes.append(bbox_from_mask(arr.astype(_np.uint8)))
        except Exception:
            medklip_bboxes = []

        per_image = {}
        ious_for_image = []
        # also track sam/medklip ious separately for this image
        sam_ious_for_image = []
        medklip_ious_for_image = []
        used_pred = set()
        for idx, (label, gt_box) in enumerate(gts):
            # gt_box assumed [x,y,w,h]
            chosen_box = None
            best_score = 0.0
            # if clipseg produced masks for this label, prefer SAM mask overlapping with clip mask
            if label in clip_masks and clip_masks[label] is not None and not args.clip_rerank:
                cm = clip_masks[label]
                for j, m in enumerate(inst_masks):
                    if j in used_pred:
                        continue
                    inter = np.logical_and(m, cm).sum()
                    if inter > best_score:
                        best_score = inter
                        chosen_box = inst_bboxes[j]
                        best_idx = j
                if chosen_box is not None:
                    used_pred.add(best_idx)
            # if clip rerank is enabled, compute CLIP similarity between each inst mask crop and the label text, pick best
            if args.clip_rerank:
                # Prefer an actual CLIP model if provided; otherwise fall back to clipseg overlap if available
                if args.clip_model_path and have_clip_model:
                    try:
                        proc = CLIPProcessor.from_pretrained(args.clip_model_path, local_files_only=True)
                        cmodel = CLIPModel.from_pretrained(args.clip_model_path, local_files_only=True)
                        scores = clip_similarity_rerank(img_arr, inst_masks, label, clip_processor=proc, clip_model=cmodel, device=args.device, batch_size=args.clip_batch_size, use_fp16=args.fp16)
                        for j, sc in enumerate(scores):
                            if j in used_pred:
                                continue
                            if sc > best_score:
                                best_score = sc
                                chosen_box = inst_bboxes[j]
                                best_idx = j
                        if chosen_box is not None:
                            used_pred.add(best_idx)
                    except Exception:
                        pass
                else:
                    # fallback: use clipseg masks overlap if available
                    if label in clip_masks and clip_masks[label] is not None:
                        cm = clip_masks[label]
                        for j, m in enumerate(inst_masks):
                            if j in used_pred:
                                continue
                            inter = np.logical_and(m, cm).sum()
                            if inter > best_score:
                                best_score = inter
                                chosen_box = inst_bboxes[j]
                                best_idx = j
                        if chosen_box is not None:
                            used_pred.add(best_idx)
            # fallback: choose instance mask with max IoU to gt box
            if chosen_box is None:
                for j, ib in enumerate(inst_bboxes):
                    if ib is None or j in used_pred:
                        continue
                    score = box_iou(ib, gt_box)
                    if score > best_score:
                        best_score = score
                        chosen_box = ib
                        best_idx = j
                if chosen_box is not None:
                    used_pred.add(best_idx)

            # combined chosen box IoU
            iou_val = box_iou(chosen_box, gt_box) if chosen_box is not None else 0.0
            per_image[label] = {"gt_box": gt_box, "pred_box": chosen_box, "iou": float(iou_val)}
            ious_for_image.append(float(iou_val))
            # SAM-only IoU: best SAM instance bbox IoU to gt
            best_sam = 0.0
            for ib in inst_bboxes:
                if ib is None:
                    continue
                best_sam = max(best_sam, box_iou(ib, gt_box))
            sam_ious_for_image.append(float(best_sam))
            # MedKLIP-only IoU: best medklip bbox IoU to gt
            best_med = 0.0
            for mb in medklip_bboxes:
                if mb is None:
                    continue
                best_med = max(best_med, box_iou(mb, gt_box))
            medklip_ious_for_image.append(float(best_med))

        # record
        sid = d.name
        # record combined
        sid = d.name
        results["models"]["combined"]["per_image"][sid] = per_image
        results["models"]["combined"]["raw_ious"].extend(ious_for_image)
        # record sam-only and medklip-only
        results["models"]["sam"]["per_image"][sid] = {"per_gt_ious": sam_ious_for_image, "n_gt": len(sam_ious_for_image)}
        results["models"]["sam"]["raw_ious"].extend(sam_ious_for_image)
        results["models"]["medklip"]["per_image"][sid] = {"per_gt_ious": medklip_ious_for_image, "n_gt": len(medklip_ious_for_image)}
        results["models"]["medklip"]["raw_ious"].extend(medklip_ious_for_image)

    # summarize
    stats = summarize_ious(results["models"]["combined"]["raw_ious"])
    results["models"]["combined"]["stats"] = stats
    # sam/medklip stats
    results["models"]["sam"]["stats"] = summarize_ious(results["models"]["sam"]["raw_ious"])
    results["models"]["medklip"]["stats"] = summarize_ious(results["models"]["medklip"]["raw_ious"])

    out_json = out_root / "results_combined.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("Saved combined results to", out_json)
    # also write per-model results separately for SAM and MedKLIP
    try:
        if "sam" in results["models"]:
            sam_out = out_root / "sam_results.json"
            sam_out.write_text(json.dumps({"models": {"sam": results["models"]["sam"]}}, indent=2))
            print("Saved SAM results to", sam_out)
        if "medklip" in results["models"]:
            med_out = out_root / "medklip_results.json"
            med_out.write_text(json.dumps({"models": {"medklip": results["models"]["medklip"]}}, indent=2))
            print("Saved MedKLIP results to", med_out)
    except Exception:
        pass

    # try plotting using existing utility (it expects models[].raw_ious)
    try:
        plot_script = Path(__file__).parent / "plot_slake_results.py"
        if plot_script.exists():
            from subprocess import Popen
            Popen(["python3", str(plot_script), "--results-json", str(out_json), "--out-dir", str(out_root / "plots_combined")])
    except Exception:
        pass


if __name__ == "__main__":
    main()

