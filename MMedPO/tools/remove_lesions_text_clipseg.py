import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import re

def load_annotations(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_prompts(report):
    text = report.lower()
    prompts = []
    mapping = {
        "nodule": ["nodule"],
        "effusion": ["effusion", "pleural effusion"],
        "pneumothorax": ["pneumothorax"],
        "atelectasis": ["atelectasis"],
        "granuloma": ["granuloma"],
        "scar": ["scarring", "scar"],
        "hernia": ["hiatal hernia", "hernia"],
        "fracture": ["fracture"],
        "opacity": ["opacity", "opacities", "consolidation"],
    }
    for arr in mapping.values():
        for kw in arr:
            if kw in text:
                prompts.append(kw)
                break
    if not prompts:
        prompts.append("abnormality")
    return prompts

def build_report_roi(report, w, h):
    t = report.lower()
    left = any(x in t for x in ["left", "lt", "lobe left"]) and not any(x in t for x in ["right"])
    right = any(x in t for x in ["right", "rt", "lobe right"]) and not any(x in t for x in ["left"])
    bilateral = ("biapical" in t) or ("bilateral" in t) or (not left and not right)
    upper = any(x in t for x in ["upper", "apical", "apex"]) or ("pneumothorax" in t)
    lower = any(x in t for x in ["lower", "base", "bases", "basilar"]) or ("effusion" in t) or ("atelectasis" in t)
    mediastinum = "mediastinum" in t
    clavicle = "clavicle" in t
    hernia = "hiatal hernia" in t or "hernia" in t
    mask = np.zeros((h, w), dtype=np.uint8)
    x0_l, x1_l = 0, w // 2
    x0_r, x1_r = w // 2, w
    y_upper = int(h * 0.33)
    y_apical = int(h * 0.2)
    y_lower = int(h * 0.66)
    y_bases = int(h * 0.8)
    y_clav = int(h * 0.15)
    y_dia = int(h * 0.85)
    x_med0, x_med1 = int(w * 0.4), int(w * 0.6)
    def add_rect(x0, y0, x1, y1):
        x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1)); y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
        mask[y0:y1, x0:x1] = 1
    if mediastinum:
        add_rect(x_med0, 0, x_med1, h)
    if clavicle:
        add_rect(0, 0, w, y_clav)
    if hernia:
        add_rect(int(w * 0.3), y_dia, int(w * 0.7), h)
    if bilateral or (left and right):
        x_pairs = [(x0_l, x1_l), (x0_r, x1_r)]
    elif left:
        x_pairs = [(x0_l, x1_l)]
    elif right:
        x_pairs = [(x0_r, x1_r)]
    else:
        x_pairs = [(0, w)]
    for x0, x1 in x_pairs:
        if upper:
            add_rect(x0, 0, x1, y_upper)
            add_rect(x0, 0, x1, y_apical)
        if lower:
            add_rect(x0, y_lower, x1, h)
            add_rect(x0, y_bases, x1, h)
        if not upper and not lower and not mediastinum and not clavicle and not hernia:
            add_rect(x0, 0, x1, h)
    if mask.sum() == 0:
        mask[:, :] = 1
    return mask

def clipseg_mask_batch(model, processor, images, prompts_per_sample, device):
    union_prompts = sorted({p for lst in prompts_per_sample for p in lst})
    bsz = len(images)
    combined = [np.zeros((images[i].size[1], images[i].size[0]), dtype=np.uint8) for i in range(bsz)]
    with torch.no_grad():
        for p in union_prompts:
            inputs = processor(text=[p]*bsz, images=images, return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            out = model(**inputs)
            masks = torch.sigmoid(out.logits)
            for i in range(bsz):
                mi = masks[i].unsqueeze(0).unsqueeze(0)
                mi = F.interpolate(mi, size=(images[i].size[1], images[i].size[0]), mode="bilinear", align_corners=False)
                mi = mi.squeeze(0).squeeze(0)
                if p in prompts_per_sample[i]:
                    mnp = (mi.cpu().numpy() > 0.5).astype(np.uint8)
                    combined[i] = np.maximum(combined[i], mnp)
    return combined

def medsam_predictor(ckpt_path, device):
    try:
        from segment_anything import sam_model_registry, SamPredictor
        model = sam_model_registry.get("vit_h")(checkpoint=ckpt_path)
        model.to(device)
        return SamPredictor(model)
    except Exception:
        return None

def medsam_mask(predictor, image, roi_mask):
    import numpy as np
    predictor.set_image(np.array(image))
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0:
        return np.zeros_like(roi_mask)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    if len(masks) == 0:
        return np.zeros_like(roi_mask)
    m = (masks[np.argmax(scores)] > 0).astype(np.uint8)
    return m

def apply_hole(image, mask):
    arr = np.array(image)
    if arr.ndim == 2:
        arr[mask == 1] = 0
    else:
        arr[mask == 1] = 0
    return Image.fromarray(arr)

def process_batch(samples, images_root, out_root, processor, model, device, medsam_pred=None, overlap_thr=0.1):
    images = []
    rels = []
    prompts_list = []
    rois = []
    for sample in samples:
        rel_paths = sample.get("image_path", [])
        report = sample.get("report", "")
        prompts = extract_prompts(report)
        for rel in rel_paths:
            src_path = os.path.join(images_root, rel)
            dst_path = os.path.join(out_root, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(src_path):
                continue
            images.append(Image.open(src_path).convert("RGB"))
            rels.append((src_path, dst_path))
            prompts_list.append(prompts)
            w, h = images[-1].size
            rois.append(build_report_roi(report, w, h))
    if not images:
        return
    if medsam_pred is not None:
        masks = []
        for img, roi in zip(images, rois):
            m = medsam_mask(medsam_pred, img, roi)
            masks.append(m)
    else:
        masks = clipseg_mask_batch(model, processor, images, prompts_list, device)
    for (src_path, dst_path), img, mask in zip(rels, images, masks):
        roi = rois.pop(0)
        inter = (mask.astype(np.uint8) & (roi.astype(np.uint8)))
        if inter.sum() / max(1, mask.sum()) < overlap_thr:
            out_img = img
        else:
            out_img = apply_hole(img, inter)
        out_img.save(dst_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    parser.add_argument("--medsam_ckpt", type=str, default="")
    parser.add_argument("--roi_overlap_threshold", type=float, default=0.1)
    args = parser.parse_args()

    data = load_annotations(args.annotation_json)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", local_files_only=True)
    use_cuda = torch.cuda.is_available() and (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "")
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    medsam_pred = None
    if args.medsam_ckpt:
        medsam_pred = medsam_predictor(args.medsam_ckpt, device)

    subsets = []
    for key in ["train", "val", "test"]:
        if key in data:
            subsets.extend(data[key])
    if args.limit and args.limit > 0:
        subsets = subsets[:args.limit]

    # batch over samples
    b = max(1, int(args.batch))
    for i in tqdm(range(0, len(subsets), b)):
        batch_samples = subsets[i:i+b]
        process_batch(batch_samples, args.images_root, args.output_root, processor, model, device, medsam_pred=medsam_pred, overlap_thr=float(args.roi_overlap_threshold))

if __name__ == "__main__":
    main()
