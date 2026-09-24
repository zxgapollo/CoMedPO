#!/usr/bin/env python3
"""
Generate background-randomized images by keeping lesion region (from mask)
and replacing background with other patient images or noise.

Input:
  --questions : JSONL file with entries that include an image path (relative, e.g. xmlab1/source.jpg)
  --full_image_dir : directory containing full images (e.g. data/SLAKE/imgs)
  --mask_dir : directory containing processed masks (e.g. data/SLAKE/processed_imgs)
Output:
  randomized images saved under --out_dir preserving relative paths.

Usage example:
  python generate_background_randomized.py \\
    --questions data/experiments/lesion_subset/slake_subset.jsonl \\
    --full_image_dir data/SLAKE/imgs \\
    --mask_dir data/SLAKE/processed_imgs \\
    --out_dir data/background_randomized/imgs \\
    --method other_patient --num 20 --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from PIL import Image, ImageOps


def parse_args():
    p = argparse.ArgumentParser(description="Generate background randomized images")
    p.add_argument("--questions", required=True, help="JSONL file with image entries")
    p.add_argument("--full_image_dir", required=True, help="Directory with full images")
    p.add_argument("--mask_dir", required=True, help="Directory with processed masks")
    p.add_argument("--out_dir", required=True, help="Output directory for randomized images")
    p.add_argument("--method", choices=("other_patient", "noise"), default="other_patient")
    p.add_argument(
        "--fill-target",
        choices=("preserve", "remove"),
        default="preserve",
        help="Whether to preserve lesion foreground on new background ('preserve') or remove lesion and fill mask ('remove')",
    )
    p.add_argument(
        "--fill-method",
        choices=("zero", "noise"),
        default="zero",
        help="When fill-target is 'remove', method to fill masked region (zero or noise).",
    )
    p.add_argument(
        "--noise-strength",
        type=float,
        default=0.2,
        help="Noise strength in [0,1] when --fill-method noise (higher = stronger noise).",
    )
    p.add_argument(
        "--mask-scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mask (0.5 small, 1.0 original, >1.0 enlarge).",
    )
    p.add_argument("--num", type=int, default=100, help="Max number of samples to process (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_mask_for_image(mask_dir: Path, image_rel: str) -> Path | None:
    """
    Given mask_dir and image relative path like xmlab1/source.jpg,
    try common mask filename patterns in the corresponding subdir.
    """
    subdir, fname = os.path.split(image_rel)
    stem = Path(fname).stem
    candidates = [
        f"{stem}_reversed_mask.jpg",
        f"{stem}_mask.jpg",
        f"{stem}_processed.jpg",
        f"{stem}_mask.png",
        f"{stem}_reversed_mask.png",
    ]
    for c in candidates:
        p = mask_dir / subdir / c
        if p.exists():
            return p
    return None


def load_questions(qfile: Path):
    with open(qfile, "r", encoding="utf-8") as fr:
        # detect JSON array vs JSONL
        head = fr.read(2)
        fr.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(fr)
            for obj in data:
                yield obj
        else:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def ensure_mask_binary(mask: Image.Image) -> Image.Image:
    m = mask.convert("L")
    # threshold to binary
    m = m.point(lambda x: 255 if x > 128 else 0)
    # ensure lesion is white (255) - if most of mask is white, it's okay; otherwise invert
    bbox = m.getbbox()
    if bbox is None:
        return m
    white_pixels = sum(1 for v in m.getdata() if v > 128)
    black_pixels = m.size[0] * m.size[1] - white_pixels
    if white_pixels < black_pixels:
        m = ImageOps.invert(m)
    return m


def scale_mask(mask: Image.Image, scale: float) -> Image.Image:
    """
    Scale the mask around its center bbox by 'scale' factor.
    """
    if scale == 1.0:
        return ensure_mask_binary(mask)
    m = mask.convert("L")
    bbox = m.getbbox()
    if bbox is None:
        return ensure_mask_binary(mask)
    left, upper, right, lower = bbox
    cx = (left + right) / 2.0
    cy = (upper + lower) / 2.0
    w = (right - left) * scale
    h = (lower - upper) * scale
    new_left = int(max(0, cx - w / 2.0))
    new_upper = int(max(0, cy - h / 2.0))
    new_right = int(min(m.width, cx + w / 2.0))
    new_lower = int(min(m.height, cy + h / 2.0))
    # create new blank mask and paste resized bbox region
    new_mask = Image.new("L", m.size, 0)
    region = m.crop((left, upper, right, lower)).resize((int(new_right - new_left), int(new_lower - new_upper)))
    new_mask.paste(region, (new_left, new_upper))
    return ensure_mask_binary(new_mask)


def fill_mask_with_noise(image: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
    """Fill masked region in image with Gaussian noise. strength in [0,1] maps to std."""
    import numpy as np

    img = image.convert("RGB")
    m = mask.convert("L").point(lambda x: 255 if x > 128 else 0)
    arr = np.array(img).astype("float32")
    noise_std = max(1.0, strength * 80.0)
    noise = np.random.normal(loc=128.0, scale=noise_std, size=arr.shape).astype("float32")
    mask_arr = np.array(m) / 255.0
    mask_arr = mask_arr[..., None]
    out = arr * (1.0 - mask_arr) + noise * (mask_arr)
    out = np.clip(out, 0, 255).astype("uint8")
    return Image.fromarray(out)


def composite_foreground_on_background(fg_image: Image.Image, mask: Image.Image, bg_image: Image.Image) -> Image.Image:
    fg = fg_image.convert("RGBA")
    bg = bg_image.convert("RGBA").resize(fg.size)
    m = mask.convert("L").resize(fg.size)
    m = ensure_mask_binary(m)
    # create alpha from mask (lesion region is white -> keep)
    alpha = m
    fg.putalpha(alpha)
    # composite
    out = Image.alpha_composite(bg, fg)
    return out.convert("RGB")


def main():
    args = parse_args()
    random.seed(args.seed)
    qpath = Path(args.questions)
    full_dir = Path(args.full_image_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect candidate backgrounds if needed
    bg_candidates = []
    if args.method == "other_patient":
        for root, _, files in os.walk(full_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    bg_candidates.append(os.path.join(root, f))

    processed = 0
    entries = list(load_questions(qpath))
    if args.num > 0:
        entries = entries[: args.num]

    for obj in entries:
        # determine image relative path
        image_rel = None
        # possible fields
        for key in ("image", "full_image_path", "full_image", "image_path"):
            if key in obj and isinstance(obj[key], str):
                image_rel = obj[key]
                break
        if image_rel is None:
            continue
        # normalize: if absolute path points inside full_dir, convert to relative
        p = Path(image_rel)
        if p.is_absolute() and str(full_dir) in str(p):
            try:
                image_rel = str(Path(p).relative_to(full_dir))
            except Exception:
                image_rel = os.path.join(*p.parts[-2:])

        src_img_path = full_dir / image_rel
        if not src_img_path.exists():
            # try alternative base paths
            src_img_path = full_dir / os.path.basename(image_rel)
            if not src_img_path.exists():
                # skip if source image not found
                print(f"[warn] source image not found: {full_dir}/{image_rel}")
                continue

        mask_path = find_mask_for_image(mask_dir, image_rel)
        if mask_path is None:
            print(f"[warn] mask not found for {image_rel}")
            continue

        try:
            fg = Image.open(src_img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"[warn] failed to open image/mask for {image_rel}: {e}")
            continue

        # apply mask scaling
        scaled_mask = scale_mask(mask, args.mask_scale)

        if args.fill_target == "preserve":
            # original behavior: composite foreground (lesion) on another background
            if args.method == "other_patient":
                # choose a random background different from source
                bg_path = None
                attempts = 0
                while attempts < 10:
                    candidate = random.choice(bg_candidates)
                    if os.path.abspath(candidate) != os.path.abspath(src_img_path):
                        bg_path = candidate
                        break
                    attempts += 1
                if not bg_path:
                    print(f"[warn] no background candidate for {image_rel}")
                    continue
                bg = Image.open(bg_path).convert("RGB")
            else:
                # noise background
                bg = Image.effect_noise(fg.size, 100).convert("RGB")
            out_img = composite_foreground_on_background(fg, scaled_mask, bg)
        else:
            # fill-target == 'remove': fill masked region in the original image
            if args.fill_method == "zero":
                # fill with mid-gray (128) by default
                fill_color = (128, 128, 128)
                base = fg.copy()
                m = scaled_mask.convert("L")
                base_arr = base.copy()
                # paste a solid color where mask is white
                solid = Image.new("RGB", base.size, fill_color)
                base.paste(solid, mask=m)
                out_img = base
            else:
                # noise fill
                out_img = fill_mask_with_noise(fg, scaled_mask, args.noise_strength)

        out_path = out_dir / image_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, quality=95)
        processed += 1

    print(f"[generate_background_randomized] Processed {processed} images, output -> {out_dir}")


if __name__ == "__main__":
    main()

