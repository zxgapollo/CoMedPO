"""Aligned factual/counterfactual visual conditions for CoMedPO training."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CAUSAL_PROMPT = (
    "Both images belong to the same patient. Please analyze each and then "
    "provide a joint clinical conclusion."
)


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl" else json.loads(text))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training data must be a nonempty JSON array or JSONL file")
    return rows


def normalize_record(row):
    """Accept explicit strings or the existing single-turn MMedPO schema."""
    row = dict(row)
    if "conversations" in row:
        chosen, rejected = row["conversations"], row["rejected_conversations"]
        for conv in (chosen, rejected):
            if len(conv) != 2 or [x["from"] for x in conv] != ["human", "gpt"]:
                raise ValueError("Only single-turn human/gpt preference pairs are supported")
        if chosen[0]["value"] != rejected[0]["value"]:
            raise ValueError("Chosen and rejected responses must share the same question")
        row.update(question=chosen[0]["value"], chosen=chosen[1]["value"],
                   rejected=rejected[1]["value"])
    for key in ("image", "question", "chosen", "rejected"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing nonempty string field: {key}")
    row["question"] = row["question"].replace("<image>", "").strip()
    if not row["question"] or row["chosen"].strip() == row["rejected"].strip():
        raise ValueError("Question must be nonempty and preference answers must differ")
    if not row.get("background_image") and not row.get("lesion_mask"):
        raise ValueError("Provide background_image or lesion_mask; no intervention is inferred")
    if bool(row.get("factual_image")) != bool(row.get("counterfactual_image")):
        raise ValueError("Precomputed factual_image and counterfactual_image must be paired")
    return row


def stitch(left, right):
    if left.size != right.size:
        raise ValueError("Background and original/corrupted images must have equal dimensions")
    result = Image.new("RGB", (left.width * 2, left.height))
    result.paste(left, (0, 0))
    result.paste(right, (left.width, 0))
    return result


class CoMedPODataset(Dataset):
    """Return six aligned (question, answer, RGB image) branches per record.

    Nonzero mask pixels identify lesions to zero out. If composites are absent,
    a fixed per-record RNG perturbs the original image with additive Gaussian
    noise in [0, 1] pixel units. Both composites have exactly the same left panel.
    Paths resolve relative to image_root (default: the manifest directory).
    """

    def __init__(self, data_path, image_root=None, noise_std=0.1, seed=42):
        if not math.isfinite(noise_std) or noise_std <= 0:
            raise ValueError("noise_std must be finite and positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.rows = [normalize_record(row) for row in read_records(data_path)]
        self.root = Path(image_root) if image_root else Path(data_path).resolve().parent
        self.noise_std, self.seed = noise_std, seed
        for row in self.rows:
            for key in ("image", "background_image", "lesion_mask", "factual_image", "counterfactual_image"):
                if row.get(key) and not (self.root / row[key]).is_file():
                    raise FileNotFoundError(self.root / row[key])

    def __len__(self):
        return len(self.rows)

    def _open(self, name, mode="RGB"):
        with Image.open(self.root / name) as image:
            return image.convert(mode)

    def __getitem__(self, index):
        row = self.rows[index]
        original = self._open(row["image"])
        if row.get("background_image"):
            background = self._open(row["background_image"])
        else:
            mask = self._open(row["lesion_mask"], "L")
            if mask.size != original.size:
                raise ValueError("Lesion mask must match the original image dimensions")
            array = np.array(original)
            array[np.asarray(mask) > 0] = 0
            background = Image.fromarray(array)
        if background.size != original.size:
            raise ValueError("Background image must match original image dimensions")
        if row.get("factual_image"):
            factual = self._open(row["factual_image"])
            counterfactual = self._open(row["counterfactual_image"])
            if factual.size != counterfactual.size:
                raise ValueError("Factual and counterfactual composites must have equal dimensions")
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
            pixels = np.asarray(original, dtype=np.float32) / 255.0
            corrupted = Image.fromarray((np.clip(
                pixels + rng.normal(0, self.noise_std, pixels.shape), 0, 1
            ) * 255).astype(np.uint8))
            factual = stitch(background, original)
            counterfactual = stitch(background, corrupted)
        question = row["question"]
        causal_question = row.get("causal_question") or f"{CAUSAL_PROMPT}\n{question}"
        chosen, rejected = row["chosen"], row["rejected"]
        return [
            (question, chosen, original), (question, rejected, background),
            (causal_question, chosen, factual), (causal_question, chosen, counterfactual),
            (causal_question, rejected, factual), (causal_question, rejected, counterfactual),
        ]
