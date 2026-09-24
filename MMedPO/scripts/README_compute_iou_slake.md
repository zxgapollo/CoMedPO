Compute IOU for SLAKE predictions — README

Purpose
-------
This README explains how to run `compute_iou_slake.py` to generate a per-sample IOU CSV for SLAKE predictions (supports mask and bbox formats).

Quickstart
----------
1. Ensure Python 3.8+ is installed.
2. Create a venv and install deps:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pandas numpy pillow
   ```

3. Example usage (masks in directories):

   ```bash
   python compute_iou_slake.py \
     --gt-mask-dir /absolute/path/to/gt_masks \
     --pred SAM:/absolute/path/to/sam_pred_masks \
     --pred MedKLIP:/absolute/path/to/medklip_pred_masks \
     --out /absolute/path/to/outputs/slake_iou_per_sample.csv \
     --repo-root /   # optional, used to resolve relative mask_path entries
   ```

4. Example usage (CSV predictions):

   - GT CSV should have columns: `sample_id,mask_path` (mask_path can be absolute or relative to repo-root)
   - Pred CSV should have `sample_id,mask_path` or bbox columns `x1,y1,x2,y2`

   ```bash
   python compute_iou_slake.py \
     --gt-csv /path/to/gt.csv \
     --pred SAM:/path/to/sam_preds.csv \
     --pred MedKLIP:/path/to/medklip_preds.jsonl \
     --out outputs/slake_iou_per_sample.csv
   ```

Output
------
- JSON with structure:
  - `models` -> `<model_name>` -> `per_image` (mapping sample_id -> per-GT IoUs), `stats` (mean/median/count/pct_gt50), `raw_ious` (flat list of IoU values)
  - Example path: `outputs/slake/results.json`
- Plots saved under `outputs/slake/plots/`:
  - `iou_histograms.png` (fixed 5 bins: 0–20,20–40,40–60,60–80,80–100)
  - `iou_threshold_curve.png` (coverage vs IoU threshold)

Notes & troubleshooting
-----------------------
- If no IOU rows are produced, check that `sample_id` keys match between GT and pred files, or that mask files exist and are readable.
- The script supports masks (PNG/JPG) and simple bbox (x1,y1,x2,y2). If you have RLE or other mask encodings, convert to binary mask images first.

Notes on binning
----------------
- This evaluation uses **fixed** five equal-width bins (0–20,20–40,40–60,60–80,80–100) for histograms and reporting by default.

If you want, I can:
- Add a small example GT/pred sample CSVs to the repo.
- Produce a Jupyter Notebook that reads the JSON and plots the distribution (PNG) per model.

Combined SAM + text-guided box evaluation
-----------------------------------------
We added a combined evaluation script that uses SAM (and optionally CLIPSeg / text prompts) to detect lesion candidates in `source` images, converts instance masks to boxes, and matches those boxes against SLAKE `detection.json` entries.

Usage example:

```bash
# Run combined pipeline (requires gt-mask-dir pointing to SLAKE imgs root)
python3 eval_slake_combined.py \
  --data-root /share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs \
  --sam-checkpoint /share_docker/workspace/Med/Med-main/Med-main/MMedPO/models/sam/sam_vit_h_4b8939.pth \
  --device cuda \
  --output-dir /share_docker/workspace/Med/Med-main/Med-main/MMedPO/eval_outputs/slake_combined
```You can also run the integrated helper script:```bash
./run_compute_iou_slake.sh --gt-mask-dir /share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE/imgs --sam-pred /share_docker/workspace/Med/Med-main/Med-main/MMedPO/models/sam/sam_vit_h_4b8939.pth --run-combined --out /share_docker/workspace/Med/Med-main/Med-main/MMedPO/eval_outputs/slake_combined --skip-install
```Notes:
- The combined pipeline skips any image "hole" / masking edits and compares final predicted boxes to GT boxes (as requested).
- If CLIPSeg models are available in the local cache, the script will use text prompts (GT labels) to guide matching; otherwise it falls back to IoU-based matching.
- Output: `results_combined.json` and plots under `plots_combined/`.
