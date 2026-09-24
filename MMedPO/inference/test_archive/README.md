# TIE-ANKER DPO test archive

This directory retains historical tests, outputs, and reports for preference-pair construction. It is separate from the CoMedPO training regression tests in the repository's `tests/` directory.

## Contents

- `scripts/`: Pair-construction tests, export tests, and debugging scripts, including `test_optimized_implementation.py`, `test_final_optimized.py`, `test_dpo_pairs_logic.py`, `test_dpo_export.py`, `debug_optimized_implementation.py`, and `debug_dpo_export.py`.
- `results/`: Generated JSON preference pairs and validation results.
- `reports/`: Pair analysis and optimization reports.
- `backup/`: Historical implementation copies where present.

## Historical notes

The original notes describe validation of both preferred and dispreferred answers, additional failure diagnostics, and threshold tuning. They report zero generated pairs from three initial examples and two pairs from five later examples. These small, different test sets do not establish a general improvement rate.

Recorded parameter changes were `tau_gamma_strong` from 0.5 to 1.5, `tau_gamma_weak` from 0.1 to 0.2, `tau_v` from 0.5 to 0.3, and `tau_n_percentile` from 75 to 70.0. These are historical curation settings, not the CoMedPO loss hyperparameters.

## Running archived scripts

After configuring their environment-specific paths:

```bash
cd MMedPO/inference/test_archive/scripts
python test_final_optimized.py
python debug_optimized_implementation.py
```

Inspect `../results/final_optimized_pairs.json` and `../reports/optimization_summary_report.md` for the corresponding archived output and report. Preserve useful results when changing or rerunning these historical scripts.
