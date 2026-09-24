# MMedPO utilities

This directory contains inherited dataset reconstruction and weighting scripts. They support historical weighted-DPO experiments. The main CoMedPO objective does not use their sample weights; see the [project README](../../README.md).

## Weight recomputation

- `recompute_dpo_weights.py`: Uses per-case normalization. The historical notes report a high mean weight of approximately 0.96 and mark this version as deprecated.
- `improved_dpo_weights.py`: Uses global normalization and adjusted parameters. Configure the input/output paths in the script before running it from the `MMedPO` directory.

```bash
python utils/improved_dpo_weights.py
```

## Historical output

The original experiment notes describe `outputs/tie_dpo_dataset_improved.jsonl`, a JSONL file with 4,919 samples, weights in `[0.01, 0.99]`, mean 0.4540, median 0.4400, and standard deviation 0.2284. They report 1,961 weights above 0.5, 729 above 0.7, and 154 above 0.9. These are archived statistics, not results recomputed during the CoMedPO implementation audit.

To rerun a legacy weighted-DPO experiment, point its `--data_path` argument to the generated JSONL file. This file alone does not provide the aligned factual/counterfactual inputs required by CoMedPO.
