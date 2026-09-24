# baselines — Baseline Methods

This directory contains implementations of preference optimization methods compared against MMedPO / CaMedPO.


## SimPO

- Paper: *SimPO: Simple Preference Optimization with a Reference-Free Reward* ([arXiv:2405.14734](https://arxiv.org/abs/2405.14734); PDF at `../literature/SimPO Simple Preference Optimization.pdf`)
- Key idea: a reference-free preference optimization method that uses the length-normalized average log probability as the implicit reward, avoiding the memory and compute cost of a reference model.

Structure:

```
SimPO/
├── alignment/          # Core training and alignment code
├── scripts/            # Training / evaluation launch scripts
├── training_configs/   # Training hyper-parameter configs
├── accelerate_configs/ # Distributed training configs
├── eval/               # Evaluation code
├── on_policy_data_gen/ # On-policy data generation
├── generate.py         # Generation script
└── environment.yaml    # Environment dependencies
```

See [`SimPO/README.md`](SimPO/README.md) for the official usage.

## Adding a new baseline

1. Create a subdirectory named after the method (e.g. `baselines/xxx/`)
2. Keep its original LICENSE and README, and add a one-line description here
