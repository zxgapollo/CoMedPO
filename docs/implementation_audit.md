# CoMedPO implementation audit

## Source and scope

This release starts from the user-provided `Med-main.zip`, whose archive comment and the source repository HEAD identify commit `e9debbce807d7e009ec1d872f4d7d0aac30b79f1` of `FAyyn/Med`. The audit used the supplied main manuscript, *Towards Reliable Medical Large Vision-Language Models via Counterfactual Preference Optimization*, and its supplementary material. These PDFs are source material for the audit and are not added to this repository.

## Findings and implementation mapping

| Paper component | Source snapshot | Released implementation |
| --- | --- | --- |
| Main Eq. (13), different visual conditions for chosen/rejected responses | `dpo_trainer_weighted.py` has policy/reference DPO with separate images and optional weights | `comedpo/loss.py` columns 0-1 implement the unweighted base term |
| Main Eqs. (11)-(12), symmetric counterfactual contrast | No complete four-branch policy/reference contrast found in the available trainers | `comedpo/loss.py` columns 2-5 implement corrections for both responses |
| Main Eq. (14), joint base plus causal objective | No connected joint training entry point found | `comedpo/model.py` and `comedpo/train.py` compute both terms in every update |
| Four image conditions with common background and composite query | Historical masking, stitching, and TIE inference tools exist | `comedpo/data.py` consumes masks/backgrounds and creates or loads aligned composites |
| Frozen SFT reference | Legacy DPO has reference-model machinery | New LoRA adapters are trained against the frozen adapter-disabled base/SFT checkpoint |
| Answer-only sequence likelihood | Legacy DPO sums likelihoods; the MedGemma script uses a weighted mean-loss contrast without a reference ratio | New LLaVA integration aligns labels after visual-token insertion and sums only response-token log likelihoods |
| Supplement Eq. (23), causal ratio weighting | Related TIE weighting experiments exist | Kept separate from the main-paper objective; no claim that historical heuristic weights reproduce Eq. (23) exactly |
| Training settings, main Section 5 | Several historical scripts and environment-specific configurations exist | Reported LoRA/optimizer/epoch settings are the new defaults; omitted beta/noise values are explicitly documented |

`MMedPO/train/dpo/train_dpo_dual_gpu.py` references `tool.dual_gpu_dpo_trainer`, which is absent from the provided archive. `MMedPO/scripts/run_sppo.py` also relies on external `alignment` and `trainer` modules not supplied for that entry point. Therefore, those launchers cannot establish that the submitted CoMedPO equations were implemented in the available snapshot. Existing weighted or causal pair-ranking scripts are not interchangeable with the symmetric online objective.

## Scope of changes

- Added an isolated CoMedPO package, launcher, requirements, and focused tests.
- Replaced the project README and converted the remaining Chinese README text to English, reusing existing English copies where available.
- Replaced hardcoded API/W&B credential literals in three inherited launch/training files and one launcher backup with environment-variable reads before publishing the new snapshot.
- Retained all other inherited source, baseline algorithms, evaluation logic, data, and original license notices. Existing Python caches and ignored artifacts are excluded by Git ignore rules.

## Reproducibility boundaries

The objective follows main-paper Eqs. (11)-(14), not the causal-ratio weighting ablation in supplement S6.1. The supplied paper leaves beta, Gaussian noise strength, exact preprocessing details, and the complete training manifests unspecified. The new defaults and explicit data contract make those choices inspectable; they are not evidence of reproducing the reported numerical results.

The new end-to-end adapter targets LLaVA-Med Mistral. It supports fresh CoMedPO LoRA adapters on an original or merged SFT checkpoint, single-device/DDP training, answer-only likelihoods, and recovery from the last completed epoch. It does not claim new end-to-end implementations for MedGemma, mDPO, SPPO, FSDP, or DeepSpeed. The inherited implementations remain available as historical code.

Local validation uses a real tiny randomly initialized LLaVA-Mistral model and local CLIP tower, in addition to loss/data regression tests. Full medical datasets, trained checkpoints, benchmark reproduction, and eight-A100 execution were not available in this audit.

## Validation results

- 13 focused CPU tests passed with PyTorch 2.5.1, Transformers 4.37.2, PEFT 0.9.0, Accelerate 0.27.2, and Pillow 11.3.0.
- A separate two-process CPU DDP smoke run completed one optimizer update and saved the adapter and checkpoint. The initial total/base/causal losses were 1.38629436 / 0.69314718 / 0.69314718, as expected when policy and reference start identically.
- Training CLI help, Python compilation, launcher shell syntax, new README local links, and new-file whitespace checks passed.
- New code and all README files contain English text; no Chinese README text remains.
- Full CUDA/A100 runs and benchmark scores remain unverified.
