# CoMedPO

Official code for **Towards Reliable Medical Large Vision-Language Models via Counterfactual Preference Optimization**.

CoMedPO combines ordinary-input preference optimization with a symmetric factual/counterfactual preference contrast. It aims to reduce background shortcuts while preserving useful lesion-background context in medical vision-language models.

This repository builds on the [original project workspace](https://github.com/FAyyn/Med), [MMedPO](https://github.com/aiming-lab/MMedPO), and [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit). The CoMedPO implementation is in [`MMedPO/comedpo`](MMedPO/comedpo). The inherited baseline, curation, inference, and evaluation code is retained. See the [implementation audit](docs/implementation_audit.md) for the correspondence between the manuscript and the code.

## Method

For a clinical question `T`, preferred response `y_w`, and rejected response `y_l`, training uses four visual conditions:

| Condition | Image | Question |
| --- | --- | --- |
| `x` | Original image `I` | `T` |
| `x*` | Lesion-masked background `I_b` | `T` |
| `x_c` | Horizontal composite `I_b ⊕ I` | `T'` |
| `x_c*` | Horizontal composite `I_b ⊕ I_null` | `T'` |

`I_null` is a Gaussian-noise-perturbed version of the original image. The left background panel is identical in both composites. `T'` prepends: "Both images belong to the same patient. Please analyze each and then provide a joint clinical conclusion."

Define `a(y, x) = log π_θ(y|x) - log π_ref(y|x)`. Equations (11)-(14) of the manuscript become:

```text
base_margin   = a(y_w, x) - a(y_l, x*)
causal_margin = [a(y_w, x_c) - a(y_w, x_c*)]
              - [a(y_l, x_c) - a(y_l, x_c*)]

L_DPO    = mean(-log sigmoid(beta * base_margin))
L_cDPO   = mean(-log sigmoid(beta * causal_margin))
L_CoMedPO = L_DPO + causal_weight * L_cDPO
```

`beta = eta^{-1}` in the paper; it is separate from the optimizer learning rate. Both responses receive the counterfactual correction. Sequence log probabilities are **summed over answer tokens**, with prompt, visual, and padding tokens excluded. The reference is the frozen base/SFT checkpoint. Clinical-relevance and TIE sample weights are not applied to this objective.

## Installation

Use a separate Python 3.10-3.12 environment for CoMedPO training:

```bash
git clone https://github.com/zxgapollo/CoMedPO.git
cd CoMedPO
python -m venv .venv
source .venv/bin/activate
python -m pip install -r MMedPO/comedpo/requirements.txt
```

Install a CUDA-compatible PyTorch 2.5.1 build on the training machine if necessary. The new entry point imports the bundled LLaVA implementation directly. It does not require installing the legacy `MMedPO/train/dpo/pyproject.toml`, whose Transformers pin predates its Mistral implementation. Keep MedEvalKit and other legacy tools in their own environments using their existing requirements.

Provide a [LLaVA-Med v1.5 Mistral checkpoint](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b), or a compatible **merged SFT checkpoint**. Its tokenizer, multimodal projector, and configured CLIP vision tower must be available. An SFT LoRA adapter must first be merged into its base model; disabling the newly created CoMedPO adapter then recovers the exact frozen SFT reference.

## Training data

Supply a JSON array or JSONL file. Image paths resolve relative to `--image-root`, or to the manifest directory when that option is omitted.

```json
[
  {
    "id": "case-001",
    "image": "images/case-001.png",
    "lesion_mask": "masks/case-001.png",
    "question": "Is pleural effusion present?",
    "chosen": "Yes, pleural effusion is present.",
    "rejected": "No pleural effusion is present."
  }
]
```

The example illustrates the schema; it is not a clinical annotation. Supply curated responses and lesion localization from your own data pipeline.

- Nonzero pixels in `lesion_mask` mark lesions to be filled with zero-valued pixels. The mask and original image must have the same dimensions.
- Alternatively, supply `background_image`, an already prepared lesion-masked image with the original dimensions.
- Optionally supply both `factual_image` and `counterfactual_image`, containing precomputed stitched composites. You are responsible for ensuring they preserve the same background and implement the intended intervention. `causal_question` can specify the exact composite prompt used during curation.
- Without precomputed composites, the loader generates a fixed, seeded additive Gaussian perturbation per record. `--noise-std` is measured in `[0, 1]` pixel units, followed by clipping to that range. The two complete panels are preserved by square-padding before CLIP preprocessing.
- Existing MMedPO single-turn `conversations` / `rejected_conversations` records are accepted after adding `lesion_mask` or `background_image`. Existing `weighted_score` fields are ignored. A weighted preference file by itself does **not** supply the required causal visual conditions.

The repository includes historical manifests and curation tools, but it does not bundle a complete, validated CoMedPO training release for every benchmark. Obtain the corresponding dataset images, masks, and model weights separately. Existing localization tools are under `MMedPO/curation` and `MMedPO/tools`; the paper uses Med-SAM, while inherited scripts also include MedKLIP. The new trainer consumes their resulting masks rather than performing segmentation or paid response generation automatically.

## Training

From the repository root, on one GPU:

```bash
bash MMedPO/scripts/train_comedpo.sh \
  --model-path /path/to/merged-llava-med-sft \
  --data-path /path/to/comedpo-train.json \
  --image-root /path/to/dataset \
  --output-dir /path/to/runs/comedpo \
  --beta 0.1 --noise-std 0.1
```

For eight GPUs with distributed data parallel training:

```bash
NUM_PROCESSES=8 bash MMedPO/scripts/train_comedpo.sh \
  --model-path /path/to/merged-llava-med-sft \
  --data-path /path/to/comedpo-train.json \
  --image-root /path/to/dataset \
  --output-dir /path/to/runs/comedpo \
  --beta 0.1 --noise-std 0.1
```

The defaults reported in the manuscript are LoRA rank 128, scaling 256, dropout 0.05, no bias, three epochs, per-device batch size two, gradient accumulation one, learning rate `1e-6`, cosine scheduling, warmup ratio `0.03`, zero weight decay, and causal weight `1.0`. The paper reports eight A100 GPUs. This entry point supports single-device and DDP training; FSDP and DeepSpeed are not implemented here.

Maximum combined text/image length defaults to 2048. Overlength examples raise an error instead of silently losing response tokens. Gradient checkpointing is enabled by default. The available new model integration is LLaVA-Med Mistral; the loss function is architecture-independent, but MedGemma requires its own model-specific adapter.

Each completed epoch replaces `checkpoint-last/` and `adapter/`; only the latest checkpoint is retained. `checkpoint-last/` contains optimizer, scheduler, and RNG state. Resume an interrupted multi-epoch run with the **same arguments** plus `--resume`. Recovery starts after the last fully saved epoch; partial-epoch progress is not saved. `training_config.json` records the arguments. Use `--causal-weight 0` for the unweighted ordinary-input objective ablation.

## Inference and evaluation

CoMedPO inference uses the **original single image and original clinical query**. Causal composites are needed for training, not for ordinary inference.

Merge the saved `adapter/` into the same base/SFT checkpoint with PEFT before using the inherited inference scripts:

```bash
PYTHONPATH=MMedPO/train/dpo python - <<'PY'
from peft import PeftModel
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

base_path = "/path/to/merged-llava-med-sft"
adapter_path = "/path/to/runs/comedpo/adapter"
merged_path = "/path/to/merged-comedpo"
base = LlavaMistralForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
model.save_pretrained(merged_path)
AutoTokenizer.from_pretrained(base_path).save_pretrained(merged_path)
PY
```

The merged model retains its configured vision tower identifier/path. Keep that tower available. Configure model and dataset paths in the inherited inference scripts under [`MMedPO/inference`](MMedPO/inference). For benchmark evaluation, see [`MedEvalKit/Readme.md`](MedEvalKit/Readme.md); report metrics are also provided under [`evaluation`](evaluation/README.md). These legacy tools have separate dependencies and environment-specific paths.

## Tests

```bash
python -m pip install pytest
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=true \
  HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/test_comedpo.py tests/test_comedpo_llava.py
```

The tests check equation values, all four causal gradient signs, reference detachment, `lambda=0`, masked sequence likelihoods, paired visual construction, and a real randomly initialized tiny LLaVA-Mistral/CLIP training step with checkpoint save/load. They require no external model downloads.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `MMedPO/comedpo/` | CoMedPO loss, data, LLaVA integration, and training entry point |
| `MMedPO/train/`, `MMedPO/scripts/` | Inherited SFT/DPO variants and training/curation launchers |
| `MMedPO/inference/`, `MMedPO/curation/`, `MMedPO/utils/` | Inference, localization, and historical data processing |
| `MedEvalKit/` | Inherited medical model evaluation framework |
| `baselines/`, `evaluation/`, `analysis/` | Baselines, metric tools, and experiment analysis |
| `tests/`, `docs/implementation_audit.md` | CoMedPO regression tests and paper-to-code audit |

Some historical launchers depend on files that were absent from the source snapshot. In particular, `train_dpo_dual_gpu.py` imports the missing `tool.dual_gpu_dpo_trainer`. Those legacy paths are not the new CoMedPO entry point.

## Acknowledgments and references

We thank the authors of **MMedPO** for their clinical-aware multimodal preference optimization framework and released code, which provide the foundation for the inherited preference curation and training infrastructure. We also thank the **Lingshu / MedEvalKit** authors for their unified medical evaluation framework. We cite the formally published versions below:

- Kangyu Zhu, Peng Xia, Yun Li, Hongtu Zhu, Sheng Wang, and Huaxiu Yao. **MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization.** ICML 2025, PMLR 267, pp. 80207-80222. [Published paper](https://proceedings.mlr.press/v267/zhu25v.html) · [Code](https://github.com/aiming-lab/MMedPO).
- Weiwen Xu et al. **Lingshu: Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, early access. DOI: [10.1109/TPAMI.2026.3730310](https://doi.org/10.1109/TPAMI.2026.3730310). This paper introduces MedEvalKit. [Bibliographic record](https://pubmed.ncbi.nlm.nih.gov/42685180/) · [Code](https://github.com/alibaba-damo-academy/MedEvalKit).

```bibtex
@inproceedings{zhu2025mmedpo,
  title = {{MMedPO}: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization},
  author = {Zhu, Kangyu and Xia, Peng and Li, Yun and Zhu, Hongtu and Wang, Sheng and Yao, Huaxiu},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {267},
  pages = {80207--80222},
  year = {2025},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v267/zhu25v.html}
}

@article{xu2026lingshu,
  title = {Lingshu: Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning},
  author = {Xu, Weiwen and Chan, Hou Pong and Li, Long and Aljunied, Mahani and Yuan, Ruifeng and Wang, Jianyu and Xiao, Chenghao and Chen, Guizhen and Liu, Chaoqun and Li, Zhaodonghui and Sun, Yu and Shen, Junao and Wang, Chaojun and Hu, Can and Pan, Siwei and Tan, Jie and Xu, Tingyang and Zhao, Deli and Zhang, Hao and Rong, Yu},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2026},
  note = {Early access},
  doi = {10.1109/TPAMI.2026.3730310}
}
```

We also acknowledge LLaVA-Med/LLaVA and the other upstream libraries included in this workspace. Original copyright notices and component licenses are retained, including [`MMedPO/LICENSE`](MMedPO/LICENSE) and [`MMedPO/train/dpo/LICENSE`](MMedPO/train/dpo/LICENSE). Refer to each component's license for its terms.
