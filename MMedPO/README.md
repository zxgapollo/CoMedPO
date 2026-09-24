# [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](https://proceedings.mlr.press/v267/zhu25v.html)


This document describes inherited MMedPO tools. For the CoMedPO objective and new training entry point, see the [root README](../README.md).

## 💡 Overview

<div align=left>
<img src=assets/logo.png width=90% />
</div>

## 📦 Requirements

1. Clone this repository and navigate to MMedPO folder

```bash
git clone https://github.com/aiming-lab/MMedPO.git
cd MMedPO
```

2. Install Package: Create conda environment

```Shell
conda create -n MMedPO python=3.10 -y
conda activate MMedPO
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install trl
```

3. Download the required model checkpoints [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.

4. For model checkpoints, we released four [checkpoints](https://huggingface.co/zky11235/mmedpo_checkpoints) of MMedPO in the huggingface.

5. For all the medical datasets, you need firstly apply for the right of access and then download the dataset.

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) (Thanks to [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) for sharing the file)
- [VQA-RAD](https://osf.io/89kps/)
- [SLAKE](https://www.med-vqa.com/slake/)

## 🪧 Data Curation

We use MedKLIP to generate visual preference data. Use the following command or the script `inference_attention-map_score.sh` at `./scripts`

```Shell
python ./inference_attention-map_score.py \
    --config ./MedKLIP_config.yaml \
    --model_path /path/to/MedKLIP_model.pth \
    --dataset_name /dataset/name \
    --dataset_type caption \
    --image_root /path/to/dataset/image_folder \
    --annotation_save_root /path/to/save/annotation \
    --noised_image_save_root /path/to/save/noised_image \
```

## 🏋️ Train

Use the script `train_dpo_visual-text.sh` in `./scripts` or the following command, make sure to specify the necessary data paths and the checkpoint saving location.

```
deepspeed --include localhost:0,1,2,3 ./train/dpo/train_dpo_visual-text.py \
    --model_name_or_path /path/to/llava-med_model_checkpoint \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /path/to/data_json \
    --image_folder /path/to/img_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path/to/output_checkpoint_saving_location \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
```

## 🚀 Inference

The inference script is at `scripts` folder. You can run after specifying relevant paths:

```
bash scripts/inference_llava-med_{vqa/report}.sh
```

## 📚 Citation

```bibtex
@article{zhu2024mmedpo,
  title={MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization},
  author={Zhu, Kangyu and Xia, Peng and Li, Yun and Zhu, Hongtu and Wang, Sheng and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2412.06141},
  year={2024}
}
```

## 🙏 Acknowledgement

We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [RULE](https://github.com/richard-peng-xia/RULE), [MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP). We thank the authors for releasing their code.

---

# Workspace Extensions (Scripts)

> The following describes scripts added or extended in this repository, complementing the official README.

## Directory guide

| Directory | Content |
|-----------|---------|
| `scripts/` | One-click scripts and tools for training, inference, evaluation and data augmentation |
| `train/dpo/` | DPO training code (`train_dpo_weighted.py`, `train_dpo_medgemma.py`, `train_dpo_dual_gpu.py`, `llava_trainer_weighted.py`, `dpo_trainer_weighted.py`) |
| `train/rl/` | Reinforcement learning training (`train_grpo_stage3.py`, GRPO stage) |
| `inference/` | Preference pair construction and inference (`build_dpo_pairs_*.py`, `analyze_*`, `generate_master_question_set.py`) |
| `eval/` | Evaluation scripts (`eval_vqa.py`, `eval_report.py`, `run_eval.sh`, `model_download.py`) |
| `utils/` | Common utilities: tie weight computation, data format conversion, DPO weight recomputation, LoRA merging, etc. |
| `tools/` | Image processing tools (`remove_lesions_text_clipseg.py`, CLIPSeg-based lesion removal) |
| `curation/` | Data curation scripts |
| `rebuttal/` | Additional experiments and results for the rebuttal |
| `assets/` | Images and other resources |
| `data/` | Pre-built preference datasets and data conversion scripts |

## Common workflow

```bash
# 1. Build preference pairs (method 1: tie weights)
bash scripts/run_inference_visual_indirect.sh

# 2. Training
bash scripts/train_sft.sh            # SFT
bash scripts/train_dpo_visual-text.sh  # DPO
bash scripts/train_sspo.sh            # SSPO
bash scripts/train_tie_sspo.sh        # SSPO (dynamic w)

# 3. Inference
bash scripts/inference_llava-med_vqa.sh      # VQA
bash scripts/inference_llava-med_report.sh   # Report generation

# 4. Evaluation
bash eval/run_eval.sh                        # MedEvalKit evaluation
bash scripts/run_compute_iou_slake.sh        # SLAKE IoU
bash scripts/run_evaluate_llmjudge.sh        # LLM-as-judge

# 5. Result collection
python scripts/plot_slake_results.py
python scripts/evaluate_and_make_latex.py    # Generate LaTeX result tables
```

## Debugging and diagnostic scripts

`scripts/` also contains troubleshooting scripts for locating issues when training behaves abnormally: `check_lora_weights.py`, `debug_model_type.py`, `debug_training_environment.py`, `test_device_check.py`, `test_dpo_device_debug.py`, `test_dual_gpu_dpo.py`, `test_fixed_dpo_trainer.py`, `fix_disable_adapter_issue.py`, `verify_policy_reference_diff.py`, etc.

## Data augmentation

- `generate_background_randomized.py` — generate randomized backgrounds (for constructing visual bias samples)
- `generate_lesion_subset.py` — generate a lesion subset
- `generate_simulated_iou.py` — generate simulated IoU data
