#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# Available datasets: MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="SLAKE" 
# Fix: For the VQA_RAD dataset, use the specific JSONL file path instead of the directory path
DATASETS_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data"
# The DATASETS_PATH should be the parents directory of the dataset directory
# E.g., if the dataset directory is /workspace/MMedPO/datasets/SLAKE, then the DATASETS_PATH should be /workspace/MMedPO/datasets
# And you should rename your slake1.0 folder to SLAKE, this was decided by the EVAL_DATASETS="SLAKE"

OUTPUT_PATH="/share_docker/workspace/Med/Med-main/Med-main/MedEvalKit/Eval_Results/SFT_medgemma_SLAKE"
# Available models: TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="MedGemma"
# Path to LoRA checkpoint (contains mm_projector.bin or non_lora_trainables.bin)
# Now you could use the MODEL_PATH to evaluate the checkpoint
# 使用训练转换后的本地HF权重目录
MODEL_PATH="/share_docker/workspace/Med/Med-main/Med-main/MMedPO/checkpoint/sft_model_lora_medgemma_slake/reexport_hf"


# Path to base model (e.g., llava-med-v1.5-mistral-7b merged or original base)
# 基座模型在MedGemma下不需要（非LoRA），保持与MODEL_PATH一致或留空
BASE_MODEL_PATH=""

# BASE_MODEL_PATH could be online path in huggingface too.

# VLLM setting
CUDA_VISIBLE_DEVICES="0,1,2,3"
TENSOR_PARALLEL_SIZE="4"
USE_VLLM="false"

# vLLM 多卡必要环境
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TOKENIZERS_PARALLELISM="false"

# Evaluation setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Model LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting - 优化后的配置
USE_LLM_JUDGE="True"
JUDGE_MODEL_TYPE="deepseek"  # openai or gemini or deepseek or claude

# DeepSeek API 配置 - 需要确保账户余额充足
# # gpt api model name
# GPT_MODEL="gpt-4.1-2025-04-14"
GPT_MODEL="deepseek-chat"  # DeepSeek聊天模型
API_KEY="${API_KEY:-}"  # Supply the API key through the environment.
BASE_URL="https://api.deepseek.com"

# JUDGE_MODEL_TYPE="gemini"  # openai or gemini or deepseek or claude
# GPT_MODEL="gemini-2.5-flash"  # DeepSeek聊天模型
# API_KEY="AIzaSyAqkZ2cavUcwj01mcE_PmsMir3j8s5lWhk"  # 注意：确保账户有足够余额
# BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"


# pass hyperparameters and run python sccript
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
/share_docker/conda_envs/MMedPO/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jsonlines pydash
/share_docker/conda_envs/MMedPO/bin/python "$SCRIPT_DIR/eval.py" \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --model_base "$BASE_MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
