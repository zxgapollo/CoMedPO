#!/bin/bash

# === 一键推理 + 评估脚本（支持多卡 0,1,2,3）===
# 使用方法：
#   bash eval/run_eval.sh [gpu_ids]
# 示例：
#   bash eval/run_eval.sh          # 自动使用所有可用 GPU（最多 4 张）
#   bash eval/run_eval.sh 0,1      # 指定 GPU 0,1

set -e

# ----------- 参数解析 -----------
GPU_IDS=${1:-0,1,2,3}                # 默认只使用 GPU 0 (减少内存使用)
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

VISIBLE_GPUS=$(IFS=,; echo "${GPUS[*]}")
export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS

# 设置 PyTorch CUDA 内存管理参数
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

echo "🎯 使用 GPU: $VISIBLE_GPUS  (共 $NUM_GPUS 张)"
echo "🔧 内存优化: max_split_size_mb=128"

# ----------- 路径配置 -----------
GT_FILE="/workspace/MMedPO/datasets/SLAKE/master_question_with_dpo.json"
PRED_RAW="/workspace/MMedPO/outputs/DPO/master_question_all_raw.jsonl"
PRED_FIXED="/workspace/MMedPO/outputs/DPO/master_question_all_pred.json"
CSV_OUTPUT="/workspace/MMedPO/outputs/DPO/master_question_all_eval.csv"

# ----------- 1. 多卡推理 -----------
export BASE_MODEL="/workspace/llava-med-v1.5-mistral-7b" # 基础模型（本地缓存路径）
export MODEL_BASE="/workspace/MMedPO/outputs/enhanced_tie_dpo_model" # DPO训练后的adapter权重
echo "🔄 正在运行推理"
echo "   基础模型: $BASE_MODEL (将从HuggingFace在线下载)"
echo "   DPO权重: $MODEL_BASE"

if (( NUM_GPUS == 1 )); then
    # 单卡：直接 python
    python /workspace/MMedPO/inference/llava-med-1.5_vqa.py \
        --model-path "$MODEL_BASE" \
        --model-base "$BASE_MODEL" \
        --question-file "$GT_FILE" \
        --answers-file "$PRED_RAW"
else
    # 多卡：torchrun
    torchrun --nproc_per_node=$NUM_GPUS \
        /workspace/MMedPO/inference/llava-med-1.5_vqa.py \
        --model-path "$MODEL_BASE" \
        --model-base "$BASE_MODEL" \
        --question-file "$GT_FILE" \
        --answers-file "$PRED_RAW"
fi

# 2. 转换为评估所需格式（JSON 数组）
echo "🔄 正在转换格式..."
python3 - << EOF
import json, sys
raw_path = "$PRED_RAW"
fixed_path = "$PRED_FIXED"

# 读取预测结果（已包含gt_answer）
with open(raw_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

# 提取 question_id, answer 和 gt_answer
converted = []
for item in lines:
    qid = item.get("id")
    pred_answer = item.get("answer", "").strip()
    gt_answer = item.get("gt_answer", "").strip()
    
    converted.append({
        "question_id": qid,
        "answer": pred_answer,
        "gt_answer": gt_answer
    })

with open(fixed_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成: {len(converted)} 条记录")
print(f"   - 预测文件: {raw_path}")
print(f"   - 输出文件: {fixed_path}")
EOF

# 3. 运行评估
echo "🔄 正在评估..."
python eval_vqa_restored.py \
  --gt-file "$GT_FILE" \
  --pred-file "$PRED_FIXED" \
  --csv-output "$CSV_OUTPUT"

echo "✅ 评估完成，结果保存在：$CSV_OUTPUT"