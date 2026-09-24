#!/usr/bin/env python3
"""
修复VQA-RAD数据集的完整性和对应关系
确保vqa-rad_slake_format.json包含所有问题，并与其他数据文件完全对应
"""

import json
import os
import shutil
from pathlib import Path

def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3)

def extract_question_from_conversation(conversation):
    """从对话中提取问题"""
    for turn in conversation:
        if turn['from'] == 'human':
            # 移除<image>标签
            question = turn['value'].replace('<image>\n', '').strip()
            return question
    return ""

def extract_answer_from_conversation(conversation):
    """从对话中提取答案"""
    for turn in conversation:
        if turn['from'] == 'gpt':
            return turn['value'].strip()
    return ""

def determine_answer_type(question):
    """根据问题内容判断答案类型"""
    question_lower = question.lower()
    
    # 开放性问题的关键词
    open_keywords = ['what', 'where', 'how', 'which', 'describe', 'explain']
    
    # 封闭性问题的关键词
    closed_keywords = ['is', 'are', 'does', 'do', 'can', 'will', 'was', 'were']
    
    # 检查是否以开放性关键词开头
    for keyword in open_keywords:
        if question_lower.startswith(keyword):
            return "OPEN"
    
    # 检查是否以封闭性关键词开头
    for keyword in closed_keywords:
        if question_lower.startswith(keyword):
            return "CLOSED"
    
    # 默认为封闭性问题
    return "CLOSED"

def main():
    print("开始修复VQA-RAD数据集...")
    
    # 文件路径
    slake_file = '/workspace/MMedPO/data/vqa-rad_slake_format.json'
    dpo_file = '/workspace/MMedPO/data/vqa-rad_dpo_weighted.json'
    backup_file = '/workspace/MMedPO/data/vqa-rad_slake_format_backup.json'
    
    image_dir = '/workspace/MMedPO/datasets/VQA_RAD/VQA_RAD_Image_Folder'
    masked_dir = '/workspace/MMedPO/datasets/VQA_RAD/outputs/masked_images'
    
    # 备份原文件
    print("备份原始文件...")
    shutil.copy2(slake_file, backup_file)
    
    # 加载数据
    print("加载数据文件...")
    slake_data = load_json(slake_file)
    dpo_data = load_json(dpo_file)
    
    # 创建映射
    slake_by_qid = {item['qid']: item for item in slake_data}
    dpo_by_id = {item['id']: item for item in dpo_data}
    
    # 获取现有的qid和id
    slake_qids = set(item['qid'] for item in slake_data)
    dpo_ids = set(item['id'] for item in dpo_data)
    
    # 找出缺失的问题
    missing_in_slake = dpo_ids - slake_qids
    print(f"发现 {len(missing_in_slake)} 个缺失的问题需要添加到slake_format中")
    
    # 获取可用的图像文件
    available_images = set(os.listdir(image_dir))
    available_masked = set(os.listdir(masked_dir))
    
    # 添加缺失的问题
    new_items = []
    for missing_id in sorted(missing_in_slake):
        dpo_item = dpo_by_id[missing_id]
        
        # 提取问题和答案
        question = extract_question_from_conversation(dpo_item['conversations'])
        positive_answer = extract_answer_from_conversation(dpo_item['conversations'])
        negative_answer = extract_answer_from_conversation(dpo_item['rejected_conversations'])
        
        # 确定图像路径
        image_name = dpo_item['image']
        full_image_path = f"/workspace/MMedPO/datasets/VQA_RAD/VQA_RAD_Image_Folder/{image_name}"
        masked_image_path = f"/workspace/MMedPO/datasets/VQA_RAD/outputs/masked_images/{image_name}"
        
        # 检查图像文件是否存在
        has_positive = bool(positive_answer.strip())
        has_negative = bool(negative_answer.strip())
        
        # 如果masked图像不存在，尝试复制原图
        if image_name not in available_masked and image_name in available_images:
            print(f"警告: {image_name} 的masked图像不存在，将使用原图")
            # 这里可以选择复制原图或者标记为缺失
        
        # 创建新的数据项
        new_item = {
            "qid": missing_id,
            "question": question,
            "answer_type": determine_answer_type(question),
            "full_image_path": full_image_path,
            "masked_image_path": masked_image_path,
            "weighted_score": dpo_item.get('weighted_score', 1.0),
            "positive_answer": positive_answer,
            "negative_answer": negative_answer,
            "has_positive": has_positive,
            "has_negative": has_negative
        }
        
        new_items.append(new_item)
    
    # 合并数据
    print(f"添加 {len(new_items)} 个新问题...")
    all_items = slake_data + new_items
    
    # 按qid排序
    all_items.sort(key=lambda x: x['qid'])
    
    # 验证数据完整性
    print("验证数据完整性...")
    final_qids = set(item['qid'] for item in all_items)
    max_qid = max(final_qids)
    expected_qids = set(range(max_qid + 1))
    still_missing = expected_qids - final_qids
    
    if still_missing:
        print(f"警告: 仍有 {len(still_missing)} 个qid缺失: {sorted(still_missing)[:10]}...")
    else:
        print("✓ 所有qid都已包含")
    
    # 检查图像文件对应关系
    print("检查图像文件对应关系...")
    missing_images = []
    missing_masked = []
    
    for item in all_items:
        image_name = os.path.basename(item['full_image_path'])
        masked_name = os.path.basename(item['masked_image_path'])
        
        if image_name not in available_images:
            missing_images.append(image_name)
        if masked_name not in available_masked:
            missing_masked.append(masked_name)
    
    if missing_images:
        print(f"警告: {len(set(missing_images))} 个原图文件缺失")
    if missing_masked:
        print(f"警告: {len(set(missing_masked))} 个masked图文件缺失")
    
    # 保存修复后的数据
    print("保存修复后的数据...")
    save_json(all_items, slake_file)
    
    print("=== 修复完成 ===")
    print(f"原始数据: {len(slake_data)} 条")
    print(f"修复后数据: {len(all_items)} 条")
    print(f"新增数据: {len(new_items)} 条")
    print(f"备份文件: {backup_file}")
    
    # 生成统计报告
    print("\n=== 统计报告 ===")
    answer_types = {}
    weighted_scores = []
    
    for item in all_items:
        answer_type = item.get('answer_type', 'UNKNOWN')
        answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
        weighted_scores.append(item.get('weighted_score', 1.0))
    
    print(f"答案类型分布: {answer_types}")
    print(f"权重分数范围: {min(weighted_scores):.2f} - {max(weighted_scores):.2f}")
    print(f"平均权重分数: {sum(weighted_scores)/len(weighted_scores):.2f}")

if __name__ == "__main__":
    main()