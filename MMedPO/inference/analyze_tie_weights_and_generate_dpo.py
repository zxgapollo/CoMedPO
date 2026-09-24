#!/usr/bin/env python3
"""
分析TIE结果中的tie_weight分布并生成符合DPO训练要求的数据集
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse

def load_dpo_pairs(file_path: str) -> List[Dict[str, Any]]:
    """加载DPO pairs数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_tie_weight_distribution(dpo_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析tie_weight分布"""
    tie_weights = [pair['tie_weight'] for pair in dpo_pairs]
    
    stats = {
        'count': len(tie_weights),
        'min': min(tie_weights),
        'max': max(tie_weights),
        'mean': np.mean(tie_weights),
        'median': np.median(tie_weights),
        'std': np.std(tie_weights),
        'percentiles': {
            '25': np.percentile(tie_weights, 25),
            '75': np.percentile(tie_weights, 75),
            '90': np.percentile(tie_weights, 90),
            '95': np.percentile(tie_weights, 95),
            '99': np.percentile(tie_weights, 99)
        }
    }
    
    return stats, tie_weights

def normalize_tie_weights(tie_weights: List[float], method: str = 'min_max') -> List[float]:
    """归一化tie_weight"""
    tie_weights = np.array(tie_weights)
    
    if method == 'min_max':
        # Min-Max归一化到[0, 1]
        min_val = np.min(tie_weights)
        max_val = np.max(tie_weights)
        if max_val == min_val:
            return [0.5] * len(tie_weights)  # 如果所有值相同，返回0.5
        normalized = (tie_weights - min_val) / (max_val - min_val)
    elif method == 'z_score':
        # Z-score标准化
        mean_val = np.mean(tie_weights)
        std_val = np.std(tie_weights)
        if std_val == 0:
            return [0.5] * len(tie_weights)
        normalized = (tie_weights - mean_val) / std_val
        # 将z-score转换到[0, 1]范围
        normalized = 1 / (1 + np.exp(-normalized))  # sigmoid函数
    elif method == 'robust':
        # 基于分位数的鲁棒归一化
        q25 = np.percentile(tie_weights, 25)
        q75 = np.percentile(tie_weights, 75)
        if q75 == q25:
            return [0.5] * len(tie_weights)
        normalized = (tie_weights - q25) / (q75 - q25)
        normalized = np.clip(normalized, 0, 1)  # 限制在[0, 1]范围内
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.tolist()

def check_image_exists(case_id: str, composites_dir: str) -> bool:
    """检查对应的图像文件是否存在"""
    image_path = os.path.join(composites_dir, case_id, "source_mask_plus_full.jpg")
    return os.path.exists(image_path)

def convert_to_dpo_format(dpo_pairs: List[Dict[str, Any]], 
                         normalized_weights: List[float],
                         composites_dir: str) -> List[Dict[str, Any]]:
    """将TIE结果转换为DPO格式"""
    dpo_dataset = []
    
    for i, (pair, weight) in enumerate(zip(dpo_pairs, normalized_weights)):
        case_id = pair['case_id']
        
        # 检查图像文件是否存在
        if not check_image_exists(case_id, composites_dir):
            print(f"Warning: Image not found for case_id {case_id}, skipping...")
            continue
        
        # 构建图像路径
        image_path = f"{case_id}/source_mask_plus_full.jpg"
        
        # 构建DPO格式的数据
        dpo_entry = {
            "id": i,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{pair['question']}"
                },
                {
                    "from": "gpt",
                    "value": pair['chosen']
                }
            ],
            "rejected_conversations": [
                {
                    "from": "human", 
                    "value": f"<image>\n{pair['question']}"
                },
                {
                    "from": "gpt",
                    "value": pair['rejected']
                }
            ],
            "weighted_score": round(weight, 4)
        }
        
        dpo_dataset.append(dpo_entry)
    
    return dpo_dataset

def save_dpo_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """保存DPO数据集为JSONL格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate DPO dataset from TIE results')
    parser.add_argument('--dpo_pairs_file', 
                       default='/workspace/MMedPO/outputs/dpo_pairs_from_tie_with_gt_fallback/dpo_pairs_from_tie_results_with_gt_fallback.json',
                       help='Path to DPO pairs JSON file')
    parser.add_argument('--composites_dir',
                       default='/workspace/MMedPO/outputs/tie_results_1/composites',
                       help='Path to composites directory containing images')
    parser.add_argument('--output_file',
                       default='/workspace/MMedPO/outputs/tie_dpo_dataset.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--normalization_method',
                       choices=['min_max', 'z_score', 'robust'],
                       default='min_max',
                       help='Normalization method for tie_weights')
    
    args = parser.parse_args()
    
    print("Loading DPO pairs...")
    dpo_pairs = load_dpo_pairs(args.dpo_pairs_file)
    print(f"Loaded {len(dpo_pairs)} DPO pairs")
    
    print("Analyzing tie_weight distribution...")
    stats, tie_weights = analyze_tie_weight_distribution(dpo_pairs)
    
    print("\nTie Weight Distribution Statistics:")
    print(f"Count: {stats['count']}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print("Percentiles:")
    for p, v in stats['percentiles'].items():
        print(f"  {p}%: {v:.4f}")
    
    print(f"\nNormalizing tie_weights using {args.normalization_method} method...")
    normalized_weights = normalize_tie_weights(tie_weights, args.normalization_method)
    
    print("Converting to DPO format...")
    dpo_dataset = convert_to_dpo_format(dpo_pairs, normalized_weights, args.composites_dir)
    print(f"Generated {len(dpo_dataset)} DPO entries")
    
    print(f"Saving dataset to {args.output_file}...")
    save_dpo_dataset(dpo_dataset, args.output_file)
    
    print("\nDataset generation completed!")
    print(f"Final dataset contains {len(dpo_dataset)} entries")
    print(f"Normalized weight range: [{min(normalized_weights):.4f}, {max(normalized_weights):.4f}]")
    
    # 保存统计信息
    stats_file = args.output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'original_stats': stats,
            'normalization_method': args.normalization_method,
            'final_dataset_size': len(dpo_dataset),
            'normalized_weight_stats': {
                'min': min(normalized_weights),
                'max': max(normalized_weights),
                'mean': np.mean(normalized_weights),
                'std': np.std(normalized_weights)
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()