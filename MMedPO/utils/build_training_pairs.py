#!/usr/bin/env python3
"""
基于因果分析框架构建医疗影像DPO训练pairs
实现三种策略：正例选择、权重设计、锚点设计
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import argparse
from pathlib import Path

def load_data(file_path: str) -> List[Dict]:
    """加载JSONL数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_tie_metrics(data: List[Dict]) -> Dict:
    """计算TIE相关统计指标"""
    tie_differences = [item['tie_difference'] for item in data]
    tie_positives = [item['tie_positive'] for item in data]
    tie_negatives = [item['tie_negative'] for item in data]
    
    stats = {
        'tie_diff_mean': np.mean(tie_differences),
        'tie_diff_std': np.std(tie_differences),
        'tie_diff_median': np.median(tie_differences),
        'tie_pos_mean': np.mean(tie_positives),
        'tie_neg_mean': np.mean(tie_negatives),
        'tie_diff_min': np.min(tie_differences),
        'tie_diff_max': np.max(tie_differences)
    }
    
    print(f"TIE统计信息:")
    print(f"  TIE差异均值: {stats['tie_diff_mean']:.4f}")
    print(f"  TIE差异标准差: {stats['tie_diff_std']:.4f}")
    print(f"  TIE差异中位数: {stats['tie_diff_median']:.4f}")
    print(f"  TIE差异范围: [{stats['tie_diff_min']:.4f}, {stats['tie_diff_max']:.4f}]")
    
    return stats

def strategy_1_positive_selection(data: List[Dict], tie_threshold: float = -0.5) -> List[Dict]:
    """
    策略1: 正例选择
    选择TIE为负且在ground truth下表现良好的样本对
    
    Args:
        data: 原始数据
        tie_threshold: TIE阈值，选择小于此值的样本
    
    Returns:
        符合条件的训练pairs
    """
    selected_pairs = []
    
    for item in data:
        tie_diff = item['tie_difference']
        weighted_score = item['weighted_score']
        
        # 条件1: TIE差异为负（背景影响小于前景影响）
        # 条件2: weighted_score较高（在ground truth下表现好）
        if tie_diff < tie_threshold and weighted_score >= 0.8:
            pair = {
                'id': item['id'],
                'case_id': item['case_id'],
                'question': item['question'],
                'prompt': item['prompt_text'],
                'chosen': item['positive_answer'],  # 正确答案作为preferred
                'rejected': item['negative_answer'],  # 错误答案作为dispreferred
                'tie_difference': tie_diff,
                'weighted_score': weighted_score,
                'strategy': 'positive_selection',
                'selection_reason': f'TIE={tie_diff:.3f} < {tie_threshold}, score={weighted_score:.3f} >= 0.8'
            }
            selected_pairs.append(pair)
    
    print(f"策略1 - 正例选择: 选择了 {len(selected_pairs)} 个训练pairs")
    return selected_pairs

def strategy_2_weight_design(data: List[Dict], alpha: float = 2.0) -> List[Dict]:
    """
    策略2: 权重设计
    基于TIE差异计算样本对的训练权重
    
    Args:
        data: 原始数据
        alpha: 权重计算参数
    
    Returns:
        带权重的训练pairs
    """
    weighted_pairs = []
    
    # 计算所有TIE差异的绝对值，用于归一化
    tie_diffs = [abs(item['tie_difference']) for item in data]
    max_tie_diff = max(tie_diffs) if tie_diffs else 1.0
    
    for item in data:
        tie_diff = item['tie_difference']
        
        # 计算训练权重：TIE差异越大，权重越高
        weight = np.exp(alpha * abs(tie_diff) / max_tie_diff)
        
        pair = {
            'id': item['id'],
            'case_id': item['case_id'],
            'question': item['question'],
            'prompt': item['prompt_text'],
            'chosen': item['positive_answer'],
            'rejected': item['negative_answer'],
            'tie_difference': tie_diff,
            'training_weight': weight,
            'strategy': 'weight_design',
            'weight_formula': f'exp({alpha} * |{tie_diff:.3f}| / {max_tie_diff:.3f}) = {weight:.3f}'
        }
        weighted_pairs.append(pair)
    
    print(f"策略2 - 权重设计: 为 {len(weighted_pairs)} 个样本分配了训练权重")
    return weighted_pairs

def strategy_3_anchor_design(data: List[Dict], stats: Dict) -> Tuple[List[Dict], Dict]:
    """
    策略3: 锚点设计
    基于TIE统计分布设置preference和dispreference锚点
    
    Args:
        data: 原始数据
        stats: TIE统计信息
    
    Returns:
        锚点pairs和锚点信息
    """
    # 设置锚点阈值
    preference_threshold = stats['tie_diff_mean'] - stats['tie_diff_std']  # 低TIE差异
    dispreference_threshold = stats['tie_diff_mean'] + stats['tie_diff_std']  # 高TIE差异
    
    preference_anchors = []
    dispreference_anchors = []
    
    for item in data:
        tie_diff = item['tie_difference']
        
        if tie_diff <= preference_threshold:
            # 低TIE差异 -> preference anchor (前景主导)
            anchor = {
                'id': item['id'],
                'case_id': item['case_id'],
                'question': item['question'],
                'prompt': item['prompt_text'],
                'response': item['positive_answer'],
                'tie_difference': tie_diff,
                'anchor_type': 'preference',
                'anchor_reason': f'TIE={tie_diff:.3f} <= {preference_threshold:.3f} (前景主导)'
            }
            preference_anchors.append(anchor)
            
        elif tie_diff >= dispreference_threshold:
            # 高TIE差异 -> dispreference anchor (背景偏见)
            anchor = {
                'id': item['id'],
                'case_id': item['case_id'],
                'question': item['question'],
                'prompt': item['prompt_text'],
                'response': item['negative_answer'],
                'tie_difference': tie_diff,
                'anchor_type': 'dispreference',
                'anchor_reason': f'TIE={tie_diff:.3f} >= {dispreference_threshold:.3f} (背景偏见)'
            }
            dispreference_anchors.append(anchor)
    
    anchor_info = {
        'preference_threshold': preference_threshold,
        'dispreference_threshold': dispreference_threshold,
        'preference_count': len(preference_anchors),
        'dispreference_count': len(dispreference_anchors)
    }
    
    print(f"策略3 - 锚点设计:")
    print(f"  Preference锚点: {len(preference_anchors)} 个 (TIE <= {preference_threshold:.3f})")
    print(f"  Dispreference锚点: {len(dispreference_anchors)} 个 (TIE >= {dispreference_threshold:.3f})")
    
    return preference_anchors + dispreference_anchors, anchor_info

def save_pairs(pairs: List[Dict], output_path: str, strategy_name: str):
    """保存训练pairs到文件"""
    output_file = Path(output_path) / f"{strategy_name}_pairs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(pairs)} 个pairs到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='构建医疗影像DPO训练pairs')
    parser.add_argument('--input', type=str, 
                       default='/workspace/MMedPO/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl',
                       help='输入数据文件路径')
    parser.add_argument('--output', type=str, 
                       default='/workspace/MMedPO/training_pairs',
                       help='输出目录')
    parser.add_argument('--tie_threshold', type=float, default=-0.5,
                       help='策略1的TIE阈值')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='策略2的权重参数')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input}")
    data = load_data(args.input)
    print(f"总共加载 {len(data)} 条数据")
    
    # 计算TIE统计信息
    stats = calculate_tie_metrics(data)
    
    # 策略1: 正例选择
    print("\n=== 策略1: 正例选择 ===")
    positive_pairs = strategy_1_positive_selection(data, args.tie_threshold)
    save_pairs(positive_pairs, args.output, 'strategy1_positive_selection')
    
    # 策略2: 权重设计
    print("\n=== 策略2: 权重设计 ===")
    weighted_pairs = strategy_2_weight_design(data, args.alpha)
    save_pairs(weighted_pairs, args.output, 'strategy2_weight_design')
    
    # 策略3: 锚点设计
    print("\n=== 策略3: 锚点设计 ===")
    anchor_pairs, anchor_info = strategy_3_anchor_design(data, stats)
    save_pairs(anchor_pairs, args.output, 'strategy3_anchor_design')
    
    # 保存锚点信息
    anchor_info_file = Path(args.output) / 'anchor_info.json'
    with open(anchor_info_file, 'w', encoding='utf-8') as f:
        json.dump(anchor_info, f, ensure_ascii=False, indent=2)
    print(f"锚点信息已保存到: {anchor_info_file}")
    
    # 生成组合策略的示例
    print("\n=== 组合策略示例 ===")
    # 结合策略1和策略2：选择高质量样本并分配权重
    combined_pairs = []
    for pair in positive_pairs:
        # 为正例选择的样本分配权重
        tie_diff = pair['tie_difference']
        weight = np.exp(args.alpha * abs(tie_diff))
        pair['training_weight'] = weight
        pair['strategy'] = 'combined_positive_weighted'
        combined_pairs.append(pair)
    
    save_pairs(combined_pairs, args.output, 'combined_strategy')
    
    print("\n=== 构建完成 ===")
    print(f"所有训练pairs已保存到: {args.output}")

if __name__ == '__main__':
    main()