#!/usr/bin/env python3
"""
改进的DPO权重计算脚本
主要改进：
1. 使用全局标准化而非per-case标准化
2. 调整权重参数以获得更合理的分布
3. 添加更多的调试信息
"""

import json
import numpy as np
import os
from pathlib import Path
import argparse

def sigmoid(x):
    """Sigmoid function with clipping for numerical stability"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_dpo_weight_improved(
    delta_pos, delta_neg, m_v, m_n,
    delta_obj=None, sigma_gamma=None,
    # 调整权重参数
    w_gamma=1.0, w_v=0.3, w_n=0.5, w_s=0.2, w_o=0.3,
    # 调整sigmoid参数
    beta=1.0, tau=0.0, eps=1e-2,
    # 全局标准化参数
    global_stats=None
):
    """
    改进的DPO权重计算函数，使用全局标准化
    
    Args:
        delta_pos: float, Δ+ (TIE for positive answer)
        delta_neg: float, Δ- (TIE for negative answer)
        m_v: float, foreground contribution
        m_n: float, background leakage
        delta_obj: float, optional object-only gain
        sigma_gamma: float, optional std consistency penalty
        w_*: weights for different terms
        beta: sharpness parameter for sigmoid (降低以获得更平滑的分布)
        tau: threshold (bias shift)
        eps: clipping boundary for stability
        global_stats: dict, 全局统计信息用于标准化
    
    Returns:
        weight in [0,1], higher = stronger preference confidence
    """
    
    # core relative effect
    gamma = delta_pos - delta_neg
    
    # 使用全局标准化
    if global_stats is not None:
        gamma_norm = (gamma - global_stats['gamma_mean']) / (global_stats['gamma_std'] + 1e-6)
        m_v_norm = (m_v - global_stats['m_v_mean']) / (global_stats['m_v_std'] + 1e-6)
        m_n_norm = (m_n - global_stats['m_n_mean']) / (global_stats['m_n_std'] + 1e-6)
        delta_obj_norm = (delta_obj - global_stats['delta_obj_mean']) / (global_stats['delta_obj_std'] + 1e-6) if delta_obj is not None else 0.0
    else:
        # 如果没有全局统计信息，使用原始值
        gamma_norm = gamma
        m_v_norm = m_v
        m_n_norm = m_n
        delta_obj_norm = delta_obj if delta_obj is not None else 0.0
    
    # linear score
    S = (w_gamma * gamma_norm
         + w_v * m_v_norm
         - w_n * abs(m_n_norm)  # 使用绝对值，因为m_n通常是负值
         + w_o * delta_obj_norm)
    
    # sigmoid scaling + clipping
    w = sigmoid(beta * (S - tau))
    w = np.clip(w, eps, 1 - eps)
    
    return float(w)

def compute_global_stats(data):
    """计算全局统计信息"""
    print("计算全局统计信息...")
    
    delta_pos_list = [item['delta_pos'] for item in data]
    delta_neg_list = [item['delta_neg'] for item in data]
    m_v_list = [item['m_v'] for item in data]
    m_n_list = [item['m_n'] for item in data]
    delta_obj_list = [item['delta_obj'] for item in data]
    
    gamma_list = [dp - dn for dp, dn in zip(delta_pos_list, delta_neg_list)]
    
    stats = {
        'gamma_mean': np.mean(gamma_list),
        'gamma_std': np.std(gamma_list),
        'm_v_mean': np.mean(m_v_list),
        'm_v_std': np.std(m_v_list),
        'm_n_mean': np.mean(m_n_list),
        'm_n_std': np.std(m_n_list),
        'delta_obj_mean': np.mean(delta_obj_list),
        'delta_obj_std': np.std(delta_obj_list),
    }
    
    print("全局统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

def load_tie_results(file_path):
    """加载TIE结果数据"""
    print(f"Loading TIE results from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} TIE result entries")
    return data

def convert_to_dpo_format(tie_data, global_stats, output_dir):
    """将TIE结果转换为DPO格式"""
    print("Converting TIE results to DPO format...")
    
    dpo_data = []
    weights = []
    
    for item in tie_data:
        # 计算改进的DPO权重
        weight = compute_dpo_weight_improved(
            item['delta_pos'], item['delta_neg'],
            item['m_v'], item['m_n'], item['delta_obj'],
            global_stats=global_stats
        )
        
        # 四舍五入到两位小数
        weight = round(weight, 2)
        weights.append(weight)
        
        # 构建DPO格式的数据
        dpo_entry = {
            "id": item['id'],
            "image": f"{item['case_id']}/source_mask_plus_full.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{item['question']}"
                },
                {
                    "from": "gpt",
                    "value": item['positive_answer']
                }
            ],
            "rejected_conversations": [
                {
                    "from": "human", 
                    "value": f"<image>\n{item['question']}"
                },
                {
                    "from": "gpt",
                    "value": item['negative_answer']
                }
            ],
            "weighted_score": weight
        }
        
        dpo_data.append(dpo_entry)
    
    # 保存DPO数据集
    output_file = os.path.join(output_dir, "tie_dpo_dataset_improved.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(dpo_data)} DPO entries to {output_file}")
    
    # 计算并保存统计信息
    weights = np.array(weights)
    stats = {
        "total_samples": len(weights),
        "weighted_score_range": [float(weights.min()), float(weights.max())],
        "weighted_score_mean": float(weights.mean()),
        "weighted_score_median": float(np.median(weights)),
        "weighted_score_std": float(weights.std()),
        "samples_gt_0.1": int(np.sum(weights > 0.1)),
        "samples_gt_0.3": int(np.sum(weights > 0.3)),
        "samples_gt_0.5": int(np.sum(weights > 0.5)),
        "samples_gt_0.7": int(np.sum(weights > 0.7)),
        "samples_gt_0.9": int(np.sum(weights > 0.9)),
    }
    
    stats_file = os.path.join(output_dir, "tie_dpo_dataset_improved_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved statistics to {stats_file}")
    print("\n=== 改进后的权重分布统计 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"权重范围: [{stats['weighted_score_range'][0]:.4f}, {stats['weighted_score_range'][1]:.4f}]")
    print(f"平均权重: {stats['weighted_score_mean']:.4f}")
    print(f"中位数权重: {stats['weighted_score_median']:.4f}")
    print(f"权重标准差: {stats['weighted_score_std']:.4f}")
    print(f"权重 > 0.1 的样本数: {stats['samples_gt_0.1']}")
    print(f"权重 > 0.3 的样本数: {stats['samples_gt_0.3']}")
    print(f"权重 > 0.5 的样本数: {stats['samples_gt_0.5']}")
    print(f"权重 > 0.7 的样本数: {stats['samples_gt_0.7']}")
    print(f"权重 > 0.9 的样本数: {stats['samples_gt_0.9']}")
    
    return output_file, stats

def main():
    parser = argparse.ArgumentParser(description='改进的DPO权重计算和数据集生成')
    parser.add_argument('--tie_results', type=str, 
                       default='/workspace/MMedPO/outputs/tie_results_1/tie_results.json',
                       help='TIE结果文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/MMedPO/outputs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载TIE结果
    tie_data = load_tie_results(args.tie_results)
    
    # 计算全局统计信息
    global_stats = compute_global_stats(tie_data)
    
    # 转换为DPO格式
    output_file, stats = convert_to_dpo_format(tie_data, global_stats, args.output_dir)
    
    print(f"\n✅ 改进的DPO数据集生成完成!")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 统计文件: {os.path.join(args.output_dir, 'tie_dpo_dataset_improved_stats.json')}")

if __name__ == "__main__":
    main()