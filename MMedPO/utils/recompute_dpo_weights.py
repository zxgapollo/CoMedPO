#!/usr/bin/env python3
"""
重新计算DPO权重并生成新的DPO训练数据集
使用用户提供的compute_dpo_weight函数
"""

import json
import numpy as np
from scipy.special import expit as sigmoid
import os
from pathlib import Path
import argparse

def compute_dpo_weight(
    delta_pos, delta_neg, m_v, m_n,
    delta_obj=None, sigma_gamma=None,
    w_gamma=1.0, w_v=0.5, w_n=0.8, w_s=0.3, w_o=0.5,
    beta=2.0, tau=0.0, eps=1e-2
):
    """
    Compute DPO weight from TIE-related metrics.

    Args:
        delta_pos: float, Δ+ (TIE for positive answer)
        delta_neg: float, Δ- (TIE for negative answer)
        m_v: float, foreground contribution
        m_n: float, background leakage
        delta_obj: float, optional object-only gain
        sigma_gamma: float, optional std consistency penalty
        w_*: weights for different terms
        beta: sharpness parameter for sigmoid
        tau: threshold (bias shift)
        eps: clipping boundary for stability

    Returns:
        weight in [0,1], higher = stronger preference confidence
    """

    # core relative effect
    gamma = delta_pos - delta_neg

    # collect features
    feats = [gamma, m_v, m_n]
    if delta_obj is not None: feats.append(delta_obj)
    if sigma_gamma is not None: feats.append(sigma_gamma)

    # per-case z-score normalization
    feats = np.array(feats, dtype=np.float32)
    mean, std = feats.mean(), feats.std() + 1e-6
    normed = (feats - mean) / std

    # unpack normalized
    gamma_t = normed[0]
    m_v_t   = normed[1]
    m_n_t   = normed[2]
    delta_obj_t = normed[3] if delta_obj is not None else 0.0
    sigma_t     = normed[-1] if sigma_gamma is not None else 0.0

    # linear score
    S = (w_gamma * gamma_t
         + w_v * m_v_t
         - w_n * m_n_t
         - w_s * sigma_t
         + w_o * delta_obj_t)

    # sigmoid scaling + clipping
    w = sigmoid(beta * (S - tau))
    w = np.clip(w, eps, 1 - eps)

    return float(w)

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

def recompute_weights_and_convert_to_dpo(tie_data, composites_dir, output_file):
    """重新计算权重并转换为DPO格式"""
    
    dpo_entries = []
    weight_stats = []
    missing_images = 0
    
    print("Recomputing DPO weights and converting to DPO format...")
    
    for i, item in enumerate(tie_data):
        try:
            # 提取TIE相关指标
            delta_pos = item.get('delta_pos', 0.0)
            delta_neg = item.get('delta_neg', 0.0)
            m_v = item.get('m_v', 0.0)
            m_n = item.get('m_n', 0.0)
            delta_obj = item.get('delta_obj', None)
            
            # 计算新的DPO权重
            new_weight = compute_dpo_weight(
                delta_pos=delta_pos,
                delta_neg=delta_neg,
                m_v=m_v,
                m_n=m_n,
                delta_obj=delta_obj
            )
            
            # 四舍五入到两位小数
            new_weight = round(new_weight, 2)
            weight_stats.append(new_weight)
            
            # 检查图像文件是否存在
            case_id = item['case_id']
            image_path = f"{case_id}/source_mask_plus_full.jpg"
            full_image_path = os.path.join(composites_dir, image_path)
            
            if not os.path.exists(full_image_path):
                missing_images += 1
                continue
            
            # 构建DPO格式的数据
            dpo_entry = {
                "id": item['id'],
                "image": image_path,
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
                "weighted_score": new_weight
            }
            
            dpo_entries.append(dpo_entry)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(tie_data)} entries...")
                
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue
    
    # 保存DPO数据集
    print(f"Saving {len(dpo_entries)} DPO entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dpo_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 计算权重统计信息
    if weight_stats:
        stats = {
            "total_samples": len(dpo_entries),
            "missing_images": missing_images,
            "weighted_score_stats": {
                "min": float(np.min(weight_stats)),
                "max": float(np.max(weight_stats)),
                "mean": float(np.mean(weight_stats)),
                "median": float(np.median(weight_stats)),
                "std": float(np.std(weight_stats)),
                "samples_gt_0.1": int(np.sum(np.array(weight_stats) > 0.1)),
                "samples_gt_0.5": int(np.sum(np.array(weight_stats) > 0.5)),
                "samples_gt_0.8": int(np.sum(np.array(weight_stats) > 0.8)),
                "samples_eq_1.0": int(np.sum(np.array(weight_stats) == 1.0))
            }
        }
        
        # 保存统计信息
        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Missing images: {stats['missing_images']}")
        print(f"Weighted score range: [{stats['weighted_score_stats']['min']:.4f}, {stats['weighted_score_stats']['max']:.4f}]")
        print(f"Mean: {stats['weighted_score_stats']['mean']:.4f}")
        print(f"Std: {stats['weighted_score_stats']['std']:.4f}")
        print(f"Samples with weight > 0.1: {stats['weighted_score_stats']['samples_gt_0.1']}")
        print(f"Samples with weight > 0.5: {stats['weighted_score_stats']['samples_gt_0.5']}")
        print(f"Samples with weight > 0.8: {stats['weighted_score_stats']['samples_gt_0.8']}")
        print(f"Samples with weight = 1.0: {stats['weighted_score_stats']['samples_eq_1.0']}")
        print(f"Statistics saved to: {stats_file}")
    
    return dpo_entries, stats

def main():
    parser = argparse.ArgumentParser(description='Recompute DPO weights using new method')
    parser.add_argument('--tie_results', default='/workspace/MMedPO/outputs/tie_results_1/tie_results.json',
                       help='Path to TIE results JSON file')
    parser.add_argument('--composites_dir', default='/workspace/MMedPO/outputs/tie_results_1/composites',
                       help='Path to composites directory')
    parser.add_argument('--output', default='/workspace/MMedPO/outputs/tie_dpo_dataset_recomputed.jsonl',
                       help='Output DPO dataset file')
    
    args = parser.parse_args()
    
    # 加载TIE结果数据
    tie_data = load_tie_results(args.tie_results)
    
    # 重新计算权重并转换为DPO格式
    dpo_entries, stats = recompute_weights_and_convert_to_dpo(
        tie_data, args.composites_dir, args.output
    )
    
    print(f"\nSuccessfully generated DPO dataset: {args.output}")
    print(f"Total entries: {len(dpo_entries)}")

if __name__ == "__main__":
    main()