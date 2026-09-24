#!/usr/bin/env python3
"""
调试DPO pairs构建过程，分析为什么没有生成pairs
"""

import json
import os
import sys
from typing import Dict, List, Any
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_results() -> List[Dict[str, Any]]:
    """创建测试用的结果数据"""
    return [
        {
            "id": "test_1",
            "case_id": "case_001",
            "question": "What is shown in this medical image?",
            "positive_answer": "This shows a normal chest X-ray with clear lung fields.",
            "negative_answer": "This shows pneumonia with consolidation in the right lung.",
            "calculate_tie": True,
            "tie_positive": 1.25,
            "tie_negative": 0.35,
            "tie_pos_token_avg": 1.20,
            "tie_neg_token_avg": 0.30,
            "delta_pos": 1.25,
            "delta_neg": 0.35,
            "m_v": 1.10,
            "m_n": 0.05,
            "gamma": 0.90
        },
        {
            "id": "test_2", 
            "case_id": "case_002",
            "question": "Describe the pathological findings.",
            "positive_answer": "The image shows clear evidence of fracture in the femur.",
            "negative_answer": "The image appears normal with no abnormalities.",
            "calculate_tie": True,
            "tie_positive": 0.85,
            "tie_negative": -0.15,
            "tie_pos_token_avg": 0.80,
            "tie_neg_token_avg": -0.20,
            "delta_pos": 0.85,
            "delta_neg": -0.15,
            "m_v": 0.95,
            "m_n": 0.02,
            "gamma": 1.00
        },
        {
            "id": "test_3",
            "case_id": "case_003", 
            "question": "What is the diagnosis?",
            "positive_answer": "The diagnosis is acute myocardial infarction.",
            "negative_answer": "The diagnosis is normal cardiac function.",
            "calculate_tie": True,
            "tie_positive": 0.05,
            "tie_negative": 0.45,
            "tie_pos_token_avg": 0.02,
            "tie_neg_token_avg": 0.40,
            "delta_pos": 0.05,
            "delta_neg": 0.45,
            "m_v": 0.25,
            "m_n": 0.85,
            "gamma": -0.40
        }
    ]

def create_test_args():
    """创建测试用的参数"""
    class TestArgs:
        def __init__(self):
            # TIE-ANKER参数
            self.enable_tie_anker = True
            self.output_pairs_file = "/workspace/MMedPO/inference/debug_dpo_pairs.json"
            
            # TIE-ANKER权重
            self.w_gamma = 1.0
            self.w_v = 0.5
            self.w_n = 0.8
            self.w_s = 0.3
            self.w_o = 0.5
            
            # Sigmoid映射参数
            self.beta = 2.0
            self.tau = 0.0
            self.epsilon = 0.02
            
            # 评分和阈值
            self.token_avg = "true"
            self.use_per_case_zscore = "true"
            self.z_eps = 1e-6
            self.tau_pos = 0.1
            self.tau_gamma_strong = 0.5
            self.tau_gamma_weak = 0.1
            self.tau_v = 0.5
            self.tau_n_percentile = 75
            
            # 权重映射
            self.w_min = 0.05
    
    return TestArgs()

def debug_tie_anker_thresholds():
    """调试TIE-ANKER阈值过滤过程"""
    print("=== 调试TIE-ANKER阈值过滤 ===")
    
    # 导入函数
    from inference_dpo_tie_comparison import apply_tie_anker_thresholds, calculate_tie_anker_weight
    
    test_results = create_test_results()
    test_args = create_test_args()
    
    # 计算TIE分数统计
    token_avg = test_args.token_avg.lower() == "true"
    use_per_case_zscore = test_args.use_per_case_zscore.lower() == "true"
    
    tie_scores = []
    for result in test_results:
        if result.get('calculate_tie', False):
            tie_pos = result.get('tie_pos_token_avg', 0) if token_avg else result.get('tie_positive', 0)
            tie_neg = result.get('tie_neg_token_avg', 0) if token_avg else result.get('tie_negative', 0)
            tie_diff = tie_pos - tie_neg
            tie_scores.append(tie_diff)
    
    if tie_scores:
        tie_mean = np.mean(tie_scores)
        tie_std = np.std(tie_scores) + test_args.z_eps
    else:
        tie_mean = 0.0
        tie_std = 1.0
    
    print(f"TIE分数统计: mean={tie_mean:.4f}, std={tie_std:.4f}")
    
    # 逐个检查每个样本
    for i, result in enumerate(test_results):
        if not result.get('calculate_tie', False):
            continue
        
        print(f"\n--- 样本 {i+1}: {result['id']} ---")
        
        # 提取TIE分数
        tie_pos = result.get('tie_pos_token_avg', 0) if token_avg else result.get('tie_positive', 0)
        tie_neg = result.get('tie_neg_token_avg', 0) if token_avg else result.get('tie_negative', 0)
        tie_diff = tie_pos - tie_neg
        
        # Z-score标准化
        if use_per_case_zscore:
            tie_diff_norm = (tie_diff - tie_mean) / tie_std
        else:
            tie_diff_norm = tie_diff
        
        print(f"TIE分数: pos={tie_pos:.4f}, neg={tie_neg:.4f}, diff={tie_diff:.4f}, norm={tie_diff_norm:.4f}")
        
        # 计算权重
        tie_weight = calculate_tie_anker_weight(
            tie_diff_norm, 
            result.get('delta_pos', 0),
            result.get('delta_neg', 0), 
            result.get('m_v', 0),
            result.get('m_n', 0),
            result.get('gamma', 0),
            test_args
        )
        
        print(f"TIE权重: {tie_weight:.4f}")
        
        # 检查各个阈值条件
        print("阈值检查:")
        
        # 1. 正向阈值
        tau_pos_pass = tie_diff_norm >= test_args.tau_pos
        print(f"  1. tie_diff_norm ({tie_diff_norm:.4f}) >= tau_pos ({test_args.tau_pos}): {'✅' if tau_pos_pass else '❌'}")
        
        # 2. Gamma阈值
        gamma = result.get('gamma', 0)
        gamma_pass = test_args.tau_gamma_weak <= gamma <= test_args.tau_gamma_strong
        print(f"  2. tau_gamma_weak ({test_args.tau_gamma_weak}) <= gamma ({gamma:.4f}) <= tau_gamma_strong ({test_args.tau_gamma_strong}): {'✅' if gamma_pass else '❌'}")
        
        # 3. V阈值
        m_v = result.get('m_v', 0)
        v_pass = m_v >= test_args.tau_v
        print(f"  3. m_v ({m_v:.4f}) >= tau_v ({test_args.tau_v}): {'✅' if v_pass else '❌'}")
        
        # 4. N百分位阈值
        m_n = result.get('m_n', 0)
        n_pass = m_n <= test_args.tau_n_percentile / 100.0
        print(f"  4. m_n ({m_n:.4f}) <= tau_n_percentile/100 ({test_args.tau_n_percentile/100.0:.4f}): {'✅' if n_pass else '❌'}")
        
        # 应用阈值过滤
        threshold_pass = apply_tie_anker_thresholds(tie_weight, tie_diff_norm, result, test_args)
        print(f"总体阈值通过: {'✅' if threshold_pass else '❌'}")
        
        if not threshold_pass:
            print("❌ 该样本被过滤掉")
        else:
            print("✅ 该样本通过过滤")

def debug_improved_vs_original():
    """对比改进版本和原始版本的差异"""
    print("\n=== 对比改进版本和原始版本 ===")
    
    test_results = create_test_results()
    test_args = create_test_args()
    
    # 原始版本
    from inference_dpo_tie_comparison import build_tie_anker_dpo_pairs
    original_pairs = build_tie_anker_dpo_pairs(test_results, test_args)
    
    # 改进版本
    from improved_dpo_pairs_logic import build_improved_tie_anker_dpo_pairs
    improved_pairs = build_improved_tie_anker_dpo_pairs(test_results, test_args)
    
    print(f"原始版本生成的pairs: {len(original_pairs)}")
    print(f"改进版本生成的pairs: {len(improved_pairs)}")
    
    if len(original_pairs) == 0:
        print("❌ 原始版本没有生成任何pairs - 这证实了我们发现的问题")
    
    if len(improved_pairs) > 0:
        print("✅ 改进版本成功生成了pairs")
        print("\n改进版本生成的第一个pair:")
        print(json.dumps(improved_pairs[0], ensure_ascii=False, indent=2))

def main():
    """主函数"""
    print("🔍 开始调试DPO pairs构建过程...")
    
    debug_tie_anker_thresholds()
    debug_improved_vs_original()
    
    print("\n🎯 调试完成!")

if __name__ == "__main__":
    main()