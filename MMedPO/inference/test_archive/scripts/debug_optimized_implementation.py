#!/usr/bin/env python3
"""
调试优化后的DPO pairs实现
"""

import sys
import os
sys.path.append('/workspace/MMedPO/inference')

from inference_dpo_tie_comparison import (
    build_tie_anker_dpo_pairs, 
    apply_tie_anker_thresholds,
    validate_preferred_answer,
    validate_dispreferred_answer,
    calculate_tie_anker_weight
)
import argparse
import json
import numpy as np

def create_test_args():
    """创建测试参数"""
    args = argparse.Namespace()
    
    # TIE-ANKER相关参数
    args.tau_pos = 0.1
    args.tau_gamma_weak = 0.2
    args.tau_gamma_strong = 0.8
    args.tau_v = 0.3
    args.tau_n_percentile = 70.0
    args.tau_n_leak = 0.4
    
    # 权重计算参数
    args.token_avg = True
    args.use_per_case_zscore = True
    args.z_eps = 1e-8
    
    # TIE-ANKER权重参数
    args.w_gamma = 1.0
    args.w_v = 1.0
    args.w_n = 1.0
    args.w_s = 1.0
    args.w_o = 1.0
    args.beta = 1.0
    args.tau = 0.5
    args.epsilon = 0.0
    args.w_min = 0.1
    
    # 其他参数
    args.enable_tie_anker = True
    args.output_pairs_file = "debug_optimized_pairs.json"
    
    return args

def create_test_data():
    """创建测试数据"""
    return [
        {
            'id': 'test_1',
            'case_id': 'case_001',
            'question': '患者症状分析',
            'positive_answer': '正确答案1',
            'negative_answer': '错误答案1',
            'calculate_tie': True,
            'tie_positive': 0.8,
            'tie_negative': 0.3,
            'tie_pos_token_avg': 0.8,
            'tie_neg_token_avg': 0.3,
            'delta_pos': 0.5,
            'delta_neg': 0.1,  # >= 0，符合反例条件
            'm_v': 0.4,
            'm_n': 0.5,  # > tau_n_leak，符合反例条件
            'gamma': 0.0,  # <= 0，符合反例条件
            'tie_difference': 0.5
        },
        {
            'id': 'test_2',
            'case_id': 'case_002',
            'question': '诊断建议',
            'positive_answer': '正确答案2',
            'negative_answer': '错误答案2',
            'calculate_tie': True,
            'tie_positive': 0.9,
            'tie_negative': 0.2,
            'tie_pos_token_avg': 0.9,
            'tie_neg_token_avg': 0.2,
            'delta_pos': 0.7,
            'delta_neg': -0.1,  # < 0，不符合反例条件
            'm_v': 0.5,
            'm_n': 0.3,  # < tau_n_leak，不符合反例条件
            'gamma': 1.0,  # > 0，不符合反例条件
            'tie_difference': 0.7
        },
        {
            'id': 'test_3',
            'case_id': 'case_003',
            'question': '治疗方案',
            'positive_answer': '正确答案3',
            'negative_answer': '错误答案3',
            'calculate_tie': True,
            'tie_positive': 0.7,
            'tie_negative': 0.4,
            'tie_pos_token_avg': 0.7,
            'tie_neg_token_avg': 0.4,
            'delta_pos': 0.3,
            'delta_neg': 0.2,  # >= 0，符合反例条件
            'm_v': 0.6,
            'm_n': 0.6,  # > tau_n_leak，符合反例条件
            'gamma': -0.1,  # <= 0，符合反例条件
            'tie_difference': 0.3
        }
    ]

def debug_validation_process():
    """调试验证过程"""
    print("=== 调试优化后的验证过程 ===\n")
    
    test_results = create_test_data()
    args = create_test_args()
    
    print(f"参数设置:")
    print(f"  tau_pos: {args.tau_pos}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong}")
    print(f"  tau_v: {args.tau_v}")
    print(f"  tau_n_percentile: {args.tau_n_percentile}")
    print(f"  tau_n_leak: {args.tau_n_leak}")
    print()
    
    # 计算批统计信息
    tie_scores = []
    for result in test_results:
        if result.get('calculate_tie', False):
            tie_pos = result.get('tie_pos_token_avg', 0)
            tie_neg = result.get('tie_neg_token_avg', 0)
            tie_diff = tie_pos - tie_neg
            tie_scores.append(tie_diff)
    
    tie_mean = np.mean(tie_scores)
    tie_std = np.std(tie_scores) + args.z_eps
    
    print(f"批统计信息:")
    print(f"  TIE scores: {tie_scores}")
    print(f"  TIE mean: {tie_mean:.4f}")
    print(f"  TIE std: {tie_std:.4f}")
    print()
    
    # 逐个分析每个样本
    for i, result in enumerate(test_results):
        print(f"--- 样本 {i+1}: {result['id']} ---")
        
        # 计算标准化分数
        tie_pos = result.get('tie_pos_token_avg', 0)
        tie_neg = result.get('tie_neg_token_avg', 0)
        tie_diff = tie_pos - tie_neg
        tie_diff_norm = (tie_diff - tie_mean) / tie_std
        
        print(f"TIE差异: {tie_diff:.4f} -> 标准化: {tie_diff_norm:.4f}")
        
        # 计算权重
        tie_weight = calculate_tie_anker_weight(
            tie_diff_norm,
            result.get('delta_pos', 0),
            result.get('delta_neg', 0),
            result.get('m_v', 0),
            result.get('m_n', 0),
            result.get('gamma', 0),
            args
        )
        print(f"TIE权重: {tie_weight:.4f}")
        
        # 验证正例和反例
        preferred_valid, preferred_violations = validate_preferred_answer(result, args)
        dispreferred_valid, dispreferred_conditions = validate_dispreferred_answer(result, args)
        
        print(f"正例验证: {preferred_valid}")
        if not preferred_valid:
            print(f"  违反条件: {preferred_violations}")
        
        print(f"反例验证: {dispreferred_valid}")
        print(f"  满足条件: {dispreferred_conditions}")
        
        # 应用阈值过滤
        threshold_passed = apply_tie_anker_thresholds(tie_weight, tie_diff_norm, result, args)
        print(f"阈值过滤: {threshold_passed}")
        
        print(f"最终结果: {'通过' if threshold_passed else '被过滤'}")
        print()

def test_with_adjusted_parameters():
    """使用调整后的参数测试"""
    print("=== 使用调整后的参数测试 ===\n")
    
    test_results = create_test_data()
    args = create_test_args()
    
    # 调整参数使其更容易通过
    args.tau_pos = 0.05  # 降低正例阈值
    args.tau_gamma_weak = 0.1  # 降低gamma弱阈值
    args.tau_gamma_strong = 1.5  # 提高gamma强阈值
    args.tau_v = 0.2  # 降低V阈值
    args.tau_n_percentile = 80.0  # 提高N百分位阈值
    
    print(f"调整后的参数:")
    print(f"  tau_pos: {args.tau_pos}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong}")
    print(f"  tau_v: {args.tau_v}")
    print(f"  tau_n_percentile: {args.tau_n_percentile}")
    print()
    
    # 构建DPO pairs
    dpo_pairs = build_tie_anker_dpo_pairs(test_results, args)
    
    print(f"生成的DPO pairs数量: {len(dpo_pairs)}")
    
    for i, pair in enumerate(dpo_pairs):
        print(f"\n--- DPO Pair {i+1} ---")
        print(f"ID: {pair['id']}")
        print(f"TIE Weight: {pair['tie_weight']:.4f}")
        print(f"TIE Score: {pair['tie_score']:.4f}")
        
        validation = pair.get('validation_info', {})
        print(f"验证信息:")
        print(f"  正例有效: {validation.get('preferred_valid', 'N/A')}")
        print(f"  反例有效: {validation.get('dispreferred_valid', 'N/A')}")
        print(f"  反例条件: {validation.get('dispreferred_reasons', [])}")

if __name__ == "__main__":
    debug_validation_process()
    test_with_adjusted_parameters()