#!/usr/bin/env python3
"""
测试优化后的DPO pairs实现
"""

import sys
import os
sys.path.append('/workspace/MMedPO/inference')

from inference_dpo_tie_comparison import build_tie_anker_dpo_pairs
import argparse
import json

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
    args.output_pairs_file = "test_optimized_pairs.json"
    
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

def test_optimized_implementation():
    """测试优化后的实现"""
    print("=== 测试优化后的DPO pairs实现 ===\n")
    
    # 创建测试数据和参数
    test_results = create_test_data()
    args = create_test_args()
    
    print(f"测试数据: {len(test_results)} 个样本")
    print(f"参数设置:")
    print(f"  tau_pos: {args.tau_pos}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong}")
    print(f"  tau_v: {args.tau_v}")
    print(f"  tau_n_percentile: {args.tau_n_percentile}")
    print(f"  tau_n_leak: {args.tau_n_leak}")
    print()
    
    # 构建DPO pairs
    try:
        dpo_pairs = build_tie_anker_dpo_pairs(test_results, args)
        
        print(f"生成的DPO pairs数量: {len(dpo_pairs)}")
        print()
        
        # 详细分析每个pair
        for i, pair in enumerate(dpo_pairs):
            print(f"--- DPO Pair {i+1} ---")
            print(f"ID: {pair['id']}")
            print(f"Case ID: {pair['case_id']}")
            print(f"TIE Weight: {pair['tie_weight']:.4f}")
            print(f"TIE Score: {pair['tie_score']:.4f}")
            
            # 验证信息
            validation = pair.get('validation_info', {})
            print(f"验证信息:")
            print(f"  正例有效: {validation.get('preferred_valid', 'N/A')}")
            print(f"  反例有效: {validation.get('dispreferred_valid', 'N/A')}")
            print(f"  反例条件: {validation.get('dispreferred_reasons', [])}")
            
            # 元数据
            metadata = pair.get('metadata', {})
            print(f"元数据:")
            print(f"  delta_pos: {metadata.get('delta_pos', 0):.4f}")
            print(f"  delta_neg: {metadata.get('delta_neg', 0):.4f}")
            print(f"  m_v: {metadata.get('m_v', 0):.4f}")
            print(f"  m_n: {metadata.get('m_n', 0):.4f}")
            print(f"  gamma: {metadata.get('gamma', 0):.4f}")
            print()
        
        # 保存结果
        if dpo_pairs:
            with open(args.output_pairs_file, 'w', encoding='utf-8') as f:
                json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.output_pairs_file}")
        
        return dpo_pairs
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_test_results(dpo_pairs):
    """分析测试结果"""
    print("\n=== 结果分析 ===")
    
    if not dpo_pairs:
        print("未生成任何DPO pairs")
        return
    
    print(f"总共生成 {len(dpo_pairs)} 个DPO pairs")
    
    # 统计验证信息
    preferred_valid_count = sum(1 for pair in dpo_pairs 
                               if pair.get('validation_info', {}).get('preferred_valid', False))
    dispreferred_valid_count = sum(1 for pair in dpo_pairs 
                                  if pair.get('validation_info', {}).get('dispreferred_valid', False))
    
    print(f"正例验证通过: {preferred_valid_count}/{len(dpo_pairs)}")
    print(f"反例验证通过: {dispreferred_valid_count}/{len(dpo_pairs)}")
    
    # 分析反例条件
    all_conditions = []
    for pair in dpo_pairs:
        conditions = pair.get('validation_info', {}).get('dispreferred_reasons', [])
        all_conditions.extend(conditions)
    
    if all_conditions:
        print(f"反例条件统计:")
        condition_counts = {}
        for condition in all_conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        for condition, count in condition_counts.items():
            print(f"  {condition}: {count}")

if __name__ == "__main__":
    dpo_pairs = test_optimized_implementation()
    analyze_test_results(dpo_pairs)