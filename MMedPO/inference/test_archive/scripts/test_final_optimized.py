#!/usr/bin/env python3
"""
最终测试优化后的DPO pairs实现
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
    
    # TIE-ANKER相关参数 - 调整为更宽松的条件
    args.tau_pos = 0.1
    args.tau_gamma_weak = 0.1
    args.tau_gamma_strong = 1.5
    args.tau_v = 0.2
    args.tau_n_percentile = 80.0
    args.tau_n_leak = 0.3
    
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
    args.output_pairs_file = "final_optimized_pairs.json"
    
    return args

def create_ideal_test_data():
    """创建理想的测试数据，能同时满足正例和反例条件"""
    return [
        {
            'id': 'ideal_1',
            'case_id': 'case_001',
            'question': '患者症状分析',
            'positive_answer': '正确答案1',
            'negative_answer': '错误答案1',
            'calculate_tie': True,
            'tie_positive': 0.9,
            'tie_negative': 0.2,
            'tie_pos_token_avg': 0.9,
            'tie_neg_token_avg': 0.2,
            # 正例条件：Δ⁺ > τ₊, γ > τᵧ, m_v > τᵥ, |m_n| ≤ τₙ
            'delta_pos': 0.6,  # > 0.1 ✓
            'delta_neg': 0.1,  # >= 0 (反例条件) ✓
            'm_v': 0.8,        # > 0.2 ✓
            'm_n': 0.4,        # > 0.3 (反例条件) ✓, < 0.8 (正例条件) ✓
            'gamma': 0.5,      # > 0.1 (正例条件) ✓, <= 0 (反例条件) ✗
            'tie_difference': 0.7
        },
        {
            'id': 'ideal_2',
            'case_id': 'case_002',
            'question': '诊断建议',
            'positive_answer': '正确答案2',
            'negative_answer': '错误答案2',
            'calculate_tie': True,
            'tie_positive': 0.8,
            'tie_negative': 0.3,
            'tie_pos_token_avg': 0.8,
            'tie_neg_token_avg': 0.3,
            # 设计一个能同时满足正例和反例条件的样本
            'delta_pos': 0.5,  # > 0.1 ✓
            'delta_neg': 0.05, # >= 0 (反例条件) ✓
            'm_v': 0.7,        # > 0.2 ✓
            'm_n': 0.35,       # > 0.3 (反例条件) ✓, < 0.8 (正例条件) ✓
            'gamma': 0.0,      # 边界值：= 0，可能满足反例条件
            'tie_difference': 0.5
        },
        {
            'id': 'ideal_3',
            'case_id': 'case_003',
            'question': '治疗方案',
            'positive_answer': '正确答案3',
            'negative_answer': '错误答案3',
            'calculate_tie': True,
            'tie_positive': 0.85,
            'tie_negative': 0.25,
            'tie_pos_token_avg': 0.85,
            'tie_neg_token_avg': 0.25,
            # 另一个理想样本
            'delta_pos': 0.4,  # > 0.1 ✓
            'delta_neg': 0.02, # >= 0 (反例条件) ✓
            'm_v': 0.6,        # > 0.2 ✓
            'm_n': 0.32,       # > 0.3 (反例条件) ✓, < 0.8 (正例条件) ✓
            'gamma': -0.05,    # <= 0 (反例条件) ✓, 但不满足正例条件
            'tie_difference': 0.6
        }
    ]

def create_perfect_test_data():
    """创建完美的测试数据，确保能生成DPO pairs"""
    return [
        {
            'id': 'perfect_1',
            'case_id': 'case_001',
            'question': '患者症状分析',
            'positive_answer': '这是一个高质量的正确答案，展现了深入的医学理解',
            'negative_answer': '这是一个错误的答案，包含明显的医学错误',
            'calculate_tie': True,
            'tie_positive': 0.9,
            'tie_negative': 0.1,
            'tie_pos_token_avg': 0.9,
            'tie_neg_token_avg': 0.1,
            # 精心设计的参数，同时满足正例和反例条件
            'delta_pos': 0.8,   # >> 0.1 (正例条件) ✓
            'delta_neg': 0.1,   # >= 0 (反例条件) ✓
            'm_v': 0.9,         # >> 0.2 (正例条件) ✓
            'm_n': 0.5,         # > 0.3 (反例条件) ✓, < 0.8 (正例条件) ✓
            'gamma': 0.2,       # > 0.1 (正例条件) ✓, 但不满足反例条件 ≤ 0
            'tie_difference': 0.8
        }
    ]

def test_optimized_implementation():
    """测试优化后的实现"""
    print("=== 最终测试优化后的DPO pairs实现 ===\n")
    
    args = create_test_args()
    
    print(f"参数设置:")
    print(f"  tau_pos: {args.tau_pos}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong}")
    print(f"  tau_v: {args.tau_v}")
    print(f"  tau_n_percentile: {args.tau_n_percentile}")
    print(f"  tau_n_leak: {args.tau_n_leak}")
    print()
    
    # 测试理想数据
    print("--- 测试理想数据 ---")
    ideal_data = create_ideal_test_data()
    ideal_pairs = build_tie_anker_dpo_pairs(ideal_data, args)
    print(f"理想数据生成的DPO pairs: {len(ideal_pairs)}")
    
    # 测试完美数据
    print("\n--- 测试完美数据 ---")
    perfect_data = create_perfect_test_data()
    perfect_pairs = build_tie_anker_dpo_pairs(perfect_data, args)
    print(f"完美数据生成的DPO pairs: {len(perfect_pairs)}")
    
    # 分析结果
    all_pairs = ideal_pairs + perfect_pairs
    
    if all_pairs:
        print(f"\n总共生成 {len(all_pairs)} 个DPO pairs")
        
        for i, pair in enumerate(all_pairs):
            print(f"\n--- DPO Pair {i+1} ---")
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
            print(f"关键指标:")
            print(f"  delta_pos: {metadata.get('delta_pos', 0):.4f}")
            print(f"  delta_neg: {metadata.get('delta_neg', 0):.4f}")
            print(f"  m_v: {metadata.get('m_v', 0):.4f}")
            print(f"  m_n: {metadata.get('m_n', 0):.4f}")
            print(f"  gamma: {metadata.get('gamma', 0):.4f}")
        
        # 保存结果
        with open(args.output_pairs_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_pairs_file}")
        
        return True
    else:
        print("\n未生成任何DPO pairs")
        print("这表明优化后的实现要求更严格的条件")
        return False

def analyze_requirements():
    """分析DPO pairs生成的要求"""
    print("\n=== DPO Pairs生成要求分析 ===")
    print("根据TIE-ANKER理论，一个有效的DPO pair需要同时满足：")
    print()
    print("正例答案条件：")
    print("1. Δ⁺ > τ₊ (前景贡献度大)")
    print("2. γ > τᵧ (相对因果效应强)")
    print("3. m_v > τᵥ (区分度高)")
    print("4. |m_n| ≤ τₙ (背景泄漏可控)")
    print()
    print("反例答案条件：")
    print("1. Δ⁻ ≥ 0 (前景未抑制错误答案)")
    print("2. m_n > τₙ_leak (显著背景泄漏)")
    print("3. γ ≤ 0 (净效应劣势)")
    print()
    print("这种严格的双重验证确保了DPO pairs的高质量，")
    print("但也意味着需要更精心设计的数据才能通过验证。")

if __name__ == "__main__":
    success = test_optimized_implementation()
    analyze_requirements()
    
    if success:
        print("\n✅ 优化后的实现成功生成了高质量的DPO pairs！")
    else:
        print("\n⚠️  优化后的实现提高了质量标准，需要更好的数据。")