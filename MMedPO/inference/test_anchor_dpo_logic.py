#!/usr/bin/env python3
"""
测试基于 Anchor 的 DPO pair 选取逻辑
"""

import sys
import os
import json
import numpy as np
from argparse import Namespace

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修改后的函数
from inference_dpo_tie_comparison import (
    calculate_semantic_similarity,
    check_conversation_validity,
    calculate_dpo_weight,
    calculate_batch_statistics,
    build_tie_anker_dpo_pairs
)

def create_test_data():
    """创建测试数据"""
    test_results = [
        {
            'id': 'test_1',
            'case_id': 'case_001',
            'question': 'What is the diagnosis based on this medical image?',
            'positive_answer': 'The image shows signs of pneumonia with consolidation in the right lower lobe.',
            'negative_answer': 'This appears to be a normal chest X-ray with no abnormalities.',
            'anchor_answer': 'Pneumonia with right lower lobe consolidation and pleural effusion.',
            'is_anchor': True,
            'calculate_tie': True,
            'tie_positive': 0.8,
            'tie_negative': 0.2,
            'tie_pos_token_avg': 0.75,
            'tie_neg_token_avg': 0.25,
            'delta_pos': 0.6,
            'delta_neg': 0.1,
            'm_v': 0.4,
            'm_n': 0.2,
            'gamma': 0.2
        },
        {
            'id': 'test_2',
            'case_id': 'case_001',
            'question': 'What is the diagnosis based on this medical image?',
            'positive_answer': 'Pneumonia is visible in the lung with inflammatory changes.',
            'negative_answer': 'The chest X-ray shows clear lungs without pathology.',
            'calculate_tie': True,
            'tie_positive': 0.7,
            'tie_negative': 0.3,
            'tie_pos_token_avg': 0.65,
            'tie_neg_token_avg': 0.35,
            'delta_pos': 0.5,
            'delta_neg': 0.2,
            'm_v': 0.3,
            'm_n': 0.15,
            'gamma': 0.15
        },
        {
            'id': 'test_3',
            'case_id': 'case_002',
            'question': 'Describe the findings in this CT scan.',
            'positive_answer': 'The CT scan reveals a mass lesion in the liver with enhancement.',
            'negative_answer': 'No significant abnormalities are seen in this CT scan.',
            'calculate_tie': True,
            'tie_positive': 0.9,
            'tie_negative': 0.1,
            'tie_pos_token_avg': 0.85,
            'tie_neg_token_avg': 0.15,
            'delta_pos': 0.7,
            'delta_neg': 0.05,
            'm_v': 0.5,
            'm_n': 0.1,
            'gamma': 0.4
        }
    ]
    return test_results

def create_test_args():
    """创建测试参数"""
    args = Namespace()
    args.enable_tie_anker = True
    args.token_avg = 'true'
    args.use_per_case_zscore = 'true'
    args.z_eps = 1e-6
    args.lambda_anchor = 0.7
    
    # TIE-ANKER 参数
    args.w_gamma = 1.0
    args.w_v = 0.5
    args.w_n = 0.8
    args.w_s = 0.3
    args.w_o = 0.5
    args.beta = 2.0
    args.tau = 0.0
    args.epsilon = 0.02
    
    # 权重映射参数
    args.w_min = 0.01
    args.w_max = 10.0
    args.p_min = 0.1
    args.p_max = 0.9
    
    # 阈值参数
    args.tie_threshold = 0.1
    args.weight_threshold = 0.5
    args.confidence_threshold = 0.6
    
    # 其他必要参数
    args.tau_n_percentile = 75
    args.strong_dispreference_threshold = 0.2
    args.tau_pos = 0.1
    args.tau_gamma_weak = 0.1
    args.tau_gamma_strong = 0.5
    args.tau_v = 0.1
    args.tau_n_percentile = 75
    args.w_min = 0.05
    
    return args

def test_semantic_similarity():
    """测试语义相似度计算"""
    print("=== 测试语义相似度计算 ===")
    
    text1 = "The image shows signs of pneumonia with consolidation in the right lower lobe."
    text2 = "Pneumonia with right lower lobe consolidation and pleural effusion."
    text3 = "This appears to be a normal chest X-ray with no abnormalities."
    
    sim1 = calculate_semantic_similarity(text1, text2, method='combined')
    sim2 = calculate_semantic_similarity(text1, text3, method='combined')
    
    print(f"相似文本相似度: {sim1:.3f}")
    print(f"不相似文本相似度: {sim2:.3f}")
    print(f"相似度差异: {sim1 - sim2:.3f}")
    
    assert sim1 > sim2, "相似文本的相似度应该更高"
    print("✓ 语义相似度计算测试通过")

def test_conversation_validity():
    """测试对话有效性检查"""
    print("\n=== 测试对话有效性检查 ===")
    
    question = "What is the diagnosis based on this medical image?"
    good_answer = "The image shows signs of pneumonia with consolidation."
    bad_answer = "Yes."
    irrelevant_answer = "The weather is nice today."
    
    valid1 = check_conversation_validity(question, good_answer)
    valid2 = check_conversation_validity(question, bad_answer)
    valid3 = check_conversation_validity(question, irrelevant_answer)
    
    print(f"好答案有效性: {valid1}")
    print(f"短答案有效性: {valid2}")
    print(f"无关答案有效性: {valid3}")
    
    assert valid1 == True, "好答案应该被认为有效"
    assert valid2 == False, "过短答案应该被认为无效"
    print("✓ 对话有效性检查测试通过")

def test_weight_calculation():
    """测试权重计算和归一化"""
    print("\n=== 测试权重计算和归一化 ===")
    
    # 创建测试数据
    test_weights = []
    for i in range(5):
        delta_pos = 0.5 + i * 0.1
        delta_neg = 0.2 + i * 0.05
        m_v = 0.3 + i * 0.1
        m_n = 0.1 + i * 0.02
        
        result = calculate_dpo_weight(
            delta_pos=delta_pos,
            delta_neg=delta_neg,
            m_v=m_v,
            m_n=m_n,
            normalize_weights=False
        )
        test_weights.append(result['dpo_weight'])
    
    # 计算批统计信息
    batch_results = [{'dpo_weight': w} for w in test_weights]
    batch_stats = calculate_batch_statistics(batch_results, include_weights=True)
    
    print(f"原始权重: {test_weights}")
    print(f"权重统计: mean={batch_stats['weight_mean']:.3f}, std={batch_stats['weight_std']:.3f}")
    
    # 测试归一化权重计算
    normalized_weights = []
    for i in range(5):
        delta_pos = 0.5 + i * 0.1
        delta_neg = 0.2 + i * 0.05
        m_v = 0.3 + i * 0.1
        m_n = 0.1 + i * 0.02
        
        result = calculate_dpo_weight(
            delta_pos=delta_pos,
            delta_neg=delta_neg,
            m_v=m_v,
            m_n=m_n,
            batch_stats=batch_stats,
            normalize_weights=True
        )
        normalized_weights.append(result['dpo_weight'])
    
    print(f"归一化权重: {normalized_weights}")
    print("✓ 权重计算和归一化测试通过")

def test_anchor_dpo_pairs():
    """测试基于 Anchor 的 DPO pairs 构建"""
    print("\n=== 测试基于 Anchor 的 DPO pairs 构建 ===")
    
    test_results = create_test_data()
    args = create_test_args()
    
    # 构建 DPO pairs
    dpo_pairs = build_tie_anker_dpo_pairs(test_results, args)
    
    print(f"生成的 DPO pairs 数量: {len(dpo_pairs)}")
    
    for i, pair in enumerate(dpo_pairs):
        print(f"\nDPO Pair {i+1}:")
        print(f"  问题: {pair['question'][:50]}...")
        print(f"  选择答案: {pair['chosen'][:50]}...")
        print(f"  拒绝答案: {pair['rejected'][:50]}...")
        print(f"  TIE 权重: {pair['tie_weight']}")
        print(f"  归一化权重: {pair.get('normalized_weight', 'N/A')}")
        
        if 'anchor_info' in pair:
            anchor_info = pair['anchor_info']
            print(f"  有 Anchor: {anchor_info['has_anchor']}")
            if anchor_info['has_anchor']:
                print(f"  选择答案与 Anchor 相似度: {anchor_info['chosen_anchor_similarity']:.3f}")
                print(f"  拒绝答案与 Anchor 相似度: {anchor_info['rejected_anchor_similarity']:.3f}")
        
        if 'validation_info' in pair:
            val_info = pair['validation_info']
            print(f"  对话有效性: {val_info['conversation_valid']}")
    
    assert len(dpo_pairs) > 0, "应该生成至少一个 DPO pair"
    print("✓ 基于 Anchor 的 DPO pairs 构建测试通过")

def main():
    """主测试函数"""
    print("开始测试修改后的 DPO pair 选取逻辑...")
    
    try:
        test_semantic_similarity()
        test_conversation_validity()
        test_weight_calculation()
        test_anchor_dpo_pairs()
        
        print("\n🎉 所有测试通过！修改后的逻辑工作正常。")
        
        # 保存测试结果
        test_results = create_test_data()
        args = create_test_args()
        dpo_pairs = build_tie_anker_dpo_pairs(test_results, args)
        
        output_file = "/workspace/MMedPO/inference/test_dpo_pairs_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"\n测试结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)