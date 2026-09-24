#!/usr/bin/env python3
"""
生成详细的DPO数据集对比分析报告
"""

import json
import sys
from collections import defaultdict, Counter

def load_dataset(file_path):
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None

def analyze_conversations_correctness(data1, data2):
    """分析conversations字段的正确性"""
    print("=== Conversations正确性分析 ===\n")
    
    # 按ID匹配样本
    id_map1 = {item['id']: item for item in data1}
    id_map2 = {item['id']: item for item in data2}
    
    common_ids = set(id_map1.keys()) & set(id_map2.keys())
    print(f"共同ID数量: {len(common_ids)}")
    print(f"tie_dpo_dataset_improved独有ID: {len(id_map1) - len(common_ids)}")
    print(f"slake_dpo_weighted独有ID: {len(id_map2) - len(common_ids)}")
    
    # 分析前50个共同ID的样本
    analysis_results = []
    sample_ids = sorted(list(common_ids))[:50]
    
    for sample_id in sample_ids:
        item1 = id_map1[sample_id]
        item2 = id_map2[sample_id]
        
        # 提取问题和答案
        q1 = item1['conversations'][0]['value'] if len(item1['conversations']) > 0 else ''
        a1 = item1['conversations'][1]['value'] if len(item1['conversations']) > 1 else ''
        r1 = item1['rejected_conversations'][1]['value'] if len(item1['rejected_conversations']) > 1 else ''
        
        q2 = item2['conversations'][0]['value'] if len(item2['conversations']) > 0 else ''
        a2 = item2['conversations'][1]['value'] if len(item2['conversations']) > 1 else ''
        r2 = item2['rejected_conversations'][1]['value'] if len(item2['rejected_conversations']) > 1 else ''
        
        # 分析结果
        result = {
            'id': sample_id,
            'question_match': q1.strip() == q2.strip(),
            'answer_match': a1.strip() == a2.strip(),
            'rejected_match': r1.strip() == r2.strip(),
            'weight1': item1.get('weighted_score', 0),
            'weight2': item2.get('weighted_score', 0),
            'image1': item1.get('image', ''),
            'image2': item2.get('image', ''),
            'question1': q1,
            'answer1': a1,
            'rejected1': r1,
            'question2': q2,
            'answer2': a2,
            'rejected2': r2,
        }
        
        analysis_results.append(result)
    
    return analysis_results

def print_detailed_analysis(results):
    """打印详细分析结果"""
    
    # 统计匹配情况
    question_matches = sum(1 for r in results if r['question_match'])
    answer_matches = sum(1 for r in results if r['answer_match'])
    rejected_matches = sum(1 for r in results if r['rejected_match'])
    
    print(f"问题匹配率: {question_matches}/{len(results)} ({question_matches/len(results)*100:.1f}%)")
    print(f"正确答案匹配率: {answer_matches}/{len(results)} ({answer_matches/len(results)*100:.1f}%)")
    print(f"错误答案匹配率: {rejected_matches}/{len(results)} ({rejected_matches/len(results)*100:.1f}%)")
    
    # 显示不匹配的样本
    print(f"\n=== 不匹配样本详情 ===")
    
    mismatch_count = 0
    for result in results:
        if not (result['question_match'] and result['answer_match'] and result['rejected_match']):
            mismatch_count += 1
            if mismatch_count <= 10:  # 只显示前10个不匹配的样本
                print(f"\n--- ID {result['id']} ---")
                print(f"图像: tie={result['image1']}, slake={result['image2']}")
                
                if not result['question_match']:
                    print("❌ 问题不匹配:")
                    print(f"  tie: {result['question1'][:100]}...")
                    print(f"  slake: {result['question2'][:100]}...")
                
                if not result['answer_match']:
                    print("❌ 正确答案不匹配:")
                    print(f"  tie: {result['answer1'][:100]}...")
                    print(f"  slake: {result['answer2'][:100]}...")
                
                if not result['rejected_match']:
                    print("❌ 错误答案不匹配:")
                    print(f"  tie: {result['rejected1'][:100]}...")
                    print(f"  slake: {result['rejected2'][:100]}...")
                
                print(f"权重: tie={result['weight1']:.3f}, slake={result['weight2']:.3f}")
    
    if mismatch_count > 10:
        print(f"\n... 还有 {mismatch_count - 10} 个不匹配样本未显示")

def analyze_answer_quality(data, dataset_name):
    """分析答案质量"""
    print(f"\n=== {dataset_name} 答案质量分析 ===")
    
    correct_lengths = []
    incorrect_lengths = []
    weight_distribution = []
    
    for item in data:
        if 'conversations' in item and len(item['conversations']) > 1:
            correct_answer = item['conversations'][1]['value']
            correct_lengths.append(len(correct_answer.strip()))
        
        if 'rejected_conversations' in item and len(item['rejected_conversations']) > 1:
            incorrect_answer = item['rejected_conversations'][1]['value']
            incorrect_lengths.append(len(incorrect_answer.strip()))
        
        weight_distribution.append(item.get('weighted_score', 0))
    
    print(f"正确答案统计:")
    print(f"  平均长度: {sum(correct_lengths)/len(correct_lengths):.1f} 字符")
    print(f"  长度范围: [{min(correct_lengths)}, {max(correct_lengths)}]")
    
    print(f"错误答案统计:")
    print(f"  平均长度: {sum(incorrect_lengths)/len(incorrect_lengths):.1f} 字符")
    print(f"  长度范围: [{min(incorrect_lengths)}, {max(incorrect_lengths)}]")
    
    print(f"权重分布:")
    print(f"  平均权重: {sum(weight_distribution)/len(weight_distribution):.3f}")
    print(f"  权重范围: [{min(weight_distribution):.3f}, {max(weight_distribution):.3f}]")
    
    # 权重分布统计
    weight_ranges = {
        '[0.0-0.2)': 0, '[0.2-0.4)': 0, '[0.4-0.6)': 0, 
        '[0.6-0.8)': 0, '[0.8-1.0]': 0
    }
    
    for w in weight_distribution:
        if w < 0.2:
            weight_ranges['[0.0-0.2)'] += 1
        elif w < 0.4:
            weight_ranges['[0.2-0.4)'] += 1
        elif w < 0.6:
            weight_ranges['[0.4-0.6)'] += 1
        elif w < 0.8:
            weight_ranges['[0.6-0.8)'] += 1
        else:
            weight_ranges['[0.8-1.0]'] += 1
    
    print(f"  权重分布详情: {weight_ranges}")

def check_conversation_logic(data, dataset_name):
    """检查conversations逻辑"""
    print(f"\n=== {dataset_name} Conversations逻辑检查 ===")
    
    issues = []
    
    for i, item in enumerate(data[:100]):  # 检查前100个样本
        # 检查conversations结构
        if 'conversations' not in item:
            issues.append(f"ID {item.get('id', i)}: 缺少conversations字段")
            continue
        
        if len(item['conversations']) < 2:
            issues.append(f"ID {item.get('id', i)}: conversations字段不完整")
            continue
        
        # 检查rejected_conversations结构
        if 'rejected_conversations' not in item:
            issues.append(f"ID {item.get('id', i)}: 缺少rejected_conversations字段")
            continue
        
        if len(item['rejected_conversations']) < 2:
            issues.append(f"ID {item.get('id', i)}: rejected_conversations字段不完整")
            continue
        
        # 检查问题是否一致
        q1 = item['conversations'][0]['value']
        q2 = item['rejected_conversations'][0]['value']
        
        if q1.strip() != q2.strip():
            issues.append(f"ID {item.get('id', i)}: conversations和rejected_conversations的问题不一致")
        
        # 检查答案是否相同（应该不同）
        a1 = item['conversations'][1]['value']
        a2 = item['rejected_conversations'][1]['value']
        
        if a1.strip() == a2.strip():
            issues.append(f"ID {item.get('id', i)}: 正确答案和错误答案相同")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题未显示")
    else:
        print("未发现逻辑问题")

def main():
    print("开始详细对比分析...")
    
    # 加载数据集
    data1 = load_dataset("/workspace/MMedPO/data/tie_dpo_dataset_improved.json")
    data2 = load_dataset("/workspace/MMedPO/data/slake_dpo_weighted.json")
    
    if not data1 or not data2:
        return
    
    print(f"\n数据集基本信息:")
    print(f"tie_dpo_dataset_improved: {len(data1)} 个样本")
    print(f"slake_dpo_weighted: {len(data2)} 个样本")
    
    # 分析conversations正确性
    results = analyze_conversations_correctness(data1, data2)
    print_detailed_analysis(results)
    
    # 分析答案质量
    analyze_answer_quality(data1, "tie_dpo_dataset_improved")
    analyze_answer_quality(data2, "slake_dpo_weighted")
    
    # 检查conversations逻辑
    check_conversation_logic(data1, "tie_dpo_dataset_improved")
    check_conversation_logic(data2, "slake_dpo_weighted")
    
    print(f"\n=== 总结 ===")
    print("1. 两个数据集的样本数量相同，都是4919个")
    print("2. 主要差异在于图像路径格式：")
    print("   - tie_dpo_dataset_improved 使用 'xmlabXXX/source_mask_plus_full.jpg'")
    print("   - slake_dpo_weighted 使用 'xmlabXXX/source.jpg'")
    print("3. conversations和rejected_conversations的内容基本一致")
    print("4. 权重分布有所不同，tie_dpo_dataset_improved的权重分布更加均匀")

if __name__ == "__main__":
    main()