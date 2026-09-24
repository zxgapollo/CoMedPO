#!/usr/bin/env python3
"""
对比分析两个DPO数据集中conversations的正确性
"""

import json
import sys
from collections import defaultdict

def load_dataset(file_path):
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {file_path}，包含 {len(data)} 个样本")
        return data
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None

def analyze_dataset_structure(data, dataset_name):
    """分析数据集结构"""
    print(f"\n=== {dataset_name} 数据集结构分析 ===")
    
    if not data:
        return
    
    # 基本统计
    print(f"样本总数: {len(data)}")
    
    # 检查字段
    sample = data[0]
    print(f"字段: {list(sample.keys())}")
    
    # 检查图像路径模式
    image_patterns = defaultdict(int)
    for item in data[:100]:  # 只检查前100个样本
        image_path = item.get('image', '')
        if '/' in image_path:
            pattern = image_path.split('/')[0]
            image_patterns[pattern] += 1
    
    print(f"图像路径模式 (前100个样本): {dict(image_patterns)}")
    
    # 检查权重分布
    weights = [item.get('weighted_score', 0) for item in data]
    print(f"权重范围: [{min(weights):.3f}, {max(weights):.3f}]")
    print(f"平均权重: {sum(weights)/len(weights):.3f}")

def find_matching_samples(data1, data2):
    """寻找两个数据集中的匹配样本"""
    print(f"\n=== 寻找匹配样本 ===")
    
    # 按问题内容建立索引
    def get_question(item):
        if 'conversations' in item and len(item['conversations']) > 0:
            return item['conversations'][0].get('value', '').strip()
        return ''
    
    # 建立问题到样本的映射
    questions_map1 = {}
    questions_map2 = {}
    
    for item in data1:
        question = get_question(item)
        if question:
            if question not in questions_map1:
                questions_map1[question] = []
            questions_map1[question].append(item)
    
    for item in data2:
        question = get_question(item)
        if question:
            if question not in questions_map2:
                questions_map2[question] = []
            questions_map2[question].append(item)
    
    # 找到共同的问题
    common_questions = set(questions_map1.keys()) & set(questions_map2.keys())
    print(f"共同问题数量: {len(common_questions)}")
    print(f"数据集1独有问题: {len(questions_map1) - len(common_questions)}")
    print(f"数据集2独有问题: {len(questions_map2) - len(common_questions)}")
    
    return questions_map1, questions_map2, common_questions

def compare_conversations(data1, data2, questions_map1, questions_map2, common_questions):
    """对比conversations的正确性"""
    print(f"\n=== 对比conversations正确性 ===")
    
    comparison_results = []
    
    # 随机选择一些共同问题进行详细对比
    sample_questions = list(common_questions)[:20]  # 只对比前20个
    
    for question in sample_questions:
        items1 = questions_map1[question]
        items2 = questions_map2[question]
        
        # 取第一个匹配的样本
        item1 = items1[0]
        item2 = items2[0]
        
        result = {
            'question': question,
            'dataset1_answer': item1['conversations'][1]['value'] if len(item1['conversations']) > 1 else '',
            'dataset1_rejected': item1['rejected_conversations'][1]['value'] if len(item1['rejected_conversations']) > 1 else '',
            'dataset1_weight': item1.get('weighted_score', 0),
            'dataset2_answer': item2['conversations'][1]['value'] if len(item2['conversations']) > 1 else '',
            'dataset2_rejected': item2['rejected_conversations'][1]['value'] if len(item2['rejected_conversations']) > 1 else '',
            'dataset2_weight': item2.get('weighted_score', 0),
            'dataset1_image': item1.get('image', ''),
            'dataset2_image': item2.get('image', ''),
        }
        
        comparison_results.append(result)
    
    return comparison_results

def print_comparison_results(results, dataset1_name, dataset2_name):
    """打印对比结果"""
    print(f"\n=== 详细对比结果 ===")
    
    for i, result in enumerate(results):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {result['question']}")
        print(f"图像: {dataset1_name}={result['dataset1_image']}, {dataset2_name}={result['dataset2_image']}")
        print(f"\n{dataset1_name} 回答 (权重={result['dataset1_weight']:.3f}):")
        print(f"  正确: {result['dataset1_answer']}")
        print(f"  错误: {result['dataset1_rejected']}")
        print(f"\n{dataset2_name} 回答 (权重={result['dataset2_weight']:.3f}):")
        print(f"  正确: {result['dataset2_answer']}")
        print(f"  错误: {result['dataset2_rejected']}")
        
        # 简单的一致性检查
        if result['dataset1_answer'].strip() == result['dataset2_answer'].strip():
            print("✓ 正确答案一致")
        else:
            print("✗ 正确答案不一致")
        
        if result['dataset1_rejected'].strip() == result['dataset2_rejected'].strip():
            print("✓ 错误答案一致")
        else:
            print("✗ 错误答案不一致")

def analyze_answer_patterns(data, dataset_name):
    """分析答案模式"""
    print(f"\n=== {dataset_name} 答案模式分析 ===")
    
    answer_lengths = []
    rejected_lengths = []
    
    for item in data[:100]:  # 只分析前100个样本
        if 'conversations' in item and len(item['conversations']) > 1:
            answer = item['conversations'][1]['value']
            answer_lengths.append(len(answer))
        
        if 'rejected_conversations' in item and len(item['rejected_conversations']) > 1:
            rejected = item['rejected_conversations'][1]['value']
            rejected_lengths.append(len(rejected))
    
    if answer_lengths:
        print(f"正确答案平均长度: {sum(answer_lengths)/len(answer_lengths):.1f} 字符")
        print(f"正确答案长度范围: [{min(answer_lengths)}, {max(answer_lengths)}]")
    
    if rejected_lengths:
        print(f"错误答案平均长度: {sum(rejected_lengths)/len(rejected_lengths):.1f} 字符")
        print(f"错误答案长度范围: [{min(rejected_lengths)}, {max(rejected_lengths)}]")

def main():
    # 文件路径
    file1 = "/workspace/MMedPO/data/tie_dpo_dataset_improved.json"
    file2 = "/workspace/MMedPO/data/slake_dpo_weighted.json"
    
    print("开始对比分析两个DPO数据集...")
    
    # 加载数据集
    data1 = load_dataset(file1)
    data2 = load_dataset(file2)
    
    if not data1 or not data2:
        print("数据集加载失败，退出分析")
        return
    
    # 分析数据集结构
    analyze_dataset_structure(data1, "tie_dpo_dataset_improved")
    analyze_dataset_structure(data2, "slake_dpo_weighted")
    
    # 分析答案模式
    analyze_answer_patterns(data1, "tie_dpo_dataset_improved")
    analyze_answer_patterns(data2, "slake_dpo_weighted")
    
    # 寻找匹配样本
    questions_map1, questions_map2, common_questions = find_matching_samples(data1, data2)
    
    if common_questions:
        # 对比conversations
        results = compare_conversations(data1, data2, questions_map1, questions_map2, common_questions)
        print_comparison_results(results, "tie_dpo_dataset_improved", "slake_dpo_weighted")
    else:
        print("没有找到共同的问题，无法进行对比")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()