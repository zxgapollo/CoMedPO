#!/usr/bin/env python3
"""
分析GT答案与positive/negative答案的关系
"""

import json
from typing import Dict, List
from collections import defaultdict

def analyze_gt_relationships(tie_results_file: str):
    """
    分析GT答案与positive/negative答案的关系
    """
    print("=== GT答案关系分析 ===")
    
    # 统计数据
    stats = {
        "total_samples": 0,
        "gt_equals_positive": 0,
        "gt_equals_negative": 0,
        "gt_different_from_both": 0,
        "gt_empty": 0,
        "positive_empty": 0,
        "negative_empty": 0
    }
    
    # 详细分析
    relationship_examples = {
        "gt_equals_positive": [],
        "gt_equals_negative": [],
        "gt_different_from_both": []
    }
    
    # 读取数据
    with open(tie_results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"处理进度: {line_num}")
            
            try:
                result = json.loads(line.strip())
                stats["total_samples"] += 1
                
                gt_answer = result.get('gt_answer', '').strip()
                positive_answer = result.get('positive_answer', '').strip()
                negative_answer = result.get('negative_answer', '').strip()
                
                # 检查空值
                if not gt_answer:
                    stats["gt_empty"] += 1
                    continue
                if not positive_answer:
                    stats["positive_empty"] += 1
                if not negative_answer:
                    stats["negative_empty"] += 1
                
                # 分析关系
                if gt_answer == positive_answer:
                    stats["gt_equals_positive"] += 1
                    if len(relationship_examples["gt_equals_positive"]) < 5:
                        relationship_examples["gt_equals_positive"].append({
                            "id": result.get("id"),
                            "case_id": result.get("case_id"),
                            "question": result.get("question"),
                            "gt_answer": gt_answer,
                            "positive_answer": positive_answer,
                            "negative_answer": negative_answer
                        })
                elif gt_answer == negative_answer:
                    stats["gt_equals_negative"] += 1
                    if len(relationship_examples["gt_equals_negative"]) < 5:
                        relationship_examples["gt_equals_negative"].append({
                            "id": result.get("id"),
                            "case_id": result.get("case_id"),
                            "question": result.get("question"),
                            "gt_answer": gt_answer,
                            "positive_answer": positive_answer,
                            "negative_answer": negative_answer
                        })
                else:
                    stats["gt_different_from_both"] += 1
                    if len(relationship_examples["gt_different_from_both"]) < 10:
                        relationship_examples["gt_different_from_both"].append({
                            "id": result.get("id"),
                            "case_id": result.get("case_id"),
                            "question": result.get("question"),
                            "gt_answer": gt_answer,
                            "positive_answer": positive_answer,
                            "negative_answer": negative_answer
                        })
                        
            except json.JSONDecodeError as e:
                print(f"JSON解析错误在行 {line_num}: {e}")
                continue
    
    # 输出统计结果
    print(f"\n=== 统计结果 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"GT答案为空: {stats['gt_empty']} ({stats['gt_empty']/stats['total_samples']*100:.2f}%)")
    print(f"Positive答案为空: {stats['positive_empty']} ({stats['positive_empty']/stats['total_samples']*100:.2f}%)")
    print(f"Negative答案为空: {stats['negative_empty']} ({stats['negative_empty']/stats['total_samples']*100:.2f}%)")
    print(f"GT = Positive: {stats['gt_equals_positive']} ({stats['gt_equals_positive']/stats['total_samples']*100:.2f}%)")
    print(f"GT = Negative: {stats['gt_equals_negative']} ({stats['gt_equals_negative']/stats['total_samples']*100:.2f}%)")
    print(f"GT ≠ Both: {stats['gt_different_from_both']} ({stats['gt_different_from_both']/stats['total_samples']*100:.2f}%)")
    
    # 输出示例
    print(f"\n=== GT = Positive 示例 ===")
    for i, example in enumerate(relationship_examples["gt_equals_positive"], 1):
        print(f"示例{i}: ID={example['id']}, Case={example['case_id']}")
        print(f"  问题: {example['question']}")
        print(f"  GT答案: {example['gt_answer']}")
        print(f"  Positive答案: {example['positive_answer']}")
        print(f"  Negative答案: {example['negative_answer']}")
        print()
    
    print(f"\n=== GT = Negative 示例 ===")
    for i, example in enumerate(relationship_examples["gt_equals_negative"], 1):
        print(f"示例{i}: ID={example['id']}, Case={example['case_id']}")
        print(f"  问题: {example['question']}")
        print(f"  GT答案: {example['gt_answer']}")
        print(f"  Positive答案: {example['positive_answer']}")
        print(f"  Negative答案: {example['negative_answer']}")
        print()
    
    print(f"\n=== GT ≠ Both 示例 (可用于GT fallback) ===")
    for i, example in enumerate(relationship_examples["gt_different_from_both"], 1):
        print(f"示例{i}: ID={example['id']}, Case={example['case_id']}")
        print(f"  问题: {example['question']}")
        print(f"  GT答案: {example['gt_answer']}")
        print(f"  Positive答案: {example['positive_answer']}")
        print(f"  Negative答案: {example['negative_answer']}")
        print()
    
    return stats, relationship_examples

if __name__ == "__main__":
    tie_results_file = "/workspace/MMedPO/outputs/tie_results_1/tie_results.json"
    stats, examples = analyze_gt_relationships(tie_results_file)
    
    # 保存分析结果
    analysis_result = {
        "statistics": stats,
        "examples": examples
    }
    
    output_file = "/workspace/MMedPO/outputs/gt_answer_relationship_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")