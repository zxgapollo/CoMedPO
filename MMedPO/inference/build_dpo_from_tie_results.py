#!/usr/bin/env python3
"""
从现有TIE结果构建DPO pairs
使用优化后的TIE-ANKER实现，基于实际数据分布调整验证逻辑
"""

import json
import sys
import os
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入优化后的验证函数
try:
    from inference_dpo_tie_comparison import calculate_tie_anker_weight
    print("✓ 成功导入优化后的TIE-ANKER权重计算函数")
except ImportError as e:
    print(f"✗ 导入权重计算函数失败: {e}")
    sys.exit(1)

def create_data_driven_args(tie_results: List[Dict]):
    """基于实际数据分布创建参数配置"""
    args = argparse.Namespace()
    
    # TIE-ANKER权重参数
    args.w_gamma = 0.3
    args.w_v = 0.2
    args.w_n = 0.1
    args.w_s = 0.2
    args.w_o = 0.2
    
    # Sigmoid映射参数
    args.beta = 1.0
    args.tau = 0.5
    args.epsilon = 1e-8
    args.w_min = 0.1
    
    # 基于数据分布计算阈值
    gamma_values = [r.get('gamma', 0) for r in tie_results if 'gamma' in r]
    delta_pos_values = [r.get('delta_pos', 0) for r in tie_results if 'delta_pos' in r]
    m_v_values = [r.get('m_v', 0) for r in tie_results if 'm_v' in r]
    m_n_values = [abs(r.get('m_n', 0)) for r in tie_results if 'm_n' in r]
    
    # 数据驱动的阈值设定
    args.tau_pos = np.percentile(delta_pos_values, 10) if delta_pos_values else -1.0  # 10%分位数
    args.tau_gamma_weak = np.percentile(gamma_values, 10) if gamma_values else -50.0  # 10%分位数
    args.tau_gamma_strong = np.percentile(gamma_values, 75) if gamma_values else 10.0  # 75%分位数
    args.tau_v = np.percentile(m_v_values, 10) if m_v_values else -100.0  # 10%分位数
    args.tau_n_percentile = 90.0  # 使用90%分位数
    args.tau_n_leak = np.percentile(m_n_values, 90) if m_n_values else 50.0  # 90%分位数
    
    print(f"数据驱动的阈值设定:")
    print(f"  tau_pos: {args.tau_pos:.3f}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak:.3f}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong:.3f}")
    print(f"  tau_v: {args.tau_v:.3f}")
    print(f"  tau_n_leak: {args.tau_n_leak:.3f}")
    
    return args

def analyze_tie_data(tie_results: List[Dict]) -> Dict[str, Any]:
    """分析TIE结果数据"""
    print(f"\n=== 分析TIE数据 ===")
    print(f"总样本数: {len(tie_results)}")
    
    # 统计基本信息
    has_positive = sum(1 for r in tie_results if r.get('has_positive', False))
    has_negative = sum(1 for r in tie_results if r.get('has_negative', False))
    calculate_tie = sum(1 for r in tie_results if r.get('calculate_tie', False))
    
    print(f"有正例答案: {has_positive}")
    print(f"有反例答案: {has_negative}")
    print(f"计算TIE: {calculate_tie}")
    
    # 统计关键指标
    gamma_values = [r.get('gamma', 0) for r in tie_results if 'gamma' in r]
    delta_pos_values = [r.get('delta_pos', 0) for r in tie_results if 'delta_pos' in r]
    delta_neg_values = [r.get('delta_neg', 0) for r in tie_results if 'delta_neg' in r]
    m_v_values = [r.get('m_v', 0) for r in tie_results if 'm_v' in r]
    m_n_values = [r.get('m_n', 0) for r in tie_results if 'm_n' in r]
    
    analysis = {
        "total_samples": len(tie_results),
        "has_positive": has_positive,
        "has_negative": has_negative,
        "calculate_tie": calculate_tie,
        "gamma_stats": {
            "count": len(gamma_values),
            "min": min(gamma_values) if gamma_values else 0,
            "max": max(gamma_values) if gamma_values else 0,
            "avg": sum(gamma_values) / len(gamma_values) if gamma_values else 0,
            "percentiles": {
                "10": np.percentile(gamma_values, 10) if gamma_values else 0,
                "25": np.percentile(gamma_values, 25) if gamma_values else 0,
                "50": np.percentile(gamma_values, 50) if gamma_values else 0,
                "75": np.percentile(gamma_values, 75) if gamma_values else 0,
                "90": np.percentile(gamma_values, 90) if gamma_values else 0
            }
        },
        "delta_pos_stats": {
            "count": len(delta_pos_values),
            "min": min(delta_pos_values) if delta_pos_values else 0,
            "max": max(delta_pos_values) if delta_pos_values else 0,
            "avg": sum(delta_pos_values) / len(delta_pos_values) if delta_pos_values else 0,
            "positive_count": sum(1 for v in delta_pos_values if v > 0),
            "percentiles": {
                "10": np.percentile(delta_pos_values, 10) if delta_pos_values else 0,
                "25": np.percentile(delta_pos_values, 25) if delta_pos_values else 0,
                "50": np.percentile(delta_pos_values, 50) if delta_pos_values else 0,
                "75": np.percentile(delta_pos_values, 75) if delta_pos_values else 0,
                "90": np.percentile(delta_pos_values, 90) if delta_pos_values else 0
            }
        },
        "m_v_stats": {
            "count": len(m_v_values),
            "min": min(m_v_values) if m_v_values else 0,
            "max": max(m_v_values) if m_v_values else 0,
            "avg": sum(m_v_values) / len(m_v_values) if m_v_values else 0,
            "percentiles": {
                "10": np.percentile(m_v_values, 10) if m_v_values else 0,
                "25": np.percentile(m_v_values, 25) if m_v_values else 0,
                "50": np.percentile(m_v_values, 50) if m_v_values else 0,
                "75": np.percentile(m_v_values, 75) if m_v_values else 0,
                "90": np.percentile(m_v_values, 90) if m_v_values else 0
            }
        },
        "m_n_stats": {
            "count": len(m_n_values),
            "min": min(m_n_values) if m_n_values else 0,
            "max": max(m_n_values) if m_n_values else 0,
            "avg": sum(m_n_values) / len(m_n_values) if m_n_values else 0,
            "abs_percentiles": {
                "10": np.percentile([abs(v) for v in m_n_values], 10) if m_n_values else 0,
                "25": np.percentile([abs(v) for v in m_n_values], 25) if m_n_values else 0,
                "50": np.percentile([abs(v) for v in m_n_values], 50) if m_n_values else 0,
                "75": np.percentile([abs(v) for v in m_n_values], 75) if m_n_values else 0,
                "90": np.percentile([abs(v) for v in m_n_values], 90) if m_n_values else 0
            }
        }
    }
    
    print(f"Gamma分位数: 10%={analysis['gamma_stats']['percentiles']['10']:.3f}, 50%={analysis['gamma_stats']['percentiles']['50']:.3f}, 90%={analysis['gamma_stats']['percentiles']['90']:.3f}")
    print(f"Delta_pos分位数: 10%={analysis['delta_pos_stats']['percentiles']['10']:.3f}, 50%={analysis['delta_pos_stats']['percentiles']['50']:.3f}, 90%={analysis['delta_pos_stats']['percentiles']['90']:.3f}")
    print(f"Delta_pos正值数量: {analysis['delta_pos_stats']['positive_count']}/{len(delta_pos_values)}")
    print(f"M_v分位数: 10%={analysis['m_v_stats']['percentiles']['10']:.3f}, 50%={analysis['m_v_stats']['percentiles']['50']:.3f}, 90%={analysis['m_v_stats']['percentiles']['90']:.3f}")
    print(f"|M_n|分位数: 10%={analysis['m_n_stats']['abs_percentiles']['10']:.3f}, 50%={analysis['m_n_stats']['abs_percentiles']['50']:.3f}, 90%={analysis['m_n_stats']['abs_percentiles']['90']:.3f}")
    
    return analysis

def validate_preferred_answer_data_driven(result: Dict, args) -> Tuple[bool, List[str]]:
    """
    基于数据分布的正例验证规则
    """
    violations = []
    
    # 条件1: Δ⁺ > τ₊ (前景贡献度) - 使用10%分位数
    delta_pos = result.get('delta_pos', 0)
    if delta_pos <= args.tau_pos:
        violations.append(f"前景贡献度不足: Δ⁺({delta_pos:.4f}) ≤ τ₊({args.tau_pos:.4f})")
    
    # 条件2: γ > τᵧ (相对因果效应) - 使用10%分位数
    gamma = result.get('gamma', 0)
    if gamma <= args.tau_gamma_weak:
        violations.append(f"相对因果效应过弱: γ({gamma:.4f}) ≤ τᵧ({args.tau_gamma_weak:.4f})")
    
    # 条件3: m_v > τᵥ (区分度) - 使用10%分位数
    m_v = result.get('m_v', 0)
    if m_v <= args.tau_v:
        violations.append(f"区分度不足: m_v({m_v:.4f}) ≤ τᵥ({args.tau_v:.4f})")
    
    # 条件4: |m_n| ≤ τₙ (背景泄漏) - 使用90%分位数
    m_n = result.get('m_n', 0)
    if abs(m_n) > args.tau_n_leak:
        violations.append(f"背景泄漏过大: |m_n|({abs(m_n):.4f}) > τₙ({args.tau_n_leak:.4f})")
    
    return len(violations) == 0, violations

def validate_dispreferred_answer_data_driven(result: Dict, args) -> Tuple[bool, List[str]]:
    """
    基于数据分布的反例验证规则
    """
    conditions = []
    
    # 条件1: γ ≤ 0 (净效应劣势)
    gamma = result.get('gamma', 0)
    if gamma <= 0:
        conditions.append(f"净效应劣势: γ({gamma:.4f}) ≤ 0")
        return True, conditions
    
    # 条件2: Δ⁻ < 0 (背景干扰强)
    delta_neg = result.get('delta_neg', 0)
    if delta_neg < 0:
        conditions.append(f"背景干扰强: Δ⁻({delta_neg:.4f}) < 0")
        return True, conditions
    
    # 条件3: gamma较小但为正
    if gamma < args.tau_gamma_strong:
        conditions.append(f"相对效应较弱: γ({gamma:.4f}) < τᵧ_strong({args.tau_gamma_strong:.4f})")
        return True, conditions
    
    return False, ["不满足反例选择条件"]

def apply_tie_anker_thresholds_relaxed(tie_weight: float, tie_diff: float, result: Dict, args) -> bool:
    """
    放宽的TIE-ANKER阈值过滤
    """
    # 基本权重阈值
    if tie_weight < args.w_min:
        return False
    
    # 非常宽松的TIE差异阈值
    if abs(tie_diff) < 0.05:  # 进一步降低要求
        return False
    
    # 综合质量评估 - 更宽松的条件
    gamma = result.get('gamma', 0)
    delta_pos = result.get('delta_pos', 0)
    
    # 任何一个条件满足即可
    if delta_pos > args.tau_pos:  # 前景贡献度足够
        return True
    
    if gamma > args.tau_gamma_weak:  # gamma值合理
        return True
    
    if abs(tie_diff) > 0.3:  # TIE差异足够大
        return True
    
    return False

def build_dpo_pairs_from_tie_results(tie_results: List[Dict], args) -> Tuple[List[Dict], Dict]:
    """从TIE结果构建DPO pairs"""
    print(f"\n=== 构建DPO Pairs ===")
    
    dpo_pairs = []
    validation_stats = {
        "total_processed": 0,
        "preferred_valid": 0,
        "dispreferred_valid": 0,
        "both_valid": 0,
        "dpo_pairs_generated": 0,
        "validation_failures": []
    }
    
    for i, result in enumerate(tie_results):
        validation_stats["total_processed"] += 1
        
        # 检查必要字段
        if not all(key in result for key in ['has_positive', 'has_negative', 'calculate_tie']):
            continue
            
        if not (result.get('has_positive') and result.get('has_negative') and result.get('calculate_tie')):
            continue
        
        # 提取TIE分数
        tie_pos = result.get('tie_positive', 0)
        tie_neg = result.get('tie_negative', 0)
        tie_diff = tie_pos - tie_neg
        
        # 计算TIE-ANKER权重
        tie_weight = calculate_tie_anker_weight(
            tie_diff,
            result.get('delta_pos', 0),
            result.get('delta_neg', 0),
            result.get('m_v', 0),
            result.get('m_n', 0),
            result.get('gamma', 0),
            args
        )
        
        # 使用数据驱动的验证函数
        preferred_valid, preferred_violations = validate_preferred_answer_data_driven(result, args)
        dispreferred_valid, dispreferred_conditions = validate_dispreferred_answer_data_driven(result, args)
        
        # 更新统计
        if preferred_valid:
            validation_stats["preferred_valid"] += 1
        if dispreferred_valid:
            validation_stats["dispreferred_valid"] += 1
        if preferred_valid and dispreferred_valid:
            validation_stats["both_valid"] += 1
        
        # 应用放宽的阈值过滤
        if preferred_valid and dispreferred_valid and apply_tie_anker_thresholds_relaxed(tie_weight, tie_diff, result, args):
            # 构建DPO对
            dpo_pair = {
                "id": result.get('id'),
                "case_id": result.get('case_id'),
                "question": result.get('question'),
                "image": f"{result.get('case_id', '')}/{result.get('case_id', '')}.jpg",
                "chosen": result.get('positive_answer'),
                "rejected": result.get('negative_answer'),
                "tie_weight": float(tie_weight),
                "tie_score": float(tie_diff),
                "validation_info": {
                    "preferred_valid": preferred_valid,
                    "dispreferred_valid": dispreferred_valid,
                    "preferred_violations": preferred_violations if not preferred_valid else [],
                    "dispreferred_reasons": dispreferred_conditions
                },
                "metadata": {
                    "tie_positive": result.get('tie_positive', 0),
                    "tie_negative": result.get('tie_negative', 0),
                    "delta_pos": result.get('delta_pos', 0),
                    "delta_neg": result.get('delta_neg', 0),
                    "m_v": result.get('m_v', 0),
                    "m_n": result.get('m_n', 0),
                    "gamma": result.get('gamma', 0),
                    "original_weighted_score": result.get('weighted_score', 0),
                    "dpo_weight": result.get('dpo_weight', 0)
                }
            }
            dpo_pairs.append(dpo_pair)
            validation_stats["dpo_pairs_generated"] += 1
            
            if len(dpo_pairs) <= 10:  # 只显示前10个
                print(f"✓ 生成DPO pair {len(dpo_pairs)}: case_id={result.get('case_id')}, "
                      f"tie_weight={tie_weight:.4f}, gamma={result.get('gamma', 0):.3f}")
        else:
            # 只记录前100个失败案例以避免文件过大
            if len(validation_stats["validation_failures"]) < 100:
                failure_reason = {
                    "case_id": result.get('case_id'),
                    "preferred_valid": preferred_valid,
                    "dispreferred_valid": dispreferred_valid,
                    "preferred_violations": preferred_violations if not preferred_valid else [],
                    "dispreferred_reasons": dispreferred_conditions if dispreferred_valid else [],
                    "gamma": result.get('gamma', 0),
                    "delta_pos": result.get('delta_pos', 0),
                    "tie_weight": tie_weight,
                    "tie_diff": tie_diff
                }
                validation_stats["validation_failures"].append(failure_reason)
    
    return dpo_pairs, validation_stats

def save_results(dpo_pairs: List[Dict], validation_stats: Dict, analysis: Dict, output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存DPO pairs
    dpo_output_path = os.path.join(output_dir, "dpo_pairs_from_tie_results.json")
    with open(dpo_output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
    
    # 保存详细报告
    report = {
        "generation_report": {
            "timestamp": datetime.now().isoformat(),
            "source_file": "/workspace/MMedPO/outputs/tie_results_1/tie_results.json",
            "data_driven_thresholds_used": True,
            "tie_data_analysis": analysis,
            "validation_statistics": validation_stats,
            "dpo_pairs_count": len(dpo_pairs),
            "generation_rate": f"{validation_stats['dpo_pairs_generated']}/{validation_stats['total_processed']} ({validation_stats['dpo_pairs_generated']/validation_stats['total_processed']*100:.2f}%)" if validation_stats['total_processed'] > 0 else "0%"
        }
    }
    
    report_path = os.path.join(output_dir, "dpo_generation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return dpo_output_path, report_path

def main():
    """主函数"""
    print("开始从TIE结果构建DPO pairs（使用数据驱动的阈值）...")
    
    # 读取TIE结果
    tie_results_path = "/workspace/MMedPO/outputs/tie_results_1/tie_results.json"
    
    if not os.path.exists(tie_results_path):
        print(f"✗ TIE结果文件不存在: {tie_results_path}")
        return False
    
    print(f"读取TIE结果文件: {tie_results_path}")
    
    # 读取JSON Lines格式的文件
    tie_results = []
    with open(tie_results_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    result = json.loads(line)
                    tie_results.append(result)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
    
    print(f"成功读取 {len(tie_results)} 条TIE结果")
    
    if not tie_results:
        print("✗ 没有有效的TIE结果数据")
        return False
    
    # 分析数据
    analysis = analyze_tie_data(tie_results)
    
    # 基于数据分布创建参数
    args = create_data_driven_args(tie_results)
    
    # 构建DPO pairs
    dpo_pairs, validation_stats = build_dpo_pairs_from_tie_results(tie_results, args)
    
    # 保存结果
    output_dir = "/workspace/MMedPO/outputs/dpo_pairs_from_tie"
    dpo_path, report_path = save_results(dpo_pairs, validation_stats, analysis, output_dir)
    
    # 输出总结
    print(f"\n=== 构建完成 ===")
    print(f"处理样本数: {validation_stats['total_processed']}")
    print(f"正例有效: {validation_stats['preferred_valid']}")
    print(f"反例有效: {validation_stats['dispreferred_valid']}")
    print(f"双重有效: {validation_stats['both_valid']}")
    print(f"生成DPO pairs: {validation_stats['dpo_pairs_generated']}")
    print(f"生成率: {validation_stats['dpo_pairs_generated']/validation_stats['total_processed']*100:.2f}%")
    print(f"\nDPO pairs保存到: {dpo_path}")
    print(f"详细报告保存到: {report_path}")
    
    return len(dpo_pairs) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)