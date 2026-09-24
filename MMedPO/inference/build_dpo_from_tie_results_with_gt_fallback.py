#!/usr/bin/env python3
"""
从现有TIE结果构建DPO pairs - 增强版本
使用优化后的TIE-ANKER实现，基于实际数据分布调整验证逻辑
新增功能：当positive和negative答案都表现不佳时，使用gt_answer作为chosen的fallback机制
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
    
    # GT fallback参数
    args.enable_gt_fallback = True  # 启用GT fallback
    args.gt_fallback_weight = 0.3   # GT fallback的默认权重
    
    print(f"数据驱动的阈值设定:")
    print(f"  tau_pos: {args.tau_pos:.3f}")
    print(f"  tau_gamma_weak: {args.tau_gamma_weak:.3f}")
    print(f"  tau_gamma_strong: {args.tau_gamma_strong:.3f}")
    print(f"  tau_v: {args.tau_v:.3f}")
    print(f"  tau_n_leak: {args.tau_n_leak:.3f}")
    print(f"  GT fallback enabled: {args.enable_gt_fallback}")
    print(f"  GT fallback weight: {args.gt_fallback_weight}")
    
    return args

def analyze_tie_data(tie_results: List[Dict]) -> Dict[str, Any]:
    """分析TIE结果数据"""
    print(f"\n=== 分析TIE数据 ===")
    print(f"总样本数: {len(tie_results)}")
    
    # 统计基本信息
    has_positive = sum(1 for r in tie_results if r.get('has_positive', False))
    has_negative = sum(1 for r in tie_results if r.get('has_negative', False))
    calculate_tie = sum(1 for r in tie_results if r.get('calculate_tie', False))
    has_gt = sum(1 for r in tie_results if r.get('gt_answer'))
    
    print(f"有正例答案: {has_positive}")
    print(f"有反例答案: {has_negative}")
    print(f"计算TIE: {calculate_tie}")
    print(f"有GT答案: {has_gt}")
    
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
        "has_gt": has_gt,
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
    
    # 条件3: γ < τᵧ_strong (相对效应较弱)
    if gamma < args.tau_gamma_strong:
        conditions.append(f"相对效应较弱: γ({gamma:.4f}) < τᵧ_strong({args.tau_gamma_strong:.4f})")
        return True, conditions
    
    return False, conditions

def apply_tie_anker_thresholds_relaxed(tie_weight: float, tie_diff: float, result: Dict, args) -> bool:
    """
    应用放宽的TIE-ANKER阈值过滤
    """
    # 基本阈值检查
    if abs(tie_diff) < 0.05:  # 差异太小
        return False
    
    # 放宽的条件：满足任一条件即可
    delta_pos = result.get('delta_pos', 0)
    gamma = result.get('gamma', 0)
    
    # 条件1: 前景贡献度足够
    if delta_pos > args.tau_pos:
        return True
    
    # 条件2: 相对因果效应足够强
    if gamma > args.tau_gamma_weak:
        return True
    
    # 条件3: TIE差异显著
    if abs(tie_diff) > 0.3:
        return True
    
    return False

def check_gt_fallback_eligibility(result: Dict, args, preferred_valid: bool, dispreferred_valid: bool) -> Tuple[bool, str]:
    """
    检查是否符合GT fallback条件
    返回: (是否符合条件, 选择的rejected答案类型)
    """
    if not args.enable_gt_fallback:
        return False, ""
    
    # 必须有GT答案
    if not result.get('gt_answer'):
        return False, ""
    
    # GT答案不能为空或过短
    gt_answer = result.get('gt_answer', '').strip()
    if len(gt_answer) < 2:
        return False, ""
    
    # 只有在两个答案都验证失败时才使用GT fallback
    if preferred_valid or dispreferred_valid:
        return False, ""
    
    # 选择rejected答案：优先选择negative答案，如果没有则选择positive答案
    positive_answer = result.get('positive_answer', '').strip()
    negative_answer = result.get('negative_answer', '').strip()
    
    if negative_answer and len(negative_answer) >= 2:
        return True, "negative"
    elif positive_answer and len(positive_answer) >= 2:
        return True, "positive"
    else:
        return False, ""

def build_dpo_pairs_with_gt_fallback(tie_results: List[Dict], args) -> Tuple[List[Dict], Dict]:
    """
    构建DPO pairs，包含GT fallback机制
    """
    dpo_pairs = []
    validation_stats = {
        "total_processed": len(tie_results),
        "preferred_valid": 0,
        "dispreferred_valid": 0,
        "both_valid": 0,
        "gt_fallback_used": 0,
        "dpo_pairs_generated": 0,
        "validation_failures": []
    }
    
    print(f"\n=== 构建DPO pairs ===")
    
    for i, result in enumerate(tie_results):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(tie_results)}")
        
        # 计算TIE权重
        try:
            tie_diff = result.get('tie_positive', 0) - result.get('tie_negative', 0)
            tie_weight = calculate_tie_anker_weight(
                tie_diff,
                result.get('delta_pos', 0),
                result.get('delta_neg', 0),
                result.get('m_v', 0),
                result.get('m_n', 0),
                result.get('gamma', 0),
                args
            )
        except Exception as e:
            print(f"计算TIE权重失败 {result.get('case_id', 'unknown')}: {e}")
            continue
        
        # 验证preferred和dispreferred答案
        preferred_valid, preferred_violations = validate_preferred_answer_data_driven(result, args)
        dispreferred_valid, dispreferred_conditions = validate_dispreferred_answer_data_driven(result, args)
        
        # 更新统计
        if preferred_valid:
            validation_stats["preferred_valid"] += 1
        if dispreferred_valid:
            validation_stats["dispreferred_valid"] += 1
        if preferred_valid and dispreferred_valid:
            validation_stats["both_valid"] += 1
        
        # 尝试构建标准DPO pair
        if preferred_valid and dispreferred_valid and apply_tie_anker_thresholds_relaxed(tie_weight, tie_diff, result, args):
            # 构建标准DPO对
            dpo_pair = {
                "id": result.get('id'),
                "case_id": result.get('case_id'),
                "question": result.get('question'),
                "image": f"{result.get('case_id', '')}/{result.get('case_id', '')}.jpg",
                "chosen": result.get('positive_answer'),
                "rejected": result.get('negative_answer'),
                "tie_weight": float(tie_weight),
                "tie_score": float(tie_diff),
                "pair_type": "standard",  # 标记为标准pair
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
                print(f"✓ 生成标准DPO pair {len(dpo_pairs)}: case_id={result.get('case_id')}, "
                      f"tie_weight={tie_weight:.4f}, gamma={result.get('gamma', 0):.3f}")
        
        # 尝试GT fallback
        else:
            gt_eligible, rejected_type = check_gt_fallback_eligibility(result, args, preferred_valid, dispreferred_valid)
            if gt_eligible:
                # 根据rejected_type选择rejected答案
                rejected_answer = result.get('negative_answer') if rejected_type == "negative" else result.get('positive_answer')
                
                dpo_pair = {
                    "id": result.get('id'),
                    "case_id": result.get('case_id'),
                    "question": result.get('question'),
                    "image": f"{result.get('case_id', '')}/{result.get('case_id', '')}.jpg",
                    "chosen": result.get('gt_answer'),  # 使用GT答案作为chosen
                    "rejected": rejected_answer,
                    "tie_weight": float(args.gt_fallback_weight),  # 使用默认权重
                    "tie_score": 1.0,  # GT默认优于其他答案
                    "pair_type": "gt_fallback",  # 标记为GT fallback pair
                    "validation_info": {
                        "preferred_valid": preferred_valid,
                        "dispreferred_valid": dispreferred_valid,
                        "preferred_violations": preferred_violations,
                        "dispreferred_reasons": dispreferred_conditions,
                        "gt_fallback_reason": f"两个答案都验证失败，使用GT作为chosen，{rejected_type}答案作为rejected",
                        "rejected_type": rejected_type
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
                        "dpo_weight": result.get('dpo_weight', 0),
                        "original_positive_answer": result.get('positive_answer'),
                        "original_negative_answer": result.get('negative_answer')
                    }
                }
                dpo_pairs.append(dpo_pair)
                validation_stats["dpo_pairs_generated"] += 1
                validation_stats["gt_fallback_used"] += 1
                
                if validation_stats["gt_fallback_used"] <= 10:  # 只显示前10个GT fallback
                    print(f"✓ 生成GT fallback DPO pair {len(dpo_pairs)}: case_id={result.get('case_id')}, "
                          f"GT作为chosen，{rejected_type}答案作为rejected")
            else:
                # 只记录前100个失败案例以避免文件过大
                if len(validation_stats["validation_failures"]) < 100:
                    failure_reason = {
                        "case_id": result.get('case_id'),
                        "preferred_valid": preferred_valid,
                        "dispreferred_valid": dispreferred_valid,
                        "preferred_violations": preferred_violations if not preferred_valid else [],
                        "dispreferred_reasons": dispreferred_conditions if dispreferred_valid else [],
                        "gt_fallback_eligible": check_gt_fallback_eligibility(result, args, preferred_valid, dispreferred_valid)[0],
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
    dpo_output_path = os.path.join(output_dir, "dpo_pairs_from_tie_results_with_gt_fallback.json")
    with open(dpo_output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
    
    # 保存详细报告
    report = {
        "generation_report": {
            "timestamp": datetime.now().isoformat(),
            "source_file": "/workspace/MMedPO/outputs/tie_results_1/tie_results.json",
            "data_driven_thresholds_used": True,
            "gt_fallback_enabled": True,
            "tie_data_analysis": analysis,
            "validation_statistics": validation_stats
        }
    }
    
    report_path = os.path.join(output_dir, "dpo_generation_report_with_gt_fallback.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 结果保存 ===")
    print(f"DPO pairs保存至: {dpo_output_path}")
    print(f"详细报告保存至: {report_path}")
    
    # 统计信息
    standard_pairs = sum(1 for pair in dpo_pairs if pair.get('pair_type') == 'standard')
    gt_fallback_pairs = sum(1 for pair in dpo_pairs if pair.get('pair_type') == 'gt_fallback')
    
    print(f"\n=== 生成统计 ===")
    print(f"总DPO pairs: {len(dpo_pairs)}")
    print(f"标准pairs: {standard_pairs}")
    print(f"GT fallback pairs: {gt_fallback_pairs}")
    print(f"生成率: {len(dpo_pairs)/validation_stats['total_processed']*100:.2f}%")
    print(f"GT fallback使用率: {gt_fallback_pairs/validation_stats['total_processed']*100:.2f}%")

def main():
    """主函数"""
    # 输入输出路径
    input_file = "/workspace/MMedPO/outputs/tie_results_1/tie_results.json"
    output_dir = "/workspace/MMedPO/outputs/dpo_pairs_from_tie_with_gt_fallback"
    
    print("=== 从TIE结果构建DPO pairs (增强版本 - 支持GT fallback) ===")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # 读取TIE结果 (JSONL格式)
    print(f"\n=== 读取TIE结果 ===")
    tie_results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tie_results.append(json.loads(line))
    print(f"读取到 {len(tie_results)} 个TIE结果")
    
    # 分析数据并创建参数
    analysis = analyze_tie_data(tie_results)
    args = create_data_driven_args(tie_results)
    
    # 构建DPO pairs
    dpo_pairs, validation_stats = build_dpo_pairs_with_gt_fallback(tie_results, args)
    
    # 保存结果
    save_results(dpo_pairs, validation_stats, analysis, output_dir)
    
    print(f"\n=== 完成 ===")
    print(f"成功生成 {len(dpo_pairs)} 个DPO pairs")

if __name__ == "__main__":
    main()