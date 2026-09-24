#!/usr/bin/env python3
"""
测试DPO pairs构建逻辑是否符合理论规则

根据您提供的理论规则验证：
1. 正例选择规则：Δ⁺ > τ₊, γ > τᵧ, m_v > τᵥ, m_n ≈ 0
2. 反例选择规则：Δ⁻ ≥ 0 或 m_n > τₙ 或 γ ≤ 0
"""

import json
import numpy as np
from typing import Dict, List, Any
import argparse

class DPOPairsValidator:
    """DPO pairs构建逻辑验证器"""
    
    def __init__(self):
        self.validation_results = {
            "total_samples": 0,
            "valid_preferred": 0,
            "valid_dispreferred": 0,
            "valid_pairs": 0,
            "rule_violations": {
                "preferred": [],
                "dispreferred": []
            },
            "statistics": {}
        }
    
    def validate_preferred_selection(self, result: Dict[str, Any], args) -> Dict[str, Any]:
        """
        验证正例选择规则
        
        正例条件：
        1. Δ⁺ > τ₊ (前景贡献度大)
        2. γ > τᵧ (相对因果效应强)  
        3. m_v > τᵥ (区分度高)
        4. m_n ≈ 0 (背景泄漏可控)
        """
        validation = {
            "is_valid": True,
            "violations": [],
            "metrics": {}
        }
        
        # 提取指标
        delta_pos = result.get('delta_pos', 0)
        gamma = result.get('gamma', 0)
        m_v = result.get('m_v', 0)
        m_n = result.get('m_n', 0)
        
        validation["metrics"] = {
            "delta_pos": delta_pos,
            "gamma": gamma,
            "m_v": m_v,
            "m_n": m_n
        }
        
        # 规则1: Δ⁺ > τ₊
        if delta_pos <= args.tau_pos:
            validation["is_valid"] = False
            validation["violations"].append(f"delta_pos ({delta_pos:.4f}) <= tau_pos ({args.tau_pos})")
        
        # 规则2: γ > τᵧ_weak (相对因果效应)
        if gamma <= args.tau_gamma_weak:
            validation["is_valid"] = False
            validation["violations"].append(f"gamma ({gamma:.4f}) <= tau_gamma_weak ({args.tau_gamma_weak})")
        
        # 规则3: m_v > τᵥ (区分度)
        if m_v <= args.tau_v:
            validation["is_valid"] = False
            validation["violations"].append(f"m_v ({m_v:.4f}) <= tau_v ({args.tau_v})")
        
        # 规则4: m_n 背景泄漏可控 (简化为百分位阈值)
        tau_n_threshold = args.tau_n_percentile / 100.0
        if m_n > tau_n_threshold:
            validation["is_valid"] = False
            validation["violations"].append(f"m_n ({m_n:.4f}) > tau_n_threshold ({tau_n_threshold:.4f})")
        
        return validation
    
    def validate_dispreferred_selection(self, result: Dict[str, Any], args) -> Dict[str, Any]:
        """
        验证反例选择规则
        
        反例条件（满足任一即可）：
        1. Δ⁻ ≥ 0 (负贡献或无提升)
        2. m_n > τₙ (泄漏效应显著)
        3. γ ≤ 0 (净效应劣势)
        """
        validation = {
            "is_valid": False,  # 需要满足至少一个条件
            "satisfied_conditions": [],
            "metrics": {}
        }
        
        # 提取指标
        delta_neg = result.get('delta_neg', 0)
        gamma = result.get('gamma', 0)
        m_n = result.get('m_n', 0)
        
        validation["metrics"] = {
            "delta_neg": delta_neg,
            "gamma": gamma,
            "m_n": m_n
        }
        
        # 条件1: Δ⁻ ≥ 0
        if delta_neg >= 0:
            validation["is_valid"] = True
            validation["satisfied_conditions"].append(f"delta_neg ({delta_neg:.4f}) >= 0")
        
        # 条件2: m_n > τₙ (泄漏效应)
        tau_n_threshold = args.tau_n_percentile / 100.0
        if m_n > tau_n_threshold:
            validation["is_valid"] = True
            validation["satisfied_conditions"].append(f"m_n ({m_n:.4f}) > tau_n_threshold ({tau_n_threshold:.4f})")
        
        # 条件3: γ ≤ 0 (净效应劣势)
        if gamma <= 0:
            validation["is_valid"] = True
            validation["satisfied_conditions"].append(f"gamma ({gamma:.4f}) <= 0")
        
        return validation
    
    def analyze_current_implementation(self, result: Dict[str, Any], args) -> Dict[str, Any]:
        """分析当前实现的过滤逻辑"""
        current_logic = {
            "passes_current_filter": True,
            "filter_checks": []
        }
        
        # 当前实现的过滤条件
        tie_pos = result.get('tie_positive', 0)
        tie_neg = result.get('tie_negative', 0)
        tie_diff = tie_pos - tie_neg
        
        # 1. tie_diff < tau_pos
        if tie_diff < args.tau_pos:
            current_logic["passes_current_filter"] = False
            current_logic["filter_checks"].append(f"FAIL: tie_diff ({tie_diff:.4f}) < tau_pos ({args.tau_pos})")
        else:
            current_logic["filter_checks"].append(f"PASS: tie_diff ({tie_diff:.4f}) >= tau_pos ({args.tau_pos})")
        
        # 2. gamma范围检查
        gamma = result.get('gamma', 0)
        if gamma < args.tau_gamma_weak or gamma > args.tau_gamma_strong:
            current_logic["passes_current_filter"] = False
            current_logic["filter_checks"].append(f"FAIL: gamma ({gamma:.4f}) not in [{args.tau_gamma_weak}, {args.tau_gamma_strong}]")
        else:
            current_logic["filter_checks"].append(f"PASS: gamma ({gamma:.4f}) in range [{args.tau_gamma_weak}, {args.tau_gamma_strong}]")
        
        # 3. m_v阈值
        m_v = result.get('m_v', 0)
        if m_v < args.tau_v:
            current_logic["passes_current_filter"] = False
            current_logic["filter_checks"].append(f"FAIL: m_v ({m_v:.4f}) < tau_v ({args.tau_v})")
        else:
            current_logic["filter_checks"].append(f"PASS: m_v ({m_v:.4f}) >= tau_v ({args.tau_v})")
        
        # 4. m_n阈值
        m_n = result.get('m_n', 0)
        tau_n_threshold = args.tau_n_percentile / 100.0
        if m_n > tau_n_threshold:
            current_logic["passes_current_filter"] = False
            current_logic["filter_checks"].append(f"FAIL: m_n ({m_n:.4f}) > tau_n_threshold ({tau_n_threshold:.4f})")
        else:
            current_logic["filter_checks"].append(f"PASS: m_n ({m_n:.4f}) <= tau_n_threshold ({tau_n_threshold:.4f})")
        
        return current_logic
    
    def validate_sample(self, result: Dict[str, Any], args) -> Dict[str, Any]:
        """验证单个样本"""
        sample_validation = {
            "id": result.get('id', 'unknown'),
            "case_id": result.get('case_id', 'unknown'),
            "preferred_validation": self.validate_preferred_selection(result, args),
            "dispreferred_validation": self.validate_dispreferred_selection(result, args),
            "current_implementation": self.analyze_current_implementation(result, args)
        }
        
        # 更新统计
        self.validation_results["total_samples"] += 1
        
        if sample_validation["preferred_validation"]["is_valid"]:
            self.validation_results["valid_preferred"] += 1
        else:
            self.validation_results["rule_violations"]["preferred"].append({
                "id": sample_validation["id"],
                "violations": sample_validation["preferred_validation"]["violations"]
            })
        
        if sample_validation["dispreferred_validation"]["is_valid"]:
            self.validation_results["valid_dispreferred"] += 1
        else:
            self.validation_results["rule_violations"]["dispreferred"].append({
                "id": sample_validation["id"],
                "satisfied_conditions": sample_validation["dispreferred_validation"]["satisfied_conditions"]
            })
        
        # 如果正例和反例都有效，则为有效pair
        if (sample_validation["preferred_validation"]["is_valid"] and 
            sample_validation["dispreferred_validation"]["is_valid"]):
            self.validation_results["valid_pairs"] += 1
        
        return sample_validation
    
    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        total = self.validation_results["total_samples"]
        if total == 0:
            return {"error": "No samples to validate"}
        
        report = {
            "summary": {
                "total_samples": total,
                "valid_preferred_rate": self.validation_results["valid_preferred"] / total,
                "valid_dispreferred_rate": self.validation_results["valid_dispreferred"] / total,
                "valid_pairs_rate": self.validation_results["valid_pairs"] / total,
                "preferred_violations": len(self.validation_results["rule_violations"]["preferred"]),
                "dispreferred_violations": len(self.validation_results["rule_violations"]["dispreferred"])
            },
            "detailed_violations": self.validation_results["rule_violations"],
            "recommendations": []
        }
        
        # 生成建议
        if report["summary"]["valid_preferred_rate"] < 0.5:
            report["recommendations"].append("正例选择规则过于严格，考虑调整阈值参数")
        
        if report["summary"]["valid_dispreferred_rate"] < 0.5:
            report["recommendations"].append("反例选择规则可能需要调整，很多样本不满足反例条件")
        
        if report["summary"]["valid_pairs_rate"] < 0.3:
            report["recommendations"].append("整体pair构建成功率较低，建议重新审视阈值设置")
        
        return report

def create_mock_results(num_samples=10):
    """创建模拟测试数据"""
    np.random.seed(42)
    results = []
    
    for i in range(num_samples):
        # 模拟不同类型的样本
        if i < 3:  # 理想的正例样本
            result = {
                "id": f"ideal_{i}",
                "case_id": f"case_{i}",
                "calculate_tie": True,
                "delta_pos": np.random.uniform(0.5, 2.0),  # 高正向贡献
                "delta_neg": np.random.uniform(-0.5, 0.2),  # 低或负贡献
                "gamma": np.random.uniform(0.3, 1.5),  # 正向净效应
                "m_v": np.random.uniform(0.6, 1.2),  # 高区分度
                "m_n": np.random.uniform(-0.2, 0.3),  # 低背景泄漏
                "tie_positive": np.random.uniform(1.0, 3.0),
                "tie_negative": np.random.uniform(-1.0, 0.5),
                "positive_answer": "Correct medical diagnosis",
                "negative_answer": "Incorrect diagnosis"
            }
        elif i < 6:  # 边界情况
            result = {
                "id": f"boundary_{i}",
                "case_id": f"case_{i}",
                "calculate_tie": True,
                "delta_pos": np.random.uniform(0.05, 0.15),  # 接近阈值
                "delta_neg": np.random.uniform(-0.1, 0.1),
                "gamma": np.random.uniform(0.05, 0.15),  # 接近阈值
                "m_v": np.random.uniform(0.4, 0.6),  # 接近阈值
                "m_n": np.random.uniform(0.6, 0.8),  # 较高背景泄漏
                "tie_positive": np.random.uniform(0.5, 1.5),
                "tie_negative": np.random.uniform(0.2, 1.0),
                "positive_answer": "Borderline diagnosis",
                "negative_answer": "Alternative diagnosis"
            }
        else:  # 不理想的样本
            result = {
                "id": f"poor_{i}",
                "case_id": f"case_{i}",
                "calculate_tie": True,
                "delta_pos": np.random.uniform(-0.5, 0.05),  # 低或负贡献
                "delta_neg": np.random.uniform(0.2, 0.8),  # 高正贡献（不理想）
                "gamma": np.random.uniform(-0.5, 0.05),  # 负或低净效应
                "m_v": np.random.uniform(0.1, 0.4),  # 低区分度
                "m_n": np.random.uniform(0.8, 1.2),  # 高背景泄漏
                "tie_positive": np.random.uniform(-0.5, 0.5),
                "tie_negative": np.random.uniform(0.5, 2.0),
                "positive_answer": "Uncertain diagnosis",
                "negative_answer": "Confident wrong diagnosis"
            }
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test DPO pairs construction logic")
    parser.add_argument("--input-file", type=str, help="Input JSON file with results")
    parser.add_argument("--output-file", type=str, default="dpo_pairs_validation_report.json", 
                       help="Output validation report file")
    parser.add_argument("--use-mock-data", action="store_true", 
                       help="Use mock data for testing")
    
    # 阈值参数（与主脚本保持一致）
    parser.add_argument("--tau-pos", type=float, default=0.1, help="Positive threshold")
    parser.add_argument("--tau-gamma-strong", type=float, default=0.5, help="Strong gamma threshold")
    parser.add_argument("--tau-gamma-weak", type=float, default=0.1, help="Weak gamma threshold")
    parser.add_argument("--tau-v", type=float, default=0.5, help="V threshold")
    parser.add_argument("--tau-n-percentile", type=float, default=75, help="N percentile threshold")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.use_mock_data or not args.input_file:
        print("Using mock data for testing...")
        results = create_mock_results(20)
    else:
        print(f"Loading data from {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    # 验证
    validator = DPOPairsValidator()
    sample_validations = []
    
    print(f"Validating {len(results)} samples...")
    for result in results:
        if result.get('calculate_tie', False):
            validation = validator.validate_sample(result, args)
            sample_validations.append(validation)
    
    # 生成报告
    report = validator.generate_report()
    
    # 输出结果
    full_report = {
        "validation_summary": report,
        "sample_validations": sample_validations[:5],  # 只保存前5个详细样本
        "parameters": {
            "tau_pos": args.tau_pos,
            "tau_gamma_weak": args.tau_gamma_weak,
            "tau_gamma_strong": args.tau_gamma_strong,
            "tau_v": args.tau_v,
            "tau_n_percentile": args.tau_n_percentile
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n=== DPO Pairs Construction Validation Report ===")
    print(f"Total samples: {report['summary']['total_samples']}")
    print(f"Valid preferred rate: {report['summary']['valid_preferred_rate']:.2%}")
    print(f"Valid dispreferred rate: {report['summary']['valid_dispreferred_rate']:.2%}")
    print(f"Valid pairs rate: {report['summary']['valid_pairs_rate']:.2%}")
    print(f"Preferred violations: {report['summary']['preferred_violations']}")
    print(f"Dispreferred violations: {report['summary']['dispreferred_violations']}")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"- {rec}")
    
    print(f"\nDetailed report saved to: {args.output_file}")

if __name__ == "__main__":
    main()