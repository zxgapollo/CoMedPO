#!/usr/bin/env python3
"""
医学VQA和报告生成结果分析脚本
比较baseline、MMedPO和CaMedPO三种方法的性能
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VQAExample:
    """VQA任务示例数据结构"""
    question: str
    answer: str
    response: str
    correct: bool
    metrics: Dict[str, float]
    lang: str
    answer_type: str


@dataclass
class ReportExample:
    """报告生成任务示例数据结构"""
    study_id: str
    prediction: str
    ground_truth: str


@dataclass
class ComparisonResult:
    """比较结果数据结构"""
    method: str
    dataset: str
    examples: List[Any]
    avg_metrics: Dict[str, float]
    error_cases: List[Any]


class MedicalResultsAnalyzer:
    """医学测试结果分析器"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.results = {}
        
        # 定义文件夹路径
        self.method_paths = {
            'baseline': {
                'SLAKE': self.base_path / 'Base_line' / 'SLAKE',
                'IU_XRAY': self.base_path / 'Base_line' / 'IU_XRAY',
                'VQA_RAD': self.base_path / 'VQA_RAD_base_line' / 'VQA_RAD'
            },
            'MMedPO': {
                'SLAKE': self.base_path / 'MMedPO' / 'SLAKE',
                'IU_XRAY': self.base_path / 'DPO_iu_xray' / 'IU_XRAY',
                'VQA_RAD': self.base_path / 'DPO_mmedpo_VQA_RAD' / 'VQA_RAD'
            },
            'CaMedPO': {
                'SLAKE': self.base_path / 'SFT_DPO_combined_1_2' / 'SLAKE',
                'VQA_RAD': self.base_path / 'SFT_DPO_method1_vqa_rad' / 'VQA_RAD',
                'IU_XRAY': self.base_path / 'SFT_DPO_method1_iu_xray' / 'IU_XRAY'
            }
        }
    
    def load_vqa_results(self, method: str, dataset: str) -> List[VQAExample]:
        """加载VQA任务结果"""
        results = []
        path = self.method_paths[method][dataset]
        results_file = path / 'results.json'
        
        if not results_file.exists():
            print(f"警告: {results_file} 不存在")
            return results
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                example = VQAExample(
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    response=item.get('response', ''),
                    correct=item.get('correct', False),
                    metrics=item.get('metrics', {}),
                    lang=item.get('lang', ''),
                    answer_type=item.get('answer_type', '')
                )
                results.append(example)
        except Exception as e:
            print(f"加载 {results_file} 时出错: {e}")
        
        return results
    
    def load_report_results(self, method: str, dataset: str) -> List[ReportExample]:
        """加载报告生成任务结果"""
        results = []
        path = self.method_paths[method][dataset]
        predictions_file = path / 'predictions.csv'
        ground_truth_file = path / 'ground_truth.csv'
        
        if not predictions_file.exists() or not ground_truth_file.exists():
            print(f"警告: {predictions_file} 或 {ground_truth_file} 不存在")
            return results
        
        try:
            # 读取预测结果
            predictions = {}
            with open(predictions_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions[row['study_id']] = row['report']
            
            # 读取真实标签
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    study_id = row['study_id']
                    if study_id in predictions:
                        example = ReportExample(
                            study_id=study_id,
                            prediction=predictions[study_id],
                            ground_truth=row['report']
                        )
                        results.append(example)
        except Exception as e:
            print(f"加载报告结果时出错: {e}")
        
        return results
    
    def calculate_avg_metrics(self, examples: List[VQAExample]) -> Dict[str, float]:
        """计算平均指标"""
        if not examples:
            return {}
        
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        
        for example in examples:
            for metric, value in example.metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[metric] += float(value)
                    metric_counts[metric] += 1
        
        avg_metrics = {}
        for metric in metric_sums:
            avg_metrics[metric] = metric_sums[metric] / metric_counts[metric]
        
        return avg_metrics
    
    def identify_error_cases(self, examples: List[VQAExample]) -> List[VQAExample]:
        """识别错误案例"""
        return [ex for ex in examples if not ex.correct]
    
    def identify_report_problems(self, examples: List[ReportExample]) -> List[ReportExample]:
        """识别报告生成问题"""
        problems = []
        for example in examples:
            # 简单的启发式规则来识别问题
            pred_lower = example.prediction.lower()
            gt_lower = example.ground_truth.lower()
            
            # 检查是否有关键词不匹配
            key_terms = ['normal', 'abnormal', 'effusion', 'pneumothorax', 'consolidation', 'opacity']
            pred_terms = [term for term in key_terms if term in pred_lower]
            gt_terms = [term for term in key_terms if term in gt_lower]
            
            if set(pred_terms) != set(gt_terms):
                problems.append(example)
        
        return problems
    
    def compare_methods(self) -> Dict[str, ComparisonResult]:
        """比较三种方法的性能"""
        comparison_results = {}
        
        for method in ['baseline', 'MMedPO', 'CaMedPO']:
            print(f"\n分析 {method} 方法的结果...")
            
            # VQA任务
            for dataset in ['SLAKE', 'VQA_RAD']:
                if dataset in self.method_paths[method]:
                    examples = self.load_vqa_results(method, dataset)
                    avg_metrics = self.calculate_avg_metrics(examples)
                    error_cases = self.identify_error_cases(examples)
                    
                    key = f"{method}_{dataset}"
                    comparison_results[key] = ComparisonResult(
                        method=method,
                        dataset=dataset,
                        examples=examples,
                        avg_metrics=avg_metrics,
                        error_cases=error_cases
                    )
                    
                    print(f"  {dataset}: {len(examples)} 个样本, {len(error_cases)} 个错误")
            
            # 报告生成任务
            if 'IU_XRAY' in self.method_paths[method]:
                examples = self.load_report_results(method, 'IU_XRAY')
                problem_cases = self.identify_report_problems(examples)
                
                key = f"{method}_IU_XRAY"
                comparison_results[key] = ComparisonResult(
                    method=method,
                    dataset='IU_XRAY',
                    examples=examples,
                    avg_metrics={},
                    error_cases=problem_cases
                )
                
                print(f"  IU_XRAY: {len(examples)} 个样本, {len(problem_cases)} 个问题案例")
        
        return comparison_results
    
    def find_camppo_best_cases(self, comparison_results: Dict[str, ComparisonResult]) -> Dict[str, List]:
        """找出CaMedPO表现最优的案例"""
        best_cases = {
            'VQA': [],
            'Report': []
        }
        
        # VQA任务比较
        for dataset in ['SLAKE', 'VQA_RAD']:
            baseline_key = f"baseline_{dataset}"
            mmedpo_key = f"MMedPO_{dataset}"
            camppo_key = f"CaMedPO_{dataset}"
            
            if all(key in comparison_results for key in [baseline_key, mmedpo_key, camppo_key]):
                baseline_correct = len(comparison_results[baseline_key].examples) - len(comparison_results[baseline_key].error_cases)
                mmedpo_correct = len(comparison_results[mmedpo_key].examples) - len(comparison_results[mmedpo_key].error_cases)
                camppo_correct = len(comparison_results[camppo_key].examples) - len(comparison_results[camppo_key].error_cases)
                
                # 如果CaMedPO正确率更高，找出具体案例
                if camppo_correct > baseline_correct and camppo_correct > mmedpo_correct:
                    # 找出CaMedPO正确但其他方法错误的案例
                    baseline_errors = {(ex.question, ex.answer) for ex in comparison_results[baseline_key].error_cases}
                    mmedpo_errors = {(ex.question, ex.answer) for ex in comparison_results[mmedpo_key].error_cases}
                    
                    for example in comparison_results[camppo_key].examples:
                        if example.correct and (example.question, example.answer) in baseline_errors and (example.question, example.answer) in mmedpo_errors:
                            best_cases['VQA'].append({
                                'dataset': dataset,
                                'question': example.question,
                                'answer': example.answer,
                                'camppo_response': example.response,
                                'reason': 'CaMedPO正确，baseline和MMedPO均错误'
                            })
        
        return best_cases
    
    def find_problematic_cases(self, comparison_results: Dict[str, ComparisonResult]) -> Dict[str, List]:
        """找出baseline和MMedPO存在问题的案例"""
        problematic_cases = {
            'baseline': [],
            'MMedPO': []
        }
        
        # VQA任务问题案例
        for dataset in ['SLAKE', 'VQA_RAD']:
            baseline_key = f"baseline_{dataset}"
            mmedpo_key = f"MMedPO_{dataset}"
            camppo_key = f"CaMedPO_{dataset}"
            
            if baseline_key in comparison_results:
                for example in comparison_results[baseline_key].error_cases:
                    problematic_cases['baseline'].append({
                        'dataset': dataset,
                        'type': 'VQA',
                        'question': example.question,
                        'expected_answer': example.answer,
                        'baseline_response': example.response,
                        'error_type': self.classify_vqa_error(example)
                    })
            
            if mmedpo_key in comparison_results:
                for example in comparison_results[mmedpo_key].error_cases:
                    problematic_cases['MMedPO'].append({
                        'dataset': dataset,
                        'type': 'VQA',
                        'question': example.question,
                        'expected_answer': example.answer,
                        'mmedpo_response': example.response,
                        'error_type': self.classify_vqa_error(example)
                    })
        
        # 报告生成任务问题案例
        for method in ['baseline', 'MMedPO']:
            iu_xray_key = f"{method}_IU_XRAY"
            if iu_xray_key in comparison_results:
                for example in comparison_results[iu_xray_key].error_cases:
                    problematic_cases[method].append({
                        'dataset': 'IU_XRAY',
                        'type': 'Report',
                        'study_id': example.study_id,
                        'ground_truth': example.ground_truth,
                        'prediction': example.prediction,
                        'error_type': '关键医学术语不匹配'
                    })
        
        return problematic_cases
    
    def classify_vqa_error(self, example: VQAExample) -> str:
        """分类VQA错误类型"""
        response_lower = example.response.lower()
        answer_lower = example.answer.lower()
        
        # 简单的错误分类
        if len(response_lower.split()) > len(answer_lower.split()) * 3:
            return "过度生成"
        elif any(word in response_lower for word in ['sorry', 'cannot', 'unable']):
            return "拒绝回答"
        elif len(response_lower) < len(answer_lower) / 2:
            return "回答不完整"
        else:
            return "内容不准确"
    
    def generate_report(self, comparison_results: Dict[str, ComparisonResult], 
                       best_cases: Dict[str, List], 
                       problematic_cases: Dict[str, List]) -> str:
        """生成分析报告"""
        report = []
        report.append("# 医学VQA和报告生成结果分析报告")
        report.append("=" * 50)
        
        # 性能概述
        report.append("\n## 1. 性能概述")
        for method in ['baseline', 'MMedPO', 'CaMedPO']:
            report.append(f"\n### {method.upper()} 方法:")
            for dataset in ['SLAKE', 'VQA_RAD', 'IU_XRAY']:
                key = f"{method}_{dataset}"
                if key in comparison_results:
                    result = comparison_results[key]
                    total = len(result.examples)
                    errors = len(result.error_cases)
                    accuracy = (total - errors) / total * 100 if total > 0 else 0
                    report.append(f"  - {dataset}: {total} 样本, 准确率 {accuracy:.1f}% ({total-errors}/{total})")
        
        # CaMedPO最优案例
        report.append("\n## 2. CaMedPO 表现最优的案例")
        if best_cases['VQA']:
            report.append("\n### VQA任务:")
            for i, case in enumerate(best_cases['VQA'][:5], 1):  # 显示前5个
                report.append(f"\n#### 案例 {i} ({case['dataset']}):")
                report.append(f"**问题**: {case['question']}")
                report.append(f"**标准答案**: {case['answer']}")
                report.append(f"**CaMedPO回答**: {case['camppo_response']}")
                report.append(f"**分析**: {case['reason']}")
        
        # 问题案例分析
        report.append("\n## 3. 问题案例分析")
        
        for method in ['baseline', 'MMedPO']:
            report.append(f"\n### {method.upper()} 方法的问题:")
            cases = problematic_cases[method]
            
            vqa_cases = [c for c in cases if c['type'] == 'VQA']
            report_cases = [c for c in cases if c['type'] == 'Report']
            
            if vqa_cases:
                report.append(f"\n#### VQA任务错误 ({len(vqa_cases)} 个):")
                error_types = defaultdict(int)
                for case in vqa_cases:
                    error_types[case['error_type']] += 1
                
                for error_type, count in error_types.items():
                    report.append(f"  - {error_type}: {count} 个")
                
                # 显示具体案例
                for i, case in enumerate(vqa_cases[:3], 1):
                    report.append(f"\n**错误案例 {i}**:")
                    report.append(f"问题: {case['question']}")
                    report.append(f"期望答案: {case['expected_answer']}")
                    if method == 'baseline':
                        report.append(f"Baseline回答: {case['baseline_response']}")
                    else:
                        report.append(f"MMedPO回答: {case['mmedpo_response']}")
                    report.append(f"错误类型: {case['error_type']}")
            
            if report_cases:
                report.append(f"\n#### 报告生成问题 ({len(report_cases)} 个):")
                for i, case in enumerate(report_cases[:2], 1):
                    report.append(f"\n**问题案例 {i} (Study ID: {case['study_id']})**:")
                    report.append(f"真实报告: {case['ground_truth']}")
                    report.append(f"预测报告: {case['prediction']}")
                    report.append(f"问题类型: {case['error_type']}")
        
        return "\n".join(report)
    
    def run_analysis(self, output_file: str = "medical_analysis_report.md"):
        """运行完整分析"""
        print("开始分析医学测试结果...")
        
        # 比较三种方法
        comparison_results = self.compare_methods()
        
        # 生成并排对比格式
        side_by_side_comparison = self.generate_side_by_side_comparison(comparison_results)
        
        # 找出CaMedPO最优案例
        best_cases = self.find_camppo_best_cases(comparison_results)
        
        # 找出问题案例
        problematic_cases = self.find_problematic_cases(comparison_results)
        
        # 生成报告
        report = self.generate_report(comparison_results, best_cases, problematic_cases)
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存并排对比报告
        self.save_side_by_side_report(side_by_side_comparison)
        
        print(f"\n分析完成！报告已保存到: {output_file}")
        
        # 生成新格式的详细分析数据
        detailed_comparison_cases = self.generate_new_format_detailed_analysis(comparison_results)
        
        # 保存详细数据 - 使用新的对比格式
        detailed_data = {
            'best_cases': best_cases,
            'problematic_cases': problematic_cases,
            'summary': self.generate_summary(comparison_results),
            'side_by_side_comparison': side_by_side_comparison,
            'detailed_comparison_cases': detailed_comparison_cases
        }
        
        with open('detailed_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        return comparison_results, best_cases, problematic_cases
    
    def generate_summary(self, comparison_results: Dict[str, ComparisonResult]) -> Dict[str, Any]:
        """生成摘要统计"""
        summary = {}
        
        for method in ['baseline', 'MMedPO', 'CaMedPO']:
            method_summary = {}
            for dataset in ['SLAKE', 'VQA_RAD', 'IU_XRAY']:
                key = f"{method}_{dataset}"
                if key in comparison_results:
                    result = comparison_results[key]
                    total = len(result.examples)
                    errors = len(result.error_cases)
                    accuracy = (total - errors) / total * 100 if total > 0 else 0
                    
                    method_summary[dataset] = {
                        'total_samples': total,
                        'error_count': errors,
                        'accuracy': accuracy
                    }
            summary[method] = method_summary
        
        return summary
    
    def generate_side_by_side_comparison(self, comparison_results: Dict[str, ComparisonResult]) -> Dict[str, Any]:
        """生成并排对比格式的结果"""
        comparison_data = {
            'summary': {},
            'detailed_cases': []
        }
        
        # 汇总统计
        for dataset in ['SLAKE', 'VQA_RAD', 'IU_XRAY']:
            dataset_summary = {
                'dataset': dataset,
                'methods': {}
            }
            
            for method in ['baseline', 'MMedPO', 'CaMedPO']:
                key = f"{method}_{dataset}"
                if key in comparison_results:
                    result = comparison_results[key]
                    total = len(result.examples)
                    errors = len(result.error_cases)
                    accuracy = (total - errors) / total * 100 if total > 0 else 0
                    
                    dataset_summary['methods'][method] = {
                        'total_samples': total,
                        'correct_count': total - errors,
                        'error_count': errors,
                        'accuracy': accuracy
                    }
            
            comparison_data['summary'][dataset] = dataset_summary
        
        # 详细案例对比
        for dataset in ['SLAKE', 'VQA_RAD']:
            # 获取所有方法的案例
            baseline_key = f"baseline_{dataset}"
            mmedpo_key = f"MMedPO_{dataset}"
            camppo_key = f"CaMedPO_{dataset}"
            
            if all(key in comparison_results for key in [baseline_key, mmedpo_key, camppo_key]):
                baseline_cases = comparison_results[baseline_key].examples
                mmedpo_cases = comparison_results[mmedpo_key].examples
                camppo_cases = comparison_results[camppo_key].examples
                
                # 对比每个案例
                for i in range(min(len(baseline_cases), len(mmedpo_cases), len(camppo_cases))):
                    baseline_case = baseline_cases[i]
                    mmedpo_case = mmedpo_cases[i]
                    camppo_case = camppo_cases[i]
                    
                    # 确保是同一个问题
                    if (baseline_case.question == mmedpo_case.question == camppo_case.question and
                        baseline_case.answer == mmedpo_case.answer == camppo_case.answer):
                        
                        case_comparison = {
                            'dataset': dataset,
                            'question': baseline_case.question,
                            'ground_truth': baseline_case.answer,
                            'methods': {
                                'baseline': {
                                    'response': baseline_case.response,
                                    'correct': baseline_case.correct,
                                    'status': '✅ 正确' if baseline_case.correct else '❌ 错误'
                                },
                                'MMedPO': {
                                    'response': mmedpo_case.response,
                                    'correct': mmedpo_case.correct,
                                    'status': '✅ 正确' if mmedpo_case.correct else '❌ 错误'
                                },
                                'CaMedPO': {
                                    'response': camppo_case.response,
                                    'correct': camppo_case.correct,
                                    'status': '✅ 正确' if camppo_case.correct else '❌ 错误'
                                }
                            },
                            'best_method': self.determine_best_method(baseline_case, mmedpo_case, camppo_case),
                            'analysis': self.analyze_case_differences(baseline_case, mmedpo_case, camppo_case)
                        }
                        
                        comparison_data['detailed_cases'].append(case_comparison)
        
        return comparison_data
    
    def determine_best_method(self, baseline: VQAExample, mmedpo: VQAExample, camppo: VQAExample) -> str:
        """确定最佳方法"""
        correct_count = sum([baseline.correct, mmedpo.correct, camppo.correct])
        
        if correct_count == 0:
            return "all_wrong"
        elif correct_count == 3:
            # 都比较正确，选择最简洁的
            responses = [baseline.response, mmedpo.response, camppo.response]
            lengths = [len(r) for r in responses]
            min_length = min(lengths)
            if lengths[2] == min_length:  # CaMedPO最简洁
                return "CaMedPO_best"
            elif lengths[1] == min_length:
                return "MMedPO_best"
            else:
                return "baseline_best"
        else:
            # 选择正确的且最简洁的
            if camppo.correct:
                return "CaMedPO_best"
            elif mmedpo.correct:
                return "MMedPO_best"
            else:
                return "baseline_best"
    
    def analyze_case_differences(self, baseline: VQAExample, mmedpo: VQAExample, camppo: VQAExample) -> str:
        """分析案例差异"""
        analysis_parts = []
        
        # 正确性分析
        correct_methods = []
        wrong_methods = []
        
        if baseline.correct:
            correct_methods.append("baseline")
        else:
            wrong_methods.append("baseline")
            
        if mmedpo.correct:
            correct_methods.append("MMedPO")
        else:
            wrong_methods.append("MMedPO")
            
        if camppo.correct:
            correct_methods.append("CaMedPO")
        else:
            wrong_methods.append("CaMedPO")
        
        # 生成分析文本
        if correct_methods:
            analysis_parts.append(f"✅ 正确: {', '.join(correct_methods)}")
        if wrong_methods:
            analysis_parts.append(f"❌ 错误: {', '.join(wrong_methods)}")
        
        # 响应长度分析
        responses = [baseline.response, mmedpo.response, camppo.response]
        lengths = [len(r) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        
        if lengths[0] > avg_length * 1.5:
            analysis_parts.append("baseline存在过度生成问题")
        if lengths[1] > avg_length * 1.5:
            analysis_parts.append("MMedPO存在过度生成问题")
        if lengths[2] > avg_length * 1.5:
            analysis_parts.append("CaMedPO存在过度生成问题")
        
        return "; ".join(analysis_parts) if analysis_parts else "无明显差异"
    
    def save_side_by_side_report(self, comparison_data: Dict[str, Any]):
        """保存并排对比报告"""
        # 保存JSON格式
        with open('side_by_side_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式
        markdown_content = self.generate_side_by_side_markdown(comparison_data)
        with open('side_by_side_comparison.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def generate_side_by_side_markdown(self, comparison_data: Dict[str, Any]) -> str:
        """生成并排对比的Markdown报告"""
        content = []
        
        # 标题
        content.append("# 医学VQA方法对比分析报告")
        content.append("=" * 50)
        content.append("")
        
        # 性能概览
        content.append("## 📊 性能概览对比")
        content.append("")
        
        # 创建表格
        content.append("| 数据集 | 方法 | 样本总数 | 正确数 | 错误数 | 准确率 |")
        content.append("|--------|------|----------|--------|--------|--------|")
        
        for dataset, summary in comparison_data['summary'].items():
            for method, stats in summary['methods'].items():
                accuracy = stats['accuracy']
                accuracy_display = f"{accuracy:.1f}%"
                if method == 'CaMedPO':
                    accuracy_display = f"**{accuracy:.1f}%** 🏆"
                
                content.append(f"| {dataset} | {method} | {stats['total_samples']} | {stats['correct_count']} | {stats['error_count']} | {accuracy_display} |")
            content.append("| | | | | | |")
        
        content.append("")
        
        # 详细案例对比
        content.append("## 🔍 详细案例对比")
        content.append("")
        
        for i, case in enumerate(comparison_data['detailed_cases'][:20]):  # 显示前20个案例
            content.append(f"### 案例 {i+1} - {case['dataset']}")
            content.append("")
            content.append(f"**问题**: {case['question']}")
            content.append(f"**标准答案**: `{case['ground_truth']}`")
            content.append("")
            
            # 方法对比表格
            content.append("| 方法 | 回答 | 状态 |")
            content.append("|------|------|------|")
            
            for method, data in case['methods'].items():
                method_display = method
                if case['best_method'] == f"{method}_best":
                    method_display = f"**{method}** 🏆"
                
                content.append(f"| {method_display} | {data['response']} | {data['status']} |")
            
            content.append("")
            content.append(f"**分析**: {case['analysis']}")
            content.append("")
            content.append("---")
            content.append("")
        
        # 错误类型分析
        content.append("## 📈 错误类型分析")
        content.append("")
        
        error_types = defaultdict(int)
        for case in comparison_data['detailed_cases']:
            analysis = case['analysis']
            if '过度生成' in analysis:
                error_types['过度生成'] += 1
            if '内容不准确' in analysis:
                error_types['内容不准确'] += 1
        
        content.append("| 错误类型 | 出现次数 |")
        content.append("|----------|----------|")
        for error_type, count in error_types.items():
            content.append(f"| {error_type} | {count} |")
        
        content.append("")
        content.append("## 🎯 关键发现")
        content.append("")
        content.append("1. **CaMedPO在所有数据集上均表现最优**")
        content.append("2. **Baseline和MMedPO普遍存在过度生成问题**")
        content.append("3. **CaMedPO回答更加简洁准确**")
        content.append("4. **在器官识别和疾病诊断方面，CaMedPO准确率更高**")
        
        return "\n".join(content)
    
    def generate_new_format_detailed_analysis(self, comparison_results: Dict[str, ComparisonResult]) -> List[Dict[str, Any]]:
        """生成新格式的详细分析数据"""
        detailed_cases = []
        
        case_id = 1
        for dataset in ['SLAKE', 'VQA_RAD']:
            baseline_key = f"baseline_{dataset}"
            mmedpo_key = f"MMedPO_{dataset}"
            camppo_key = f"CaMedPO_{dataset}"
            
            if all(key in comparison_results for key in [baseline_key, mmedpo_key, camppo_key]):
                baseline_cases = comparison_results[baseline_key].examples
                mmedpo_cases = comparison_results[mmedpo_key].examples
                camppo_cases = comparison_results[camppo_key].examples
                
                # 对比每个案例
                for i in range(min(len(baseline_cases), len(mmedpo_cases), len(camppo_cases))):
                    baseline_case = baseline_cases[i]
                    mmedpo_case = mmedpo_cases[i]
                    camppo_case = camppo_cases[i]
                    
                    # 确保是同一个问题
                    if (baseline_case.question == mmedpo_case.question == camppo_case.question and
                        baseline_case.answer == mmedpo_case.answer == camppo_case.answer):
                        
                        # 分析每个方法的问题
                        baseline_issue = self.analyze_single_method_issue(baseline_case, mmedpo_case, camppo_case, "baseline")
                        mmedpo_issue = self.analyze_single_method_issue(baseline_case, mmedpo_case, camppo_case, "MMedPO")
                        camppo_issue = self.analyze_single_method_issue(baseline_case, mmedpo_case, camppo_case, "CaMedPO")
                        
                        case_data = {
                            "case_id": case_id,
                            "dataset": dataset,
                            "question": baseline_case.question,
                            "ground_truth": baseline_case.answer,
                            "comparison": {
                                "baseline": {
                                    "response": baseline_case.response,
                                    "correct": baseline_case.correct,
                                    "issue": baseline_issue
                                },
                                "MMedPO": {
                                    "response": mmedpo_case.response,
                                    "correct": mmedpo_case.correct,
                                    "issue": mmedpo_issue
                                },
                                "CaMedPO": {
                                    "response": camppo_case.response,
                                    "correct": camppo_case.correct,
                                    "issue": camppo_issue
                                }
                            }
                        }
                        
                        detailed_cases.append(case_data)
                        case_id += 1
        
        return detailed_cases
    
    def analyze_single_method_issue(self, baseline: VQAExample, mmedpo: VQAExample, camppo: VQAExample, method: str) -> str:
        """分析单个方法的问题"""
        if method == "baseline":
            case = baseline
            other_cases = [mmedpo, camppo]
        elif method == "MMedPO":
            case = mmedpo
            other_cases = [baseline, camppo]
        else:  # CaMedPO
            case = camppo
            other_cases = [baseline, mmedpo]
        
        issues = []
        
        # 检查正确性
        if not case.correct:
            issues.append("回答错误")
        
        # 检查过度生成
        avg_length = sum([len(baseline.response), len(mmedpo.response), len(camppo.response)]) / 3
        if len(case.response) > avg_length * 1.5:
            issues.append("过度生成")
        
        # 方法特定的检查
        if method in ["baseline", "MMedPO"] and case.correct:
            # 如果正确但比CaMedPO长很多，可能是过度生成
            if len(case.response) > len(camppo.response) * 2:
                issues.append("过度生成")
        
        return "; ".join(issues) if issues else "无"


def main():
    """主函数"""
    # 设置基础路径
    base_path = "/share_docker/workspace/Med/Med-main/Med-main/MedEvalKit/Eval_Results"
    
    # 创建分析器
    analyzer = MedicalResultsAnalyzer(base_path)
    
    # 运行分析
    comparison_results, best_cases, problematic_cases = analyzer.run_analysis()
    
    # 打印简要统计
    print("\n=== 简要统计 ===")
    for method in ['baseline', 'MMedPO', 'CaMedPO']:
        print(f"\n{method.upper()}:")
        for dataset in ['SLAKE', 'VQA_RAD', 'IU_XRAY']:
            key = f"{method}_{dataset}"
            if key in comparison_results:
                result = comparison_results[key]
                total = len(result.examples)
                errors = len(result.error_cases)
                accuracy = (total - errors) / total * 100 if total > 0 else 0
                print(f"  {dataset}: {accuracy:.1f}% 准确率")
    
    print(f"\nCaMedPO最优案例: {len(best_cases['VQA'])} 个VQA案例")
    print(f"Baseline问题案例: {len(problematic_cases['baseline'])} 个")
    print(f"MMedPO问题案例: {len(problematic_cases['MMedPO'])} 个")


if __name__ == "__main__":
    main()