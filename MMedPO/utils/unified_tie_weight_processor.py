#!/usr/bin/env python3
"""
统一TIE权重处理器
整合所有TIE权重计算功能的统一接口

基于链接中的TIE评分规律，提供完整的权重计算解决方案
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# 导入各个专门的处理器
from tie_weighted_score_calculator import TIEWeightedScoreCalculator
from background_tie_processor import BackgroundTIEProcessor
from anchor_weight_processor import AnchorWeightProcessor
from tie_diff_calculator import TIEDiffCalculator

class UnifiedTIEWeightProcessor:
    """
    统一TIE权重处理器
    
    整合所有TIE权重计算方法，提供统一的处理接口
    支持以下功能：
    1. (y_gt|I) vs (y_gen|I)的TIE diff权重计算（positive>0.05）
    2. (y_gt|I+I_B) vs (y_gt|I+I_B)的TIE分数权重计算
    3. m_v负例和符合条件m_n的-anchor形式设置
    4. (y_gt|I+I_B)和(y_gt|I+I_O)的TIE差值计算
    """
    
    def __init__(self, 
                 tie_calculator_config: Dict = None,
                 background_processor_config: Dict = None,
                 anchor_processor_config: Dict = None,
                 diff_calculator_config: Dict = None,
                 weight_combination_method: str = "weighted_sum"):
        """
        初始化统一TIE权重处理器
        
        Args:
            tie_calculator_config: TIE权重计算器配置
            background_processor_config: 背景TIE处理器配置
            anchor_processor_config: Anchor权重处理器配置
            diff_calculator_config: TIE差值计算器配置
            weight_combination_method: 权重组合方法
        """
        # 初始化各个处理器
        self.tie_calculator = TIEWeightedScoreCalculator(
            **(tie_calculator_config or {})
        )
        
        self.background_processor = BackgroundTIEProcessor(
            base_calculator=self.tie_calculator,
            **(background_processor_config or {})
        )
        
        self.anchor_processor = AnchorWeightProcessor(
            **(anchor_processor_config or {})
        )
        
        self.diff_calculator = TIEDiffCalculator(
            **(diff_calculator_config or {})
        )
        
        self.weight_combination_method = weight_combination_method
    
    def process_single_item(self, item: Dict) -> Dict:
        """
        处理单个数据项
        
        Args:
            item: 包含TIE分数的数据项
            
        Returns:
            处理后的数据项
        """
        processed_item = item.copy()
        
        # 提取基础TIE分数
        tff_pos = item.get('ll_positive_with_background', 0)
        tfn_pos = item.get('ll_positive_with_white_background', 0)
        tff_neg = item.get('ll_negative_with_background', 0)
        tfn_neg = item.get('ll_negative_with_white_background', 0)
        
        # 1. 基础TIE权重计算
        if any([tff_pos, tfn_pos, tff_neg, tfn_neg]):
            tie_gt = item.get('tie_gt', None)
            tie_gen = item.get('tie_gen', None)
            
            basic_weight_result = self.tie_calculator.calculate_comprehensive_weight(
                tff_pos, tfn_pos, tff_neg, tfn_neg, tie_gt, tie_gen
            )
            processed_item.update({
                'basic_tie_weight': basic_weight_result['final_weight'],
                **{f'basic_{k}': v for k, v in basic_weight_result.items()}
            })
        
        # 2. 背景TIE权重计算
        tie_bg_1 = item.get('tie_with_background_1', tff_pos)
        tie_bg_2 = item.get('tie_with_background_2', item.get('ll_positive_with_background_repeat', tie_bg_1))
        tie_no_bg = item.get('tie_without_background', tfn_pos if tfn_pos != 0 else None)
        
        if tie_bg_1 != 0 or tie_bg_2 != 0:
            bg_weight_result = self.background_processor.calculate_background_tie_weight(
                tie_bg_1, tie_bg_2, tie_no_bg
            )
            processed_item.update({
                'background_tie_weight': bg_weight_result['final_weight'],
                **{f'bg_{k}': v for k, v in bg_weight_result.items()}
            })
        
        # 3. Anchor权重计算
        m_v = processed_item.get('basic_m_v', 0)
        m_n = processed_item.get('basic_m_n', 0)
        gamma = processed_item.get('basic_gamma', None)
        
        if m_v != 0 or m_n != 0:
            anchor_weight_result = self.anchor_processor.calculate_combined_anchor_weight(
                m_v, m_n, gamma
            )
            processed_item.update({
                'anchor_weight': anchor_weight_result['final_anchor_weight'],
                **{f'anchor_{k}': v for k, v in anchor_weight_result.items()}
            })
        
        # 4. TIE差值权重计算
        tie_with_bg = item.get('tie_with_bg', tff_pos)
        tie_with_other = item.get('tie_with_other', item.get('ll_positive_with_other_background', tie_with_bg))
        
        if tie_with_bg != tie_with_other:
            diff_weight_result = self.diff_calculator.calculate_comprehensive_tie_diff_weight(
                tie_with_bg, tie_with_other, tie_no_bg
            )
            processed_item.update({
                'tie_diff_weight': diff_weight_result['final_tie_diff_weight'],
                **{f'diff_{k}': v for k, v in diff_weight_result.items()}
            })
        
        # 5. 综合权重计算
        final_weight = self._calculate_combined_weight(processed_item)
        processed_item['unified_tie_weight'] = final_weight
        
        return processed_item
    
    def _calculate_combined_weight(self, item: Dict) -> float:
        """
        计算综合权重
        
        Args:
            item: 包含各种权重的数据项
            
        Returns:
            综合权重分数
        """
        # 提取各种权重
        basic_weight = item.get('basic_tie_weight', 0.0)
        bg_weight = item.get('background_tie_weight', 0.0)
        anchor_weight = item.get('anchor_weight', 0.0)
        diff_weight = item.get('tie_diff_weight', 0.0)
        
        if self.weight_combination_method == "weighted_sum":
            # 加权求和方法
            weights = [basic_weight, bg_weight, diff_weight]
            coefficients = [0.4, 0.3, 0.3]  # 可调整的权重系数
            
            combined_weight = sum(w * c for w, c in zip(weights, coefficients))
            combined_weight += anchor_weight  # anchor权重直接加上
            
        elif self.weight_combination_method == "max_pooling":
            # 最大池化方法
            combined_weight = max(basic_weight, bg_weight, diff_weight) + anchor_weight
            
        elif self.weight_combination_method == "adaptive":
            # 自适应组合方法
            # 根据各权重的置信度进行自适应组合
            weights = [basic_weight, bg_weight, diff_weight]
            confidences = [
                item.get('basic_p_pref', 0.5),
                item.get('bg_consistency_score', 0.5),
                item.get('diff_is_significant', False)
            ]
            
            # 将布尔值转换为数值
            confidences = [float(c) if isinstance(c, bool) else c for c in confidences]
            
            # 归一化置信度
            total_confidence = sum(confidences) + 1e-8
            normalized_confidences = [c / total_confidence for c in confidences]
            
            combined_weight = sum(w * c for w, c in zip(weights, normalized_confidences))
            combined_weight += anchor_weight
            
        else:
            # 默认简单平均
            non_zero_weights = [w for w in [basic_weight, bg_weight, diff_weight] if w != 0]
            if non_zero_weights:
                combined_weight = np.mean(non_zero_weights) + anchor_weight
            else:
                combined_weight = anchor_weight
        
        # 限制权重范围
        combined_weight = max(0.0, min(3.0, combined_weight))
        
        return combined_weight
    
    def process_dataset(self, data_items: List[Dict]) -> List[Dict]:
        """
        处理整个数据集
        
        Args:
            data_items: 数据项列表
            
        Returns:
            处理后的数据项列表
        """
        processed_items = []
        
        print(f"使用统一TIE权重处理器处理 {len(data_items)} 条数据...")
        
        for i, item in enumerate(data_items):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(data_items)}")
            
            processed_item = self.process_single_item(item)
            processed_items.append(processed_item)
        
        return processed_items
    
    def analyze_comprehensive_results(self, processed_items: List[Dict]) -> Dict:
        """
        分析综合处理结果
        
        Args:
            processed_items: 处理后的数据项列表
            
        Returns:
            综合分析结果
        """
        total_items = len(processed_items)
        
        # 收集各种权重
        basic_weights = [item.get('basic_tie_weight', 0) for item in processed_items]
        bg_weights = [item.get('background_tie_weight', 0) for item in processed_items]
        anchor_weights = [item.get('anchor_weight', 0) for item in processed_items]
        diff_weights = [item.get('tie_diff_weight', 0) for item in processed_items]
        unified_weights = [item.get('unified_tie_weight', 0) for item in processed_items]
        
        # 统计anchor样本
        anchor_samples = sum(1 for item in processed_items if item.get('anchor_is_anchor_sample', False))
        
        # 统计显著差异样本
        significant_diff_samples = sum(1 for item in processed_items if item.get('diff_is_significant', False))
        
        # 统计稳定背景样本
        stable_bg_samples = sum(1 for item in processed_items if item.get('bg_is_stable', False))
        
        return {
            'total_items': total_items,
            'anchor_samples': anchor_samples,
            'anchor_ratio': anchor_samples / total_items if total_items > 0 else 0,
            'significant_diff_samples': significant_diff_samples,
            'significant_diff_ratio': significant_diff_samples / total_items if total_items > 0 else 0,
            'stable_bg_samples': stable_bg_samples,
            'stable_bg_ratio': stable_bg_samples / total_items if total_items > 0 else 0,
            
            # 权重统计
            'basic_weight_stats': {
                'mean': np.mean(basic_weights),
                'std': np.std(basic_weights),
                'min': np.min(basic_weights),
                'max': np.max(basic_weights)
            },
            'bg_weight_stats': {
                'mean': np.mean(bg_weights),
                'std': np.std(bg_weights),
                'min': np.min(bg_weights),
                'max': np.max(bg_weights)
            },
            'anchor_weight_stats': {
                'mean': np.mean(anchor_weights),
                'std': np.std(anchor_weights),
                'min': np.min(anchor_weights),
                'max': np.max(anchor_weights)
            },
            'diff_weight_stats': {
                'mean': np.mean(diff_weights),
                'std': np.std(diff_weights),
                'min': np.min(diff_weights),
                'max': np.max(diff_weights)
            },
            'unified_weight_stats': {
                'mean': np.mean(unified_weights),
                'std': np.std(unified_weights),
                'min': np.min(unified_weights),
                'max': np.max(unified_weights)
            }
        }

def process_unified_tie_weights(input_file: str, output_file: str,
                              processor_config: Dict = None) -> None:
    """
    使用统一处理器处理TIE权重数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        processor_config: 处理器配置
    """
    if processor_config is None:
        processor_config = {
            'tie_calculator_config': {
                'w1': 1.0, 'w2': 0.5, 'w3': 0.5,
                'beta': 1.0, 'lambda_disp': 1.0, 'mu_disp': 0.5,
                'tie_diff_threshold': 0.05
            },
            'background_processor_config': {
                'background_weight': 1.0,
                'context_weight': 0.8,
                'stability_threshold': 0.02
            },
            'anchor_processor_config': {
                'mv_negative_threshold': 0.0,
                'mn_positive_threshold': 0.1,
                'anchor_strength': 1.0,
                'anchor_decay': 0.9,
                'min_anchor_weight': 0.1
            },
            'diff_calculator_config': {
                'diff_threshold': 0.05,
                'background_sensitivity': 1.0,
                'context_stability': 0.8,
                'adaptive_scaling': True
            },
            'weight_combination_method': 'adaptive'
        }
    
    # 创建统一处理器
    processor = UnifiedTIEWeightProcessor(**processor_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"开始统一TIE权重处理，共 {len(data)} 条数据")
    
    # 处理数据
    processed_data = processor.process_dataset(data)
    
    # 分析结果
    analysis_result = processor.analyze_comprehensive_results(processed_data)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存分析报告
    analysis_file = output_path.with_suffix('.analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存统一TIE权重结果到: {output_file}")
    print(f"已保存分析报告到: {analysis_file}")
    
    # 打印统计信息
    print(f"\n=== 统一TIE权重处理结果 ===")
    print(f"总样本数: {analysis_result['total_items']}")
    print(f"Anchor样本: {analysis_result['anchor_samples']} ({analysis_result['anchor_ratio']:.2%})")
    print(f"显著差异样本: {analysis_result['significant_diff_samples']} ({analysis_result['significant_diff_ratio']:.2%})")
    print(f"稳定背景样本: {analysis_result['stable_bg_samples']} ({analysis_result['stable_bg_ratio']:.2%})")
    
    print(f"\n=== 权重统计 ===")
    for weight_type in ['basic', 'bg', 'anchor', 'diff', 'unified']:
        stats = analysis_result[f'{weight_type}_weight_stats']
        print(f"{weight_type.upper()}权重: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}]")

def main():
    """
    主函数：演示统一TIE权重处理
    """
    input_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_unified_tie_weights.jsonl"
    
    print("开始统一TIE权重处理...")
    
    process_unified_tie_weights(input_file, output_file)
    
    print("\n统一TIE权重处理完成！")

if __name__ == "__main__":
    main()