#!/usr/bin/env python3
"""
TIE差值计算器
实现(y_gt|I+I_B)和(y_gt|I+I_O)的TIE差值计算

基于链接中的TIE评分规律，计算不同背景条件下的TIE差值权重
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tie_weighted_score_calculator import TIEWeightedScoreCalculator

class TIEDiffCalculator:
    """
    TIE差值计算器
    
    专门处理不同背景条件下的TIE差值计算
    主要用于(y_gt|I+I_B) vs (y_gt|I+I_O)的对比分析
    
    其中:
    - I+I_B: 图像+背景图像
    - I+I_O: 图像+其他背景图像
    """
    
    def __init__(self, 
                 diff_threshold: float = 0.05,        # TIE差值阈值
                 background_sensitivity: float = 1.0,  # 背景敏感性权重
                 context_stability: float = 0.8,      # 上下文稳定性权重
                 adaptive_scaling: bool = True):       # 自适应缩放
        """
        初始化TIE差值计算器
        
        Args:
            diff_threshold: TIE差值的显著性阈值
            background_sensitivity: 背景敏感性权重
            context_stability: 上下文稳定性权重
            adaptive_scaling: 是否使用自适应缩放
        """
        self.diff_threshold = diff_threshold
        self.background_sensitivity = background_sensitivity
        self.context_stability = context_stability
        self.adaptive_scaling = adaptive_scaling
    
    def calculate_basic_tie_diff(self, 
                                tie_with_bg: float,    # (y_gt|I+I_B)
                                tie_with_other: float  # (y_gt|I+I_O)
                                ) -> Dict[str, float]:
        """
        计算基础TIE差值
        
        Args:
            tie_with_bg: 带原始背景的TIE分数
            tie_with_other: 带其他背景的TIE分数
            
        Returns:
            基础差值计算结果
        """
        # 计算绝对差值和相对差值
        abs_diff = tie_with_bg - tie_with_other
        rel_diff = abs_diff / (abs(tie_with_bg) + 1e-8)  # 避免除零
        
        # 计算差值的显著性
        is_significant = abs(abs_diff) > self.diff_threshold
        
        # 计算差值强度
        diff_magnitude = abs(abs_diff)
        diff_intensity = 1 - np.exp(-diff_magnitude / self.diff_threshold)
        
        return {
            'tie_with_bg': tie_with_bg,
            'tie_with_other': tie_with_other,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'diff_magnitude': diff_magnitude,
            'diff_intensity': diff_intensity,
            'is_significant': is_significant
        }
    
    def calculate_background_sensitivity_score(self, 
                                             tie_with_bg: float,
                                             tie_with_other: float,
                                             tie_without_bg: Optional[float] = None) -> Dict[str, float]:
        """
        计算背景敏感性分数
        
        分析模型对不同背景的敏感程度
        
        Args:
            tie_with_bg: 带原始背景的TIE分数
            tie_with_other: 带其他背景的TIE分数
            tie_without_bg: 不带背景的TIE分数(可选)
            
        Returns:
            背景敏感性分析结果
        """
        # 基础差值计算
        basic_diff = self.calculate_basic_tie_diff(tie_with_bg, tie_with_other)
        
        # 背景敏感性指标
        bg_sensitivity = abs(basic_diff['abs_diff']) * self.background_sensitivity
        
        # 如果有无背景的TIE分数，计算更详细的敏感性分析
        bg_impact_original = 0.0
        bg_impact_other = 0.0
        bg_impact_diff = 0.0
        
        if tie_without_bg is not None:
            bg_impact_original = tie_with_bg - tie_without_bg
            bg_impact_other = tie_with_other - tie_without_bg
            bg_impact_diff = bg_impact_original - bg_impact_other
            
            # 更新背景敏感性分数
            bg_sensitivity = abs(bg_impact_diff) * self.background_sensitivity
        
        # 敏感性等级分类
        if bg_sensitivity < 0.02:
            sensitivity_level = "low"
        elif bg_sensitivity < 0.1:
            sensitivity_level = "medium"
        else:
            sensitivity_level = "high"
        
        result = {
            **basic_diff,
            'bg_sensitivity': bg_sensitivity,
            'sensitivity_level': sensitivity_level
        }
        
        if tie_without_bg is not None:
            result.update({
                'tie_without_bg': tie_without_bg,
                'bg_impact_original': bg_impact_original,
                'bg_impact_other': bg_impact_other,
                'bg_impact_diff': bg_impact_diff
            })
        
        return result
    
    def calculate_context_stability_weight(self, 
                                         sensitivity_result: Dict[str, float]) -> Dict[str, float]:
        """
        计算上下文稳定性权重
        
        基于背景敏感性计算样本的稳定性权重
        
        Args:
            sensitivity_result: 背景敏感性分析结果
            
        Returns:
            上下文稳定性权重结果
        """
        bg_sensitivity = sensitivity_result['bg_sensitivity']
        diff_intensity = sensitivity_result['diff_intensity']
        is_significant = sensitivity_result['is_significant']
        
        # 基础稳定性权重：敏感性越低，稳定性越高
        base_stability = np.exp(-bg_sensitivity / self.diff_threshold)
        
        # 上下文一致性调整
        if is_significant:
            # 显著差异时，根据差异方向调整权重
            abs_diff = sensitivity_result['abs_diff']
            if abs_diff > 0:  # 原始背景TIE更高
                context_adjustment = self.context_stability * diff_intensity
            else:  # 其他背景TIE更高
                context_adjustment = -self.context_stability * diff_intensity
        else:
            # 非显著差异时，给予稳定性奖励
            context_adjustment = self.context_stability * base_stability
        
        # 最终稳定性权重
        stability_weight = base_stability + context_adjustment
        
        # 自适应缩放
        if self.adaptive_scaling:
            # 根据整体敏感性水平进行缩放
            scaling_factor = 1.0 / (1.0 + bg_sensitivity)
            stability_weight *= scaling_factor
        
        # 限制权重范围
        stability_weight = max(0.0, min(2.0, stability_weight))
        
        return {
            'base_stability': base_stability,
            'context_adjustment': context_adjustment,
            'stability_weight': stability_weight,
            'scaling_factor': 1.0 / (1.0 + bg_sensitivity) if self.adaptive_scaling else 1.0
        }
    
    def calculate_comprehensive_tie_diff_weight(self, 
                                              tie_with_bg: float,
                                              tie_with_other: float,
                                              tie_without_bg: Optional[float] = None) -> Dict[str, float]:
        """
        计算综合TIE差值权重
        
        整合所有TIE差值分析方法
        
        Args:
            tie_with_bg: 带原始背景的TIE分数
            tie_with_other: 带其他背景的TIE分数
            tie_without_bg: 不带背景的TIE分数(可选)
            
        Returns:
            综合TIE差值权重结果
        """
        # 计算背景敏感性
        sensitivity_result = self.calculate_background_sensitivity_score(
            tie_with_bg, tie_with_other, tie_without_bg
        )
        
        # 计算上下文稳定性权重
        stability_result = self.calculate_context_stability_weight(sensitivity_result)
        
        # 整合结果
        result = {
            **sensitivity_result,
            **stability_result,
            'final_tie_diff_weight': stability_result['stability_weight']
        }
        
        return result
    
    def process_tie_diff_dataset(self, data_items: List[Dict]) -> List[Dict]:
        """
        处理TIE差值数据集
        
        Args:
            data_items: 包含TIE分数的数据项列表
            
        Returns:
            处理后的数据项列表
        """
        processed_items = []
        
        for item in data_items:
            # 提取TIE分数
            tie_with_bg = item.get('tie_with_bg', item.get('ll_positive_with_background', 0))
            tie_with_other = item.get('tie_with_other', item.get('ll_positive_with_other_background', tie_with_bg))
            tie_without_bg = item.get('tie_without_bg', item.get('ll_positive_with_white_background', None))
            
            # 计算TIE差值权重
            tie_diff_result = self.calculate_comprehensive_tie_diff_weight(
                tie_with_bg, tie_with_other, tie_without_bg
            )
            
            # 创建新的数据项
            new_item = item.copy()
            new_item.update({
                'tie_diff_weight': tie_diff_result['final_tie_diff_weight'],
                **tie_diff_result
            })
            
            processed_items.append(new_item)
        
        return processed_items
    
    def analyze_tie_diff_distribution(self, data_items: List[Dict]) -> Dict[str, Union[int, float]]:
        """
        分析TIE差值分布
        
        Args:
            data_items: 处理后的数据项列表
            
        Returns:
            TIE差值分布分析结果
        """
        total_items = len(data_items)
        
        # 统计敏感性等级
        sensitivity_counts = {'low': 0, 'medium': 0, 'high': 0}
        significant_count = 0
        
        # 收集数值统计
        abs_diffs = []
        bg_sensitivities = []
        stability_weights = []
        
        for item in data_items:
            # 敏感性等级统计
            sensitivity_level = item.get('sensitivity_level', 'low')
            if sensitivity_level in sensitivity_counts:
                sensitivity_counts[sensitivity_level] += 1
            
            # 显著性统计
            if item.get('is_significant', False):
                significant_count += 1
            
            # 数值统计
            if 'abs_diff' in item:
                abs_diffs.append(abs(item['abs_diff']))
            if 'bg_sensitivity' in item:
                bg_sensitivities.append(item['bg_sensitivity'])
            if 'stability_weight' in item:
                stability_weights.append(item['stability_weight'])
        
        return {
            'total_items': total_items,
            'significant_items': significant_count,
            'significant_ratio': significant_count / total_items if total_items > 0 else 0,
            **{f'{k}_sensitivity_count': v for k, v in sensitivity_counts.items()},
            **{f'{k}_sensitivity_ratio': v / total_items if total_items > 0 else 0 
               for k, v in sensitivity_counts.items()},
            'avg_abs_diff': np.mean(abs_diffs) if abs_diffs else 0,
            'std_abs_diff': np.std(abs_diffs) if abs_diffs else 0,
            'avg_bg_sensitivity': np.mean(bg_sensitivities) if bg_sensitivities else 0,
            'std_bg_sensitivity': np.std(bg_sensitivities) if bg_sensitivities else 0,
            'avg_stability_weight': np.mean(stability_weights) if stability_weights else 0,
            'std_stability_weight': np.std(stability_weights) if stability_weights else 0
        }

def process_tie_diff_dataset(input_file: str, output_file: str,
                           calculator_config: Dict = None) -> None:
    """
    处理TIE差值数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        calculator_config: 计算器配置
    """
    if calculator_config is None:
        calculator_config = {
            'diff_threshold': 0.05,
            'background_sensitivity': 1.0,
            'context_stability': 0.8,
            'adaptive_scaling': True
        }
    
    # 创建计算器
    calculator = TIEDiffCalculator(**calculator_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"处理 {len(data)} 条TIE差值数据...")
    
    # 处理数据
    processed_data = calculator.process_tie_diff_dataset(data)
    
    # 分析分布
    distribution_analysis = calculator.analyze_tie_diff_distribution(processed_data)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存TIE差值权重结果到: {output_file}")
    
    # 打印统计信息
    weights = [item['tie_diff_weight'] for item in processed_data]
    print(f"\nTIE差值权重统计:")
    print(f"  均值: {np.mean(weights):.4f}")
    print(f"  标准差: {np.std(weights):.4f}")
    print(f"  范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    
    print(f"\n分布分析:")
    print(f"  总样本数: {distribution_analysis['total_items']}")
    print(f"  显著差异样本: {distribution_analysis['significant_items']} ({distribution_analysis['significant_ratio']:.2%})")
    print(f"  低敏感性: {distribution_analysis['low_sensitivity_count']} ({distribution_analysis['low_sensitivity_ratio']:.2%})")
    print(f"  中敏感性: {distribution_analysis['medium_sensitivity_count']} ({distribution_analysis['medium_sensitivity_ratio']:.2%})")
    print(f"  高敏感性: {distribution_analysis['high_sensitivity_count']} ({distribution_analysis['high_sensitivity_ratio']:.2%})")
    print(f"  平均背景敏感性: {distribution_analysis['avg_bg_sensitivity']:.4f}")

def main():
    """
    主函数：演示TIE差值计算
    """
    input_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_diff_weights.jsonl"
    
    calculator_config = {
        'diff_threshold': 0.05,
        'background_sensitivity': 1.0,
        'context_stability': 0.8,
        'adaptive_scaling': True
    }
    
    print("开始TIE差值权重计算...")
    print(f"配置: {calculator_config}")
    
    process_tie_diff_dataset(input_file, output_file, calculator_config)
    
    print("\nTIE差值权重计算完成！")

if __name__ == "__main__":
    main()