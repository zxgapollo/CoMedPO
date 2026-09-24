#!/usr/bin/env python3
"""
改进的TIE (Total Intervention Effect) 计算方法
基于因果推理理论，结合医疗影像的前景-背景分离

核心改进:
1. 引入因果干预强度权重
2. 考虑答案长度归一化
3. 添加置信度调整
4. 实现多层次TIE计算
"""

import json
import numpy as np
import math
from typing import List, Dict, Tuple
from pathlib import Path

class ImprovedTIECalculator:
    """
    改进的TIE计算器
    
    基于因果推理框架，实现更准确的Total Intervention Effect计算
    """
    
    def __init__(self, 
                 intervention_strength: float = 1.0,
                 length_normalization: bool = True,
                 confidence_adjustment: bool = True,
                 multi_level_tie: bool = True):
        """
        初始化TIE计算器
        
        Args:
            intervention_strength: 干预强度权重 (0.5-2.0)
            length_normalization: 是否进行答案长度归一化
            confidence_adjustment: 是否进行置信度调整
            multi_level_tie: 是否计算多层次TIE
        """
        self.intervention_strength = intervention_strength
        self.length_normalization = length_normalization
        self.confidence_adjustment = confidence_adjustment
        self.multi_level_tie = multi_level_tie
    
    def calculate_basic_tie(self, 
                           ll_with_bg: float, 
                           ll_with_white: float,
                           answer_length: int = 1) -> float:
        """
        计算基础TIE值
        
        Args:
            ll_with_bg: 带背景的对数似然
            ll_with_white: 白背景的对数似然
            answer_length: 答案长度（token数）
        
        Returns:
            基础TIE值
        """
        # 原始TIE计算
        raw_tie = ll_with_bg - ll_with_white
        
        # 长度归一化
        if self.length_normalization and answer_length > 0:
            normalized_tie = raw_tie / answer_length
        else:
            normalized_tie = raw_tie
            
        return normalized_tie
    
    def calculate_intervention_weighted_tie(self,
                                          ll_pos_bg: float,
                                          ll_pos_white: float,
                                          ll_neg_bg: float,
                                          ll_neg_white: float,
                                          pos_length: int,
                                          neg_length: int) -> Dict[str, float]:
        """
        计算干预加权的TIE
        
        基于因果干预理论，考虑正负样本的相对重要性
        
        Returns:
            包含各种TIE指标的字典
        """
        # 计算基础TIE
        tie_pos_basic = self.calculate_basic_tie(ll_pos_bg, ll_pos_white, pos_length)
        tie_neg_basic = self.calculate_basic_tie(ll_neg_bg, ll_neg_white, neg_length)
        
        # 计算干预强度权重
        pos_intervention_weight = self._calculate_intervention_weight(ll_pos_bg, ll_pos_white)
        neg_intervention_weight = self._calculate_intervention_weight(ll_neg_bg, ll_neg_white)
        
        # 加权TIE
        tie_pos_weighted = tie_pos_basic * pos_intervention_weight * self.intervention_strength
        tie_neg_weighted = tie_neg_basic * neg_intervention_weight * self.intervention_strength
        
        # 置信度调整
        if self.confidence_adjustment:
            pos_confidence = self._calculate_confidence(ll_pos_bg, ll_pos_white)
            neg_confidence = self._calculate_confidence(ll_neg_bg, ll_neg_white)
            
            tie_pos_adjusted = tie_pos_weighted * pos_confidence
            tie_neg_adjusted = tie_neg_weighted * neg_confidence
        else:
            tie_pos_adjusted = tie_pos_weighted
            tie_neg_adjusted = tie_neg_weighted
        
        # 计算差异指标
        tie_difference_basic = tie_pos_basic - tie_neg_basic
        tie_difference_weighted = tie_pos_weighted - tie_neg_weighted
        tie_difference_adjusted = tie_pos_adjusted - tie_neg_adjusted
        
        # 多层次TIE计算
        if self.multi_level_tie:
            multilevel_metrics = self._calculate_multilevel_tie(
                ll_pos_bg, ll_pos_white, ll_neg_bg, ll_neg_white,
                pos_length, neg_length
            )
        else:
            multilevel_metrics = {}
        
        return {
            # 基础指标
            'tie_positive_basic': tie_pos_basic,
            'tie_negative_basic': tie_neg_basic,
            'tie_difference_basic': tie_difference_basic,
            
            # 加权指标
            'tie_positive_weighted': tie_pos_weighted,
            'tie_negative_weighted': tie_neg_weighted,
            'tie_difference_weighted': tie_difference_weighted,
            
            # 调整后指标
            'tie_positive_adjusted': tie_pos_adjusted,
            'tie_negative_adjusted': tie_neg_adjusted,
            'tie_difference_adjusted': tie_difference_adjusted,
            
            # 干预权重
            'pos_intervention_weight': pos_intervention_weight,
            'neg_intervention_weight': neg_intervention_weight,
            
            # 置信度
            'pos_confidence': pos_confidence if self.confidence_adjustment else 1.0,
            'neg_confidence': neg_confidence if self.confidence_adjustment else 1.0,
            
            # 多层次指标
            **multilevel_metrics
        }
    
    def _calculate_intervention_weight(self, ll_bg: float, ll_white: float) -> float:
        """
        计算干预强度权重
        
        基于背景移除对模型预测的影响程度
        """
        # 计算相对变化幅度
        if abs(ll_white) > 1e-8:  # 避免除零
            relative_change = abs(ll_bg - ll_white) / abs(ll_white)
        else:
            relative_change = abs(ll_bg - ll_white)
        
        # 使用sigmoid函数将变化幅度映射到权重
        weight = 2.0 / (1.0 + math.exp(-relative_change)) - 1.0
        return max(0.1, min(2.0, weight))  # 限制权重范围
    
    def _calculate_confidence(self, ll_bg: float, ll_white: float) -> float:
        """
        计算置信度调整因子
        
        基于对数似然的绝对值，较高的绝对值表示更高的置信度
        """
        # 计算平均对数似然的绝对值
        avg_ll_abs = (abs(ll_bg) + abs(ll_white)) / 2.0
        
        # 使用对数函数计算置信度
        if avg_ll_abs > 1.0:
            confidence = 1.0 - math.exp(-avg_ll_abs / 10.0)
        else:
            confidence = avg_ll_abs
        
        return max(0.1, min(1.0, confidence))  # 限制置信度范围
    
    def _calculate_multilevel_tie(self,
                                 ll_pos_bg: float, ll_pos_white: float,
                                 ll_neg_bg: float, ll_neg_white: float,
                                 pos_length: int, neg_length: int) -> Dict[str, float]:
        """
        计算多层次TIE指标
        
        包括：
        1. 前景直接效应 (Foreground Direct Effect, FDE)
        2. 背景上下文效应 (Background Context Effect, BCE)
        3. 背景影响比率 (Background Influence Ratio, BIR)
        """
        # 前景直接效应：正确答案在白背景下的表现
        fde = -ll_pos_white / pos_length if pos_length > 0 else -ll_pos_white
        
        # 背景上下文效应：背景对错误答案的促进作用
        bce_raw = ll_neg_white - ll_neg_bg  # 背景移除后错误答案似然的变化
        bce = bce_raw / neg_length if neg_length > 0 else bce_raw
        
        # 背景影响比率：背景效应相对于前景效应的比率
        if abs(fde) > 1e-8:
            bir = abs(bce) / abs(fde)
        else:
            bir = float('inf') if abs(bce) > 1e-8 else 0.0
        
        # 因果强度指标：衡量因果关系的强度
        causal_strength = abs(ll_pos_bg - ll_pos_white) + abs(ll_neg_bg - ll_neg_white)
        causal_strength_normalized = causal_strength / (pos_length + neg_length)
        
        # 干预效果指标：衡量背景移除的整体效果
        intervention_effect = (ll_pos_white - ll_pos_bg) - (ll_neg_white - ll_neg_bg)
        intervention_effect_normalized = intervention_effect / (pos_length + neg_length)
        
        return {
            'foreground_direct_effect': fde,
            'background_context_effect': bce,
            'background_influence_ratio': bir,
            'causal_strength': causal_strength_normalized,
            'intervention_effect': intervention_effect_normalized
        }

def process_dataset_with_improved_tie(input_file: str, output_file: str, 
                                    calculator_config: Dict = None) -> None:
    """
    使用改进的TIE计算方法处理数据集
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        calculator_config: TIE计算器配置
    """
    if calculator_config is None:
        calculator_config = {
            'intervention_strength': 1.2,
            'length_normalization': True,
            'confidence_adjustment': True,
            'multi_level_tie': True
        }
    
    calculator = ImprovedTIECalculator(**calculator_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"处理 {len(data)} 条数据...")
    
    # 处理每条数据
    processed_data = []
    for item in data:
        # 提取原始数据
        ll_pos_bg = item['ll_positive_with_background']
        ll_pos_white = item['ll_positive_with_white_background']
        ll_neg_bg = item['ll_negative_with_background']
        ll_neg_white = item['ll_negative_with_white_background']
        pos_length = item.get('positive_answer_token_len', 1)
        neg_length = item.get('negative_answer_token_len', 1)
        
        # 计算改进的TIE指标
        tie_metrics = calculator.calculate_intervention_weighted_tie(
            ll_pos_bg, ll_pos_white, ll_neg_bg, ll_neg_white,
            pos_length, neg_length
        )
        
        # 创建新的数据项
        new_item = item.copy()
        
        # 添加改进的TIE指标
        new_item.update({
            # 保留原始指标
            'tie_positive_original': item.get('tie_positive', 0),
            'tie_negative_original': item.get('tie_negative', 0),
            'tie_difference_original': item.get('tie_difference', 0),
            
            # 添加改进指标
            **tie_metrics,
            
            # 添加计算配置
            'tie_calculation_config': calculator_config
        })
        
        processed_data.append(new_item)
    
    # 保存处理后的数据
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存改进的TIE计算结果到: {output_file}")
    
    # 打印统计信息
    print_tie_statistics(processed_data)

def print_tie_statistics(data: List[Dict]) -> None:
    """
    打印TIE统计信息
    """
    metrics = [
        'tie_difference_original', 'tie_difference_basic', 
        'tie_difference_weighted', 'tie_difference_adjusted',
        'background_influence_ratio', 'causal_strength', 'intervention_effect'
    ]
    
    print("\n=== TIE指标统计对比 ===")
    for metric in metrics:
        if metric in data[0]:
            values = [item[metric] for item in data if not math.isnan(item.get(metric, 0)) and not math.isinf(item.get(metric, 0))]
            if values:
                print(f"{metric}:")
                print(f"  均值: {np.mean(values):.4f}")
                print(f"  标准差: {np.std(values):.4f}")
                print(f"  范围: [{np.min(values):.4f}, {np.max(values):.4f}]")
                print()

def main():
    """
    主函数：演示改进的TIE计算
    """
    input_file = "/workspace/MMedPO/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO/datasets/Slake1.0/dpo_tie_improved_calculation.jsonl"
    
    # 配置改进的TIE计算器
    calculator_config = {
        'intervention_strength': 1.2,  # 适度增强干预强度
        'length_normalization': True,   # 启用长度归一化
        'confidence_adjustment': True,  # 启用置信度调整
        'multi_level_tie': True        # 启用多层次TIE计算
    }
    
    print("开始改进的TIE计算...")
    print(f"配置: {calculator_config}")
    
    process_dataset_with_improved_tie(input_file, output_file, calculator_config)
    
    print("\n改进的TIE计算完成！")
    print("\n主要改进点:")
    print("1. 干预强度权重：基于背景移除的影响程度动态调整权重")
    print("2. 长度归一化：消除答案长度对TIE计算的影响")
    print("3. 置信度调整：基于对数似然的绝对值调整置信度")
    print("4. 多层次TIE：计算前景直接效应、背景上下文效应、背景影响比率等")
    print("5. 因果强度指标：量化因果关系的强度")
    print("6. 干预效果指标：衡量背景移除的整体效果")

if __name__ == "__main__":
    main()