#!/usr/bin/env python3
"""
背景TIE处理器
专门处理(y_gt|I+I_B) vs (y_gt|I+I_B)的TIE分数权重计算

基于链接中的TIE评分规律，实现带背景图像的TIE权重计算
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tie_weighted_score_calculator import TIEWeightedScoreCalculator

class BackgroundTIEProcessor:
    """
    背景TIE处理器
    
    专门处理带背景图像的TIE分数计算和权重生成
    主要用于(y_gt|I+I_B) vs (y_gt|I+I_B)的对比分析
    """
    
    def __init__(self, 
                 base_calculator: TIEWeightedScoreCalculator = None,
                 background_weight: float = 1.0,
                 context_weight: float = 0.8,
                 stability_threshold: float = 0.02):
        """
        初始化背景TIE处理器
        
        Args:
            base_calculator: 基础TIE权重计算器
            background_weight: 背景影响权重
            context_weight: 上下文一致性权重
            stability_threshold: 稳定性阈值
        """
        if base_calculator is None:
            self.calculator = TIEWeightedScoreCalculator()
        else:
            self.calculator = base_calculator
            
        self.background_weight = background_weight
        self.context_weight = context_weight
        self.stability_threshold = stability_threshold
    
    def calculate_background_consistency(self, 
                                       tie_with_bg_1: float,
                                       tie_with_bg_2: float) -> Dict[str, float]:
        """
        计算背景一致性指标
        
        对于(y_gt|I+I_B) vs (y_gt|I+I_B)的情况，
        理想情况下两个TIE分数应该相近
        
        Args:
            tie_with_bg_1: 第一次带背景的TIE分数
            tie_with_bg_2: 第二次带背景的TIE分数
            
        Returns:
            包含一致性指标的字典
        """
        # 计算绝对差异
        abs_diff = abs(tie_with_bg_1 - tie_with_bg_2)
        
        # 计算相对差异
        avg_tie = (tie_with_bg_1 + tie_with_bg_2) / 2
        rel_diff = abs_diff / (abs(avg_tie) + 1e-8)  # 避免除零
        
        # 一致性分数 (差异越小，一致性越高)
        consistency_score = np.exp(-abs_diff / self.stability_threshold)
        
        # 稳定性判断
        is_stable = abs_diff < self.stability_threshold
        
        return {
            'tie_with_bg_1': tie_with_bg_1,
            'tie_with_bg_2': tie_with_bg_2,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'avg_tie': avg_tie,
            'consistency_score': consistency_score,
            'is_stable': is_stable
        }
    
    def calculate_background_tie_weight(self, 
                                      tie_with_bg_1: float,
                                      tie_with_bg_2: float,
                                      tie_without_bg: Optional[float] = None) -> Dict[str, float]:
        """
        计算基于背景TIE的权重
        
        Args:
            tie_with_bg_1: 第一次带背景的TIE分数
            tie_with_bg_2: 第二次带背景的TIE分数  
            tie_without_bg: 不带背景的TIE分数(可选)
            
        Returns:
            权重计算结果
        """
        # 计算一致性指标
        consistency = self.calculate_background_consistency(tie_with_bg_1, tie_with_bg_2)
        
        # 基础权重：基于平均TIE分数
        avg_tie = consistency['avg_tie']
        base_weight = torch.sigmoid(torch.tensor(avg_tie)).item()
        
        # 一致性调整：一致性高的样本权重更高
        consistency_adjustment = consistency['consistency_score'] * self.context_weight
        
        # 背景效应分析
        background_effect = 0.0
        if tie_without_bg is not None:
            # 计算背景对TIE的影响
            bg_impact_1 = tie_with_bg_1 - tie_without_bg
            bg_impact_2 = tie_with_bg_2 - tie_without_bg
            avg_bg_impact = (bg_impact_1 + bg_impact_2) / 2
            
            # 背景效应权重调整
            background_effect = self.background_weight * torch.sigmoid(torch.tensor(avg_bg_impact)).item()
        
        # 最终权重计算
        final_weight = base_weight + consistency_adjustment + background_effect
        final_weight = max(0.0, min(2.0, final_weight))  # 限制在合理范围
        
        result = {
            **consistency,
            'base_weight': base_weight,
            'consistency_adjustment': consistency_adjustment,
            'background_effect': background_effect,
            'final_weight': final_weight
        }
        
        if tie_without_bg is not None:
            result.update({
                'tie_without_bg': tie_without_bg,
                'bg_impact_1': bg_impact_1,
                'bg_impact_2': bg_impact_2,
                'avg_bg_impact': avg_bg_impact
            })
        
        return result
    
    def process_background_pairs(self, data_items: List[Dict]) -> List[Dict]:
        """
        处理背景TIE对比数据
        
        Args:
            data_items: 包含TIE分数的数据项列表
            
        Returns:
            处理后的数据项列表
        """
        processed_items = []
        
        for item in data_items:
            # 提取背景TIE分数
            tie_bg_1 = item.get('tie_with_background_1', item.get('ll_positive_with_background', 0))
            tie_bg_2 = item.get('tie_with_background_2', item.get('ll_positive_with_background_repeat', tie_bg_1))
            tie_no_bg = item.get('tie_without_background', item.get('ll_positive_with_white_background', None))
            
            # 计算背景TIE权重
            bg_weight_result = self.calculate_background_tie_weight(tie_bg_1, tie_bg_2, tie_no_bg)
            
            # 创建新的数据项
            new_item = item.copy()
            new_item.update({
                'background_tie_weight': bg_weight_result['final_weight'],
                **bg_weight_result
            })
            
            processed_items.append(new_item)
        
        return processed_items
    
    def analyze_background_stability(self, data_items: List[Dict]) -> Dict[str, float]:
        """
        分析整个数据集的背景稳定性
        
        Args:
            data_items: 数据项列表
            
        Returns:
            稳定性分析结果
        """
        consistency_scores = []
        abs_diffs = []
        stable_count = 0
        
        for item in data_items:
            if 'consistency_score' in item:
                consistency_scores.append(item['consistency_score'])
            if 'abs_diff' in item:
                abs_diffs.append(item['abs_diff'])
            if item.get('is_stable', False):
                stable_count += 1
        
        total_items = len(data_items)
        stability_ratio = stable_count / total_items if total_items > 0 else 0
        
        return {
            'total_items': total_items,
            'stable_items': stable_count,
            'stability_ratio': stability_ratio,
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'avg_abs_diff': np.mean(abs_diffs) if abs_diffs else 0,
            'std_abs_diff': np.std(abs_diffs) if abs_diffs else 0
        }

def process_background_tie_dataset(input_file: str, output_file: str,
                                  processor_config: Dict = None) -> None:
    """
    处理背景TIE数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        processor_config: 处理器配置
    """
    if processor_config is None:
        processor_config = {
            'background_weight': 1.0,
            'context_weight': 0.8,
            'stability_threshold': 0.02
        }
    
    # 创建处理器
    processor = BackgroundTIEProcessor(**processor_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"处理 {len(data)} 条背景TIE数据...")
    
    # 处理数据
    processed_data = processor.process_background_pairs(data)
    
    # 分析稳定性
    stability_analysis = processor.analyze_background_stability(processed_data)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存背景TIE权重结果到: {output_file}")
    
    # 打印统计信息
    weights = [item['background_tie_weight'] for item in processed_data]
    print(f"\n背景TIE权重统计:")
    print(f"  均值: {np.mean(weights):.4f}")
    print(f"  标准差: {np.std(weights):.4f}")
    print(f"  范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    
    print(f"\n稳定性分析:")
    print(f"  总样本数: {stability_analysis['total_items']}")
    print(f"  稳定样本数: {stability_analysis['stable_items']}")
    print(f"  稳定性比例: {stability_analysis['stability_ratio']:.2%}")
    print(f"  平均一致性: {stability_analysis['avg_consistency']:.4f}")
    print(f"  平均绝对差异: {stability_analysis['avg_abs_diff']:.4f}")

def main():
    """
    主函数：演示背景TIE处理
    """
    input_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_background_tie_weights.jsonl"
    
    processor_config = {
        'background_weight': 1.0,
        'context_weight': 0.8,
        'stability_threshold': 0.02
    }
    
    print("开始背景TIE权重计算...")
    print(f"配置: {processor_config}")
    
    process_background_tie_dataset(input_file, output_file, processor_config)
    
    print("\n背景TIE权重计算完成！")

if __name__ == "__main__":
    main()