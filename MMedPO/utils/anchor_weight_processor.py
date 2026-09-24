#!/usr/bin/env python3
"""
Anchor权重处理器
实现m_v负例和符合条件m_n的-anchor形式设置

基于链接中的TIE评分规律，实现anchor形式的权重计算
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tie_weighted_score_calculator import TIEWeightedScoreCalculator

class AnchorWeightProcessor:
    """
    Anchor权重处理器
    
    专门处理m_v负例和符合条件m_n的-anchor形式权重设置
    实现基于TIE指标的anchor样本识别和权重分配
    """
    
    def __init__(self, 
                 mv_negative_threshold: float = 0.0,  # m_v负例阈值
                 mn_positive_threshold: float = 0.1,  # m_n正例阈值
                 anchor_strength: float = 1.0,        # anchor强度
                 anchor_decay: float = 0.9,           # anchor衰减因子
                 min_anchor_weight: float = 0.1):     # 最小anchor权重
        """
        初始化Anchor权重处理器
        
        Args:
            mv_negative_threshold: m_v负例判定阈值
            mn_positive_threshold: m_n正例判定阈值
            anchor_strength: anchor权重强度
            anchor_decay: anchor权重衰减因子
            min_anchor_weight: 最小anchor权重
        """
        self.mv_negative_threshold = mv_negative_threshold
        self.mn_positive_threshold = mn_positive_threshold
        self.anchor_strength = anchor_strength
        self.anchor_decay = anchor_decay
        self.min_anchor_weight = min_anchor_weight
    
    def identify_mv_negative_anchors(self, m_v: float) -> Dict[str, Union[bool, float]]:
        """
        识别m_v负例anchor
        
        当m_v < 0时，表示在全图像条件下负答案的TIE分数高于正答案
        这种情况下应该设置为负anchor，降低该样本的权重
        
        Args:
            m_v: 视觉区分边际 (Tff+ - Tff-)
            
        Returns:
            m_v负例anchor信息
        """
        is_mv_negative = m_v < self.mv_negative_threshold
        
        if is_mv_negative:
            # 计算负anchor权重，m_v越负，anchor权重越大
            anchor_magnitude = abs(m_v)
            anchor_weight = -self.anchor_strength * (1 - np.exp(-anchor_magnitude))
            anchor_weight = max(anchor_weight, -2.0)  # 限制最大负权重
        else:
            anchor_weight = 0.0
        
        return {
            'is_mv_negative_anchor': is_mv_negative,
            'mv_anchor_weight': anchor_weight,
            'mv_anchor_magnitude': abs(m_v) if is_mv_negative else 0.0
        }
    
    def identify_mn_positive_anchors(self, m_n: float) -> Dict[str, Union[bool, float]]:
        """
        识别m_n正例anchor
        
        当m_n显著为正时，表示存在文本/背景先验泄漏
        这种情况下也应该设置为负anchor，因为模型可能过度依赖非视觉信息
        
        Args:
            m_n: 泄漏指标 (Tfn+ - Tfn-)
            
        Returns:
            m_n正例anchor信息
        """
        is_mn_positive = m_n > self.mn_positive_threshold
        
        if is_mn_positive:
            # 计算负anchor权重，m_n越大，anchor权重越大
            anchor_magnitude = m_n - self.mn_positive_threshold
            anchor_weight = -self.anchor_strength * (1 - np.exp(-anchor_magnitude))
            anchor_weight = max(anchor_weight, -2.0)  # 限制最大负权重
        else:
            anchor_weight = 0.0
        
        return {
            'is_mn_positive_anchor': is_mn_positive,
            'mn_anchor_weight': anchor_weight,
            'mn_anchor_magnitude': anchor_magnitude if is_mn_positive else 0.0
        }
    
    def calculate_combined_anchor_weight(self, 
                                       m_v: float, 
                                       m_n: float,
                                       gamma: float = None) -> Dict[str, Union[bool, float]]:
        """
        计算组合anchor权重
        
        综合考虑m_v负例和m_n正例的anchor效应
        
        Args:
            m_v: 视觉区分边际
            m_n: 泄漏指标
            gamma: 前景差分(可选，用于进一步调整)
            
        Returns:
            组合anchor权重信息
        """
        # 识别m_v负例anchor
        mv_anchor = self.identify_mv_negative_anchors(m_v)
        
        # 识别m_n正例anchor
        mn_anchor = self.identify_mn_positive_anchors(m_n)
        
        # 计算组合anchor权重
        total_anchor_weight = mv_anchor['mv_anchor_weight'] + mn_anchor['mn_anchor_weight']
        
        # 如果提供了gamma，进行进一步调整
        gamma_adjustment = 0.0
        if gamma is not None:
            # 当gamma为负时(负答案获得更多视觉增益)，增加负anchor权重
            if gamma < 0:
                gamma_adjustment = -0.5 * abs(gamma) * self.anchor_strength
        
        final_anchor_weight = total_anchor_weight + gamma_adjustment
        
        # 应用衰减因子
        if final_anchor_weight < 0:
            final_anchor_weight *= self.anchor_decay
        
        # 限制最小权重
        if abs(final_anchor_weight) < self.min_anchor_weight and final_anchor_weight != 0:
            final_anchor_weight = -self.min_anchor_weight if final_anchor_weight < 0 else self.min_anchor_weight
        
        # 判断是否为anchor样本
        is_anchor_sample = mv_anchor['is_mv_negative_anchor'] or mn_anchor['is_mn_positive_anchor']
        
        return {
            **mv_anchor,
            **mn_anchor,
            'is_anchor_sample': is_anchor_sample,
            'total_anchor_weight': total_anchor_weight,
            'gamma_adjustment': gamma_adjustment,
            'final_anchor_weight': final_anchor_weight,
            'anchor_type': self._determine_anchor_type(mv_anchor['is_mv_negative_anchor'], 
                                                     mn_anchor['is_mn_positive_anchor'])
        }
    
    def _determine_anchor_type(self, is_mv_negative: bool, is_mn_positive: bool) -> str:
        """
        确定anchor类型
        
        Args:
            is_mv_negative: 是否为m_v负例
            is_mn_positive: 是否为m_n正例
            
        Returns:
            anchor类型字符串
        """
        if is_mv_negative and is_mn_positive:
            return "mixed_anchor"  # 混合anchor
        elif is_mv_negative:
            return "mv_negative_anchor"  # m_v负例anchor
        elif is_mn_positive:
            return "mn_positive_anchor"  # m_n正例anchor
        else:
            return "no_anchor"  # 非anchor
    
    def process_anchor_weights(self, data_items: List[Dict]) -> List[Dict]:
        """
        处理数据集的anchor权重
        
        Args:
            data_items: 包含TIE指标的数据项列表
            
        Returns:
            处理后的数据项列表
        """
        processed_items = []
        
        for item in data_items:
            # 提取TIE指标
            m_v = item.get('m_v', 0.0)
            m_n = item.get('m_n', 0.0)
            gamma = item.get('gamma', None)
            
            # 如果没有预计算的指标，尝试从原始TIE分数计算
            if m_v == 0.0 and m_n == 0.0:
                tff_pos = item.get('ll_positive_with_background', 0)
                tfn_pos = item.get('ll_positive_with_white_background', 0)
                tff_neg = item.get('ll_negative_with_background', 0)
                tfn_neg = item.get('ll_negative_with_white_background', 0)
                
                if tff_pos != 0 or tfn_pos != 0 or tff_neg != 0 or tfn_neg != 0:
                    m_v = tff_pos - tff_neg
                    m_n = tfn_pos - tfn_neg
                    if gamma is None:
                        g_pos = tff_pos - tfn_pos
                        g_neg = tff_neg - tfn_neg
                        gamma = g_pos - g_neg
            
            # 计算anchor权重
            anchor_result = self.calculate_combined_anchor_weight(m_v, m_n, gamma)
            
            # 创建新的数据项
            new_item = item.copy()
            new_item.update({
                'anchor_weight': anchor_result['final_anchor_weight'],
                **anchor_result
            })
            
            processed_items.append(new_item)
        
        return processed_items
    
    def analyze_anchor_distribution(self, data_items: List[Dict]) -> Dict[str, Union[int, float]]:
        """
        分析anchor样本分布
        
        Args:
            data_items: 处理后的数据项列表
            
        Returns:
            anchor分布分析结果
        """
        total_items = len(data_items)
        anchor_counts = {
            'mv_negative_anchor': 0,
            'mn_positive_anchor': 0,
            'mixed_anchor': 0,
            'no_anchor': 0
        }
        
        anchor_weights = []
        
        for item in data_items:
            anchor_type = item.get('anchor_type', 'no_anchor')
            anchor_counts[anchor_type] += 1
            
            if item.get('is_anchor_sample', False):
                anchor_weights.append(item.get('anchor_weight', 0.0))
        
        total_anchors = sum(anchor_counts.values()) - anchor_counts['no_anchor']
        
        return {
            'total_items': total_items,
            'total_anchors': total_anchors,
            'anchor_ratio': total_anchors / total_items if total_items > 0 else 0,
            **{f'{k}_count': v for k, v in anchor_counts.items()},
            **{f'{k}_ratio': v / total_items if total_items > 0 else 0 for k, v in anchor_counts.items()},
            'avg_anchor_weight': np.mean(anchor_weights) if anchor_weights else 0,
            'std_anchor_weight': np.std(anchor_weights) if anchor_weights else 0,
            'min_anchor_weight': np.min(anchor_weights) if anchor_weights else 0,
            'max_anchor_weight': np.max(anchor_weights) if anchor_weights else 0
        }

def process_anchor_weight_dataset(input_file: str, output_file: str,
                                 processor_config: Dict = None) -> None:
    """
    处理anchor权重数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        processor_config: 处理器配置
    """
    if processor_config is None:
        processor_config = {
            'mv_negative_threshold': 0.0,
            'mn_positive_threshold': 0.1,
            'anchor_strength': 1.0,
            'anchor_decay': 0.9,
            'min_anchor_weight': 0.1
        }
    
    # 创建处理器
    processor = AnchorWeightProcessor(**processor_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"处理 {len(data)} 条anchor权重数据...")
    
    # 处理数据
    processed_data = processor.process_anchor_weights(data)
    
    # 分析anchor分布
    anchor_analysis = processor.analyze_anchor_distribution(processed_data)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存anchor权重结果到: {output_file}")
    
    # 打印统计信息
    weights = [item['anchor_weight'] for item in processed_data]
    print(f"\nAnchor权重统计:")
    print(f"  均值: {np.mean(weights):.4f}")
    print(f"  标准差: {np.std(weights):.4f}")
    print(f"  范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    
    print(f"\nAnchor分布分析:")
    print(f"  总样本数: {anchor_analysis['total_items']}")
    print(f"  Anchor样本数: {anchor_analysis['total_anchors']}")
    print(f"  Anchor比例: {anchor_analysis['anchor_ratio']:.2%}")
    print(f"  m_v负例anchor: {anchor_analysis['mv_negative_anchor_count']} ({anchor_analysis['mv_negative_anchor_ratio']:.2%})")
    print(f"  m_n正例anchor: {anchor_analysis['mn_positive_anchor_count']} ({anchor_analysis['mn_positive_anchor_ratio']:.2%})")
    print(f"  混合anchor: {anchor_analysis['mixed_anchor_count']} ({anchor_analysis['mixed_anchor_ratio']:.2%})")
    print(f"  平均anchor权重: {anchor_analysis['avg_anchor_weight']:.4f}")

def main():
    """
    主函数：演示anchor权重处理
    """
    input_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_anchor_weights.jsonl"
    
    processor_config = {
        'mv_negative_threshold': 0.0,
        'mn_positive_threshold': 0.1,
        'anchor_strength': 1.0,
        'anchor_decay': 0.9,
        'min_anchor_weight': 0.1
    }
    
    print("开始anchor权重计算...")
    print(f"配置: {processor_config}")
    
    process_anchor_weight_dataset(input_file, output_file, processor_config)
    
    print("\nAnchor权重计算完成！")

if __name__ == "__main__":
    main()