#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于改进TIE计算的医疗影像DPO训练pairs构建脚本

本脚本基于改进的TIE（Total Intervention Effect）计算结果，
构建更精确的医疗影像DPO（Direct Preference Optimization）训练pairs。

改进的TIE计算包含:
1. 干预强度权重 - 基于背景移除的影响程度动态调整
2. 长度归一化 - 消除答案长度对TIE计算的影响
3. 置信度调整 - 基于对数似然的绝对值调整置信度
4. 多层次TIE - 前景直接效应、背景上下文效应、背景影响比率等
5. 因果强度指标 - 量化因果关系的强度
6. 干预效果指标 - 衡量背景移除的整体效果
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTIETrainingPairsBuilder:
    """基于改进TIE计算的训练pairs构建器"""
    
    def __init__(self, data_path: str, output_dir: str = "training_pairs_improved"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 改进的TIE指标配置
        self.tie_metrics = {
            'adjusted': 'tie_difference_adjusted',  # 主要使用调整后的TIE
            'weighted': 'tie_difference_weighted',  # 权重调整的TIE
            'basic': 'tie_difference_basic',        # 基础归一化TIE
            'original': 'tie_difference_original'   # 原始TIE
        }
        
        # 新增因果指标
        self.causal_metrics = {
            'background_influence': 'background_influence_ratio',
            'causal_strength': 'causal_strength',
            'intervention_effect': 'intervention_effect'
        }
        
    def load_data(self) -> List[Dict[str, Any]]:
        """加载改进TIE计算后的数据"""
        logger.info(f"从 {self.data_path} 加载数据...")
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    
    def analyze_improved_tie_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析改进TIE指标的统计信息"""
        logger.info("分析改进TIE指标统计信息...")
        
        stats = {}
        
        # 分析所有TIE指标
        for metric_name, metric_key in self.tie_metrics.items():
            values = [item[metric_key] for item in data if metric_key in item]
            if values:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        # 分析因果指标
        for metric_name, metric_key in self.causal_metrics.items():
            values = [item[metric_key] for item in data if metric_key in item]
            if values:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return stats
    
    def strategy1_enhanced_positive_selection(self, data: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略1: 增强的正例选择策略
        
        基于改进的TIE指标和因果强度进行更精确的正例选择
        """
        logger.info("执行策略1: 增强的正例选择...")
        
        pairs = []
        
        # 使用调整后的TIE作为主要指标
        tie_threshold = stats['adjusted']['q25']  # 使用25分位数作为阈值
        causal_threshold = stats['causal_strength']['median']  # 因果强度中位数
        
        for item in data:
            # 条件1: 调整后的TIE差异显著（前景主导）
            tie_adjusted = item.get('tie_difference_adjusted', 0)
            
            # 条件2: 因果强度足够
            causal_strength = item.get('causal_strength', 0)
            
            # 条件3: 背景影响比率较低（减少背景偏见）
            bg_influence = item.get('background_influence_ratio', 1)
            
            # 条件4: 干预效果明显
            intervention_effect = item.get('intervention_effect', 0)
            
            if (tie_adjusted < tie_threshold and 
                causal_strength > causal_threshold and
                bg_influence < 0.02 and  # 背景影响小于2%
                intervention_effect > 0.01):  # 干预效果大于1%
                
                pair = {
                    'id': item['id'],
                    'case_id': item['case_id'],
                    'question': item['question'],
                    'prompt': item.get('prompt_text', ''),
                    'chosen': item['positive_answer'],
                    'rejected': item['negative_answer'],
                    
                    # TIE指标
                    'tie_difference_adjusted': tie_adjusted,
                    'tie_difference_weighted': item.get('tie_difference_weighted', 0),
                    'tie_difference_original': item.get('tie_difference_original', 0),
                    
                    # 因果指标
                    'causal_strength': causal_strength,
                    'background_influence_ratio': bg_influence,
                    'intervention_effect': intervention_effect,
                    
                    # 元信息
                    'weighted_score': item.get('weighted_score', 0),
                    'strategy': 'enhanced_positive_selection',
                    'selection_reason': f'TIE_adj={tie_adjusted:.4f}<{tie_threshold:.4f}, causal={causal_strength:.4f}>{causal_threshold:.4f}, bg_inf={bg_influence:.4f}<0.02'
                }
                pairs.append(pair)
        
        logger.info(f"策略1生成 {len(pairs)} 个训练pairs")
        return pairs
    
    def strategy2_adaptive_weight_design(self, data: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略2: 自适应权重设计策略
        
        基于多个TIE指标和因果强度设计自适应训练权重
        """
        logger.info("执行策略2: 自适应权重设计...")
        
        pairs = []
        
        for item in data:
            # 获取多个TIE指标
            tie_adjusted = item.get('tie_difference_adjusted', 0)
            tie_weighted = item.get('tie_difference_weighted', 0)
            tie_original = item.get('tie_difference_original', 0)
            
            # 获取因果指标
            causal_strength = item.get('causal_strength', 0)
            bg_influence = item.get('background_influence_ratio', 0)
            intervention_effect = item.get('intervention_effect', 0)
            
            # 计算自适应权重
            # 基础权重：基于调整后的TIE
            base_weight = np.exp(-abs(tie_adjusted))
            
            # 因果强度权重：因果关系越强，权重越大
            causal_weight = 1 + causal_strength
            
            # 背景影响惩罚：背景影响越大，权重越小
            bg_penalty = 1 / (1 + 10 * bg_influence)
            
            # 干预效果奖励：干预效果越明显，权重越大
            intervention_bonus = 1 + intervention_effect
            
            # 综合权重
            training_weight = base_weight * causal_weight * bg_penalty * intervention_bonus
            
            # 权重归一化到[0.1, 2.0]范围
            training_weight = np.clip(training_weight, 0.1, 2.0)
            
            pair = {
                'id': item['id'],
                'case_id': item['case_id'],
                'question': item['question'],
                'prompt': item.get('prompt_text', ''),
                'chosen': item['positive_answer'],
                'rejected': item['negative_answer'],
                
                # TIE指标
                'tie_difference_adjusted': tie_adjusted,
                'tie_difference_weighted': tie_weighted,
                'tie_difference_original': tie_original,
                
                # 因果指标
                'causal_strength': causal_strength,
                'background_influence_ratio': bg_influence,
                'intervention_effect': intervention_effect,
                
                # 权重信息
                'training_weight': training_weight,
                'base_weight': base_weight,
                'causal_weight': causal_weight,
                'bg_penalty': bg_penalty,
                'intervention_bonus': intervention_bonus,
                
                # 元信息
                'weighted_score': item.get('weighted_score', 0),
                'strategy': 'adaptive_weight_design',
                'weight_formula': 'exp(-|TIE_adj|) * (1+causal) * (1/(1+10*bg_inf)) * (1+intervention)'
            }
            pairs.append(pair)
        
        logger.info(f"策略2生成 {len(pairs)} 个训练pairs")
        return pairs
    
    def strategy3_multi_level_anchor_design(self, data: List[Dict[str, Any]], stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """策略3: 多层次锚点设计策略
        
        基于多个TIE指标和因果强度设计多层次锚点
        """
        logger.info("执行策略3: 多层次锚点设计...")
        
        pairs = []
        
        # 计算多层次阈值
        tie_adj_q25 = stats['adjusted']['q25']
        tie_adj_q75 = stats['adjusted']['q75']
        
        causal_q25 = stats['causal_strength']['q25']
        causal_q75 = stats['causal_strength']['q75']
        
        bg_inf_median = stats['background_influence']['median']
        
        anchor_info = {
            'tie_adjusted_thresholds': {
                'strong_preference': tie_adj_q25,
                'weak_preference': tie_adj_q75
            },
            'causal_strength_thresholds': {
                'weak_causal': causal_q25,
                'strong_causal': causal_q75
            },
            'background_influence_threshold': bg_inf_median,
            'anchor_counts': {
                'strong_preference': 0,
                'weak_preference': 0,
                'strong_dispreference': 0,
                'weak_dispreference': 0,
                'causal_strong': 0,
                'causal_weak': 0
            }
        }
        
        for item in data:
            tie_adjusted = item.get('tie_difference_adjusted', 0)
            causal_strength = item.get('causal_strength', 0)
            bg_influence = item.get('background_influence_ratio', 0)
            intervention_effect = item.get('intervention_effect', 0)
            
            # 确定锚点类型
            anchor_type = None
            anchor_reason = ""
            
            # TIE锚点判断
            if tie_adjusted < tie_adj_q25:
                if causal_strength > causal_q75:
                    anchor_type = 'strong_preference'
                    anchor_reason = f'强前景主导(TIE_adj={tie_adjusted:.4f}<{tie_adj_q25:.4f}) + 强因果关系({causal_strength:.4f}>{causal_q75:.4f})'
                    anchor_info['anchor_counts']['strong_preference'] += 1
                else:
                    anchor_type = 'weak_preference'
                    anchor_reason = f'前景主导(TIE_adj={tie_adjusted:.4f}<{tie_adj_q25:.4f}) + 弱因果关系'
                    anchor_info['anchor_counts']['weak_preference'] += 1
            
            elif tie_adjusted > tie_adj_q75:
                if bg_influence > bg_inf_median:
                    anchor_type = 'strong_dispreference'
                    anchor_reason = f'背景主导(TIE_adj={tie_adjusted:.4f}>{tie_adj_q75:.4f}) + 高背景影响({bg_influence:.4f}>{bg_inf_median:.4f})'
                    anchor_info['anchor_counts']['strong_dispreference'] += 1
                else:
                    anchor_type = 'weak_dispreference'
                    anchor_reason = f'背景主导(TIE_adj={tie_adjusted:.4f}>{tie_adj_q75:.4f}) + 低背景影响'
                    anchor_info['anchor_counts']['weak_dispreference'] += 1
            
            # 因果强度锚点
            if causal_strength > causal_q75:
                anchor_info['anchor_counts']['causal_strong'] += 1
            elif causal_strength < causal_q25:
                anchor_info['anchor_counts']['causal_weak'] += 1
            
            if anchor_type:
                pair = {
                    'id': item['id'],
                    'case_id': item['case_id'],
                    'question': item['question'],
                    'prompt': item.get('prompt_text', ''),
                    'response': item['positive_answer'],
                    
                    # TIE指标
                    'tie_difference_adjusted': tie_adjusted,
                    'tie_difference_weighted': item.get('tie_difference_weighted', 0),
                    'tie_difference_original': item.get('tie_difference_original', 0),
                    
                    # 因果指标
                    'causal_strength': causal_strength,
                    'background_influence_ratio': bg_influence,
                    'intervention_effect': intervention_effect,
                    
                    # 锚点信息
                    'anchor_type': anchor_type,
                    'anchor_reason': anchor_reason,
                    
                    # 元信息
                    'weighted_score': item.get('weighted_score', 0),
                    'strategy': 'multi_level_anchor_design'
                }
                pairs.append(pair)
        
        logger.info(f"策略3生成 {len(pairs)} 个训练pairs")
        return pairs, anchor_info
    
    def save_results(self, pairs: List[Dict[str, Any]], filename: str):
        """保存训练pairs到文件"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {len(pairs)} 个训练pairs到: {output_path}")
    
    def build_all_strategies(self):
        """构建所有策略的训练pairs"""
        logger.info("开始构建基于改进TIE的训练pairs...")
        
        # 加载数据
        data = self.load_data()
        
        # 分析统计信息
        stats = self.analyze_improved_tie_statistics(data)
        
        # 打印统计信息
        print("\n=== 改进TIE指标统计信息 ===")
        for metric_name, metric_stats in stats.items():
            if metric_name in ['adjusted', 'weighted', 'original']:
                print(f"{metric_name}_tie:")
                print(f"  均值: {metric_stats['mean']:.4f}")
                print(f"  标准差: {metric_stats['std']:.4f}")
                print(f"  范围: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
                print(f"  中位数: {metric_stats['median']:.4f}")
                print(f"  25%-75%分位数: [{metric_stats['q25']:.4f}, {metric_stats['q75']:.4f}]")
                print()
        
        print("=== 因果指标统计信息 ===")
        for metric_name, metric_stats in stats.items():
            if metric_name in ['causal_strength', 'background_influence', 'intervention_effect']:
                print(f"{metric_name}:")
                print(f"  均值: {metric_stats['mean']:.4f}")
                print(f"  标准差: {metric_stats['std']:.4f}")
                print(f"  范围: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
                print(f"  中位数: {metric_stats['median']:.4f}")
                print()
        
        # 保存统计信息
        stats_path = self.output_dir / "improved_tie_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 策略1: 增强的正例选择
        strategy1_pairs = self.strategy1_enhanced_positive_selection(data, stats)
        self.save_results(strategy1_pairs, "strategy1_enhanced_positive_selection.jsonl")
        
        # 策略2: 自适应权重设计
        strategy2_pairs = self.strategy2_adaptive_weight_design(data, stats)
        self.save_results(strategy2_pairs, "strategy2_adaptive_weight_design.jsonl")
        
        # 策略3: 多层次锚点设计
        strategy3_pairs, anchor_info = self.strategy3_multi_level_anchor_design(data, stats)
        self.save_results(strategy3_pairs, "strategy3_multi_level_anchor_design.jsonl")
        
        # 保存锚点信息
        anchor_path = self.output_dir / "improved_anchor_info.json"
        with open(anchor_path, 'w', encoding='utf-8') as f:
            json.dump(anchor_info, f, ensure_ascii=False, indent=2)
        
        # 创建组合策略示例
        combined_pairs = []
        
        # 从策略1选择高质量样本
        high_quality = [p for p in strategy1_pairs if p['causal_strength'] > stats['causal_strength']['q75']]
        combined_pairs.extend(high_quality[:20])  # 取前20个
        
        # 从策略2选择高权重样本
        high_weight = sorted(strategy2_pairs, key=lambda x: x['training_weight'], reverse=True)[:30]
        combined_pairs.extend(high_weight)
        
        # 从策略3选择强偏好锚点
        strong_anchors = [p for p in strategy3_pairs if p['anchor_type'] == 'strong_preference']
        combined_pairs.extend(strong_anchors[:15])
        
        self.save_results(combined_pairs, "combined_improved_strategy_pairs.jsonl")
        
        print(f"\n=== 改进训练pairs构建完成 ===")
        print(f"策略1 (增强正例选择): {len(strategy1_pairs)} pairs")
        print(f"策略2 (自适应权重设计): {len(strategy2_pairs)} pairs")
        print(f"策略3 (多层次锚点设计): {len(strategy3_pairs)} pairs")
        print(f"组合策略: {len(combined_pairs)} pairs")
        print(f"\n所有文件已保存到: {self.output_dir}")

def main():
    """主函数"""
    # 数据路径
    data_path = "/workspace/MMedPO/datasets/Slake1.0/dpo_tie_improved_calculation.jsonl"
    
    # 构建训练pairs
    builder = ImprovedTIETrainingPairsBuilder(data_path)
    builder.build_all_strategies()

if __name__ == "__main__":
    main()