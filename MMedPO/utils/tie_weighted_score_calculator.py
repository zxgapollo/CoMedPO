#!/usr/bin/env python3
"""
基于TIE评分规律的权重计算器
根据 https://chatgpt.com/share/68c11835-092c-800a-9bd9-221d57223d18 中的规律实现

核心指标:
1. γ (gamma): 前景差分 = (Tff+ - Tfn+) - (Tff- - Tfn-)
2. m_v: 视觉区分边际 = Tff+ - Tff-
3. m_n: 泄漏(文本/背景先验) = Tfn+ - Tfn-
4. g+: 正答案视觉增益 = Tff+ - Tfn+
5. g-: 负答案视觉增益 = Tff- - Tfn-
"""

import json
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
import torch.nn.functional as F

class TIEWeightedScoreCalculator:
    """
    基于TIE评分规律的权重计算器
    
    实现链接中提到的核心评分公式和权重计算方法
    """
    
    def __init__(self, 
                 w1: float = 1.0,  # γ权重
                 w2: float = 0.5,  # m_v权重
                 w3: float = 0.5,  # m_n惩罚权重
                 beta: float = 1.0,  # sigmoid缩放因子
                 lambda_disp: float = 1.0,  # 反偏好λ
                 mu_disp: float = 0.5,  # 反偏好μ
                 tie_diff_threshold: float = 0.05):  # TIE差值阈值
        """
        初始化权重计算器
        
        Args:
            w1: γ(前景差分)的权重
            w2: m_v(视觉区分边际)的权重  
            w3: m_n(泄漏惩罚)的权重
            beta: sigmoid函数的缩放因子
            lambda_disp: 反偏好计算中的λ参数
            mu_disp: 反偏好计算中的μ参数
            tie_diff_threshold: TIE差值的阈值(positive>0.05)
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.beta = beta
        self.lambda_disp = lambda_disp
        self.mu_disp = mu_disp
        self.tie_diff_threshold = tie_diff_threshold
    
    def calculate_core_metrics(self, 
                              tff_pos: float,  # Tff+: bg+full image下正解的TIE
                              tfn_pos: float,  # Tfn+: bg+null image下正解的TIE
                              tff_neg: float,  # Tff-: bg+full image下负解的TIE
                              tfn_neg: float   # Tfn-: bg+null image下负解的TIE
                              ) -> Dict[str, float]:
        """
        计算核心TIE指标
        
        Args:
            tff_pos: 带背景+完整图像下正确答案的TIE分数
            tfn_pos: 带背景+空白图像下正确答案的TIE分数
            tff_neg: 带背景+完整图像下错误答案的TIE分数
            tfn_neg: 带背景+空白图像下错误答案的TIE分数
            
        Returns:
            包含所有核心指标的字典
        """
        # 视觉增益(单答案)
        g_pos = tff_pos - tfn_pos  # g+: 正答案视觉增益
        g_neg = tff_neg - tfn_neg  # g-: 负答案视觉增益
        
        # 视觉区分边际
        m_v = tff_pos - tff_neg    # 全图像下正>负的边际
        
        # 泄漏(文本/背景先验)
        m_n = tfn_pos - tfn_neg    # 空白图像下正>负的边际
        
        # 核心"前景差分"(正对负的前景相对增益)
        gamma = g_pos - g_neg      # γ = (Tff+ - Tfn+) - (Tff- - Tfn-)
        
        return {
            'g_pos': g_pos,
            'g_neg': g_neg,
            'm_v': m_v,
            'm_n': m_n,
            'gamma': gamma,
            'tff_pos': tff_pos,
            'tfn_pos': tfn_pos,
            'tff_neg': tff_neg,
            'tfn_neg': tfn_neg
        }
    
    def calculate_preference_score(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        计算偏好分数
        
        基于公式: S_pref = w1*γ + w2*m_v - w3*max(0, m_n)
        """
        gamma = metrics['gamma']
        m_v = metrics['m_v']
        m_n = metrics['m_n']
        
        # 偏好分数计算
        max_m_n = max(0, m_n)
        s_pref = self.w1 * gamma + self.w2 * m_v - self.w3 * max_m_n
        
        # 映射到[0,1]的偏好概率
        p_pref = torch.sigmoid(torch.tensor(self.beta * s_pref)).item()
        
        return {
            's_pref': s_pref,
            'p_pref': p_pref,
            'preference_score_raw': s_pref,
            'preference_probability': p_pref,
            'pref_w1': self.w1,
            'pref_w2': self.w2,
            'pref_w3': self.w3,
            'pref_beta': self.beta,
            'max_m_n': max_m_n
        }
    
    def calculate_dispreference_score(self, metrics: Dict[str, float], 
                                    tau_n: float = None) -> Dict[str, float]:
        """
        计算反偏好分数
        
        基于公式: S_disp = max(0,-m_v) + λ*max(0,g--g+) + μ*max(0,m_n-τ_n)
        """
        m_v = metrics['m_v']
        g_pos = metrics['g_pos']
        g_neg = metrics['g_neg']
        m_n = metrics['m_n']
        
        # 如果没有提供τ_n，使用m_n的75分位数作为阈值
        if tau_n is None:
            tau_n = max(0, m_n * 0.75)  # 简化处理
        
        # 反偏好分数计算
        term1 = max(0, -m_v)  # 全图像下负>正
        term2 = self.lambda_disp * max(0, g_neg - g_pos)  # 前景更"帮助"负解
        term3 = self.mu_disp * max(0, m_n - tau_n)  # 显著泄漏
        
        s_disp = term1 + term2 + term3
        
        # 映射到[0,1]的反偏好概率
        p_disp = torch.sigmoid(torch.tensor(self.beta * s_disp)).item()
        
        return {
            's_disp': s_disp,
            'p_disp': p_disp,
            'tau_n': tau_n,
            'dispreference_score_raw': s_disp,
            'dispreference_probability': p_disp,
            'disp_term1': term1,
            'disp_term2': term2,
            'disp_term3': term3,
            'disp_lambda': self.lambda_disp,
            'disp_mu': self.mu_disp
        }
    
    def calculate_tie_diff_weight(self, 
                                 tie_gt: float,    # (y_gt|I)的TIE
                                 tie_gen: float    # (y_gen|I)的TIE
                                 ) -> Dict[str, float]:
        """
        计算(y_gt|I) vs (y_gen|I)的TIE差值权重
        
        当positive > 0.05时给予权重
        """
        tie_diff = tie_gt - tie_gen
        
        # 只有当差值大于阈值时才给予正权重
        if tie_diff > self.tie_diff_threshold:
            weight = min(1.0, tie_diff / (2 * self.tie_diff_threshold))  # 归一化到[0,1]
        else:
            weight = 0.0
        
        return {
            'tie_diff': tie_diff,
            'tie_diff_weight': weight,
            'meets_threshold': tie_diff > self.tie_diff_threshold
        }
    
    def calculate_anchor_weights(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        计算anchor形式的权重
        
        m_v的负例和符合条件的m_n都可以设置成-anchor的形式
        """
        m_v = metrics['m_v']
        m_n = metrics['m_n']
        
        # m_v负例权重 (当m_v < 0时，视为负anchor)
        mv_anchor_weight = -m_v if m_v < 0 else 0.0
        
        # m_n符合条件时的负anchor权重 (当m_n显著为正时)
        mn_threshold = 0.1  # 可调整的阈值
        mn_anchor_weight = -m_n if m_n > mn_threshold else 0.0
        
        return {
            'mv_anchor_weight': mv_anchor_weight,
            'mn_anchor_weight': mn_anchor_weight,
            'total_anchor_weight': mv_anchor_weight + mn_anchor_weight
        }
    
    def calculate_comprehensive_weight(self, 
                                     tff_pos: float, tfn_pos: float,
                                     tff_neg: float, tfn_neg: float,
                                     tie_gt: Optional[float] = None,
                                     tie_gen: Optional[float] = None) -> Dict[str, float]:
        """
        计算综合权重分数
        
        整合所有权重计算方法
        """
        # 计算核心指标
        metrics = self.calculate_core_metrics(tff_pos, tfn_pos, tff_neg, tfn_neg)
        
        # 计算偏好分数
        pref_scores = self.calculate_preference_score(metrics)
        
        # 计算反偏好分数
        disp_scores = self.calculate_dispreference_score(metrics)
        
        # 计算anchor权重
        anchor_weights = self.calculate_anchor_weights(metrics)
        
        # 如果提供了TIE差值数据，计算TIE差值权重
        tie_diff_scores = {}
        if tie_gt is not None and tie_gen is not None:
            tie_diff_scores = self.calculate_tie_diff_weight(tie_gt, tie_gen)
        
        # 综合权重计算
        base_weight = pref_scores['p_pref'] - disp_scores['p_disp']
        anchor_adjustment = anchor_weights['total_anchor_weight']
        
        # 最终权重
        final_weight = base_weight + anchor_adjustment
        
        # 确保权重在合理范围内
        final_weight = max(0.0, min(2.0, final_weight))
        
        # 整合所有结果
        result = {
            **metrics,
            **pref_scores,
            **disp_scores,
            **anchor_weights,
            **tie_diff_scores,
            'base_weight': base_weight,
            'final_weight': final_weight
        }
        
        return result

def process_dataset_with_tie_weights(input_file: str, output_file: str,
                                   calculator_config: Dict = None) -> None:
    """
    使用TIE权重计算器处理数据集
    """
    if calculator_config is None:
        calculator_config = {
            'w1': 1.0, 'w2': 0.5, 'w3': 0.5,
            'beta': 1.0, 'lambda_disp': 1.0, 'mu_disp': 0.5,
            'tie_diff_threshold': 0.05
        }
    
    calculator = TIEWeightedScoreCalculator(**calculator_config)
    
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"处理 {len(data)} 条数据...")
    
    # 处理每条数据
    processed_data = []
    for item in data:
        # 提取TIE分数
        tff_pos = item.get('ll_positive_with_background', 0)
        tfn_pos = item.get('ll_positive_with_white_background', 0)
        tff_neg = item.get('ll_negative_with_background', 0)
        tfn_neg = item.get('ll_negative_with_white_background', 0)
        
        # 提取TIE差值数据(如果有)
        tie_gt = item.get('tie_gt', None)
        tie_gen = item.get('tie_gen', None)
        
        # 计算权重
        weight_result = calculator.calculate_comprehensive_weight(
            tff_pos, tfn_pos, tff_neg, tfn_neg, tie_gt, tie_gen
        )
        
        # 创建新的数据项
        new_item = item.copy()
        new_item.update({
            'tie_weighted_score': weight_result['final_weight'],
            **weight_result,
            'calculator_config': calculator_config
        })
        
        processed_data.append(new_item)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已保存TIE权重计算结果到: {output_file}")
    
    # 打印统计信息
    weights = [item['tie_weighted_score'] for item in processed_data]
    print(f"\n权重统计:")
    print(f"  均值: {np.mean(weights):.4f}")
    print(f"  标准差: {np.std(weights):.4f}")
    print(f"  范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

def main():
    """
    主函数：演示TIE权重计算
    """
    input_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_comparison_base_model.jsonl"
    output_file = "/workspace/MMedPO_changed/datasets/Slake1.0/dpo_tie_weighted_scores.jsonl"
    
    calculator_config = {
        'w1': 1.0,  # γ权重
        'w2': 0.5,  # m_v权重
        'w3': 0.5,  # m_n惩罚权重
        'beta': 1.0,  # sigmoid缩放
        'lambda_disp': 1.0,  # 反偏好λ
        'mu_disp': 0.5,  # 反偏好μ
        'tie_diff_threshold': 0.05  # TIE差值阈值
    }
    
    print("开始TIE权重计算...")
    print(f"配置: {calculator_config}")
    
    process_dataset_with_tie_weights(input_file, output_file, calculator_config)
    
    print("\nTIE权重计算完成！")

if __name__ == "__main__":
    main()