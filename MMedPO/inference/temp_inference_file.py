#!/usr/bin/env python3
import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import sys
import time
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import warnings
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 添加GPU监控
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("Warning: nvidia-ml-py3 not available, GPU monitoring disabled")

def get_gpu_info():
    """获取GPU信息"""
    if not GPU_MONITORING:
        return "GPU监控不可用"
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / 1024**3
        total_gb = mem_info.total / 1024**3
        return f"GPU利用率: {util.gpu}%, 内存: {used_gb:.1f}/{total_gb:.1f}GB"
    except:
        return "GPU信息获取失败"

def get_gpu_utilization():
    """获取GPU利用率"""
    if not GPU_MONITORING:
        return "N/A"
    
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        return f"{util.gpu}%"
    except:
        return "N/A"

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if not GPU_MONITORING:
        return "N/A"
    
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / 1024**3
        total_gb = mem_info.total / 1024**3
        return f"{used_gb:.1f}/{total_gb:.1f}GB"
    except:
        return "N/A"

class PerformanceMonitor:
    """性能监控类"""
    def __init__(self):
        self.start_time = time.time()
        self.processed_count = 0
        self.last_update_time = time.time()
        self.last_processed_count = 0
        
    def update(self, processed_count):
        """更新处理进度"""
        self.processed_count = processed_count
        current_time = time.time()
        
        # 计算总体统计
        total_elapsed = current_time - self.start_time
        if total_elapsed > 0:
            avg_speed = processed_count / total_elapsed * 3600  # 样本/小时
        else:
            avg_speed = 0
            
        # 计算瞬时速度
        time_diff = current_time - self.last_update_time
        count_diff = processed_count - self.last_processed_count
        if time_diff > 0:
            instant_speed = count_diff / time_diff * 3600  # 样本/小时
        else:
            instant_speed = 0
            
        self.last_update_time = current_time
        self.last_processed_count = processed_count
        
        return {
            'avg_speed': avg_speed,
            'instant_speed': instant_speed,
            'total_elapsed': total_elapsed,
            'gpu_util': get_gpu_utilization(),
            'gpu_memory': get_gpu_memory()
        }


# --- Original Python Path Setup ---
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)
import llava.model.language_model.llava_mistral
# --- End Path Setup ---

warnings.simplefilter(action="ignore", category=FutureWarning)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
)
from PIL import Image
import math
from transformers import set_seed, logging
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable

logging.set_verbosity_error()

def normalize_text(text):
    """
    标准化文本用于相似度计算
    """
    if not text:
        return ""
    # 转换为小写
    text = text.lower()
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def calculate_semantic_similarity(text1, text2, method='tfidf'):
    """
    计算两个文本之间的语义相似度
    
    Args:
        text1, text2: 要比较的文本
        method: 相似度计算方法 ('tfidf', 'sequence', 'combined')
        
    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # 标准化文本
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    if not norm_text1 or not norm_text2:
        return 0.0
    
    if method == 'sequence':
        # 使用序列匹配器
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    elif method == 'tfidf':
        # 使用TF-IDF向量化
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([norm_text1, norm_text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # 如果TF-IDF失败，回退到序列匹配
            return SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    elif method == 'combined':
        # 组合多种方法
        seq_sim = SequenceMatcher(None, norm_text1, norm_text2).ratio()
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([norm_text1, norm_text2])
            tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return (seq_sim + tfidf_sim) / 2.0
        except:
            return seq_sim
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def check_conversation_validity(question, answer, min_similarity=0.1, min_length=10):
    """
    检查问题和答案的逻辑一致性
    
    Args:
        question: 问题文本
        answer: 答案文本
        min_similarity: 最小相似度阈值（降低到0.1）
        min_length: 最小答案长度
        
    Returns:
        bool: 是否有效
    """
    if not question or not answer:
        return False
    
    # 检查答案长度是否合理
    if len(answer.strip()) < min_length:
        return False
    
    # 检查是否包含明显的错误模式
    answer_lower = answer.lower().strip()
    question_lower = question.lower().strip()
    
    # 过滤明显无效的答案
    invalid_patterns = ['yes', 'no', 'ok', 'sure', 'maybe', 'i don\'t know']
    if answer_lower in invalid_patterns:
        return False
    
    # 检查答案是否与问题相关（使用较低的阈值）
    similarity = calculate_semantic_similarity(question, answer, method='combined')
    
    # 避免完全无关的答案
    if similarity < min_similarity:
        return False
    
    return True

def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
    if image1.height != image2.height:
        new_height = max(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))

    width1, height1 = image1.size
    width2, height2 = image2.size
    combined_image = Image.new('RGB', (width1 + width2, height1))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))
    return combined_image

def calculate_dpo_weight(delta_pos, delta_neg, m_v=None, m_n=None, delta_obj=None, 
                        alpha=0.5, beta=0.5, eta=0.3, tau_n_percentile=80,
                        K=10, beta_T=1.0, kappa=0.5, w_min=0.01, w_max=10.0,
                        tau_flip=0.3, rho=0.1, batch_stats=None, normalize_weights=True):
    """
    计算基于TIE特征的DPO权重（修复版本，包含权重归一化）
    
    根据TIE_formula_summary.md第10节的公式实现DPO权重计算
    
    Args:
        delta_pos: Δ⁺ (tie_positive_token_avg)
        delta_neg: Δ⁻ (tie_negative_token_avg)  
        m_v: LL_pos(bg+full) - LL_neg(bg+full) (可选)
        m_n: LL_pos(bg+white) - LL_neg(bg+white) (可选)
        delta_obj: Δ⁺_obj - Δ⁻_obj (object-only差分，可选)
        alpha, beta, eta: 合成权重参数
        tau_n_percentile: m_n的百分位阈值
        K: 标准化裁剪上限
        beta_T: 温度参数
        kappa: 幂缩放参数
        w_min, w_max: 权重裁剪范围
        tau_flip: 翻转阈值
        rho: 翻转时的权重缩放因子
        batch_stats: 批统计信息 (mu, sigma, tau_n, weight_mean, weight_std)
        normalize_weights: 是否对最终权重进行归一化
        
    Returns:
        dict: 包含计算结果的字典
    """
    # 计算gamma (preference strength: m_v - m_n)
    gamma = m_v - m_n if (m_v is not None and m_n is not None) else 0.0
    
    # 合成偏好原始分 S_raw = Δ⁺ - Δ⁻ + α*m_v - β*max(0, m_n - τ_n) + η*δ
    # 基础项：Δ⁺ - Δ⁻ (即 delta_obj)
    S_raw = delta_obj if delta_obj is not None else (delta_pos - delta_neg)
    
    # 添加辅助项
    if m_v is not None:
        S_raw += alpha * m_v
    if m_n is not None and batch_stats is not None and 'tau_n' in batch_stats:
        tau_n = batch_stats['tau_n']
        S_raw -= beta * max(0, m_n - tau_n)
    # 注意：delta_obj已经包含在基础项中，不需要再次添加
    
    # 批内标准化
    if batch_stats is not None:
        mu = batch_stats.get('mu', 0.0)
        sigma = batch_stats.get('sigma', 1.0)
        S_normalized = np.clip((S_raw - mu) / sigma, -K, K)
    else:
        # 如果没有批统计，使用原始值
        S_normalized = np.clip(S_raw, -K, K)
    
    # Sigmoid映射到偏好概率
    p = 1.0 / (1.0 + np.exp(-beta_T * S_normalized))
    
    # 幂缩放
    w_raw = np.clip(p ** kappa, w_min, w_max)
    
    # 强反偏好处理 (flip)
    if delta_neg - delta_pos > tau_flip:
        w_raw = rho * w_raw
    
    # 权重归一化（修复归一化问题）
    if normalize_weights and batch_stats is not None:
        weight_mean = batch_stats.get('weight_mean', 1.0)
        weight_std = batch_stats.get('weight_std', 1.0)
        
        # Z-score 归一化
        w_normalized = (w_raw - weight_mean) / weight_std
        
        # 重新映射到合理范围 [w_min, w_max]
        # 使用 sigmoid 函数将归一化权重映射回原始范围
        w_sigmoid = 1.0 / (1.0 + np.exp(-w_normalized))
        w_final = w_min + (w_max - w_min) * w_sigmoid
    else:
        w_final = w_raw
    
    return {
        'gamma': gamma,
        'S_raw': S_raw,
        'S_normalized': S_normalized,
        'preference_prob': p,
        'dpo_weight_raw': round(w_raw, 4),
        'dpo_weight': round(w_final, 4),
        'normalized': normalize_weights and batch_stats is not None,
        'flipped': delta_neg - delta_pos > tau_flip
    }

def calculate_batch_statistics(all_results, tau_n_percentile=80, include_weights=True):
    """
    计算批统计信息用于DPO权重标准化（修复版本）
    
    Args:
        all_results: 包含TIE计算结果的列表
        tau_n_percentile: m_n的百分位阈值
        include_weights: 是否计算权重统计信息
        
    Returns:
        dict: 批统计信息，包含权重归一化参数
    """
    if not all_results:
        return {
            'mu': 0.0, 
            'sigma': 1.0, 
            'tau_n': 0.0,
            'weight_mean': 1.0,
            'weight_std': 1.0,
            'weight_min': 0.01,
            'weight_max': 10.0
        }
    
    # 提取所有的S_raw值用于计算均值和标准差
    S_raw_values = []
    m_n_values = []
    weight_values = []
    
    for result in all_results:
        if 'S_raw' in result:
            S_raw_values.append(result['S_raw'])
        if 'm_n' in result:
            m_n_values.append(result['m_n'])
        
        # 如果需要计算权重统计，提取权重值
        if include_weights:
            if 'dpo_weight_raw' in result:
                weight_values.append(result['dpo_weight_raw'])
            elif 'dpo_weight' in result:
                weight_values.append(result['dpo_weight'])
    
    # 计算基础统计量
    mu = np.mean(S_raw_values) if S_raw_values else 0.0
    sigma = np.std(S_raw_values) + 1e-8 if S_raw_values else 1.0  # 添加小值避免除零
    tau_n = np.percentile(m_n_values, tau_n_percentile) if m_n_values else 0.0
    
    # 计算权重统计量（修复归一化问题）
    if include_weights and weight_values:
        weight_mean = np.mean(weight_values)
        weight_std = np.std(weight_values) + 1e-8  # 避免除零
        weight_min = np.min(weight_values)
        weight_max = np.max(weight_values)
    else:
        weight_mean = 1.0
        weight_std = 1.0
        weight_min = 0.01
        weight_max = 10.0
    
    return {
        'mu': mu,
        'sigma': sigma,
        'tau_n': tau_n,
        'weight_mean': weight_mean,
        'weight_std': weight_std,
        'weight_min': weight_min,
        'weight_max': weight_max,
        'n_samples': len(S_raw_values),
        'n_weights': len(weight_values) if include_weights else 0
    }

def build_tie_anker_dpo_pairs(results, args):
    """
    构建基于 Anchor 的 TIE-ANKER DPO训练对
    
    Args:
        results: 包含TIE分数和其他指标的结果列表
        args: 命令行参数，包含TIE-ANKER配置
        
    Returns:
        list: DPO训练对列表
    """
    if not args.enable_tie_anker:
        return []
    
    # 转换字符串参数为布尔值
    token_avg = args.token_avg.lower() == 'true' if isinstance(args.token_avg, str) else args.token_avg
    use_per_case_zscore = args.use_per_case_zscore.lower() == 'true' if isinstance(args.use_per_case_zscore, str) else args.use_per_case_zscore
    
    dpo_pairs = []
    
    # 计算批统计信息用于权重标准化
    all_weights = []
    for result in results:
        if result.get('calculate_tie', False):
            # 计算原始权重
            tie_pos = result.get('tie_pos_token_avg', 0) if token_avg else result.get('tie_positive', 0)
            tie_neg = result.get('tie_neg_token_avg', 0) if token_avg else result.get('tie_negative', 0)
            tie_diff = tie_pos - tie_neg
            
            weight = calculate_tie_anker_weight(
                tie_diff, 
                result.get('delta_pos', 0),
                result.get('delta_neg', 0), 
                result.get('m_v', 0),
                result.get('m_n', 0),
                result.get('gamma', 0),
                args
            )
            all_weights.append(weight)
    
    # 计算权重归一化参数
    if all_weights:
        weight_mean = np.mean(all_weights)
        weight_std = np.std(all_weights) + 1e-8
        weight_min = np.min(all_weights)
        weight_max = np.max(all_weights)
    else:
        weight_mean = 0.0
        weight_std = 1.0
        weight_min = 0.0
        weight_max = 1.0
    
    # 按问题分组处理候选答案
    question_groups = {}
    for result in results:
        if not result.get('calculate_tie', False):
            continue
        
        question = result.get('question', '')
        if question not in question_groups:
            question_groups[question] = []
        question_groups[question].append(result)
    
    for question, candidates in question_groups.items():
        if len(candidates) < 2:  # 至少需要2个候选才能构成pair
            continue
        
        # 检查是否有 Anchor 答案
        anchor_answer = None
        anchor_candidate = None
        
        # 寻找 Anchor（可以是标注的正确答案或高置信度答案）
        for candidate in candidates:
            # 检查是否有明确的 anchor 标记
            if candidate.get('is_anchor', False) or candidate.get('anchor_answer'):
                anchor_answer = candidate.get('anchor_answer') or candidate.get('positive_answer')
                anchor_candidate = candidate
                break
        
        # 如果没有明确的 anchor，选择 TIE 分数最高的作为 anchor
        if not anchor_answer:
            best_candidate = max(candidates, key=lambda x: x.get('tie_positive', 0) - x.get('tie_negative', 0))
            anchor_answer = best_candidate.get('positive_answer')
            anchor_candidate = best_candidate
        
        # 为每个候选计算分数
        candidate_scores = []
        for candidate in candidates:
            # 计算 TIE 权重
            tie_pos = candidate.get('tie_pos_token_avg', 0) if token_avg else candidate.get('tie_positive', 0)
            tie_neg = candidate.get('tie_neg_token_avg', 0) if token_avg else candidate.get('tie_negative', 0)
            tie_diff = tie_pos - tie_neg
            
            tie_weight = calculate_tie_anker_weight(
                tie_diff, 
                candidate.get('delta_pos', 0),
                candidate.get('delta_neg', 0), 
                candidate.get('m_v', 0),
                candidate.get('m_n', 0),
                candidate.get('gamma', 0),
                args
            )
            
            # 归一化权重
            normalized_weight = (tie_weight - weight_mean) / weight_std
            
            # 计算与 Anchor 的相似度
            pos_answer = candidate.get('positive_answer', '')
            neg_answer = candidate.get('negative_answer', '')
            
            anchor_sim_pos = calculate_semantic_similarity(anchor_answer, pos_answer, method='combined')
            anchor_sim_neg = calculate_semantic_similarity(anchor_answer, neg_answer, method='combined')
            
            # 检查 conversation 有效性
            pos_valid = check_conversation_validity(question, pos_answer)
            neg_valid = check_conversation_validity(question, neg_answer)
            
            candidate_scores.append({
                'candidate': candidate,
                'tie_weight': tie_weight,
                'normalized_weight': normalized_weight,
                'anchor_sim_pos': anchor_sim_pos,
                'anchor_sim_neg': anchor_sim_neg,
                'pos_valid': pos_valid,
                'neg_valid': neg_valid,
                'pos_answer': pos_answer,
                'neg_answer': neg_answer
            })
        
        # 基于 Anchor 选择正反例
        if anchor_answer:
            # 有 Anchor 的情况：优先 Anchor 对齐 + TIE 权重
            lambda_anchor = getattr(args, 'lambda_anchor', 0.7)  # Anchor 相似度权重
            
            # 选择正例：与 Anchor 相似且 TIE 权重高
            preferred_scores = []
            for score_info in candidate_scores:
                if score_info['pos_valid']:
                    combined_score = (lambda_anchor * score_info['anchor_sim_pos'] + 
                                    (1 - lambda_anchor) * score_info['normalized_weight'])
                    preferred_scores.append((combined_score, score_info))
            
            # 选择反例：与 Anchor 差异大或 TIE 权重低
            dispreferred_scores = []
            for score_info in candidate_scores:
                if score_info['neg_valid']:
                    # 反例分数：低相似度 + 低权重
                    combined_score = -(lambda_anchor * score_info['anchor_sim_neg'] + 
                                     (1 - lambda_anchor) * score_info['normalized_weight'])
                    dispreferred_scores.append((combined_score, score_info))
        else:
            # 无 Anchor 的情况：纯基于 TIE 权重
            preferred_scores = [(info['normalized_weight'], info) for info in candidate_scores if info['pos_valid']]
            dispreferred_scores = [(-info['normalized_weight'], info) for info in candidate_scores if info['neg_valid']]
        
        # 排序并选择最佳正反例
        if preferred_scores and dispreferred_scores:
            preferred_scores.sort(reverse=True)
            dispreferred_scores.sort(reverse=True)
            
            best_preferred = preferred_scores[0][1]
            best_dispreferred = dispreferred_scores[0][1]
            
            # 确保正反例不是同一个候选
            if best_preferred['candidate'] == best_dispreferred['candidate']:
                if len(dispreferred_scores) > 1:
                    best_dispreferred = dispreferred_scores[1][1]
                elif len(preferred_scores) > 1:
                    best_preferred = preferred_scores[1][1]
                else:
                    continue  # 跳过这个问题
            
            # 验证正例和反例选择规则
            preferred_valid, preferred_violations = validate_preferred_answer(best_preferred['candidate'], args)
            dispreferred_valid, dispreferred_conditions = validate_dispreferred_answer(best_dispreferred['candidate'], args)
            
            # 应用阈值过滤
            if apply_tie_anker_thresholds(best_preferred['tie_weight'], 
                                        best_preferred['normalized_weight'], 
                                        best_preferred['candidate'], args):
                
                # 构建DPO对
                dpo_pair = {
                    "id": best_preferred['candidate'].get('id'),
                    "case_id": best_preferred['candidate'].get('case_id'),
                    "question": question,
                    "image": best_preferred['candidate'].get('case_id', '') + "/" + best_preferred['candidate'].get('case_id', '') + ".jpg",
                    "chosen": best_preferred['pos_answer'],
                    "rejected": best_dispreferred['neg_answer'],
                    "tie_weight": float(best_preferred['tie_weight']),
                    "normalized_weight": float(best_preferred['normalized_weight']),
                    "anchor_info": {
                        "has_anchor": anchor_answer is not None,
                        "anchor_answer": anchor_answer,
                        "chosen_anchor_similarity": best_preferred['anchor_sim_pos'],
                        "rejected_anchor_similarity": best_dispreferred['anchor_sim_neg']
                    },
                    "validation_info": {
                        "preferred_valid": preferred_valid,
                        "dispreferred_valid": dispreferred_valid,
                        "preferred_violations": preferred_violations,
                        "dispreferred_reasons": dispreferred_conditions,
                        "conversation_valid": best_preferred['pos_valid'] and best_dispreferred['neg_valid']
                    },
                    "metadata": {
                        "tie_positive": best_preferred['candidate'].get('tie_positive', 0),
                        "tie_negative": best_preferred['candidate'].get('tie_negative', 0),
                        "delta_pos": best_preferred['candidate'].get('delta_pos', 0),
                        "delta_neg": best_preferred['candidate'].get('delta_neg', 0),
                        "m_v": best_preferred['candidate'].get('m_v', 0),
                        "m_n": best_preferred['candidate'].get('m_n', 0),
                        "gamma": best_preferred['candidate'].get('gamma', 0)
                    }
                }
                dpo_pairs.append(dpo_pair)
    
    return dpo_pairs

def calculate_tie_anker_weight(tie_diff, delta_pos, delta_neg, m_v, m_n, gamma, args):
    """
    计算TIE-ANKER权重
    
    Args:
        tie_diff: 标准化的TIE差异分数
        delta_pos, delta_neg: 正负样本的delta值
        m_v, m_n: 视觉和噪声指标
        gamma: gamma值
        args: 包含权重参数的命令行参数
        
    Returns:
        float: 计算得到的权重
    """
    # 组合权重计算
    weighted_score = (
        args.w_gamma * gamma +
        args.w_v * m_v +
        args.w_n * m_n +
        args.w_s * tie_diff +
        args.w_o * (delta_pos - delta_neg)
    )
    
    # Sigmoid映射
    sigmoid_input = args.beta * (weighted_score - args.tau)
    weight = 1.0 / (1.0 + np.exp(-sigmoid_input + args.epsilon))
    
    # 应用最小权重约束
    weight = max(weight, args.w_min)
    
    return weight

def validate_preferred_answer(result, args):
    """
    验证正例答案是否符合TIE-ANKER理论要求
    
    正例应满足所有条件：
    1. Δ⁺ > τ₊ (前景贡献度大)
    2. γ > τᵧ (相对因果效应强)  
    3. m_v > τᵥ (区分度高)
    4. |m_n| ≤ τₙ (背景泄漏可控)
    
    Args:
        result: 结果字典
        args: 命令行参数
        
    Returns:
        tuple: (是否有效, 违规条件列表)
    """
    violations = []
    
    # 条件1: Δ⁺ > τ₊ (前景贡献度大)
    delta_pos = result.get('delta_pos', 0)
    if delta_pos <= args.tau_pos:
        violations.append(f"前景贡献度不足: Δ⁺({delta_pos:.4f}) ≤ τ₊({args.tau_pos})")
    
    # 条件2: γ > τᵧ (相对因果效应强)
    gamma = result.get('gamma', 0)
    if gamma <= args.tau_gamma_weak:
        violations.append(f"相对因果效应弱: γ({gamma:.4f}) ≤ τᵧ({args.tau_gamma_weak})")
    
    # 条件3: m_v > τᵥ (区分度高)
    m_v = result.get('m_v', 0)
    if m_v <= args.tau_v:
        violations.append(f"区分度不足: m_v({m_v:.4f}) ≤ τᵥ({args.tau_v})")
    
    # 条件4: |m_n| ≤ τₙ (背景泄漏可控)
    m_n = result.get('m_n', 0)
    tau_n_leak = args.tau_n_percentile / 100.0
    if abs(m_n) > tau_n_leak:
        violations.append(f"背景泄漏过大: |m_n|({abs(m_n):.4f}) > τₙ({tau_n_leak:.4f})")
    
    return len(violations) == 0, violations

def validate_dispreferred_answer(result, args):
    """
    验证反例答案是否符合TIE-ANKER理论要求
    
    反例应满足至少一个条件：
    1. Δ⁻ ≥ 0 (前景未抑制错误答案)
    2. m_n > τₙ (背景泄漏显著)
    3. γ ≤ 0 (净效应劣势)
    
    Args:
        result: 结果字典
        args: 命令行参数
        
    Returns:
        tuple: (是否有效, 满足的条件列表)
    """
    satisfied_conditions = []
    
    delta_neg = result.get('delta_neg', 0)
    m_n = result.get('m_n', 0)
    gamma = result.get('gamma', 0)
    
    # 条件1: Δ⁻ ≥ 0 (负贡献或无提升)
    if delta_neg >= 0:
        satisfied_conditions.append(f"前景未抑制错误答案: Δ⁻({delta_neg:.4f}) ≥ 0")
    
    # 条件2: m_n > τₙ (泄漏效应显著)
    tau_n_leak = args.tau_n_percentile / 100.0
    if m_n > tau_n_leak:
        satisfied_conditions.append(f"背景泄漏显著: m_n({m_n:.4f}) > τₙ({tau_n_leak:.4f})")
    
    # 条件3: γ ≤ 0 (净效应劣势)
    if gamma <= 0:
        satisfied_conditions.append(f"净效应劣势: γ({gamma:.4f}) ≤ 0")
    
    return len(satisfied_conditions) > 0, satisfied_conditions

def apply_tie_anker_thresholds(weight, tie_diff, result, args):
    """
    应用改进的TIE-ANKER阈值过滤
    
    Args:
        weight: 计算得到的权重
        tie_diff: TIE差异分数
        result: 结果字典
        args: 命令行参数
        
    Returns:
        bool: 是否通过阈值过滤
    """
    # 验证正例和反例选择规则
    preferred_valid, preferred_violations = validate_preferred_answer(result, args)
    dispreferred_valid, dispreferred_conditions = validate_dispreferred_answer(result, args)
    
    # 只有当正例和反例都有效时才通过过滤
    return preferred_valid and dispreferred_valid

def compute_log_likelihood_teacher_forcing(model, sequences, input_length, image_tensor, device):
    """
    Compute log-likelihood of ground-truth answer using teacher forcing.
    """
    if sequences is None or sequences.shape[1] <= input_length:
        print(f"LL Debug: sequences is None or total_len (={0 if sequences is None else sequences.shape[1]}) <= input_len (={input_length}). Returning 0.")
        return 0.0

    with torch.inference_mode():
        # Inputs exclude the last token; labels exclude the first token
        inputs = sequences[:, :-1].to(device)
        labels = sequences[:, 1:].clone().to(device)

        # Mask out the prompt portion so loss is only computed on generated tokens
        # Prompt covers positions [0 .. input_length-1] in sequences; in shifted labels these map to [0 .. input_length-2]
        if input_length > 1:
            labels[:, : input_length - 1] = -100
        total_len = sequences.shape[1]
        gen_len = total_len - input_length
        print(f"LL Debug: total_len={total_len}, input_len={input_length}, generated_len={gen_len}")
        print(f"LL Debug: inputs shape={inputs.shape}, labels shape={labels.shape}")

        # Ensure input tensors match model precision and device
        inputs = inputs.to(device)
        labels = labels.to(device)
        if "cpu" not in str(device):
            # For GPU, ensure inputs are on the same device but keep as long (int64) for token IDs
            pass  # input_ids should remain as long integers
        
        outputs = model(
            input_ids=inputs,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device),
        )

        logits = outputs.logits  # [1, seq_len-1, vocab]
        vocab_size = logits.size(-1)

        # CrossEntropy over tokens (ignore masked labels)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        ).view(labels.size())  # [1, seq_len-1]

        valid_mask = (labels != -100)
        valid_count = int(valid_mask.sum().item())
        print(f"LL Debug: valid generated tokens={valid_count}")
        if valid_mask.any():
            nll_tensor = per_token_loss[valid_mask]
            nll_sum = nll_tensor.sum().item()
            avg_nll = nll_tensor.mean().item()
            print(f"LL Debug: nll_sum={nll_sum:.6f}, avg_nll={avg_nll:.6f}")
            # Show first few generated token ids and their per-token loss
            # Recover generated token ids from labels where valid
            gen_token_ids = labels[valid_mask].view(-1).tolist()
            preview_k = min(5, len(gen_token_ids))
            if preview_k > 0:
                print(f"LL Debug: first {preview_k} generated token ids={gen_token_ids[:preview_k]}")
            return -nll_sum
        print("LL Debug: No valid generated tokens after masking. Returning 0.")
        return 0.0

def build_prompt_text(question: str) -> str:
    return (
        f"The following image displays a version with a masked background on the left and a full medical scan on the right. "
        f"Based on this composite image, answer the following question.\n"
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}"
    )

def compute_ll_token_by_token(model, prompt_ids, answer_ids, image_tensor, device, log_states=False, tokenizer=None):
    """
    Compute log-likelihood of ground-truth answer by iteratively conditioning on
    prompt + already revealed answer tokens. Avoids alignment issues with vision tokens.
    Returns sum of log probabilities over answer tokens (optionally incl. EOS in answer_ids).
    """
    if answer_ids is None or answer_ids.numel() == 0:
        return 0.0
    total_log_likelihood = 0.0
    state_logs = {"prompt": None, "steps": []} if log_states else None
    
    # Ensure answer_ids and prompt_ids are on the correct device
    answer_ids = answer_ids.to(device)
    prompt_ids = prompt_ids.to(device)
    
    with torch.inference_mode():
        for i in range(answer_ids.shape[1]):
            # Concatenate tensors (both already on correct device)
            prefix = torch.cat([prompt_ids, answer_ids[:, :i]], dim=1)
            # Ensure prefix is on the correct device
            prefix = prefix.to(device)
            
            outputs = model(
                input_ids=prefix,
                images=image_tensor.unsqueeze(0).to(device, dtype=model.dtype),
                use_cache=True,
                output_hidden_states=log_states,
                return_dict=True,
            )
            logits = outputs.logits  # [1, prefix_len, vocab]
            next_logits = logits[:, -1, :]  # last position
            log_probs = torch.log_softmax(next_logits, dim=-1)
            token_id = answer_ids[0, i].item()
            total_log_likelihood += log_probs[0, token_id].item()

            if log_states:
                hidden_norm = None
                if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                    hidden_norm = float(torch.linalg.norm(last_hidden).item())
                topk_vals, topk_ids = torch.topk(log_probs, k=min(5, log_probs.shape[-1]), dim=-1)
                top_list = []
                for j in range(topk_ids.shape[-1]):
                    tid = int(topk_ids[0, j].item())
                    tstr = tokenizer.decode([tid]) if tokenizer is not None else ""
                    top_list.append({"id": tid, "text": tstr, "logprob": float(topk_vals[0, j].item())})
                # Save prompt state for i==0 (before adding any answer token)
                if i == 0:
                    state_logs["prompt"] = {
                        "hidden_norm": hidden_norm,
                        "topk": top_list,
                    }
                state_logs["steps"].append({
                    "step": i,
                    "target_token_id": int(token_id),
                    "target_token_text": tokenizer.decode([token_id]) if tokenizer is not None else "",
                    "hidden_norm": hidden_norm,
                    "topk": top_list,
                })
    return (total_log_likelihood, state_logs) if log_states else total_log_likelihood

def run_inference_with_image(model, tokenizer, image_processor, composite_image, prompt_text, target_answer, args, device):
    """
    Run inference on a composite image and return both text output and log likelihood.
    """
    try:
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
    except Exception as e:
        print(f"Warning: Error in tokenization: {e}")
        return "ERROR", 0.0
    
    if image_processor is not None:
        image_tensor = process_images([composite_image], image_processor, model.config)[0]
    else:
        # Fallback: try to process image without image_processor
        try:
            image_tensor = process_images([composite_image], None, model.config)[0]
        except Exception as e:
            print(f"Warning: Failed to process image: {e}")
            # Create a dummy image tensor as fallback
            image_tensor = torch.zeros((3, 224, 224))  # Dummy tensor
    
    # Ensure image tensor uses bfloat16 precision to match model
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    try:
        with torch.inference_mode():
            # Generate deterministically to stabilize LL computation
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            pad_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_id is None:
                pad_id = eos_id
            # Image tensor precision is already set above, no need to convert again
            
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device),
                do_sample=False,
                max_new_tokens=getattr(args, 'max_new_tokens', 64),
                min_new_tokens=getattr(args, 'min_new_tokens', 1),
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=False,
            )
    except Exception as e:
        print(f"Warning: Error during model generation: {e}")
        return "ERROR", 0.0

    # Prepare ground-truth target sequence: prompt + ground-truth answer
    answer_text = str(target_answer).strip() if target_answer is not None else ""
    if answer_text == "":
        # If no ground truth, we cannot compute LL; still decode model output for record
        try:
            outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
        except Exception:
            outputs = ""
        if getattr(args, 'verbose', False):
            print("LL Debug: Empty ground-truth answer; returning LL=0.")
        input_length = input_ids.shape[1]
        return outputs, 0.0

    # Tokenize target answer without adding extra specials to align with CE
    target_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"].to(device)

    # Optionally append EOS if tokenizer has it
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is not None:
        eos_tensor = torch.tensor([[eos_id]], device=device)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    # Decode a preview of the model's own generated text (optional)
    try:
        gen_only_text = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
    except Exception:
        gen_only_text = ""
    outputs = gen_only_text
    
    # Debug information
    if getattr(args, 'verbose', False):
        print(f"Gen Debug: Generated text preview: {outputs[:120]}...")
        print(f"LL Debug: Input length (prompt tokens): {input_ids.shape[1]}")
        print(f"LL Debug: Target answer length (tokens): {target_ids.shape[1]}")
    
    # Compute log likelihood using token-by-token method
    input_length = input_ids.shape[1]
    try:
        log_likelihood = compute_ll_token_by_token(
            model=model,
            prompt_ids=input_ids,
            answer_ids=target_ids,
            image_tensor=image_tensor,
            device=device,
            log_states=getattr(args, 'log_states', False),
            tokenizer=tokenizer,
        )
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"LL Debug: token-by-token failed: {e}. Falling back to TF method.")
        # Build full sequence for teacher forcing method
        full_sequence = torch.cat([input_ids, target_ids], dim=1)
        log_likelihood = compute_log_likelihood_teacher_forcing(
            model=model,
            sequences=full_sequence,
            input_length=input_length,
            image_tensor=image_tensor,
            device=device,
        )
    
    return outputs, log_likelihood

def eval_model(args):
    setup()
    # 获取rank和world_size，支持单GPU模式
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ["RANK"])
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
    print(f"Rank {rank}/{world_size} started")

    set_seed(0)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=f"cuda:{rank}"
    )
    
    # Ensure all model components use consistent precision (bfloat16)
    device = f"cuda:{rank}"
    
    # Force all model parameters to bfloat16 recursively
    def convert_to_bfloat16(module):
        for param in module.parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        for buffer in module.buffers():
            if buffer.dtype != torch.bfloat16 and buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(torch.bfloat16)
        for child in module.children():
            convert_to_bfloat16(child)
    
    # Apply recursive conversion
    convert_to_bfloat16(model)
    
    # Ensure vision tower is properly loaded after type conversion
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print(f"Rank {rank}: Vision tower not loaded, loading now...")
            vision_tower.load_model(device_map=device)
        elif vision_tower is not None and hasattr(vision_tower, 'vision_tower') and vision_tower.vision_tower is None:
            print(f"Rank {rank}: Vision tower attribute is None, reloading...")
            vision_tower.load_model(device_map=device)
    model = model.to(device)
    
    print(f"Rank {rank}: All model parameters forcibly converted to bfloat16")

    if image_processor is None:
        print(f"Rank {rank}: Image processor was not loaded...")
        try:
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
            if hasattr(model, 'get_vision_tower'):
                 model.get_vision_tower().image_processor = image_processor
        except Exception as e:
            print(f"Rank {rank}: Failed to load CLIP image processor: {e}")
            print(f"Rank {rank}: Continuing without image processor...")
            # Create a dummy image processor or use the model's default
            image_processor = None

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    
    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # 优化批处理配置
    batch_size = getattr(args, 'batch_size', 1)
    num_workers = getattr(args, 'num_workers', 4)  # 减少worker数量以避免内存问题
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    local_results = []

    # 初始化性能监控
    performance_monitor = PerformanceMonitor()
    processed = 0
    limit = getattr(args, 'limit', None)
    
    # 创建带有性能监控的进度条
    progress_bar = tqdm(dataloader, position=0, file=sys.stdout, desc=f"Rank {rank}")
    
    for i, line in enumerate(progress_bar):
        # Handle batch data from DataLoader (batch_size=1)

        # Since we're using batch_size=1, each field will be a list with one element
        idx = line["id"][0] if "id" in line and isinstance(line["id"], list) else line.get("id", i)
        
        # Extract data - handle both DPO format and extracted_where_questions format
        if "image" in line:
            # DPO format
            image_path = line["image"][0] if isinstance(line["image"], list) else line.get("image", "")
            full_image_path = os.path.join("/workspace/MMedPO/outputs/tie_results_1/composites", image_path)
            masked_image_path = full_image_path  # Use same for DPO format
        else:
            # extracted_where_questions format
            full_image_path = line["full_image_path"][0] if isinstance(line["full_image_path"], list) else line.get("full_image_path", "")
            masked_image_path = line["masked_image_path"][0] if isinstance(line["masked_image_path"], list) else line.get("masked_image_path", full_image_path)
        
        # Extract question and answers - handle both formats
        if "conversations" in line:
            # DPO format
            conversations = line["conversations"]
            original_question = ""
            gt_answer = ""
            if isinstance(conversations, list):
                for conv in conversations:
                    if isinstance(conv, dict):
                        # DataLoader wraps values in lists, so we need to extract them
                        from_field = conv.get("from", [])
                        value_field = conv.get("value", [])
                        if isinstance(from_field, list) and len(from_field) > 0:
                            from_value = from_field[0]
                        else:
                            from_value = from_field
                        if isinstance(value_field, list) and len(value_field) > 0:
                            value_value = value_field[0]
                        else:
                            value_value = value_field
                        
                        if from_value == "human":
                            original_question = value_value.replace("<image>\n", "")
                        elif from_value == "gpt":
                            gt_answer = value_value
        else:
            # extracted_where_questions format
            original_question = line["question"][0] if isinstance(line["question"], list) else line.get("question", "")
            gt_answer = ""  # Will be set from positive_answer below
        
        # Extract positive and negative answers - handle both formats
        if "rejected_conversations" in line:
            # DPO format
            rejected_conversations = line["rejected_conversations"]
            positive_answer = gt_answer  # The accepted answer is positive
            negative_answer = ""
            if isinstance(rejected_conversations, list):
                for conv in rejected_conversations:
                    if isinstance(conv, dict):
                        # DataLoader wraps values in lists, so we need to extract them
                        from_field = conv.get("from", [])
                        value_field = conv.get("value", [])
                        if isinstance(from_field, list) and len(from_field) > 0:
                            from_value = from_field[0]
                        else:
                            from_value = from_field
                        if isinstance(value_field, list) and len(value_field) > 0:
                            value_value = value_field[0]
                        else:
                            value_value = value_field
                        
                        if from_value == "gpt":
                            negative_answer = value_value
        else:
            # extracted_where_questions format
            positive_answer = line["positive_answer"][0] if isinstance(line["positive_answer"], list) else line.get("positive_answer", "")
            negative_answer = line["negative_answer"][0] if isinstance(line["negative_answer"], list) else line.get("negative_answer", "")
            gt_answer = positive_answer  # Set gt_answer for consistency
        
        answer_type = "unknown"  # Default since not provided in DPO format
        
        # Get answers from data file for TIE calculation
        # Note: weighted_score will be calculated based on TIE features after inference
        original_weighted_score = line["weighted_score"][0] if "weighted_score" in line and isinstance(line["weighted_score"], list) else line.get("weighted_score", 1.0)
        has_positive = bool(positive_answer)
        has_negative = bool(negative_answer)

        # For TIE calculation, we need both positive and negative answers
        # If we don't have negative answer, skip TIE calculation but still do basic inference
        calculate_tie = has_positive and has_negative

        try:
            full_image = Image.open(full_image_path).convert('RGB')
            masked_image = Image.open(masked_image_path).convert('RGB')
            # Create white background image
            white_background = Image.new('RGB', full_image.size, color='white')
        except FileNotFoundError as e:
            progress_bar.write(f"Warning: Could not find required image for item {idx}. Skipping. Reason: {e}")
            progress_bar.write(f"  Full image path: {full_image_path}")
            progress_bar.write(f"  Masked image path: {masked_image_path}")
            continue
        except Exception as e:
            progress_bar.write(f"Warning: Error loading images for item {idx}. Skipping. Reason: {e}")
            continue

        # Create composite images - 背景+全图 instead of 全图+背景
        composite_with_mask = stitch_images_side_by_side(masked_image, full_image)
        composite_with_white = stitch_images_side_by_side(white_background, full_image)

        # Build a single prompt text to ensure strict consistency between both runs
        prompt_text = build_prompt_text(original_question)

        # Run inference for positive answer
        try:
            # Initialize default values
            output_pos_mask = output_pos_white = output_neg_mask = output_neg_white = ""
            ll_pos_mask = ll_pos_white = ll_neg_mask = ll_neg_white = 0.0
            tie_positive = tie_negative = tie_difference = 0.0
            
            if calculate_tie:
                # Positive answer with masked background
                output_pos_mask, ll_pos_mask = run_inference_with_image(
                    model, tokenizer, image_processor, composite_with_mask, prompt_text, positive_answer, args, device
                )
                
                # Positive answer with white background
                output_pos_white, ll_pos_white = run_inference_with_image(
                    model, tokenizer, image_processor, composite_with_white, prompt_text, positive_answer, args, device
                )
                
                # Negative answer with masked background
                output_neg_mask, ll_neg_mask = run_inference_with_image(
                    model, tokenizer, image_processor, composite_with_mask, prompt_text, negative_answer, args, device
                )
                
                # Negative answer with white background
                output_neg_white, ll_neg_white = run_inference_with_image(
                    model, tokenizer, image_processor, composite_with_white, prompt_text, negative_answer, args, device
                )
                
                # Calculate TIE values (log likelihood differences)
                tie_positive = ll_pos_mask - ll_pos_white  # TIE for positive answer
                tie_negative = ll_neg_mask - ll_neg_white  # TIE for negative answer
                tie_difference = tie_positive - tie_negative  # Overall TIE difference
                
                # Calculate new TIE-based metrics according to TIE_formula_summary.md
                # Extract plain values for calculation
                ll_pos_mask_val = ll_pos_mask[0] if isinstance(ll_pos_mask, tuple) else ll_pos_mask
                ll_pos_white_val = ll_pos_white[0] if isinstance(ll_pos_white, tuple) else ll_pos_white
                ll_neg_mask_val = ll_neg_mask[0] if isinstance(ll_neg_mask, tuple) else ll_neg_mask
                ll_neg_white_val = ll_neg_white[0] if isinstance(ll_neg_white, tuple) else ll_neg_white
                
                # Calculate delta values
                delta_pos = float(tie_positive)  # δ_pos = TIE_positive
                delta_neg = float(tie_negative)  # δ_neg = TIE_negative
                delta_obj = delta_pos - delta_neg  # δ_obj = δ_pos - δ_neg
                
                # Calculate m_v and m_n (mean log-likelihoods)
                m_v = (ll_pos_mask_val + ll_pos_white_val) / 2.0  # Mean of positive answer log-likelihoods
                m_n = (ll_neg_mask_val + ll_neg_white_val) / 2.0  # Mean of negative answer log-likelihoods
                
                # Calculate gamma (preference strength)
                gamma = m_v - m_n  # γ = m_v - m_n
                
                # Calculate S_disp (dispersion measure)
                S_disp = abs(delta_pos) + abs(delta_neg)  # S_disp = |δ_pos| + |δ_neg|
                
                # Calculate DPO weight using the new formula
                dpo_result = calculate_dpo_weight(
                    delta_pos=delta_pos,
                    delta_neg=delta_neg,
                    m_v=m_v,
                    m_n=m_n,
                    delta_obj=delta_obj
                )
                
                # Extract DPO weight calculation results
                calculated_gamma = dpo_result['gamma']
                S_raw = dpo_result['S_raw']
                S_normalized = dpo_result['S_normalized']
                preference_prob = dpo_result['preference_prob']
                dpo_weight = dpo_result['dpo_weight']
                flipped = dpo_result['flipped']
                
                # Use calculated DPO weight as the new weighted_score
                weighted_score = dpo_weight
            else:
                # If we can't calculate TIE, just run basic inference with positive answer
                output_pos_mask, ll_pos_mask = run_inference_with_image(
                    model, tokenizer, image_processor, composite_with_mask, prompt_text, positive_answer, args, device
                )
                output_pos_white = output_neg_mask = output_neg_white = output_pos_mask
                ll_pos_white = ll_neg_mask = ll_neg_white = ll_pos_mask
                
                # Set default values for new metrics when TIE can't be calculated
                delta_pos = delta_neg = delta_obj = 0.0
                m_v = m_n = gamma = S_disp = 0.0
                # Set default DPO weight calculation results
                calculated_gamma = S_raw = S_normalized = preference_prob = 0.0
                dpo_weight = flipped = 0.0
                # Use original weighted_score when TIE can't be calculated
                weighted_score = original_weighted_score
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            progress_bar.write(f"Error processing item {idx}: {str(e)}")
            progress_bar.write(f"Error details: {error_details}")
            
            # 尝试继续处理下一个项目而不是完全失败
            output_pos_mask = output_pos_white = output_neg_mask = output_neg_white = "ERROR"
            ll_pos_mask = ll_pos_white = ll_neg_mask = ll_neg_white = 0.0
            tie_positive = tie_negative = tie_difference = 0.0
            # Set default values for new metrics in case of error
            delta_pos = delta_neg = delta_obj = 0.0
            m_v = m_n = gamma = S_disp = 0.0
            # Set default DPO weight calculation results
            calculated_gamma = S_raw = S_normalized = preference_prob = 0.0
            dpo_weight = flipped = 0.0
            # Use original weighted_score in case of error
            weighted_score = original_weighted_score
            
            # 记录错误但继续处理

        # Save composite images if requested
        if args.output_image_folder and rank == 0:
            try:
                case_id = os.path.basename(os.path.dirname(full_image_path))
                base_name = os.path.splitext(os.path.basename(full_image_path))[0]
                
                # Save masked composite (背景+全图)
                output_dir = os.path.join(args.output_image_folder, case_id)
                os.makedirs(output_dir, exist_ok=True)
                masked_output_path = os.path.join(output_dir, f"{base_name}_mask_plus_full.jpg")
                composite_with_mask.save(masked_output_path)
                
                # Save white background composite (白背景+全图)
                white_output_path = os.path.join(output_dir, f"{base_name}_white_plus_full.jpg")
                composite_with_white.save(white_output_path)
                
            except Exception as e:
                progress_bar.write(f"Warning: Could not save composite images for {case_id}. Reason: {e}")
                # 继续处理，不因图像保存失败而中断

        # Create result entry
        ans_id = shortuuid.uuid()
        case_id = os.path.basename(os.path.dirname(full_image_path))
        
        # Extract plain values if they are tuples (from log_states)
        ll_pos_mask_value = ll_pos_mask[0] if isinstance(ll_pos_mask, tuple) else ll_pos_mask
        ll_pos_white_value = ll_pos_white[0] if isinstance(ll_pos_white, tuple) else ll_pos_white
        ll_neg_mask_value = ll_neg_mask[0] if isinstance(ll_neg_mask, tuple) else ll_neg_mask
        ll_neg_white_value = ll_neg_white[0] if isinstance(ll_neg_white, tuple) else ll_neg_white
        
        # Calculate token lengths for normalization
        pos_token_len = neg_token_len = 0
        try:
            if positive_answer:
                pos_token_len = int(tokenizer(
                    str(positive_answer).strip(),
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].shape[1])
            if negative_answer:
                neg_token_len = int(tokenizer(
                    str(negative_answer).strip(),
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].shape[1])
        except Exception:
            pass

        # Compute token-averaged TIE values
        tie_pos_token_avg = (tie_positive / max(1, pos_token_len)) if pos_token_len > 0 else tie_positive
        tie_neg_token_avg = (tie_negative / max(1, neg_token_len)) if neg_token_len > 0 else tie_negative

        result = {
            "id": idx,
            "case_id": case_id,
            "question": original_question,
            "answer_type": answer_type,
            "gt_answer": gt_answer,
            "positive_answer": positive_answer,
            "negative_answer": negative_answer if calculate_tie else None,
            "weighted_score": weighted_score,
            "has_positive": has_positive,
            "has_negative": has_negative,
            "calculate_tie": calculate_tie,
            
            # Model outputs (combined since they're identical)
            "output_with_background": output_pos_mask,  # Same as output_neg_mask
            "output_with_white_background": output_pos_white,  # Same as output_neg_white
            
            # Log likelihoods
            "ll_positive_with_background": ll_pos_mask_value,
            "ll_positive_with_white_background": ll_pos_white_value,
            "ll_negative_with_background": ll_neg_mask_value if calculate_tie else None,
            "ll_negative_with_white_background": ll_neg_white_value if calculate_tie else None,
            
            # TIE values (Treatment Interaction Effects)
            "tie_positive": tie_positive if calculate_tie else None,
            "tie_negative": tie_negative if calculate_tie else None,
            "tie_difference": tie_difference if calculate_tie else None,
            
            # New TIE-based metrics according to TIE_formula_summary.md
            "delta_pos": delta_pos if calculate_tie else None,
            "delta_neg": delta_neg if calculate_tie else None,
            "delta_obj": delta_obj if calculate_tie else None,
            "m_v": m_v if calculate_tie else None,
            "m_n": m_n if calculate_tie else None,
            "gamma": gamma if calculate_tie else None,
            "S_disp": S_disp if calculate_tie else None,
            
            # DPO weight calculation results
            "calculated_gamma": calculated_gamma if calculate_tie else None,
            "S_raw": S_raw if calculate_tie else None,
            "S_normalized": S_normalized if calculate_tie else None,
            "preference_prob": preference_prob if calculate_tie else None,
            "dpo_weight": dpo_weight if calculate_tie else None,
            "flipped": flipped if calculate_tie else None,
            "original_weighted_score": original_weighted_score,
            
            # Token information
            "positive_answer_token_len": pos_token_len,
            "negative_answer_token_len": neg_token_len if calculate_tie else None,
            "positive_answer_char_len": len(str(positive_answer).strip()) if positive_answer else 0,
            "negative_answer_char_len": len(str(negative_answer).strip()) if negative_answer and calculate_tie else 0,
            
            # Token-averaged TIE values
            "tie_positive_token_avg": tie_pos_token_avg if calculate_tie else None,
            "tie_negative_token_avg": tie_neg_token_avg if calculate_tie else None,
            
            # Metadata
            "prompt_text": prompt_text,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {},
        }

        # Attach optional state logs if enabled and available
        if getattr(args, 'log_states', False):
            # When log_states=True, compute_ll_token_by_token returns (ll, logs)
            if isinstance(ll_pos_mask, tuple):
                result["state_logs_pos_with_background"] = ll_pos_mask[1]
            if isinstance(ll_pos_white, tuple):
                result["state_logs_pos_with_white_background"] = ll_pos_white[1]
            if isinstance(ll_neg_mask, tuple):
                result["state_logs_neg_with_background"] = ll_neg_mask[1]
            if isinstance(ll_neg_white, tuple):
                result["state_logs_neg_with_white_background"] = ll_neg_white[1]
        
        serial_result = tensor_to_serializable(result)
        local_results.append(serial_result)

        processed += 1
        
        # 更新性能监控和进度条
        if processed % 10 == 0 or processed == 1:  # 每10个样本更新一次
            perf_stats = performance_monitor.update(processed)
            progress_bar.set_postfix({
                'avg_speed': f"{perf_stats['avg_speed']:.1f}/h",
                'inst_speed': f"{perf_stats['instant_speed']:.1f}/h", 
                'gpu_util': perf_stats['gpu_util'],
                'gpu_mem': perf_stats['gpu_memory']
            })
        
        if limit is not None and processed >= limit:
            break

    # 显示最终统计
    final_stats = performance_monitor.update(processed)
    print(f"\nRank {rank} 处理完成统计:")
    print(f"  总处理样本: {processed}")
    print(f"  总耗时: {final_stats['total_elapsed']:.1f}秒")
    print(f"  平均速度: {final_stats['avg_speed']:.1f}样本/小时")
    print(f"  最终GPU利用率: {final_stats['gpu_util']}")
    print(f"  最终GPU内存: {final_stats['gpu_memory']}")

    if world_size > 1:
        dist.barrier()
        print(f"Rank {rank} reached gathering barrier")
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)
        print(f"Rank {rank} finished all_gather_object")
        
        if rank == 0:
            print(f"Rank {rank} starting to process and write results...")
            all_results = [item for sublist in gathered_results for item in sublist]
            unique_results = []
            seen_ids = set()
            for res in all_results:
                q_id = res["id"]
                if isinstance(q_id, list): q_id = q_id[0]
                if q_id not in seen_ids:
                    unique_results.append(res)
                    seen_ids.add(q_id)
    else:
        # Single GPU mode - process results directly
        print("Processing results in single GPU mode...")
        unique_results = local_results
    
    # Process results regardless of distributed or single GPU mode
    if rank == 0:
        def get_sort_key(x):
            q_id = x["id"]
            return q_id[0] if isinstance(q_id, list) else q_id
        
        unique_results.sort(key=get_sort_key)
        
        # Save to CSV/Excel
        csv_file = os.path.expanduser(args.csv_file)
        csv_dir = os.path.dirname(csv_file)
        if csv_dir:  # 只有当目录不为空时才创建
            os.makedirs(csv_dir, exist_ok=True)
        
        # Convert to DataFrame: keep ALL fields (no reduction)
        df_full = pd.DataFrame(unique_results)

        # Ensure we write to an .xlsx file
        if csv_file.lower().endswith(".csv"):
            xlsx_file = os.path.splitext(csv_file)[0] + ".xlsx"
        else:
            xlsx_file = csv_file

        try:
            with pd.ExcelWriter(xlsx_file, engine="xlsxwriter") as writer:
                # Main sheet
                sheet_name = "dpo_tie_scores"
                df_full.to_excel(writer, index=False, sheet_name=sheet_name)
                workbook  = writer.book
                worksheet = writer.sheets[sheet_name]

                # Apply conditional formatting: TIE difference < 0 shown in red (only if TIE data exists)
                red_format = workbook.add_format({"font_color": "#9C0006"})
                if "tie_difference" in df_full.columns and not df_full["tie_difference"].isna().all():
                    # Find the column index for TIE difference (0-based)
                    diff_col_idx = df_full.columns.get_loc("tie_difference")
                    # Excel columns are letters; build range like C2:C{n}
                    start_row = 2  # 1-based Excel row index, skipping header
                    end_row = len(df_full) + 1
                    # Convert column index to Excel letter(s)
                    def col_to_excel(col_idx):
                        col_str = ""
                        col_idx += 1
                        while col_idx:
                            col_idx, remainder = divmod(col_idx - 1, 26)
                            col_str = chr(65 + remainder) + col_str
                        return col_str
                    diff_col_letter = col_to_excel(diff_col_idx)
                    cell_range = f"{diff_col_letter}{start_row}:{diff_col_letter}{end_row}"
                    worksheet.conditional_format(cell_range, {"type": "cell", "criteria": "<", "value": 0, "format": red_format})

                    # Only create TIE-specific sheets if we have valid TIE data
                    if not df_full["tie_difference"].isna().all():
                        # Negatives sheet
                        negatives = df_full[df_full["tie_difference"] < 0].copy()
                        if not negatives.empty:
                            negatives.to_excel(writer, index=False, sheet_name="negative_tie_diff")
                        
                        # Positive TIE sheet
                        positive_tie = df_full[df_full["tie_positive"] > 0].copy()
                        if not positive_tie.empty:
                            positive_tie.to_excel(writer, index=False, sheet_name="positive_tie")
                        
                        # Negative TIE sheet
                        negative_tie = df_full[df_full["tie_negative"] < 0].copy()
                        if not negative_tie.empty:
                            negative_tie.to_excel(writer, index=False, sheet_name="negative_tie")
        except Exception:
            # Fallback to writing without formatting
            with pd.ExcelWriter(xlsx_file) as writer:
                df_full.to_excel(writer, index=False, sheet_name="dpo_tie_scores")
                if "tie_difference" in df_full.columns and not df_full["tie_difference"].isna().all():
                    negatives = df_full[df_full["tie_difference"] < 0]
                    if not negatives.empty:
                        negatives.to_excel(writer, index=False, sheet_name="negative_tie_diff")
                    positive_tie = df_full[df_full["tie_positive"] > 0]
                    if not positive_tie.empty:
                        positive_tie.to_excel(writer, index=False, sheet_name="positive_tie")
                    negative_tie = df_full[df_full["tie_negative"] < 0]
                    if not negative_tie.empty:
                        negative_tie.to_excel(writer, index=False, sheet_name="negative_tie")
        print(f"Rank {rank} finished writing to Excel file {xlsx_file}")
        
        # Also save JSON for compatibility
        if args.answers_file:
            answers_file = os.path.expanduser(args.answers_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            with open(answers_file, "w") as ans_file:
                for res in unique_results:
                    ans_file.write(json.dumps(res) + "\n")
            print(f"Rank {rank} finished writing to JSON file {args.answers_file}")
        
        # Generate and save TIE-ANKER DPO pairs if enabled
        if args.enable_tie_anker and args.output_pairs_file:
            print(f"Rank {rank} generating TIE-ANKER DPO pairs...")
            dpo_pairs = build_tie_anker_dpo_pairs(unique_results, args)
            
            if dpo_pairs:
                pairs_file = os.path.expanduser(args.output_pairs_file)
                os.makedirs(os.path.dirname(pairs_file), exist_ok=True)
                
                # Save DPO pairs as JSON
                with open(pairs_file, "w", encoding='utf-8') as f:
                    json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
                
                print(f"Rank {rank} finished writing {len(dpo_pairs)} DPO pairs to {pairs_file}")
                
                # Also save summary statistics
                summary_file = pairs_file.replace('.json', '_summary.json')
                summary = {
                    "total_pairs": len(dpo_pairs),
                    "total_processed": len(unique_results),
                    "selection_rate": len(dpo_pairs) / len(unique_results) if unique_results else 0,
                    "parameters": {
                        "w_gamma": args.w_gamma,
                        "w_v": args.w_v,
                        "w_n": args.w_n,
                        "w_s": args.w_s,
                        "w_o": args.w_o,
                        "beta": args.beta,
                        "tau": args.tau,
                        "epsilon": args.epsilon,
                        "tau_pos": args.tau_pos,
                        "tau_gamma_strong": args.tau_gamma_strong,
                        "tau_gamma_weak": args.tau_gamma_weak,
                        "tau_v": args.tau_v,
                        "tau_n_percentile": args.tau_n_percentile,
                        "w_min": args.w_min
                    }
                }
                
                with open(summary_file, "w", encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                
                print(f"Rank {rank} finished writing DPO pairs summary to {summary_file}")
            else:
                print(f"Rank {rank} no DPO pairs generated (all filtered out or no valid TIE data)")

    if world_size > 1:
        dist.barrier()
        print(f"Rank {rank} passed final barrier. Preparing to clean up.")
        cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/MMedPO/checkpoints_original_tie/slake_strategy1/checkpoint-54", help="Path to the model checkpoint")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path for LoRA adapter")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the DPO JSON file with positive and negative answers.")
    parser.add_argument("--csv-file", type=str, required=True, help="Path to save the TIE comparison results as CSV/Excel.")
    parser.add_argument("--answers-file", type=str, default=None, help="Optional: Path to save results as JSON.")
    parser.add_argument("--output-image-folder", type=str, default=None, help="Optional: Path to save the stitched composite images.")
    
    # TIE-ANKER DPO Pairs Construction parameters
    parser.add_argument("--enable-tie-anker", action="store_true", help="Enable TIE-ANKER DPO pairs construction")
    parser.add_argument("--output-pairs-file", type=str, default=None, help="Path to save DPO pairs JSON file")
    
    # TIE-ANKER weights
    parser.add_argument("--w-gamma", type=float, default=1.0, help="Weight for gamma component")
    parser.add_argument("--w-v", type=float, default=0.5, help="Weight for v component")
    parser.add_argument("--w-n", type=float, default=0.8, help="Weight for n component")
    parser.add_argument("--w-s", type=float, default=0.3, help="Weight for s component")
    parser.add_argument("--w-o", type=float, default=0.5, help="Weight for o component")
    
    # Sigmoid mapping parameters
    parser.add_argument("--beta", type=float, default=2.0, help="Beta parameter for sigmoid mapping")
    parser.add_argument("--tau", type=float, default=0.0, help="Tau parameter for sigmoid mapping")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Epsilon for numerical stability")
    
    # Scoring & thresholds
    parser.add_argument("--token-avg", type=str, default="true", help="Use token averaging")
    parser.add_argument("--use-per-case-zscore", type=str, default="true", help="Use per-case z-score normalization")
    parser.add_argument("--z-eps", type=float, default=1e-6, help="Z-score numerical stability parameter")
    parser.add_argument("--tau-pos", type=float, default=0.1, help="Positive threshold")
    parser.add_argument("--tau-gamma-strong", type=float, default=0.5, help="Strong gamma threshold")
    parser.add_argument("--tau-gamma-weak", type=float, default=0.1, help="Weak gamma threshold")
    parser.add_argument("--tau-v", type=float, default=0.5, help="V threshold")
    parser.add_argument("--tau-n-percentile", type=float, default=75, help="N percentile threshold")
    
    # Weight mapping
    parser.add_argument("--w-min", type=float, default=0.05, help="Minimum weight")
    
    # Dispreference
    parser.add_argument("--p-disp", type=float, default=0.10, help="Dispreference probability")
    parser.add_argument("--alpha-neg", type=float, default=1.0, help="Negative alpha")
    parser.add_argument("--alpha-leak", type=float, default=0.5, help="Leak alpha")
    parser.add_argument("--beta-disp", type=float, default=2.0, help="Dispreference beta")
    parser.add_argument("--tau-disp", type=float, default=0.0, help="Dispreference tau")
    
    # Sampling
    parser.add_argument("--k-max", type=int, default=5, help="Maximum k for sampling")
    parser.add_argument("--anchor-factor", type=float, default=0.8, help="Anchor factor")
    parser.add_argument("--strong-weak-ratio", type=float, default=3, help="Strong to weak ratio")
    
    # Anchor-based selection parameters
    parser.add_argument("--lambda-anchor", type=float, default=0.7, help="Weight for anchor similarity in positive/negative selection")
    
    # DPO training
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-tokens", type=int, default=2048, help="Batch tokens")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lambda-kl", type=float, default=0.02, help="KL divergence lambda")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--tau-score", type=float, default=1.0, help="Score tau")
    
    # 批处理和性能优化参数
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of samples for debugging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logs")
    parser.add_argument("--log_states", action="store_true", help="Log pre-output and step-wise states (hidden norms, top-k)")
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()