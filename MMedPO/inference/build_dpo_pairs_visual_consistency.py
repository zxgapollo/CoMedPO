#!/usr/bin/env python3
"""
Build DPO pairs for Method 2: Visual Consistency Preference (Ground-truth vs. Generated)

This script constructs DPO training pairs based on Visual Consistency TIE results.
The pairs compare ground-truth answers vs. model-generated answers on the same visual input.

Pair Construction:
- Preferred: Ground-truth answer on full image (y_gt, I_full)  
- Dispreferred: Generated answer on full image (y_gen, I_full)

Weight Calculation:
S_raw = (Δ+ - Δ-) + α*m_v - β*max(0, m_n - τ_n)
where:
- Δ+ = LL(y_gt|I_full) - LL(y_gt|I_bg): Foreground contribution to correct answer
- Δ- = LL(y_gen|I_full) - LL(y_gen|I_bg): Foreground contribution to generated answer  
- γ = Δ+ - Δ-: Net visual support effect
- m_v = LL(y_gt|I_full) - LL(y_gen|I_full): Full visual discrimination
- m_n = LL(y_gt|I_bg) - LL(y_gen|I_bg): Background bias
"""

import argparse
import json
import numpy as np
import logging
from typing import List, Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_scores(scores: List[float], method: str = "zscore") -> List[float]:
    """Normalize scores using specified method."""
    scores = np.array(scores)
    
    if method == "zscore":
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 1e-8:
            return ((scores - mean_score) / std_score).tolist()
        else:
            return scores.tolist()
    elif method == "minmax":
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            return ((scores - min_score) / (max_score - min_score)).tolist()
        else:
            return scores.tolist()
    else:
        return scores.tolist()

def sigmoid_transform(x: float, beta: float = 2.0, tau: float = 0.0) -> float:
    """Apply sigmoid transformation: σ(β*(x-τ))"""
    return 1.0 / (1.0 + np.exp(-beta * (x - tau)))

def calculate_visual_consistency_weight(
    delta_pos: float, delta_neg: float, m_v: float, m_n: float,
    alpha: float = 0.8, beta_penalty: float = 0.6, tau_n: float = 0.3,
    beta_sigmoid: float = 2.0, tau_sigmoid: float = 0.0,
    w_min: float = 0.05, w_max: float = 1.0
) -> float:
    """
    Calculate weight for Visual Consistency DPO pair based on TIE metrics.
    
    Formula: S_raw = (Δ+ - Δ-) + α*m_v - β*max(0, m_n - τ_n)
    
    Args:
        delta_pos: Δ+ = LL(y_gt|I_full) - LL(y_gt|I_bg)
        delta_neg: Δ- = LL(y_gen|I_full) - LL(y_gen|I_bg)  
        m_v: LL(y_gt|I_full) - LL(y_gen|I_full)
        m_n: LL(y_gt|I_bg) - LL(y_gen|I_bg)
        alpha: Weight for full visual discrimination
        beta_penalty: Penalty weight for background bias
        tau_n: Threshold for background bias penalty
        beta_sigmoid: Sigmoid steepness parameter
        tau_sigmoid: Sigmoid center bias
        w_min: Minimum weight
        w_max: Maximum weight
    """
    # Calculate raw score
    gamma = delta_pos - delta_neg  # Net visual support effect
    background_penalty = max(0, m_n - tau_n)  # Background bias penalty
    
    s_raw = gamma + alpha * m_v - beta_penalty * background_penalty
    
    # Apply sigmoid transformation
    weight = sigmoid_transform(s_raw, beta_sigmoid, tau_sigmoid)
    
    # Clip to valid range
    weight = max(w_min, min(w_max, weight))
    
    return weight

def build_visual_consistency_pairs(
    tie_results: List[Dict[str, Any]],
    tie_threshold: float = 0.0,
    alpha: float = 0.8,
    beta_penalty: float = 0.6, 
    tau_n: float = 0.3,
    beta_sigmoid: float = 2.0,
    w_min: float = 0.05,
    normalize_method: str = "zscore"
) -> List[Dict[str, Any]]:
    """
    Build DPO pairs from Visual Consistency TIE results.
    
    Args:
        tie_results: List of TIE calculation results
        tie_threshold: Minimum gamma (net visual support) threshold
        alpha: Weight for full visual discrimination
        beta_penalty: Penalty weight for background bias
        tau_n: Threshold for background bias penalty
        beta_sigmoid: Sigmoid steepness for weight mapping
        w_min: Minimum weight threshold
        normalize_method: Score normalization method
    """
    logger.info(f"Building Visual Consistency DPO pairs from {len(tie_results)} TIE results")
    
    # Filter valid results
    valid_results = []
    for result in tie_results:
        if (result.get("gamma") is not None and 
            result.get("m_v") is not None and 
            result.get("m_n") is not None and
            result.get("gt_answer") and 
            result.get("generated_answer") and
            result.get("gt_answer") != result.get("generated_answer")):
            valid_results.append(result)
    
    logger.info(f"Found {len(valid_results)} valid results for pair construction")
    
    if not valid_results:
        logger.warning("No valid results found for DPO pair construction")
        return []
    
    # Extract scores for normalization
    gamma_scores = [r["gamma"] for r in valid_results]
    m_v_scores = [r["m_v"] for r in valid_results] 
    m_n_scores = [r["m_n"] for r in valid_results]
    
    # Normalize scores
    if normalize_method != "none":
        gamma_norm = normalize_scores(gamma_scores, normalize_method)
        m_v_norm = normalize_scores(m_v_scores, normalize_method)
        m_n_norm = normalize_scores(m_n_scores, normalize_method)
    else:
        gamma_norm = gamma_scores
        m_v_norm = m_v_scores
        m_n_norm = m_n_scores
    
    # Build pairs
    pairs = []
    weights = []
    
    for i, result in enumerate(valid_results):
        # Use normalized scores for weight calculation
        delta_pos = result["delta_pos"]
        delta_neg = result["delta_neg"]
        m_v = m_v_norm[i] if normalize_method != "none" else result["m_v"]
        m_n = m_n_norm[i] if normalize_method != "none" else result["m_n"]
        gamma = gamma_norm[i] if normalize_method != "none" else result["gamma"]
        
        # Apply gamma threshold filter
        if gamma < tie_threshold:
            continue
        
        # Calculate weight
        weight = calculate_visual_consistency_weight(
            delta_pos, delta_neg, m_v, m_n,
            alpha=alpha, beta_penalty=beta_penalty, tau_n=tau_n,
            beta_sigmoid=beta_sigmoid, w_min=w_min
        )
        
        # Skip if weight is too low
        if weight < w_min:
            continue
        
        # Construct DPO pair
        pair = {
            "qid": result["qid"],
            "image": result["image"],
            "question": result["question"],
            
            # Preferred: Ground-truth answer on full image
            "preferred": {
                "answer": result["gt_answer"],
                "input_type": "full_image",
                "ll_score": result["ll_gt_full"]
            },
            
            # Dispreferred: Generated answer on full image  
            "dispreferred": {
                "answer": result["generated_answer"],
                "input_type": "full_image", 
                "ll_score": result["ll_gen_full"]
            },
            
            # TIE metrics
            "tie_metrics": {
                "delta_pos": result["delta_pos"],
                "delta_neg": result["delta_neg"], 
                "gamma": result["gamma"],
                "m_v": result["m_v"],
                "m_n": result["m_n"],
                "gamma_normalized": gamma,
                "m_v_normalized": m_v,
                "m_n_normalized": m_n
            },
            
            # Weight and metadata
            "weight": weight,
            "method": "visual_consistency",
            "comparison_type": "gt_vs_generated",
            "pair_type": "same_visual_input"
        }
        
        pairs.append(pair)
        weights.append(weight)
    
    # Log statistics
    if pairs:
        weights = np.array(weights)
        logger.info(f"Generated {len(pairs)} Visual Consistency DPO pairs")
        logger.info(f"Weight statistics: mean={weights.mean():.3f}, std={weights.std():.3f}, "
                   f"min={weights.min():.3f}, max={weights.max():.3f}")
        
        # Log gamma distribution
        gammas = [p["tie_metrics"]["gamma"] for p in pairs]
        gamma_array = np.array(gammas)
        logger.info(f"Gamma (net visual support) statistics: mean={gamma_array.mean():.3f}, "
                   f"std={gamma_array.std():.3f}, min={gamma_array.min():.3f}, max={gamma_array.max():.3f}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Build Visual Consistency DPO pairs from TIE results")
    parser.add_argument("--tie-results-file", type=str, required=True,
                       help="Path to TIE results JSONL file")
    parser.add_argument("--output-pairs-file", type=str, required=True,
                       help="Path to output DPO pairs JSONL file")
    parser.add_argument("--tie-threshold", type=float, default=0.0,
                       help="Minimum gamma (net visual support) threshold")
    parser.add_argument("--alpha", type=float, default=0.8,
                       help="Weight for full visual discrimination (m_v)")
    parser.add_argument("--beta-penalty", type=float, default=0.6,
                       help="Penalty weight for background bias")
    parser.add_argument("--tau-n", type=float, default=0.3,
                       help="Threshold for background bias penalty")
    parser.add_argument("--beta", type=float, default=2.0,
                       help="Sigmoid steepness parameter")
    parser.add_argument("--w-min", type=float, default=0.05,
                       help="Minimum weight threshold")
    parser.add_argument("--normalize-method", type=str, default="zscore",
                       choices=["zscore", "minmax", "none"],
                       help="Score normalization method")
    
    args = parser.parse_args()
    
    # Load TIE results
    logger.info(f"Loading TIE results from {args.tie_results_file}")
    tie_results = []
    with open(args.tie_results_file, 'r') as f:
        for line in f:
            tie_results.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(tie_results)} TIE results")
    
    # Build DPO pairs
    pairs = build_visual_consistency_pairs(
        tie_results,
        tie_threshold=args.tie_threshold,
        alpha=args.alpha,
        beta_penalty=args.beta_penalty,
        tau_n=args.tau_n,
        beta_sigmoid=args.beta,
        w_min=args.w_min,
        normalize_method=args.normalize_method
    )
    
    # Save pairs
    os.makedirs(os.path.dirname(args.output_pairs_file), exist_ok=True)
    
    logger.info(f"Saving {len(pairs)} DPO pairs to {args.output_pairs_file}")
    with open(args.output_pairs_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    logger.info("Visual Consistency DPO pair construction completed successfully!")
    
    # Print summary
    if pairs:
        print(f"\n=== Visual Consistency DPO Pairs Summary ===")
        print(f"Total pairs generated: {len(pairs)}")
        print(f"Method: Visual Consistency Preference")
        print(f"Comparison: Ground-truth vs. Generated answers")
        print(f"Input: Same visual input (I_full ⊕ I_bg)")
        print(f"Weight range: [{args.w_min:.3f}, 1.0]")
        print(f"Gamma threshold: {args.tie_threshold}")
        
        weights = [p["weight"] for p in pairs]
        print(f"Weight distribution: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")

if __name__ == "__main__":
    main()