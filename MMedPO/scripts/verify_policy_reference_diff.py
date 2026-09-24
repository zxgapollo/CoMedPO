#!/usr/bin/env python3
"""
验证修复后的 policy 和 reference log probabilities 是否不同
这是确保 DPO 训练正常工作的关键测试
"""

import torch
import os
import sys
from contextlib import contextmanager
import numpy as np

# 添加项目路径
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo/llava')

from llava.model.builder import load_pretrained_model
from llava.model import LlavaMistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoTokenizer
import transformers

@contextmanager
def safe_disable_adapter(model):
    """安全的 disable_adapter 上下文管理器"""
    if not isinstance(model, PeftModel):
        # 如果不是 PeftModel，直接返回
        yield model
        return
    
    try:
        # 禁用适配器
        model.disable_adapter()
        print("适配器已安全禁用")
        
        yield model
        
    finally:
        # 确保适配器被重新启用
        try:
            if hasattr(model, 'enable_adapter'):
                model.enable_adapter()
            elif hasattr(model, 'enable_adapters'):
                model.enable_adapters()
            print("适配器已安全重新启用")
        except Exception as e:
            print(f"警告：重新启用适配器时出现问题: {e}")
            # 尝试重新加载适配器
            try:
                model.set_adapter("default")
                print("通过 set_adapter 重新启用适配器")
            except:
                pass

def test_policy_reference_difference():
    """测试 policy 和 reference 模型的 log probabilities 差异"""
    
    print("=== 验证 Policy 和 Reference Log Probabilities 差异 ===")
    
    # 模拟训练参数
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["mm_projector"],
    )
    
    print(f"加载基础模型: {base_model_path}")
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print(f"加载 LoRA 适配器: {lora_checkpoint_path}")
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建测试输入
    test_text = "What is the diagnosis for this medical condition?"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    print(f"测试文本: {test_text}")
    print(f"输入 token 数量: {input_ids.shape[1]}")
    
    # 获取 policy 模型的输出（启用 LoRA）
    print("\n--- 获取 Policy 模型输出（启用 LoRA）---")
    model.eval()
    with torch.no_grad():
        policy_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        
        # 计算平均 log probability
        policy_mean_logprob = policy_log_probs.mean().item()
        print(f"Policy 平均 log probability: {policy_mean_logprob:.6f}")
    
    # 获取 reference 模型的输出（禁用 LoRA）
    print("\n--- 获取 Reference 模型输出（禁用 LoRA）---")
    with safe_disable_adapter(model):
        with torch.no_grad():
            reference_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            reference_logits = reference_outputs.logits
            reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
            
            # 计算平均 log probability
            reference_mean_logprob = reference_log_probs.mean().item()
            print(f"Reference 平均 log probability: {reference_mean_logprob:.6f}")
    
    # 计算差异
    print("\n--- 分析差异 ---")
    logits_diff = torch.abs(policy_logits - reference_logits).mean().item()
    logprob_diff = torch.abs(policy_log_probs - reference_log_probs).mean().item()
    mean_logprob_diff = abs(policy_mean_logprob - reference_mean_logprob)
    
    print(f"Logits 平均绝对差异: {logits_diff:.6f}")
    print(f"Log probabilities 平均绝对差异: {logprob_diff:.6f}")
    print(f"平均 log probability 差异: {mean_logprob_diff:.6f}")
    
    # 判断是否有显著差异
    significant_diff_threshold = 1e-4
    
    if logprob_diff > significant_diff_threshold:
        print(f"✅ Policy 和 Reference 模型有显著差异 (差异 > {significant_diff_threshold})")
        print("这表明 LoRA 适配器正在正常工作")
        return True
    else:
        print(f"❌ Policy 和 Reference 模型差异过小 (差异 <= {significant_diff_threshold})")
        print("这可能表明 LoRA 适配器没有正常工作或者适配器权重过小")
        return False

def test_token_level_differences():
    """测试 token 级别的差异"""
    
    print("\n=== Token 级别差异分析 ===")
    
    # 模拟训练参数
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["mm_projector"],
    )
    
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建测试输入
    test_text = "The patient shows symptoms of"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    model.eval()
    
    # 获取 policy 和 reference 的 top-k predictions
    with torch.no_grad():
        # Policy 模型
        policy_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits[0, -1, :]  # 最后一个 token 的 logits
        policy_probs = torch.softmax(policy_logits, dim=-1)
        policy_top_k = torch.topk(policy_probs, k=10)
        
        # Reference 模型
        with safe_disable_adapter(model):
            reference_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            reference_logits = reference_outputs.logits[0, -1, :]  # 最后一个 token 的 logits
            reference_probs = torch.softmax(reference_logits, dim=-1)
            reference_top_k = torch.topk(reference_probs, k=10)
    
    print(f"测试文本: '{test_text}'")
    print("\nPolicy 模型 Top-10 预测:")
    for i, (prob, token_id) in enumerate(zip(policy_top_k.values, policy_top_k.indices)):
        token = tokenizer.decode([token_id.item()])
        print(f"  {i+1}. '{token}' (概率: {prob.item():.4f})")
    
    print("\nReference 模型 Top-10 预测:")
    for i, (prob, token_id) in enumerate(zip(reference_top_k.values, reference_top_k.indices)):
        token = tokenizer.decode([token_id.item()])
        print(f"  {i+1}. '{token}' (概率: {prob.item():.4f})")
    
    # 计算 KL 散度
    kl_div = torch.nn.functional.kl_div(
        torch.log(reference_probs + 1e-8), 
        policy_probs, 
        reduction='sum'
    ).item()
    
    print(f"\nKL 散度 (Policy || Reference): {kl_div:.6f}")
    
    if kl_div > 0.01:
        print("✅ Policy 和 Reference 模型有显著的分布差异")
        return True
    else:
        print("❌ Policy 和 Reference 模型分布过于相似")
        return False

if __name__ == "__main__":
    print("开始验证 Policy 和 Reference 模型差异...")
    
    # 测试整体差异
    success1 = test_policy_reference_difference()
    
    # 测试 token 级别差异
    success2 = test_token_level_differences()
    
    if success1 and success2:
        print("\n🎉 验证成功！Policy 和 Reference 模型有显著差异")
        print("DPO 训练应该能够正常工作")
    else:
        print("\n❌ 验证失败，Policy 和 Reference 模型差异不足")
        print("需要检查 LoRA 适配器配置或训练过程")