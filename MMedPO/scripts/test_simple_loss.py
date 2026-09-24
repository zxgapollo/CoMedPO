#!/usr/bin/env python3
"""
简化的双模型loss计算测试
"""

import torch
import os
import sys
import time

# 添加项目路径
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo/llava')

from llava.model import LlavaMistralForCausalLM
from transformers import AutoTokenizer
import torch.nn.functional as F

def create_simple_batch(tokenizer, batch_size=1, seq_len=20):
    """创建简单的测试批次"""
    # 使用tokenizer编码一些简单文本
    text = "Hello, how are you today?"
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    # 如果序列太短，重复填充
    if tokens.shape[1] < seq_len:
        repeat_times = (seq_len // tokens.shape[1]) + 1
        tokens = tokens.repeat(1, repeat_times)[:, :seq_len]
    else:
        tokens = tokens[:, :seq_len]
    
    # 扩展到batch_size
    if batch_size > 1:
        tokens = tokens.repeat(batch_size, 1)
    
    return {
        'input_ids': tokens,
        'attention_mask': torch.ones_like(tokens),
        'labels': tokens.clone()
    }

def compute_model_loss(model, batch, device):
    """计算单个模型的loss"""
    # 将数据移动到指定设备
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    return outputs.loss

def main():
    print("=== 简化双模型Loss计算测试 ===")
    
    # 检查GPU可用性
    if torch.cuda.device_count() < 2:
        print(f"❌ 需要至少2个GPU，但只检测到 {torch.cuda.device_count()} 个")
        return False
    
    print(f"✅ 检测到 {torch.cuda.device_count()} 个GPU")
    
    # 模型路径
    policy_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    reference_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    
    try:
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("创建测试批次...")
        batch = create_simple_batch(tokenizer, batch_size=1, seq_len=20)
        
        print("加载策略模型到GPU 0...")
        policy_model = LlavaMistralForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True
        )
        policy_model.eval()
        
        print("计算策略模型loss...")
        policy_loss = compute_model_loss(policy_model, batch, "cuda:0")
        print(f"策略模型loss: {policy_loss.item():.6f}")
        
        # 清理策略模型
        del policy_model
        torch.cuda.empty_cache()
        
        print("加载参考模型到GPU 1...")
        reference_model = LlavaMistralForCausalLM.from_pretrained(
            reference_model_path,
            torch_dtype=torch.float16,
            device_map={"": 1},
            low_cpu_mem_usage=True
        )
        reference_model.eval()
        
        print("计算参考模型loss...")
        reference_loss = compute_model_loss(reference_model, batch, "cuda:1")
        print(f"参考模型loss: {reference_loss.item():.6f}")
        
        print(f"\n✅ 测试成功完成!")
        print(f"策略模型 ({policy_model_path}) loss: {policy_loss.item():.6f}")
        print(f"参考模型 ({reference_model_path}) loss: {reference_loss.item():.6f}")
        print(f"Loss差异: {abs(policy_loss.item() - reference_loss.item()):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()