#!/usr/bin/env python3
"""
检查 LoRA 权重是否正确加载和应用
这是调试 DPO 训练中 policy 和 reference 模型输出相同问题的关键
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

def check_base_model_weights():
    """检查基础模型权重"""
    
    print("=== 检查基础模型权重 ===")
    
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["mm_projector"],
    )
    
    print(f"加载基础模型: {base_model_path}")
    base_model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 检查一些关键层的权重
    print("\n基础模型关键层权重统计:")
    
    # 检查语言模型的第一层
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        first_layer = base_model.model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            q_proj_weight = first_layer.self_attn.q_proj.weight
            print(f"  第一层 q_proj 权重形状: {q_proj_weight.shape}")
            print(f"  第一层 q_proj 权重数据类型: {q_proj_weight.dtype}")
            
            # 处理量化权重
            if q_proj_weight.dtype == torch.uint8:
                print("  检测到量化权重 (uint8)，跳过统计计算")
            else:
                print(f"  第一层 q_proj 权重均值: {q_proj_weight.float().mean().item():.6f}")
                print(f"  第一层 q_proj 权重标准差: {q_proj_weight.float().std().item():.6f}")
    
    return base_model

def check_lora_checkpoint():
    """检查 LoRA 检查点内容"""
    
    print("\n=== 检查 LoRA 检查点内容 ===")
    
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 检查 adapter_config.json
    config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"LoRA 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ 未找到 adapter_config.json: {config_path}")
        return False
    
    # 检查 adapter_model.bin
    model_path = os.path.join(lora_checkpoint_path, "adapter_model.bin")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"\nLoRA 权重文件包含 {len(checkpoint)} 个参数:")
        
        total_params = 0
        for name, param in checkpoint.items():
            print(f"  {name}: {param.shape}, 均值: {param.mean().item():.6f}, 标准差: {param.std().item():.6f}")
            total_params += param.numel()
        
        print(f"总 LoRA 参数数量: {total_params:,}")
        
        # 检查是否有非零权重
        non_zero_params = sum(1 for param in checkpoint.values() if param.abs().max().item() > 1e-6)
        print(f"非零权重参数数量: {non_zero_params}/{len(checkpoint)}")
        
        return True
    else:
        print(f"❌ 未找到 adapter_model.bin: {model_path}")
        return False

def compare_before_after_lora():
    """比较加载 LoRA 前后的模型权重"""
    
    print("\n=== 比较加载 LoRA 前后的模型权重 ===")
    
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
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 获取基础模型的一些权重作为参考
    base_weights = {}
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        first_layer = base_model.model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            weight = first_layer.self_attn.q_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                base_weights['q_proj'] = weight.clone()
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'v_proj'):
            weight = first_layer.self_attn.v_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                base_weights['v_proj'] = weight.clone()
    
    print(f"基础模型权重统计:")
    for name, weight in base_weights.items():
        print(f"  {name}: 形状 {weight.shape}, 均值 {weight.float().mean().item():.6f}, 标准差 {weight.float().std().item():.6f}")
    
    # 加载 LoRA 适配器
    print(f"\n加载 LoRA 适配器: {lora_checkpoint_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    
    print(f"LoRA 模型类型: {type(lora_model)}")
    print(f"是否为 PeftModel: {isinstance(lora_model, PeftModel)}")
    
    # 检查 LoRA 配置
    if hasattr(lora_model, 'peft_config'):
        print(f"LoRA 配置: {lora_model.peft_config}")
    
    # 检查适配器状态
    if hasattr(lora_model, 'active_adapters'):
        print(f"活跃适配器: {lora_model.active_adapters}")
    
    # 获取加载 LoRA 后的权重
    lora_weights = {}
    if hasattr(lora_model, 'base_model') and hasattr(lora_model.base_model, 'model') and hasattr(lora_model.base_model.model, 'layers'):
        first_layer = lora_model.base_model.model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            weight = first_layer.self_attn.q_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                lora_weights['q_proj'] = weight.clone()
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'v_proj'):
            weight = first_layer.self_attn.v_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                lora_weights['v_proj'] = weight.clone()
    elif hasattr(lora_model, 'model') and hasattr(lora_model.model, 'layers'):
        # 直接访问模型层
        first_layer = lora_model.model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            weight = first_layer.self_attn.q_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                lora_weights['q_proj'] = weight.clone()
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'v_proj'):
            weight = first_layer.self_attn.v_proj.weight
            if weight.dtype != torch.uint8:  # 跳过量化权重
                lora_weights['v_proj'] = weight.clone()
    
    print(f"\n加载 LoRA 后权重统计:")
    for name, weight in lora_weights.items():
        print(f"  {name}: 形状 {weight.shape}, 均值 {weight.float().mean().item():.6f}, 标准差 {weight.float().std().item():.6f}")
    
    # 比较权重差异
    print(f"\n权重差异分析:")
    for name in base_weights.keys():
        if name in lora_weights:
            diff = torch.abs(base_weights[name].float() - lora_weights[name].float()).mean().item()
            print(f"  {name} 平均绝对差异: {diff:.8f}")
            
            if diff > 1e-6:
                print(f"    ✅ {name} 权重有显著变化")
            else:
                print(f"    ❌ {name} 权重几乎没有变化")
        else:
            print(f"  {name} 在 LoRA 模型中不可用（可能是量化权重）")
    
    return lora_model

def test_forward_pass_differences():
    """测试前向传播的差异"""
    
    print("\n=== 测试前向传播差异 ===")
    
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
    
    # 加载基础模型
    base_model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建测试输入
    test_text = "What is the diagnosis?"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(base_model.device)
    attention_mask = inputs["attention_mask"].to(base_model.device)
    
    # 基础模型前向传播
    print("基础模型前向传播...")
    base_model.eval()
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_outputs.logits
        base_probs = torch.softmax(base_logits[0, -1, :], dim=-1)
        base_top_k = torch.topk(base_probs, k=5)
    
    print("基础模型 Top-5 预测:")
    for i, (prob, token_id) in enumerate(zip(base_top_k.values, base_top_k.indices)):
        token = tokenizer.decode([token_id.item()])
        print(f"  {i+1}. '{token}' (概率: {prob.item():.4f})")
    
    # 加载 LoRA 模型
    print(f"\n加载 LoRA 模型...")
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    
    # LoRA 模型前向传播
    print("LoRA 模型前向传播...")
    lora_model.eval()
    with torch.no_grad():
        lora_outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
        lora_logits = lora_outputs.logits
        lora_probs = torch.softmax(lora_logits[0, -1, :], dim=-1)
        lora_top_k = torch.topk(lora_probs, k=5)
    
    print("LoRA 模型 Top-5 预测:")
    for i, (prob, token_id) in enumerate(zip(lora_top_k.values, lora_top_k.indices)):
        token = tokenizer.decode([token_id.item()])
        print(f"  {i+1}. '{token}' (概率: {prob.item():.4f})")
    
    # 计算差异
    logits_diff = torch.abs(base_logits - lora_logits).mean().item()
    probs_diff = torch.abs(base_probs - lora_probs).mean().item()
    
    print(f"\n差异分析:")
    print(f"  Logits 平均绝对差异: {logits_diff:.8f}")
    print(f"  Probabilities 平均绝对差异: {probs_diff:.8f}")
    
    if logits_diff > 1e-4:
        print("  ✅ 基础模型和 LoRA 模型有显著差异")
        return True
    else:
        print("  ❌ 基础模型和 LoRA 模型差异过小")
        return False

if __name__ == "__main__":
    print("开始检查 LoRA 权重加载和应用...")
    
    # 1. 检查基础模型
    base_model = check_base_model_weights()
    
    # 2. 检查 LoRA 检查点
    lora_exists = check_lora_checkpoint()
    
    if not lora_exists:
        print("❌ LoRA 检查点不存在或不完整，无法继续测试")
        sys.exit(1)
    
    # 3. 比较加载前后的权重
    lora_model = compare_before_after_lora()
    
    # 4. 测试前向传播差异
    has_difference = test_forward_pass_differences()
    
    if has_difference:
        print("\n🎉 LoRA 权重正确加载并应用！")
    else:
        print("\n❌ LoRA 权重可能没有正确应用")