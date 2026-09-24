#!/usr/bin/env python3
"""
简化的 LoRA 检查脚本，专注于验证 LoRA 适配器是否正确加载
"""

import torch
import os
import sys
import json

# 添加项目路径
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo/llava')

from llava.model.builder import load_pretrained_model
from llava.model import LlavaMistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoTokenizer
import transformers

def check_lora_checkpoint_details():
    """详细检查 LoRA 检查点内容"""
    
    print("=== 检查 LoRA 检查点详细信息 ===")
    
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 检查 adapter_config.json
    config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if os.path.exists(config_path):
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
        non_zero_params = 0
        
        for name, param in checkpoint.items():
            param_count = param.numel()
            total_params += param_count
            
            # 检查是否有非零权重
            max_abs_val = param.abs().max().item()
            if max_abs_val > 1e-6:
                non_zero_params += 1
            
            print(f"  {name}:")
            print(f"    形状: {param.shape}")
            print(f"    参数数量: {param_count:,}")
            print(f"    最大绝对值: {max_abs_val:.8f}")
            print(f"    均值: {param.mean().item():.8f}")
            print(f"    标准差: {param.std().item():.8f}")
            print()
        
        print(f"总 LoRA 参数数量: {total_params:,}")
        print(f"非零权重参数数量: {non_zero_params}/{len(checkpoint)}")
        
        if non_zero_params > 0:
            print("✅ LoRA 权重包含非零值，看起来是有效的")
            return True
        else:
            print("❌ 所有 LoRA 权重都接近零，可能没有经过训练")
            return False
    else:
        print(f"❌ 未找到 adapter_model.bin: {model_path}")
        return False

def test_lora_loading_without_cuda():
    """在 CPU 上测试 LoRA 加载，避免 CUDA 内存问题"""
    
    print("\n=== 在 CPU 上测试 LoRA 加载 ===")
    
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    try:
        # 在 CPU 上加载基础模型（不使用量化）
        print("在 CPU 上加载基础模型...")
        base_model = LlavaMistralForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print(f"基础模型类型: {type(base_model)}")
        print(f"基础模型设备: {next(base_model.parameters()).device}")
        
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
        
        # 检查是否可以禁用/启用适配器
        print(f"\n测试适配器禁用/启用:")
        
        # 尝试禁用适配器
        try:
            lora_model.disable_adapter()
            print("✅ 成功禁用适配器")
            
            # 尝试重新启用适配器
            try:
                if hasattr(lora_model, 'enable_adapter'):
                    lora_model.enable_adapter()
                    print("✅ 成功重新启用适配器 (enable_adapter)")
                elif hasattr(lora_model, 'enable_adapters'):
                    lora_model.enable_adapters()
                    print("✅ 成功重新启用适配器 (enable_adapters)")
                else:
                    print("❌ 没有找到启用适配器的方法")
            except Exception as e:
                print(f"❌ 重新启用适配器失败: {e}")
                
                # 尝试使用 set_adapter
                try:
                    lora_model.set_adapter("default")
                    print("✅ 通过 set_adapter 成功重新启用适配器")
                except Exception as e2:
                    print(f"❌ set_adapter 也失败: {e2}")
                    
        except Exception as e:
            print(f"❌ 禁用适配器失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False

def analyze_lora_architecture():
    """分析 LoRA 架构和目标模块"""
    
    print("\n=== 分析 LoRA 架构 ===")
    
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 读取配置
    config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if not os.path.exists(config_path):
        print("❌ 配置文件不存在")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 分析目标模块
    target_modules = config.get('target_modules', [])
    print(f"LoRA 目标模块: {target_modules}")
    
    # 读取权重文件
    model_path = os.path.join(lora_checkpoint_path, "adapter_model.bin")
    if not os.path.exists(model_path):
        print("❌ 权重文件不存在")
        return False
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 分析权重结构
    print(f"\n权重结构分析:")
    layer_stats = {}
    
    for name, param in checkpoint.items():
        # 提取层信息
        parts = name.split('.')
        if 'layers' in parts:
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = parts[i + 1]
                    break
            
            if layer_idx is not None:
                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = []
                layer_stats[layer_idx].append(name)
    
    print(f"涉及的层数: {len(layer_stats)}")
    for layer_idx, params in layer_stats.items():
        print(f"  层 {layer_idx}: {len(params)} 个参数")
        for param_name in params[:3]:  # 只显示前3个
            print(f"    {param_name}")
        if len(params) > 3:
            print(f"    ... 还有 {len(params) - 3} 个参数")
    
    return True

if __name__ == "__main__":
    print("开始简化的 LoRA 检查...")
    
    # 1. 检查 LoRA 检查点详细信息
    checkpoint_valid = check_lora_checkpoint_details()
    
    if not checkpoint_valid:
        print("❌ LoRA 检查点无效，无法继续测试")
        sys.exit(1)
    
    # 2. 分析 LoRA 架构
    analyze_lora_architecture()
    
    # 3. 在 CPU 上测试 LoRA 加载
    loading_success = test_lora_loading_without_cuda()
    
    if loading_success:
        print("\n🎉 LoRA 检查完成！")
        print("主要发现:")
        print("- LoRA 检查点存在且包含有效权重")
        print("- LoRA 适配器可以成功加载")
        print("- 适配器禁用/启用功能需要特殊处理")
    else:
        print("\n❌ LoRA 检查失败，需要进一步调试")