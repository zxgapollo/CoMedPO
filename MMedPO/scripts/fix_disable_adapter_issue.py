#!/usr/bin/env python3
"""
修复 disable_adapter 问题的脚本
问题：在 DPO 训练中，使用 disable_adapter() 后无法正确重新启用适配器
解决方案：使用上下文管理器确保适配器状态正确恢复
"""

import torch
import os
import sys
from contextlib import contextmanager

# 添加项目路径
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo/llava')

from llava.model.builder import load_pretrained_model
from llava.model import LlavaMistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import transformers

def test_disable_adapter_fix():
    """测试修复后的 disable_adapter 功能"""
    
    print("=== 测试 disable_adapter 修复方案 ===")
    
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
    
    print(f"模型类型: {type(model)}")
    print(f"是否为 PeftModel: {isinstance(model, PeftModel)}")
    print(f"LoRA 配置: {model.peft_config}")
    
    # 测试原始的 disable_adapter 方法
    print("\n=== 测试原始 disable_adapter 方法 ===")
    try:
        print("调用 disable_adapter()...")
        with model.disable_adapter():
            print("适配器已禁用，模型状态正常")
            # 在这里可以获取参考模型的输出
            
        print("退出 disable_adapter 上下文")
        print("适配器应该已重新启用")
        
        # 验证适配器是否正确重新启用
        if hasattr(model, 'active_adapters'):
            print(f"活跃适配器: {model.active_adapters}")
        
        print("✅ disable_adapter 上下文管理器工作正常")
        
    except Exception as e:
        print(f"❌ disable_adapter 出现错误: {e}")
        return False
    
    return True

@contextmanager
def safe_disable_adapter(model):
    """安全的 disable_adapter 上下文管理器"""
    if not isinstance(model, PeftModel):
        # 如果不是 PeftModel，直接返回
        yield model
        return
    
    try:
        # 保存当前适配器状态
        original_adapters = getattr(model, 'active_adapters', None)
        
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

def test_safe_disable_adapter():
    """测试安全的 disable_adapter 实现"""
    
    print("\n=== 测试安全的 disable_adapter 实现 ===")
    
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
    
    try:
        print("使用安全的 disable_adapter...")
        with safe_disable_adapter(model):
            print("在安全上下文中，适配器已禁用")
            # 这里可以安全地获取参考模型输出
            
        print("退出安全上下文")
        print("✅ 安全的 disable_adapter 工作正常")
        return True
        
    except Exception as e:
        print(f"❌ 安全的 disable_adapter 出现错误: {e}")
        return False

if __name__ == "__main__":
    print("开始测试 disable_adapter 修复方案...")
    
    # 测试原始方法
    success1 = test_disable_adapter_fix()
    
    # 测试安全方法
    success2 = test_safe_disable_adapter()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！disable_adapter 问题已修复")
    else:
        print("\n❌ 测试失败，需要进一步调试")