#!/usr/bin/env python3
"""
DPO设备调试脚本 - 模拟DPO训练器的具体操作流程
"""

import torch
import os
import sys
sys.path.append('/workspace/MMedPO/MMedPO')
sys.path.append('/workspace/MMedPO')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

from llava.model import *
from transformers import AutoTokenizer

def create_mock_batch():
    """创建模拟批次数据"""
    batch_size = 2
    seq_len = 10
    
    # 创建模拟数据，不预先分配到任何设备
    batch = {
        'chosen_input_ids': torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long),
        'chosen_attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'chosen_labels': torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long),
        'rejected_input_ids': torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long),
        'rejected_attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'rejected_labels': torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long),
    }
    
    return batch

def check_model_device_details(model, model_name):
    """详细检查模型各层的设备分配"""
    print(f"\n=== {model_name} 详细设备检查 ===")
    
    # 检查embedding层
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_device = model.model.embed_tokens.weight.device
        print(f"  embed_tokens: {embed_device}")
    
    # 检查第一层和最后一层
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        if len(model.model.layers) > 0:
            first_layer_device = next(model.model.layers[0].parameters()).device
            last_layer_device = next(model.model.layers[-1].parameters()).device
            print(f"  第一层: {first_layer_device}")
            print(f"  最后一层: {last_layer_device}")
    
    # 检查lm_head
    if hasattr(model, 'lm_head'):
        lm_head_device = model.lm_head.weight.device
        print(f"  lm_head: {lm_head_device}")
    
    return True

def test_concatenated_forward_simulation(model, batch, target_device, model_name):
    """模拟concatenated_forward操作"""
    print(f"\n=== 模拟 {model_name} concatenated_forward ===")
    
    try:
        # 将批次数据移动到目标设备
        print(f"将批次数据移动到 {target_device}")
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(target_device)
                print(f"  {key}: {value.device} -> {batch_on_device[key].device}")
            else:
                batch_on_device[key] = value
        
        # 测试chosen数据的前向传播
        print(f"\n测试chosen数据前向传播...")
        print(f"  input_ids设备: {batch_on_device['chosen_input_ids'].device}")
        print(f"  attention_mask设备: {batch_on_device['chosen_attention_mask'].device}")
        
        with torch.no_grad():
            chosen_outputs = model(
                input_ids=batch_on_device['chosen_input_ids'],
                attention_mask=batch_on_device['chosen_attention_mask']
            )
        
        print(f"  ✅ chosen前向传播成功，输出设备: {chosen_outputs.logits.device}")
        
        # 测试rejected数据的前向传播
        print(f"\n测试rejected数据前向传播...")
        print(f"  input_ids设备: {batch_on_device['rejected_input_ids'].device}")
        print(f"  attention_mask设备: {batch_on_device['rejected_attention_mask'].device}")
        
        with torch.no_grad():
            rejected_outputs = model(
                input_ids=batch_on_device['rejected_input_ids'],
                attention_mask=batch_on_device['rejected_attention_mask']
            )
        
        print(f"  ✅ rejected前向传播成功，输出设备: {rejected_outputs.logits.device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        print(f"  错误类型: {type(e).__name__}")
        return False

def main():
    print("开始DPO设备调试...")
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    
    policy_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    reference_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    
    try:
        print("\n=== 加载模型 ===")
        print("加载策略模型到GPU 0...")
        policy_model = LlavaMistralForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True
        )
        
        print("加载参考模型到GPU 1...")
        reference_model = LlavaMistralForCausalLM.from_pretrained(
            reference_model_path,
            torch_dtype=torch.float16,
            device_map={"": 1},
            low_cpu_mem_usage=True
        )
        
        # 详细检查模型设备分配
        check_model_device_details(policy_model, "策略模型")
        check_model_device_details(reference_model, "参考模型")
        
        # 创建模拟批次
        print("\n=== 创建模拟批次 ===")
        batch = create_mock_batch()
        print("批次数据创建完成，初始设备为CPU")
        
        # 测试策略模型的concatenated_forward
        success_policy = test_concatenated_forward_simulation(
            policy_model, batch, 'cuda:0', "策略模型"
        )
        
        # 测试参考模型的concatenated_forward
        success_reference = test_concatenated_forward_simulation(
            reference_model, batch, 'cuda:1', "参考模型"
        )
        
        print(f"\n=== 测试结果 ===")
        print(f"策略模型测试: {'✅ 成功' if success_policy else '❌ 失败'}")
        print(f"参考模型测试: {'✅ 成功' if success_reference else '❌ 失败'}")
        
        if success_policy and success_reference:
            print("\n🎉 所有测试通过！DPO训练器应该能正常工作。")
        else:
            print("\n⚠️  存在设备问题，需要进一步调试。")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    print("\nDPO设备调试完成!")
    return True

if __name__ == "__main__":
    main()