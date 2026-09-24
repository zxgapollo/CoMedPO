#!/usr/bin/env python3
"""
设备检查脚本 - 诊断模型各部分的设备分配情况
"""

import torch
import os
import sys
sys.path.append('/workspace/MMedPO/MMedPO')
sys.path.append('/workspace/MMedPO')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

from llava.model import *

def check_model_devices(model, model_name):
    """检查模型各部分的设备分配"""
    print(f"\n=== {model_name} 设备分配检查 ===")
    
    device_count = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_count:
            device_count[device] = 0
        device_count[device] += 1
        
        # 只显示前10个参数的设备信息
        if len(device_count) <= 2 and device_count[device] <= 5:
            print(f"  {name}: {device}")
    
    print(f"\n设备分布统计:")
    for device, count in device_count.items():
        print(f"  {device}: {count} 个参数")
    
    return device_count

def main():
    print("开始设备检查...")
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    
    policy_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    reference_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    
    try:
        print("\n加载策略模型到GPU 0...")
        policy_model = LlavaMistralForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True
        )
        
        policy_devices = check_model_devices(policy_model, "策略模型")
        
        print("\n加载参考模型到GPU 1...")
        reference_model = LlavaMistralForCausalLM.from_pretrained(
            reference_model_path,
            torch_dtype=torch.float16,
            device_map={"": 1},
            low_cpu_mem_usage=True
        )
        
        reference_devices = check_model_devices(reference_model, "参考模型")
        
        # 检查是否有设备冲突
        print("\n=== 设备冲突检查 ===")
        if len(policy_devices) > 1:
            print("⚠️  策略模型跨多个设备分布!")
        else:
            print("✅ 策略模型在单一设备上")
            
        if len(reference_devices) > 1:
            print("⚠️  参考模型跨多个设备分布!")
        else:
            print("✅ 参考模型在单一设备上")
        
        # 测试简单前向传播
        print("\n=== 前向传播测试 ===")
        
        # 创建测试输入
        test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        
        # 测试策略模型
        try:
            test_input_policy = test_input_ids.to('cuda:0')
            with torch.no_grad():
                policy_output = policy_model(input_ids=test_input_policy)
            print("✅ 策略模型前向传播成功")
        except Exception as e:
            print(f"❌ 策略模型前向传播失败: {e}")
        
        # 测试参考模型
        try:
            test_input_reference = test_input_ids.to('cuda:1')
            with torch.no_grad():
                reference_output = reference_model(input_ids=test_input_reference)
            print("✅ 参考模型前向传播成功")
        except Exception as e:
            print(f"❌ 参考模型前向传播失败: {e}")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    print("\n设备检查完成!")
    return True

if __name__ == "__main__":
    main()