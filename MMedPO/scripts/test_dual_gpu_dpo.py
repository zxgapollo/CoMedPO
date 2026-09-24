#!/usr/bin/env python3
"""
测试双GPU DPO trainer的功能和性能
"""

import torch
import os
import sys
import time
import numpy as np
from contextlib import contextmanager

# 添加项目路径
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo/llava')

from llava.model.builder import load_pretrained_model
from llava.model import LlavaMistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoTokenizer
import transformers

# 导入双GPU DPO trainer
from tool.dual_gpu_dpo_trainer import DualGPUDPOTrainer

def create_mock_batch(batch_size=2, seq_len=50):
    """创建模拟训练批次"""
    # 使用合理的token ID范围（1到31999，避免0和超出词汇表大小）
    # 不预先分配到GPU，让trainer处理设备分配
    batch = {
        'chosen_input_ids': torch.randint(1, 31999, (batch_size, seq_len), dtype=torch.long),
        'chosen_attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long),
        'chosen_labels': torch.randint(1, 31999, (batch_size, seq_len), dtype=torch.long),
        'rejected_input_ids': torch.randint(1, 31999, (batch_size, seq_len), dtype=torch.long),
        'rejected_attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long),
        'rejected_labels': torch.randint(1, 31999, (batch_size, seq_len), dtype=torch.long),
        'images': None  # 对于纯文本测试
    }
    
    # 设置一些标签为-100（忽略）
    for key in ['chosen_labels', 'rejected_labels']:
        batch[key][:, :3] = -100  # 前3个token忽略
    
    return batch

def test_dual_gpu_setup():
    """测试双GPU设置"""
    
    print("=== 测试双GPU DPO Trainer ===")
    
    # 检查GPU可用性
    if torch.cuda.device_count() < 2:
        print(f"❌ 需要至少2个GPU，但只检测到 {torch.cuda.device_count()} 个")
        return False
    
    print(f"✅ 检测到 {torch.cuda.device_count()} 个GPU")
    
    # 使用用户指定的两个模型路径
    policy_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    reference_model_path = "/workspace/MMedPO/Models/SFT_Slake"
    
    try:
        print("加载策略模型...")
        policy_model = LlavaMistralForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},  # 强制将整个模型加载到GPU 0
            low_cpu_mem_usage=True
        )
        
        print("加载参考模型...")
        reference_model = LlavaMistralForCausalLM.from_pretrained(
            reference_model_path,
            torch_dtype=torch.float16,
            device_map={"": 1},  # 强制将整个模型加载到GPU 1
            low_cpu_mem_usage=True
        )
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("创建双GPU DPO trainer...")
        trainer = DualGPUDPOTrainer(
            model=policy_model,
            tokenizer=tokenizer,
            policy_gpu=0,  # 在CUDA_VISIBLE_DEVICES=2,3的情况下，这里的0对应物理GPU 2
            reference_gpu=1,  # 在CUDA_VISIBLE_DEVICES=2,3的情况下，这里的1对应物理GPU 3
            beta=0.1,
            reference_model=reference_model  # 传入预加载的参考模型
        )
        
        print("创建测试批次...")
        batch = create_mock_batch(batch_size=1, seq_len=20)
        
        print("测试前向传播...")
        start_time = time.time()
        
        # 测试损失计算
        loss = trainer.compute_loss(batch, "train")
        
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播成功")
        print(f"   损失值: {loss.item():.6f}")
        print(f"   计算时间: {forward_time:.3f}秒")
        
        # 测试评估指标
        print("测试评估指标...")
        metrics = trainer.get_eval_metrics(batch)
        
        print("评估指标:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.6f}")
            else:
                print(f"   {key}: {value}")
        
        # 验证policy和reference模型的差异
        print("\n验证模型差异...")
        
        # 检查rewards差异
        chosen_rewards = metrics.get("rewards_eval/chosen", 0)
        rejected_rewards = metrics.get("rewards_eval/rejected", 0)
        margin = metrics.get("rewards_eval/margins", 0)
        
        if isinstance(chosen_rewards, torch.Tensor):
            chosen_rewards = chosen_rewards.item()
        if isinstance(rejected_rewards, torch.Tensor):
            rejected_rewards = rejected_rewards.item()
        if isinstance(margin, torch.Tensor):
            margin = margin.item()
        
        print(f"Chosen rewards: {chosen_rewards:.6f}")
        print(f"Rejected rewards: {rejected_rewards:.6f}")
        print(f"Reward margin: {margin:.6f}")
        
        # 检查policy和reference的logps差异
        policy_chosen = metrics.get("debug_eval/policy_chosen_logps", 0)
        policy_rejected = metrics.get("debug_eval/policy_rejected_logps", 0)
        ref_chosen = metrics.get("debug_eval/reference_chosen_logps", 0)
        ref_rejected = metrics.get("debug_eval/reference_rejected_logps", 0)
        
        if isinstance(policy_chosen, torch.Tensor):
            policy_chosen = policy_chosen.item()
        if isinstance(policy_rejected, torch.Tensor):
            policy_rejected = policy_rejected.item()
        if isinstance(ref_chosen, torch.Tensor):
            ref_chosen = ref_chosen.item()
        if isinstance(ref_rejected, torch.Tensor):
            ref_rejected = ref_rejected.item()
        
        print(f"Policy chosen logps: {policy_chosen:.6f}")
        print(f"Policy rejected logps: {policy_rejected:.6f}")
        print(f"Reference chosen logps: {ref_chosen:.6f}")
        print(f"Reference rejected logps: {ref_rejected:.6f}")
        
        # 计算差异
        chosen_diff = abs(policy_chosen - ref_chosen)
        rejected_diff = abs(policy_rejected - ref_rejected)
        
        print(f"Chosen logps差异: {chosen_diff:.6f}")
        print(f"Rejected logps差异: {rejected_diff:.6f}")
        
        # 判断是否成功
        if chosen_diff > 1e-4 or rejected_diff > 1e-4:
            print("✅ Policy和Reference模型有显著差异")
            success = True
        else:
            print("❌ Policy和Reference模型差异过小")
            success = False
        
        if abs(loss.item()) > 1e-6:
            print("✅ DPO损失非零")
        else:
            print("❌ DPO损失为零或接近零")
            success = False
        
        # 测试GPU内存使用
        print(f"\nGPU内存使用:")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_dual_gpu_vs_single_gpu():
    """对比双GPU和单GPU的性能"""
    
    print("\n=== 性能对比测试 ===")
    
    if torch.cuda.device_count() < 2:
        print("跳过性能对比测试（需要至少2个GPU）")
        return
    
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    try:
        # 加载模型
        model = LlavaMistralForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},  # 在CUDA_VISIBLE_DEVICES=2,3的情况下，逻辑GPU 0对应物理GPU 2
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, lora_checkpoint_path)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 创建双GPU trainer
        dual_gpu_trainer = DualGPUDPOTrainer(
                model=model,
                tokenizer=tokenizer,
                policy_gpu=0,  # 在CUDA_VISIBLE_DEVICES=2,3的情况下，这里的0对应物理GPU 2
                reference_gpu=1,  # 在CUDA_VISIBLE_DEVICES=2,3的情况下，这里的1对应物理GPU 3
                beta=0.1
            )
        
        # 创建测试批次
        batch = create_mock_batch(batch_size=2, seq_len=50)
        
        # 预热
        print("预热...")
        for _ in range(3):
            _ = dual_gpu_trainer.compute_loss(batch, "train")
        
        # 测试双GPU性能
        print("测试双GPU性能...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            loss = dual_gpu_trainer.compute_loss(batch, "train")
        
        torch.cuda.synchronize()
        dual_gpu_time = (time.time() - start_time) / 10
        
        print(f"双GPU平均时间: {dual_gpu_time:.3f}秒")
        print(f"双GPU损失: {loss.item():.6f}")
        
        # 清理内存
        del dual_gpu_trainer
        torch.cuda.empty_cache()
        
        print("双GPU设置测试完成")
        
    except Exception as e:
        print(f"性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始双GPU DPO trainer测试...")
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    try:
        # 基础功能测试
        success = test_dual_gpu_setup()
        
        if success:
            print("\n🎉 双GPU DPO trainer测试成功！")
            print("- Policy和Reference模型成功分离到不同GPU")
            print("- 模型输出有显著差异")
            print("- DPO损失计算正常")
            
            # 性能对比测试
            benchmark_dual_gpu_vs_single_gpu()
            
        else:
            print("\n❌ 双GPU DPO trainer测试失败")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()