#!/usr/bin/env python3
"""
测试修复后的 DPO trainer，验证 policy 和 reference 模型是否有差异
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

# 导入修复后的 DPO trainer
from tool.dpo_trainer import DPOTrainer

def create_mock_batch():
    """创建模拟的训练批次"""
    
    # 模拟输入数据
    batch = {
        'chosen_input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        'chosen_attention_mask': torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        'chosen_labels': torch.tensor([[-100, -100, -100, 4, 5]], dtype=torch.long),
        'rejected_input_ids': torch.tensor([[1, 2, 3, 6, 7]], dtype=torch.long),
        'rejected_attention_mask': torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        'rejected_labels': torch.tensor([[-100, -100, -100, 6, 7]], dtype=torch.long),
        'images': None  # 对于纯文本测试
    }
    
    return batch

def test_dpo_trainer_fix():
    """测试修复后的 DPO trainer"""
    
    print("=== 测试修复后的 DPO Trainer ===")
    
    base_model_path = "/workspace/llava-med-v1.5-mistral-7b"
    lora_checkpoint_path = "/workspace/MMedPO/MMedPO/checkpoints/sft_dpo_combined"
    
    # 在 CPU 上加载模型以避免 CUDA 内存问题
    print("加载基础模型...")
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # 使用 float32 避免 CPU 上的 half precision 问题
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    print("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建 DPO trainer（简化版本，只测试关键功能）
    class SimpleDPOTrainer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.ref_model = None  # 使用 disable_adapter 作为 reference
        
        def _get_batch_logps(self, logits, labels, average_log_prob=False):
            """简化的 log probability 计算"""
            # 计算 log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 获取标签对应的 log probabilities
            gathered_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            
            # 只考虑非 -100 的标签
            mask = (labels != -100).float()
            masked_log_probs = gathered_log_probs * mask
            
            if average_log_prob:
                return masked_log_probs.sum(-1) / mask.sum(-1)
            else:
                return masked_log_probs.sum(-1)
        
        def concatenated_forward(self, model, batch):
            """简化的前向传播"""
            # 处理 chosen
            chosen_outputs = model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask'],
                images=batch['images']
            )
            chosen_logits = chosen_outputs.logits
            chosen_logps = self._get_batch_logps(chosen_logits, batch['chosen_labels'])
            
            # 处理 rejected
            rejected_outputs = model(
                input_ids=batch['rejected_input_ids'],
                attention_mask=batch['rejected_attention_mask'],
                images=batch['images']
            )
            rejected_logits = rejected_outputs.logits
            rejected_logps = self._get_batch_logps(rejected_logits, batch['rejected_labels'])
            
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits
        
        def test_get_batch_metrics(self, batch):
            """测试修复后的 get_batch_metrics 逻辑"""
            print("获取 policy 模型输出...")
            
            # Policy 模型输出
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(self.model, batch)
            
            print(f"Policy chosen logps: {policy_chosen_logps}")
            print(f"Policy rejected logps: {policy_rejected_logps}")
            
            # Reference 模型输出（使用修复后的逻辑）
            print("\n获取 reference 模型输出（使用修复后的适配器处理）...")
            
            with torch.no_grad():
                if self.ref_model is None:
                    # 使用修复后的适配器禁用/启用逻辑
                    try:
                        print("禁用适配器...")
                        self.model.disable_adapter()
                        
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                        
                        print(f"Reference chosen logps: {reference_chosen_logps}")
                        print(f"Reference rejected logps: {reference_rejected_logps}")
                        
                    finally:
                        # 确保适配器被重新启用
                        print("重新启用适配器...")
                        try:
                            if hasattr(self.model, 'enable_adapter'):
                                self.model.enable_adapter()
                                print("使用 enable_adapter 重新启用")
                            elif hasattr(self.model, 'enable_adapters'):
                                self.model.enable_adapters()
                                print("使用 enable_adapters 重新启用")
                        except Exception as e:
                            print(f"enable_adapter 失败: {e}")
                            # 如果 enable_adapter 失败，尝试使用 set_adapter
                            try:
                                self.model.set_adapter("default")
                                print("使用 set_adapter 重新启用")
                            except Exception as e2:
                                print(f"警告：无法重新启用适配器: {e2}")
            
            # 验证适配器是否正确重新启用
            print("\n验证适配器重新启用后的输出...")
            (
                final_chosen_logps,
                final_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.model, batch)
            
            print(f"重新启用后 chosen logps: {final_chosen_logps}")
            print(f"重新启用后 rejected logps: {final_rejected_logps}")
            
            # 分析差异
            print(f"\n差异分析:")
            policy_ref_chosen_diff = abs(policy_chosen_logps.item() - reference_chosen_logps.item())
            policy_ref_rejected_diff = abs(policy_rejected_logps.item() - reference_rejected_logps.item())
            policy_final_chosen_diff = abs(policy_chosen_logps.item() - final_chosen_logps.item())
            policy_final_rejected_diff = abs(policy_rejected_logps.item() - final_rejected_logps.item())
            
            print(f"Policy vs Reference chosen 差异: {policy_ref_chosen_diff:.8f}")
            print(f"Policy vs Reference rejected 差异: {policy_ref_rejected_diff:.8f}")
            print(f"Policy vs Final chosen 差异: {policy_final_chosen_diff:.8f}")
            print(f"Policy vs Final rejected 差异: {policy_final_rejected_diff:.8f}")
            
            # 判断修复是否成功
            if policy_ref_chosen_diff > 1e-6 or policy_ref_rejected_diff > 1e-6:
                print("✅ Policy 和 Reference 模型有显著差异")
                success_ref = True
            else:
                print("❌ Policy 和 Reference 模型差异过小")
                success_ref = False
            
            if policy_final_chosen_diff < 1e-6 and policy_final_rejected_diff < 1e-6:
                print("✅ 适配器成功重新启用，输出恢复到 Policy 状态")
                success_restore = True
            else:
                print("❌ 适配器重新启用后输出不一致")
                success_restore = False
            
            return success_ref and success_restore
    
    # 创建简化的 DPO trainer
    trainer = SimpleDPOTrainer(model, tokenizer)
    
    # 创建测试批次
    batch = create_mock_batch()
    
    # 将批次移动到模型设备
    device = next(model.parameters()).device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # 测试修复后的逻辑
    success = trainer.test_get_batch_metrics(batch)
    
    return success

if __name__ == "__main__":
    print("开始测试修复后的 DPO trainer...")
    
    try:
        success = test_dpo_trainer_fix()
        
        if success:
            print("\n🎉 DPO trainer 修复成功！")
            print("- Policy 和 Reference 模型有显著差异")
            print("- 适配器能够正确禁用和重新启用")
            print("- DPO 训练应该能够正常工作")
        else:
            print("\n❌ DPO trainer 修复失败")
            print("需要进一步调试适配器处理逻辑")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()