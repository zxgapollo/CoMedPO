#!/usr/bin/env python3
"""
调试脚本：准确模拟训练环境，检查模型是否正确包装为 PeftModel
"""

import sys
import os
import torch
from transformers import BitsAndBytesConfig

# 添加必要的路径
sys.path.append('/workspace/MMedPO/MMedPO')
sys.path.append('/workspace/MMedPO/MMedPO/train')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

# 模拟训练参数
class MockArgs:
    def __init__(self):
        # 模型路径
        self.model_name_or_path = "/workspace/llava-med-v1.5-mistral-7b"
        self.lora_checkpoint_path = "/workspace/MMedPO/checkpoints/sft_dpo_new_pair_vqa_rad"
        
        # LoRA 参数
        self.lora_enable = True
        self.lora_r = 128
        self.lora_alpha = 256
        self.lora_dropout = 0.05
        self.lora_weight_path = ""
        self.lora_bias = "none"
        
        # 量化参数
        self.bits = 4
        self.double_quant = True
        self.quant_type = "nf4"
        
        # 其他参数
        self.bf16 = True
        self.tf32 = True
        self.model_max_length = 2048
        self.version = "v1"
        self.vision_tower = "openai/clip-vit-large-patch14-336"
        self.mm_projector_type = "mlp2x_gelu"
        self.mm_vision_select_layer = -2
        self.mm_use_im_start_end = False
        self.mm_use_im_patch_token = False
        self.image_aspect_ratio = "pad"

def main():
    print("=== 调试训练环境中的模型加载 ===")
    
    # 创建模拟参数
    model_args = MockArgs()
    
    print(f"基础模型路径: {model_args.model_name_or_path}")
    print(f"LoRA 检查点路径: {model_args.lora_checkpoint_path}")
    print(f"LoRA 启用: {model_args.lora_enable}")
    print(f"LoRA 参数: r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    print(f"量化: {model_args.bits}位, 类型={model_args.quant_type}")
    
    # 检查路径是否存在
    if not os.path.exists(model_args.model_name_or_path):
        print(f"错误: 基础模型路径不存在: {model_args.model_name_or_path}")
        return
    
    if not os.path.exists(model_args.lora_checkpoint_path):
        print(f"错误: LoRA 检查点路径不存在: {model_args.lora_checkpoint_path}")
        return
    
    print("\n=== 模拟训练脚本中的模型加载过程 ===")
    
    try:
        # 1. 设置量化配置（模拟训练脚本中的量化设置）
        if model_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args = {}
            if model_args.bits == 4:
                bnb_model_from_pretrained_args.update(dict(
                    device_map={"": 0},
                    load_in_4bit=True,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=model_args.double_quant,
                        bnb_4bit_quant_type=model_args.quant_type
                    )
                ))
            print("量化配置已设置")
        else:
            bnb_model_from_pretrained_args = {}
        
        # 2. 加载基础模型（模拟训练脚本中的加载过程）
        from llava.model import LlavaMistralForCausalLM
        
        print("正在加载基础模型...")
        model = LlavaMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=None,
            **bnb_model_from_pretrained_args
        )
        print(f"基础模型类型: {type(model)}")
        
        # 3. 准备模型进行 k-bit 训练（如果使用量化）
        if model_args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            print("模型已准备进行 k-bit 训练")
        
        # 4. 设置 LoRA 配置
        if model_args.lora_enable:
            from peft import LoraConfig, get_peft_model, PeftModel
            
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            print(f"LoRA 配置: {lora_config}")
            
            # 5. 应用 LoRA（模拟训练脚本中的过程）
            if model_args.lora_checkpoint_path:
                print("从检查点加载 LoRA 权重...")
                
                # 加载 non_lora_trainables.bin
                non_lora_trainables_path = os.path.join(model_args.lora_checkpoint_path, 'non_lora_trainables.bin')
                if os.path.exists(non_lora_trainables_path):
                    non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
                    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                    if any(k.startswith('model.model.') for k in non_lora_trainables):
                        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                    # 注释掉加载 non_lora_trainables，因为会导致形状不匹配
                    # model.load_state_dict(non_lora_trainables, strict=False)
                    print("跳过加载 non_lora_trainables.bin（避免形状不匹配）")
                
                # 使用 PeftModel.from_pretrained 加载 LoRA 权重
                model = PeftModel.from_pretrained(model, model_args.lora_checkpoint_path)
                print("已使用 PeftModel.from_pretrained 加载 LoRA 权重")
            else:
                # 如果没有检查点，创建新的 PEFT 模型
                model = get_peft_model(model, lora_config)
                print("已创建新的 PEFT 模型")
        
        # 6. 检查最终模型状态
        print(f"\n=== 最终模型状态 ===")
        print(f"模型类型: {type(model)}")
        print(f"是否为 PeftModel: {hasattr(model, 'peft_config')}")
        print(f"是否有 disable_adapter 方法: {hasattr(model, 'disable_adapter')}")
        
        if hasattr(model, 'peft_config'):
            print(f"PEFT 配置键: {list(model.peft_config.keys())}")
            for key, config in model.peft_config.items():
                print(f"  {key}: {config}")
        
        # 7. 测试 disable_adapter 功能
        if hasattr(model, 'disable_adapter'):
            print(f"\n=== 测试 disable_adapter 功能 ===")
            
            # 创建一个简单的输入
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
            
            # 正常模式
            model.train()
            with torch.no_grad():
                output_normal = model(input_ids=input_ids)
                logits_normal = output_normal.logits
            
            # 禁用适配器模式
            model.disable_adapter()
            with torch.no_grad():
                output_disabled = model(input_ids=input_ids)
                logits_disabled = output_disabled.logits
            
            # 重新启用适配器（使用正确的方法名）
            if hasattr(model, 'enable_adapter'):
                model.enable_adapter()
            elif hasattr(model, 'enable_adapters'):
                model.enable_adapters()
            
            # 比较输出
            diff = torch.abs(logits_normal - logits_disabled).max().item()
            print(f"正常模式与禁用适配器模式的最大差异: {diff}")
            
            if diff > 1e-6:
                print("✓ disable_adapter 功能正常工作")
            else:
                print("✗ disable_adapter 功能可能有问题")
        else:
            print("✗ 模型没有 disable_adapter 方法")
        
        print(f"\n=== 调试完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()