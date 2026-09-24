#!/usr/bin/env python3
"""
Debug script to check the model type and configuration in the training script
"""
import os
import sys
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional

# Add the training script path
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

# Import the training script components
from train_dpo_weighted import ModelArguments, TrainingArguments, DataArguments

def debug_model_loading():
    """Debug the model loading process to understand the issue"""
    
    # Simulate the training arguments from the shell script
    model_args = ModelArguments(
        model_name_or_path="/workspace/MMedPO/checkpoints/sft_dpo_new_pair_vqa_rad",
        version="v1",
        vision_tower="openai/clip-vit-large-patch14-336",
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_patch_merge_type="flat",
        lora_checkpoint_path="/workspace/MMedPO/checkpoints/sft_dpo_new_pair_vqa_rad"
    )
    
    training_args = TrainingArguments(
        output_dir="/workspace/MMedPO/checkpoints/llava-v1.5-7b-lora-dpo-tie-sppo",
        lora_enable=True,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        bits=16,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="steps",
        save_steps=50000,
        save_total_limit=1,
        evaluation_strategy="no",
        dataloader_num_workers=4,
        group_by_modality_length=True,
        beta=0.1,
        loss_type="sigmoid",
        loss_use_weight=True,
        remove_unused_columns=False,
        model_max_length=2048,
        mm_projector_lr=None,
        freeze_mm_mlp_adapter=False
    )
    
    print("=== Model Loading Debug ===")
    print(f"Model path: {model_args.model_name_or_path}")
    print(f"LoRA checkpoint path: {model_args.lora_checkpoint_path}")
    print(f"LoRA enabled: {training_args.lora_enable}")
    print(f"LoRA r: {training_args.lora_r}")
    print(f"LoRA alpha: {training_args.lora_alpha}")
    
    # Check if the model path exists
    if not os.path.exists(model_args.model_name_or_path):
        print(f"ERROR: Model path does not exist: {model_args.model_name_or_path}")
        return
    
    # Check if LoRA checkpoint exists
    if model_args.lora_checkpoint_path and not os.path.exists(model_args.lora_checkpoint_path):
        print(f"ERROR: LoRA checkpoint path does not exist: {model_args.lora_checkpoint_path}")
        return
    
    # List files in the model directory
    print(f"\nFiles in model directory:")
    for file in os.listdir(model_args.model_name_or_path):
        print(f"  {file}")
    
    # Check for LoRA-related files
    lora_files = [f for f in os.listdir(model_args.model_name_or_path) if 'lora' in f.lower() or 'adapter' in f.lower()]
    print(f"\nLoRA-related files: {lora_files}")
    
    # Check for PEFT configuration
    peft_config_path = os.path.join(model_args.model_name_or_path, 'adapter_config.json')
    if os.path.exists(peft_config_path):
        import json
        with open(peft_config_path, 'r') as f:
            peft_config = json.load(f)
        print(f"\nPEFT config found: {peft_config}")
    else:
        print(f"\nNo PEFT config found at: {peft_config_path}")
    
    # Try to load the model (simplified version)
    try:
        print("\n=== Attempting to load model ===")
        
        # Import required modules
        from llava.model import LlavaLlamaForCausalLM
        from peft import PeftModel, LoraConfig, get_peft_model
        
        # Load base model
        print("Loading base model...")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16
        )
        
        print(f"Base model type: {type(model)}")
        print(f"Base model has disable_adapter: {hasattr(model, 'disable_adapter')}")
        
        # Check if this is already a PEFT model
        if hasattr(model, 'peft_config'):
            print(f"Model is already a PEFT model: {type(model)}")
            print(f"PEFT config: {model.peft_config}")
        else:
            print("Model is not a PEFT model")
        
        # Check if we need to load LoRA from checkpoint
        if model_args.lora_checkpoint_path and os.path.exists(model_args.lora_checkpoint_path):
            print(f"Loading LoRA from checkpoint: {model_args.lora_checkpoint_path}")
            
            # Check if it's already a PeftModel
            if not hasattr(model, 'peft_config'):
                model = PeftModel.from_pretrained(model, model_args.lora_checkpoint_path)
                print(f"Loaded PEFT model type: {type(model)}")
            else:
                print("Model already has PEFT configuration")
        
        print(f"Final model type: {type(model)}")
        print(f"Final model has disable_adapter: {hasattr(model, 'disable_adapter')}")
        
        # Check if it's a PeftModel
        from peft import PeftModel
        print(f"Is PeftModel: {isinstance(model, PeftModel)}")
        
        if hasattr(model, 'peft_config'):
            print(f"Active adapter: {getattr(model, 'active_adapter', 'N/A')}")
            print(f"PEFT config keys: {list(model.peft_config.keys())}")
            
            # Check the configuration
            if hasattr(model, 'active_adapter') and model.active_adapter in model.peft_config:
                config = model.peft_config[model.active_adapter]
                print(f"Is prompt learning: {getattr(config, 'is_prompt_learning', 'N/A')}")
        
        # Test disable_adapter if available
        if hasattr(model, 'disable_adapter'):
            print("\nTesting disable_adapter...")
            try:
                with model.disable_adapter():
                    print("disable_adapter works!")
            except Exception as e:
                print(f"disable_adapter failed: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_loading()