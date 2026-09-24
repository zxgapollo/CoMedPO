#!/usr/bin/env python3
import os
import sys
sys.path.append('/workspace/MMedPO/MMedPO')
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

from train_dpo_weighted import ModelArguments, DataArguments, TrainingArguments
from llava.model import LlavaLlamaForCausalLM
from peft import PeftModel
import torch

def main():
    print("=== Model Loading Debug (Simple) ===")
    
    # Simulate the training arguments from the shell script
    model_args = ModelArguments(
        model_name_or_path="/workspace/MMedPO/checkpoints/sft_dpo_new_pair_vqa_rad",
        version="v1",
        vision_tower="openai/clip-vit-large-patch14-336",
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_patch_merge_type="flat",
        lora_checkpoint_path="/workspace/MMedPO/checkpoints/sft_dpo_new_pair_vqa_rad"
    )
    
    model_path = model_args.model_name_or_path
    lora_checkpoint_path = model_args.lora_checkpoint_path
    
    print(f"Model path: {model_path}")
    print(f"LoRA checkpoint path: {lora_checkpoint_path}")
    print(f"LoRA enabled: True")
    print(f"LoRA r: 128")
    print(f"LoRA alpha: 256")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return
    
    print("\nFiles in model directory:")
    for f in os.listdir(model_path):
        print(f"  {f}")
    
    lora_files = [f for f in os.listdir(model_path) if 'lora' in f.lower() or 'adapter' in f.lower()]
    print(f"\nLoRA-related files: {lora_files}")
    
    try:
        # Load base model
        print("\n=== Loading Base Model ===")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"Base model type: {type(model)}")
        print(f"Is PeftModel: {isinstance(model, PeftModel)}")
        
        # Check if model has LoRA adapters
        if hasattr(model, 'peft_config'):
            print(f"PEFT config keys: {list(model.peft_config.keys())}")
            print(f"PEFT type: {type(model.peft_config['default'])}")
        
        # Test disable_adapter functionality
        print("\n=== Testing disable_adapter ===")
        if hasattr(model, 'disable_adapter'):
            print("Model has disable_adapter method")
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (1, 10)).to(model.device)
            
            # Get output with adapter enabled
            with torch.no_grad():
                output_with_adapter = model(dummy_input).logits
            
            # Get output with adapter disabled
            with torch.no_grad():
                with model.disable_adapter():
                    output_without_adapter = model(dummy_input).logits
            
            # Compare outputs
            diff = torch.abs(output_with_adapter - output_without_adapter)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"Max logits difference: {max_diff:.3f}")
            print(f"Mean logits difference: {mean_diff:.3f}")
            print(f"Are logits different: {max_diff > 1e-6}")
            
        else:
            print("Model does NOT have disable_adapter method")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()