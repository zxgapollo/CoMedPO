#!/usr/bin/env python3
"""
LLaVA PEFT Model Merger using LLaVA's native model loading
This script merges LoRA checkpoints with base models using LLaVA's model loading infrastructure.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Disable quantization before importing any transformers modules
os.environ["DISABLE_QUANTIZATION"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# Add LLaVA path to sys.path
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

try:
    from transformers import AutoTokenizer, LlamaTokenizer
    from peft import PeftModel
    from llava.model.builder import load_pretrained_model
    print("Successfully imported transformers and PEFT")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install transformers peft")
    sys.exit(1)


def merge_lora_with_base_model(base_model_path, lora_checkpoint_path, output_path, device_map="auto"):
    """
    Merge LoRA checkpoint with base model using LLaVA's native model loading
    """
    print(f"🔹 Base model path: {base_model_path}")
    print(f"🔹 LoRA checkpoint path: {lora_checkpoint_path}")
    print(f"🔹 Output path: {output_path}")
    print(f"🔹 Device map: {device_map}")
    
    # Ensure we can access model configs
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Step 1: Detect model type from base model config
        import json
        base_config_path = os.path.join(base_model_path, "config.json")
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
        
        model_architecture = base_config.get("architectures", [""])[0]
        print(f"🔹 Detected model architecture: {model_architecture}")
        
        # Step 2: Load model using appropriate method based on architecture
        if "Mistral" in model_architecture:
            print("🔹 Loading Mistral-based LLaVA model with LoRA...")
            # For Mistral models, we need to use a custom approach since the builder has a bug
            from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
            
            # Load LoRA config
            lora_config = LlavaMistralConfig.from_pretrained(lora_checkpoint_path)
            tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
            
            # Load base model with LoRA config (disable quantization for merging)
            model = LlavaMistralForCausalLM.from_pretrained(
                base_model_path, 
                low_cpu_mem_usage=True, 
                config=lora_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                load_in_8bit=False,
                load_in_4bit=False,
                quantization_config=None
            )
            
            # Check if model is quantized and reload if necessary
            if hasattr(model, 'is_quantized') and model.is_quantized:
                print("🔹 Model is quantized, reloading without quantization...")
                # Reload the model without quantization
                model = LlavaMistralForCausalLM.from_pretrained(
                    base_model_path,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    load_in_8bit=False,
                    load_in_4bit=False,
                    quantization_config=None
                )
            
            # Load non-LoRA trainables
            non_lora_path = os.path.join(lora_checkpoint_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                print("🔹 Loading non-LoRA trainables...")
                non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)
            
            # Load and merge LoRA weights
            print("🔹 Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, lora_checkpoint_path)
            print("🔹 Merging LoRA weights...")
            model = model.merge_and_unload()
            
        else:
            print("🔹 Loading Llama-based LLaVA model with LoRA...")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=lora_checkpoint_path,
                model_base=base_model_path,
                model_name="llava_llama",
                device_map=device_map
            )
        
        print(f"🔹 LLaVA model loaded: {type(model)}")
        print("✅ LoRA weights merged successfully")
        
        # Step 4: Save the merged model
        print(f"🔹 Saving merged model to: {output_path}")
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print("✅ Model merge completed successfully!")
        
        # Step 5: Generate README
        readme_content = f"""# LLaVA Merged Model

This model was created by merging a LoRA checkpoint with a base model using transformers and PEFT.
All components are loaded and re-exported through the model loading mechanism.

## Model Details
- Base Model: {base_model_path}
- LoRA Checkpoint: {lora_checkpoint_path}
- Merge Method: PEFT merge_and_unload
- Output Path: {output_path}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model
model = AutoModelForCausalLM.from_pretrained(
    "{output_path}",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("{output_path}")
```

## Files Included
- Model weights (safetensors format)
- Tokenizer files
- Configuration files (auto-generated during save)
"""
        
        readme_path = os.path.join(output_path, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"🎉 Merge completed successfully!")
        print(f"🎯 Merged model saved to: {output_path}")
        
        # Verify output
        output_files = list(Path(output_path).glob('*'))
        print(f"📁 Output contains {len(output_files)} files:")
        for file_path in sorted(output_files):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  📄 {file_path.name}: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint with base model using LLaVA infrastructure")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--lora_checkpoint_path", type=str, required=True,
                        help="Path to the LoRA checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the merged model")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device mapping for model loading")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.base_model_path):
        print(f"Error: Base model path does not exist: {args.base_model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.lora_checkpoint_path):
        print(f"Error: LoRA checkpoint path does not exist: {args.lora_checkpoint_path}")
        sys.exit(1)
    
    # Perform merge
    success = merge_lora_with_base_model(
        base_model_path=args.base_model_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        output_path=args.output_path,
        device_map=args.device_map
    )
    
    if success:
        print("Merge completed successfully!")
        sys.exit(0)
    else:
        print("Merge failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()