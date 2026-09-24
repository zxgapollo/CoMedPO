#!/usr/bin/env python3

import argparse
import torch
import os
import sys

# Add the project root to Python path
sys.path.append('/workspace/MMedPO')
sys.path.append('/workspace/MMedPO/train/dpo')

# Add LLaVA path
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def test_checkpoint_loading():
    print("Testing checkpoint loading...")
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model paths
    model_path = "/workspace/MMedPO/mmedpo_checkpoints/llava-med-1.5_slake_mmedpo"
    model_base = "/workspace/llava-med-v1.5-mistral-7b"
    
    try:
        print("Loading model...")
        model_name = "llava-med-v1.5-mistral-7b"  # Use the base model name
        print(f"Model name: {model_name}")
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )
        
        print("✅ Checkpoint loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Context length: {context_len}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # Test a simple forward pass
        print("Testing model inference...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **test_input,
                max_new_tokens=10,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model inference test passed!")
        print(f"Test response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)