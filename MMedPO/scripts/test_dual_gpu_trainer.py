#!/usr/bin/env python3

import os
import sys
import torch
import transformers
from transformers import TrainingArguments

# Add the DPO training directory to Python path
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')

from dual_gpu_dpo_trainer import DualGPUDPOTrainer

def test_trainer_initialization():
    """Test basic DualGPUDPOTrainer initialization"""
    print("Testing DualGPUDPOTrainer initialization...")
    
    # Create minimal training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/test_output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        remove_unused_columns=False,
    )
    
    # Create a dummy model (just for testing initialization)
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_config(config)
    
    # Create a dummy tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy datasets
    class DummyDataset:
        def __init__(self, size=10):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return {
                'chosen_input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'chosen_labels': torch.tensor([1, 2, 3, 4, 5]),
                'rejected_input_ids': torch.tensor([1, 2, 3, 6, 7]),
                'rejected_labels': torch.tensor([1, 2, 3, 6, 7]),
            }
    
    train_dataset = DummyDataset()
    
    # Create dummy data collator
    def dummy_data_collator(features):
        return {
            'chosen_input_ids': torch.stack([f['chosen_input_ids'] for f in features]),
            'chosen_labels': torch.stack([f['chosen_labels'] for f in features]),
            'rejected_input_ids': torch.stack([f['rejected_input_ids'] for f in features]),
            'rejected_labels': torch.stack([f['rejected_labels'] for f in features]),
        }
    
    try:
        # Test trainer initialization
        trainer = DualGPUDPOTrainer(
            model=model,
            args=training_args,
            data_collator=dummy_data_collator,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            policy_gpu=0,
            reference_gpu=0,  # Use same GPU for testing
            beta=0.1,
            loss_type='sigmoid',
            loss_variant='dpo',
            sppo_eta=0.0,
            sppo_lambda=0.0,
            sppo_alpha=1.0,
            reference_free=False,
            reference_model=None,
        )
        
        print("✓ DualGPUDPOTrainer initialization successful!")
        
        # Test compute_loss method
        dummy_inputs = {
            'chosen_input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'chosen_labels': torch.tensor([[1, 2, 3, 4, 5]]),
            'rejected_input_ids': torch.tensor([[1, 2, 3, 6, 7]]),
            'rejected_labels': torch.tensor([[1, 2, 3, 6, 7]]),
        }
        
        loss = trainer.compute_loss(model, dummy_inputs)
        print(f"✓ compute_loss method works! Loss: {loss}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during trainer initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trainer_initialization()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
