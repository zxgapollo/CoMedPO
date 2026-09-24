#!/usr/bin/env python3
"""
Minimal test to verify log likelihood calculation works.
"""
import torch
import torch.nn.functional as F

def test_log_likelihood():
    """Test log likelihood calculation with minimal example."""
    
    # Create dummy data
    batch_size = 1
    vocab_size = 10
    input_length = 5
    generated_length = 3
    
    # Create dummy scores (one for each generation step)
    scores = []
    for i in range(generated_length):
        # Create random logits
        logits = torch.randn(batch_size, vocab_size)
        scores.append(logits)
    
    # Create dummy generated tokens (input + generated)
    total_length = input_length + generated_length
    generated_tokens = torch.randint(0, vocab_size, (batch_size, total_length))
    
    print("Testing log likelihood calculation...")
    print(f"Input length: {input_length}")
    print(f"Generated length: {generated_length}")
    print(f"Total length: {total_length}")
    print(f"Number of scores: {len(scores)}")
    print(f"Generated tokens: {generated_tokens[0].tolist()}")
    
    # Calculate log likelihood
    def get_log_likelihood_from_scores(scores, generated_tokens, input_length):
        if scores is None or len(scores) == 0:
            return 0.0
        
        total_log_likelihood = 0.0
        generated_length = generated_tokens.shape[1] - input_length
        
        if generated_length <= 0:
            return 0.0
        
        for i in range(generated_length):
            if i < len(scores):
                # Get the log probabilities for this generation step
                log_probs = torch.log_softmax(scores[i], dim=-1)
                # Get the actual token that was generated
                actual_token = generated_tokens[0, input_length + i].item()
                # Get the log probability of that specific token
                token_log_prob = log_probs[0, actual_token].item()
                total_log_likelihood += token_log_prob
                print(f"  Step {i}: Token {actual_token}, Log prob: {token_log_prob:.4f}")
        
        return total_log_likelihood
    
    log_likelihood = get_log_likelihood_from_scores(scores, generated_tokens, input_length)
    print(f"Total log likelihood: {log_likelihood:.4f}")
    
    # Verify it's not zero
    if log_likelihood != 0.0:
        print("✅ Log likelihood calculation is working correctly!")
        return True
    else:
        print("❌ Log likelihood calculation returned 0.0")
        return False

if __name__ == "__main__":
    success = test_log_likelihood()
    exit(0 if success else 1)
