import torch
import sys
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
from peft import PeftModel

def test_disable_adapter_detailed():
    print("Testing disable_adapter behavior in detail...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    
    # 加载一个小模型进行测试
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 添加 LoRA 适配器
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    print(f"Model type: {type(peft_model)}")
    print(f"Active adapter: {peft_model.active_adapter}")
    print(f"PEFT config: {peft_model.peft_config}")
    
    # 检查模型参数
    print("\nModel parameters before training:")
    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    # 训练一些步骤来确保 LoRA 参数有变化
    print("\nTraining model to modify LoRA parameters...")
    peft_model.train()
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-3)
    
    input_text = "Hello world"
    inputs = tokenizer(input_text, return_tensors="pt")
    target = inputs['input_ids'].clone()
    
    # 训练几步
    for step in range(5):
        optimizer.zero_grad()
        outputs = peft_model(**inputs, labels=target)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")
    
    peft_model.eval()
    
    # 现在测试 disable_adapter 的行为
    print("\nTesting disable_adapter after training...")
    
    with torch.no_grad():
        # 正常模式下的输出
        normal_output = peft_model(**inputs)
        normal_logits = normal_output.logits
        
        # 禁用适配器模式下的输出
        with peft_model.disable_adapter():
            disabled_output = peft_model(**inputs)
            disabled_logits = disabled_output.logits
    
    # 比较输出
    logits_diff = torch.abs(normal_logits - disabled_logits).max().item()
    mean_diff = torch.abs(normal_logits - disabled_logits).mean().item()
    
    print(f"Max logits difference: {logits_diff}")
    print(f"Mean logits difference: {mean_diff}")
    print(f"Are logits different: {logits_diff > 1e-6}")
    
    # 检查 base_model 的行为
    print(f"\nBase model type: {type(peft_model.base_model)}")
    print(f"Has disable_adapter_layers: {hasattr(peft_model.base_model, 'disable_adapter_layers')}")
    print(f"Has enable_adapter_layers: {hasattr(peft_model.base_model, 'enable_adapter_layers')}")
    
    # 检查 PEFT 配置
    active_config = peft_model.peft_config[peft_model.active_adapter]
    print(f"Is prompt learning: {active_config.is_prompt_learning}")
    
    return logits_diff > 1e-6

if __name__ == "__main__":
    test_disable_adapter_detailed()
