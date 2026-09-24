"""Offline integration test using a real tiny LLaVA-Mistral and CLIP tower."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MMedPO/train/dpo"))

from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, PreTrainedTokenizerFast
from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
from MMedPO.comedpo.data import CoMedPODataset
from MMedPO.comedpo.model import CoMedPOPolicy, LlavaCollator


def tiny_checkpoint(tmp_path):
    vision_path = tmp_path / "vision"
    vision = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, image_size=8, patch_size=4,
    ))
    vision.save_pretrained(vision_path)
    processor = CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
    processor.save_pretrained(vision_path)
    config = LlavaMistralConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
        mm_vision_tower=str(vision_path), mm_hidden_size=16,
        mm_vision_select_layer=-2, mm_projector_type="linear", mm_vision_select_feature="patch",
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    base = LlavaMistralForCausalLM(config)
    model_path = tmp_path / "model"
    base.save_pretrained(model_path)
    tokenizer_impl = Tokenizer(WordLevel(
        {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "Yes": 4, "No": 5,
         "Is": 6, "there": 7, "a": 8, "lesion": 9, "?": 10}, unk_token="[UNK]",
    ))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl,
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(model_path)
    Image.new("RGB", (8, 8), (100, 150, 200)).save(tmp_path / "image.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 3:5] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    data_path = tmp_path / "pairs.json"
    data_path.write_text(json.dumps([dict(image="image.png", lesion_mask="mask.png",
        question="Is there a lesion?", chosen="Yes", rejected="No")]))
    return model_path, data_path, tokenizer, processor


def test_real_llava_forward_reference_and_update(tmp_path):
    torch.manual_seed(7)
    model_path, data_path, tokenizer, processor = tiny_checkpoint(tmp_path)
    base = LlavaMistralForCausalLM.from_pretrained(model_path)
    base.get_vision_tower().load_model()
    base.config.tokenizer_model_max_length = None
    policy = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model = CoMedPOPolicy(policy, max_length=512)
    batch = LlavaCollator(tokenizer, processor, max_length=512)([CoMedPODataset(data_path)[0]])
    frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    trainable = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    result = model(batch)
    assert result["loss"].item() == pytest.approx(2 * np.log(2), rel=1e-5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    result["loss"].backward()
    optimizer.step()
    assert any(not torch.equal(p, trainable[k]) for k, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, frozen[k]) for k, p in model.named_parameters() if not p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert torch.isfinite(model(batch)["loss"])


def test_training_cli_checkpoint_and_resume(tmp_path):
    model_path, data_path, _, _ = tiny_checkpoint(tmp_path)
    output = tmp_path / "output"
    command = [sys.executable, "-m", "MMedPO.comedpo.train", "--model-path", str(model_path),
        "--data-path", str(data_path), "--output-dir", str(output), "--mixed-precision", "no",
        "--epochs", "1", "--batch-size", "1", "--gradient-accumulation-steps", "2",
        "--lora-rank", "2", "--lora-alpha", "4", "--max-length", "512"]
    import os
    env = dict(os.environ, ACCELERATE_USE_CPU="true", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    run = subprocess.run(command, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    progress = json.loads((output / "checkpoint-last/progress.json").read_text())
    assert progress["step"] == 1 and progress["completed_epochs"] == 1
    assert (output / "adapter/adapter_model.safetensors").is_file()
    resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, env=env)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
