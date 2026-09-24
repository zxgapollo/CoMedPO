import os
import pathlib
import transformers
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict

# Reuse existing trainer and dataset utilities
from tool.dual_gpu_dpo_trainer import DualGPUDPOTrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.utils import disable_torch_init

from train_dpo_weighted import (
    ModelArguments as BaseModelArguments,
    DataArguments as BaseDataArguments,
    TrainingArguments as BaseTrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_supervised_data_module,
    maybe_zero_3,
    find_all_linear_names,
)


@dataclass
class ModelArguments(BaseModelArguments):
    reference_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reference model (frozen)."})


@dataclass
class DataArguments(BaseDataArguments):
    pass


@dataclass
class TrainingArguments(BaseTrainingArguments):
    # SPPO extras
    loss_variant: str = field(default="sppo", metadata={"help": "SPPO loss variant: sppo | sppo_adv_squared | tie_sppo | tie_sppo_dynamic"})
    sppo_eta: float = field(default=1.0, metadata={"help": "SPPO shaping strength η"})
    sppo_lambda: float = field(default=0.75, metadata={"help": "SPPO-ADV positive reinforcement weight λ"})
    sppo_alpha: float = field(default=0.5, metadata={"help": "TIE-SPPO exponent/weight α for fixed or dynamic weight"})
    # Dual-GPU mapping (relative to CUDA_VISIBLE_DEVICES)
    policy_gpu: int = field(default=0, metadata={"help": "GPU index for policy model"})
    reference_gpu: int = field(default=1, metadata={"help": "GPU index for reference model"})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Guard against multi-process launches (e.g., deepspeed/torchrun).
    # This entry is designed for single-process dual-GPU: one process controls
    # both policy and reference models on separate GPUs. Multi-process will
    # duplicate model loads per rank and quickly OOM.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = getattr(training_args, 'local_rank', -1)
    if world_size > 1 or (local_rank not in (-1, 0)):
        raise RuntimeError(
            "train_dpo_dual_gpu.py expects single-process dual-GPU. "
            "Please run via `python3 train_dpo_dual_gpu.py` with CUDA_VISIBLE_DEVICES set. "
            "Launching with deepspeed/torchrun spawns multiple ranks that each load both models, causing OOM."
        )

    # Compute dtype
    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # BitsAndBytes config
    bnb_model_from_pretrained_args: Dict = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    # Disable default torch init
    disable_torch_init()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Load policy (LLaVA variants) on specified GPU
    def load_llava(path, extra_args=None):
        extra_args = extra_args or {}
        try:
            return LlavaLlamaForCausalLM.from_pretrained(path, cache_dir=training_args.cache_dir, **extra_args)
        except Exception:
            try:
                return LlavaMistralForCausalLM.from_pretrained(path, cache_dir=training_args.cache_dir, **extra_args)
            except Exception:
                return LlavaMptForCausalLM.from_pretrained(path, cache_dir=training_args.cache_dir, **extra_args)

    # Device maps: relative to CUDA_VISIBLE_DEVICES ordering
    policy_device = f"cuda:{training_args.policy_gpu}"
    ref_device = f"cuda:{training_args.reference_gpu}"

    policy_model = load_llava(
        model_args.model_name_or_path,
        extra_args={**bnb_model_from_pretrained_args, "device_map": {"": policy_device}},
    )
    policy_model.config.use_cache = False

    # Gradient checkpointing & PEFT prep for policy
    if training_args.gradient_checkpointing:
        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            policy_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Prepare model for k-bit training if quantized load (4/8-bit)
    if training_args.bits in [4, 8]:
        try:
            from peft import prepare_model_for_kbit_training
            # Align dtype config similar to weighted trainer
            policy_model.config.torch_dtype = (
                torch.float32 if training_args.fp16 else (
                    torch.bfloat16 if training_args.bf16 else torch.float32
                )
            )
            policy_model = prepare_model_for_kbit_training(
                policy_model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        except Exception as e:
            print(f"Warning: prepare_model_for_kbit_training failed: {e}")

    # LoRA setup on policy
    lora_config = None
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        # Derive target modules from current policy model to satisfy PEFT requirements
        target_modules = find_all_linear_names(policy_model)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        # Cast when training in 16-bit modes
        if training_args.bits == 16:
            if training_args.bf16:
                policy_model.to(torch.bfloat16)
            if training_args.fp16:
                policy_model.to(torch.float16)
        if model_args.lora_checkpoint_path:
            # Avoid double-wrap here; Trainer will handle saving. If needed, user can load LoRA via PeftModel externally.
            pass
        else:
            # Wrap policy model with newly created LoRA adapters
            policy_model = get_peft_model(policy_model, lora_config)

    # Freeze vision tower & mm_projector during preference training
    try:
        if hasattr(policy_model, 'get_model') and hasattr(policy_model.get_model(), 'mm_projector') and policy_model.get_model().mm_projector is not None:
            policy_model.get_model().mm_projector.requires_grad_(False)
            for p in policy_model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if hasattr(policy_model, 'get_vision_tower') and policy_model.get_vision_tower() is not None:
            vt = policy_model.get_vision_tower()
            vt.requires_grad_(False)
            for p in vt.parameters():
                p.requires_grad = False
    except Exception as e:
        print(f"Warning: failed to freeze mm_projector/vision_tower: {e}")

    # Tokenizer pads and conversation template
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=policy_model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        if tokenizer.pad_token is None:
            print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=policy_model,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]

    # Initialize vision tokenizer/config on policy (use wrapper method)
    if model_args.vision_tower is not None:
        # Ensure CLIP vision tower is fully loaded (delay_load=True at init requires this)
        try:
            policy_model.get_model().initialize_vision_modules(
                model_args=model_args, fsdp=getattr(training_args, 'fsdp', None)
            )
        except Exception as e:
            print(f"Warning: failed to initialize vision modules on policy: {e}")

        policy_model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        # Mirror train pipeline behavior: set image processor & multimodal flags
        try:
            vt = getattr(policy_model, 'get_vision_tower', lambda: None)()
            processor = None
            if vt is not None and hasattr(vt, 'image_processor') and vt.image_processor is not None:
                processor = vt.image_processor

            # Robust fallback: load processor directly from local/remote vision_tower path
            if processor is None and model_args.vision_tower is not None:
                try:
                    from transformers import AutoImageProcessor, AutoProcessor
                    if os.path.isdir(model_args.vision_tower):
                        try:
                            processor = AutoImageProcessor.from_pretrained(model_args.vision_tower)
                        except Exception:
                            processor = AutoProcessor.from_pretrained(model_args.vision_tower)
                    else:
                        processor = AutoImageProcessor.from_pretrained(model_args.vision_tower)
                except Exception as e:
                    print(f"Warning: failed to load image processor from '{model_args.vision_tower}': {e}")

            # Final safety: minimal processor to avoid AttributeError
            if processor is None:
                try:
                    import torchvision.transforms as T
                    from PIL import Image
                    import types
                    default_mean = [0.48145466, 0.4578275, 0.40821073]
                    default_std = [0.26862954, 0.26130258, 0.27577711]
                    size = 224
                    class _SimpleProcessor:
                        def __init__(self):
                            self.image_mean = default_mean
                            self.image_std = default_std
                            self.crop_size = {"height": size, "width": size}
                            self._tfm = T.Compose([
                                T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                                T.CenterCrop(size),
                                T.ToTensor(),
                                T.Normalize(mean=self.image_mean, std=self.image_std),
                            ])
                        def preprocess(self, image, return_tensors="pt"):
                            if not isinstance(image, Image.Image):
                                image = Image.fromarray(image)
                            pixel = self._tfm(image)
                            import torch
                            return {"pixel_values": pixel.unsqueeze(0)}
                    processor = _SimpleProcessor()
                    print("Info: Using SimpleProcessor fallback for image preprocessing.")
                except Exception as e:
                    print(f"Warning: failed to create SimpleProcessor fallback: {e}")

            if processor is not None:
                data_args.image_processor = processor
                data_args.is_multimodal = True

            # Sync tokenizer and vision-related config on policy model
            policy_model.config.image_aspect_ratio = data_args.image_aspect_ratio
            policy_model.config.tokenizer_padding_side = tokenizer.padding_side
            policy_model.config.tokenizer_model_max_length = tokenizer.model_max_length

            policy_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            policy_model.config.mm_projector_lr = training_args.mm_projector_lr
            training_args.use_im_start_end = model_args.mm_use_im_start_end
            policy_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        except Exception as e:
            print(f"Warning: failed to finalize vision config/image_processor: {e}")

    # Reference model path (required for dual-model SPPO)
    if not model_args.reference_model_path:
        raise ValueError("--reference_model_path must be provided for dual-GPU SPPO training.")

    ref_model = load_llava(
        model_args.reference_model_path,
        extra_args={**bnb_model_from_pretrained_args, "device_map": {"": ref_device}},
    )
    ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False
    if hasattr(ref_model, 'model'):
        ref_model.model.requires_grad_(False)

    # Place vision modules explicitly to their devices
    def place_vision(model, device_id: int):
        dev = f'cuda:{device_id}'
        if hasattr(model, 'get_vision_tower') and model.get_vision_tower() is not None:
            model.get_vision_tower().to(dev)
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
            model.get_model().mm_projector.to(dev)

    place_vision(policy_model, training_args.policy_gpu)
    place_vision(ref_model, training_args.reference_gpu)

    # Dataset and collator
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Build trainer with SPPO variants
    peft_config_to_pass = None if model_args.lora_checkpoint_path else locals().get('lora_config', None)
    trainer = DualGPUDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        policy_gpu=training_args.policy_gpu,
        reference_gpu=training_args.reference_gpu,
        beta=getattr(training_args, 'beta', 0.5),
        loss_type=getattr(training_args, 'loss_type', 'sigmoid'),
        loss_variant=getattr(training_args, 'loss_variant', 'sppo'),
        sppo_eta=getattr(training_args, 'sppo_eta', 1.0),
        sppo_lambda=getattr(training_args, 'sppo_lambda', 0.75),
        sppo_alpha=getattr(training_args, 'sppo_alpha', 0.5),
        args=training_args,
        peft_config=peft_config_to_pass,
        tokenizer=tokenizer,
        loss_use_weight=training_args.loss_use_weight,
        **data_module,
    )

    # Align dtypes of vision towers & mm_projector across models
    target_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    if hasattr(policy_model, 'get_vision_tower') and policy_model.get_vision_tower() is not None:
        vt = policy_model.get_vision_tower()
        vt.to(target_dtype)
        for module in vt.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.to(target_dtype)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.to(target_dtype)
    if hasattr(policy_model, 'get_model') and hasattr(policy_model.get_model(), 'mm_projector') and policy_model.get_model().mm_projector is not None:
        proj = policy_model.get_model().mm_projector
        proj.to(target_dtype)
        for module in proj.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.to(target_dtype)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.to(target_dtype)
    if hasattr(trainer, 'ref_model') and trainer.ref_model is not None:
        if hasattr(trainer.ref_model, 'get_vision_tower') and trainer.ref_model.get_vision_tower() is not None:
            rvt = trainer.ref_model.get_vision_tower()
            rvt.to(target_dtype)
            for module in rvt.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(target_dtype)
        if hasattr(trainer.ref_model, 'get_model') and hasattr(trainer.ref_model.get_model(), 'mm_projector') and trainer.ref_model.get_model().mm_projector is not None:
            rproj = trainer.ref_model.get_model().mm_projector
            rproj.to(target_dtype)
            for module in rproj.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(target_dtype)

    # Train and save
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    trainer.save_state()

    policy_model.config.use_cache = True

    # Save LoRA or full model
    if training_args.lora_enable:
        # Save only LoRA weights and non-LoRA trainables
        from train_dpo_weighted import (
            get_peft_state_maybe_zero_3,
            get_peft_state_non_lora_maybe_zero_3,
            safe_save_model_for_hf_trainer,
        )
        state_dict = get_peft_state_maybe_zero_3(policy_model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(policy_model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            policy_model.config.save_pretrained(training_args.output_dir)
            policy_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        from train_dpo_weighted import safe_save_model_for_hf_trainer
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()