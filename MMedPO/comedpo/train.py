"""Train CoMedPO LoRA adapters on the repository's LLaVA-Med Mistral model."""

import argparse
import json
import math
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Merged LLaVA-Med Mistral base/SFT checkpoint")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1, help="eta^{-1}; a configurable default, not reported in the paper")
    parser.add_argument("--causal-weight", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.1, help="Additive noise std in [0,1] pixel units; not reported in the paper")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--conversation", default="mistral_instruct")
    parser.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from this output directory's last completed epoch")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "lora_rank", "lora_alpha", "max_length"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("beta", "noise_std", "learning_rate"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.causal_weight) or args.causal_weight < 0:
        parser.error("causal-weight must be finite and nonnegative")
    if not 0 <= args.warmup_ratio <= 1 or not 0 <= args.lora_dropout < 1 or args.seed < 0:
        parser.error("Invalid warmup-ratio, lora-dropout, or seed")
    return args


def main():
    args = parse_args()
    # Prefer the bundled LLaVA implementation without importing legacy trainers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "dpo"))
    import torch
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import DistributedType, set_seed
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
    from .data import CoMedPODataset
    from .model import CoMedPOPolicy, LlavaCollator

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.MULTI_CPU):
        raise ValueError("This entry point supports single-device and DDP training; use neither FSDP nor DeepSpeed")
    set_seed(args.seed)
    output = Path(args.output_dir)
    checkpoint = output / "checkpoint-last"
    if args.resume:
        if not (checkpoint / "progress.json").is_file():
            raise FileNotFoundError("No completed-epoch checkpoint to resume")
        previous = json.loads((checkpoint / "progress.json").read_text())
        # Changing optimization or data settings would invalidate a resumed run.
        for key, value in vars(args).items():
            if key not in ("resume", "output_dir") and previous["args"].get(key) != value:
                raise ValueError(f"Resume argument differs from saved run: {key}")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output-dir is not empty; choose a new directory or use --resume")
    if (Path(args.model_path) / "adapter_config.json").exists():
        raise ValueError("Merge the SFT adapter into a base checkpoint before CoMedPO training")

    dataset = CoMedPODataset(args.data_path, args.image_root, args.noise_std, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("A tokenizer EOS token is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = {"no": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    # Keep trainable adapters in FP32 for mixed-precision optimizer stability.
    base = LlavaMistralForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    tower = base.get_vision_tower()
    if tower is None:
        raise ValueError("The checkpoint must include a multimodal vision tower")
    if not tower.is_loaded:
        tower.load_model()
    tower.to(dtype=dtype)
    base.config.use_cache = False
    base.config.tokenizer_padding_side = "right"
    base.config.tokenizer_model_max_length = None
    targets = [name for name, module in base.named_modules()
               if isinstance(module, torch.nn.Linear)
               and name != "lm_head"
               and not any(part in name for part in ("vision_tower", "mm_projector", "vision_resampler"))]
    policy = get_peft_model(base, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    ))
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = LlavaCollator(tokenizer, tower.image_processor, args.conversation,
                            args.max_length, getattr(base.config, "mm_use_im_start_end", False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    model = CoMedPOPolicy(policy, args.beta, args.causal_weight, args.max_length)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=0.0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(total_steps * args.warmup_ratio), total_steps,
    )
    accelerator.register_for_checkpointing(scheduler)
    start_epoch, step = 0, 0
    if args.resume:
        accelerator.load_state(str(checkpoint))
        start_epoch, step = previous["completed_epochs"], previous["step"]
    accelerator.print(f"CoMedPO: {len(dataset)} pairs, {total_steps} optimizer steps, beta={args.beta}, lambda={args.causal_weight}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        optimizer.zero_grad()
        for batch_index, batch in enumerate(loader):
            with accelerator.accumulate(model):
                result = model(batch)
                # Accelerate divides by the configured accumulation count; correct
                # the last partial window so its samples retain their mean scale.
                remainder = len(loader) % args.gradient_accumulation_steps
                partial = remainder and batch_index >= len(loader) - remainder
                scale = args.gradient_accumulation_steps / remainder if partial else 1.0
                accelerator.backward(result["loss"] * scale)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    step += 1
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                values = torch.stack([result[key].detach() for key in ("loss", "dpo_loss", "causal_loss")])
                values = accelerator.reduce(values, reduction="mean").tolist()
                accelerator.print(json.dumps(dict(epoch=epoch + 1, step=step,
                    loss=values[0], dpo_loss=values[1], causal_loss=values[2])))
        accelerator.wait_for_everyone()
        accelerator.save_state(str(checkpoint))
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped.policy.save_pretrained(output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output / "adapter")
            (checkpoint / "progress.json").write_text(json.dumps({
                "completed_epochs": epoch + 1, "step": step, "args": vars(args),
            }, indent=2) + "\n")
            (output / "training_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        accelerator.wait_for_everyone()
    accelerator.print(f"Saved adapter: {output / 'adapter'}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
