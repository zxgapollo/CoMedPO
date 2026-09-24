import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


def build_dataset(path: str, prompt_key: str, answer_key: str):
    # Expect LLaVA-style JSON list with conversations OR simple QA dicts
    # We extract a single-turn prompt and reference answer for sparse rewards.
    data = load_dataset("json", data_files=path, split="train")

    def to_prompt(ex):
        if prompt_key in ex:
            prompt = ex[prompt_key]
        elif "conversations" in ex and isinstance(ex["conversations"], list) and ex["conversations"]:
            first = ex["conversations"][0]
            prompt = first.get("value", "")
        else:
            prompt = ""
        answer = ex.get(answer_key, "")
        return {"prompt": prompt, "answer": answer}

    return data.map(to_prompt, remove_columns=[c for c in data.column_names if c not in (prompt_key, answer_key)])


# --- Reward functions ---

def format_reward(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    # Encourage concise, properly formatted answers (e.g., limit length, penalize empty)
    scores = []
    for comp in completions:
        text = comp[0]["content"] if comp and isinstance(comp[0], dict) else ""
        l = len(text.strip())
        if l == 0:
            scores.append(-1.0)
            continue
        # soft length prior: reward 0..1 for length in [10, 256]
        score = max(0.0, min(1.0, (l - 10) / 246))
        scores.append(score)
    return scores


def accuracy_reward(completions: List[List[Dict[str, Any]]], references: List[str] = None, **kwargs) -> List[float]:
    # Simple exact-match (case-insensitive, stripped)
    refs = references or kwargs.get("references") or []
    out = []
    for i, comp in enumerate(completions):
        text = comp[0]["content"].strip().lower() if comp and isinstance(comp[0], dict) else ""
        ref = (refs[i].strip().lower() if i < len(refs) and refs[i] is not None else "")
        out.append(1.0 if (ref and text == ref) else 0.0)
    return out


@dataclass
class Args:
    model_path: str
    dataset_path: str
    prompt_key: str
    answer_key: str
    output_dir: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_generations: int
    bf16: bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--prompt_key", default="question")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    ds = build_dataset(args.dataset_path, args.prompt_key, args.answer_key)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load merged model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    # Prepare references aligned with dataset order
    references = [ex.get("answer", "") for ex in ds]

    def reward_funcs(completions, **kwargs):
        # Combine rewards with weights
        r1 = format_reward(completions)
        r2 = accuracy_reward(completions, references=references)
        return [0.3 * a + 0.7 * b for a, b in zip(r1, r2)]

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16,
        remove_unused_columns=False,
        max_prompt_length=512,
        max_completion_length=128,
        num_generations=args.num_generations,
        report_to=["none"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,  # expects a column named "prompt"
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        formatting_func=lambda x: x["prompt"],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()