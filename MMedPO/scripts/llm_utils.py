#!/usr/bin/env python3
"""
LLM utilities: attempt to use a local transformers-compatible model to clean/extract short prompts.
Falls back to a simple heuristic if model loading fails or is unavailable.
"""
from pathlib import Path
import re
import json

def simple_clean(text: str) -> str:
    # very small heuristic: keep words with length>2 and remove common stopwords/negations
    stop = {"no", "not", "without", "none", "the", "a", "an", "of", "and", "or", "in", "on", "with", "show", "shows"}
    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)
    filtered = [w for w in words if w.lower() not in stop and len(w) > 2]
    if len(filtered) == 0:
        return text.strip()
    return " ".join(filtered[:8])


def clean_prompt_with_llm(raw_text: str, llm_path: str, max_tokens: int = 64, device: str = "cpu") -> str:
    """
    Try to load a local HF-style causal LM from `llm_path` and generate a short cleaned prompt.
    If loading or generation fails, return a heuristic-cleaned string.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, pipeline
        import torch
        model_path = str(Path(llm_path).expanduser())
        # load tokenizer and model with local_files_only to avoid network
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device != "cpu" and torch.cuda.is_available() else -1)
        prompt = f"Extract the main lesion and concise phrase from the following medical report. Output a short noun phrase only.\n\nReport: {raw_text}\n\nPhrase:"
        out = gen(prompt, max_length=len(tokenizer.encode(prompt)) + max_tokens, do_sample=False, num_return_sequences=1)
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            txt = out[0]["generated_text"]
            # get tail after "Phrase:"
            if "Phrase:" in txt:
                txt = txt.split("Phrase:")[-1].strip()
            # keep first line
            txt = txt.splitlines()[0].strip()
            if len(txt) == 0:
                return simple_clean(raw_text)
            return txt
    except Exception:
        return simple_clean(raw_text)
    return simple_clean(raw_text)


def generate_ensemble_prompts(cleaned_phrase: str):
    """
    Given a cleaned phrase, produce an ensemble of prompt variations including synonyms and short templates.
    """
    syns = ["lesion", "nodule", "mass", "abnormality", "opacity", "opacity region"]
    prompts = []
    base = cleaned_phrase.strip()
    if base:
        prompts.append(base)
    for s in syns:
        prompts.append(f"{base} {s}".strip())
        prompts.append(f"{s} in {base}".strip())
    # ensure uniqueness and limit
    seen = set()
    out = []
    for p in prompts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
        if len(out) >= 8:
            break
    return out

