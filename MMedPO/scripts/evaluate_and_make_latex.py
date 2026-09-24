#!/usr/bin/env python3
"""
Compute closed accuracy and open recall for baseline and multiple conditions,
and output a LaTeX table to the specified directory.

Usage:
  python evaluate_and_make_latex.py --baseline baseline.jsonl --conds cond1:preds1.jsonl cond2:preds2.jsonl ... --out_dir /path/to/rebuttal
"""
from __future__ import annotations
import argparse
import json
import os
import re
from typing import Dict, List, Tuple
import urllib.request
import urllib.error
import time
import datetime
import torch
from typing import Optional
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None


def load_jsonl(path: str) -> Dict[str, dict]:
    d = {}
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = obj.get("image") or ""
            if isinstance(img, list):
                img = img[0] if img else ""
            d[img] = obj
    return d


def is_closed_question(obj: dict) -> bool:
    # heuristics: presence of 'answer_type' == 'CLOSED' or 'positive_answer' field
    if obj.get("answer_type", "").upper() == "CLOSED":
        return True
    if "positive_answer" in obj:
        return True
    gt = (obj.get("gt_answer") or obj.get("positive_answer") or obj.get("answer") or "")
    if isinstance(gt, str) and len(re.split(r"\W+", gt)) <= 4:
        return True
    return False


def open_recall(gt: str, pred: str) -> bool:
    if not gt:
        return False
    gt = gt.strip().lower()
    pred = pred.strip().lower()
    toks = [t for t in re.split(r"\W+", gt) if len(t) > 2]
    if not toks:
        return False
    return any(tok in pred for tok in toks)


def llm_judge_correct_deepseek(gt: str, pred: str, api_key: str, base_url: str, model: str, timeout: int = 10) -> bool:
    """
    Call DeepSeek-style chat API to ask whether the baseline prediction (pred) is a correct answer for gt.
    Returns True if judged correct, False otherwise. On any failure, returns False as conservative fallback.
    """
    if not api_key or not base_url or not model:
        return False

    # Build a strict instruction prompting a JSON-only reply.
    # Require the model to set correct=true only when PRED is semantically equivalent to GT.
    user_msg = (
        "You are a strict evaluator for CLOSED questions. Read GT (ground truth) and PRED (model prediction). "
        "Return ONLY a single valid JSON object and nothing else. The JSON must contain the boolean field "
        "`correct` which must be true only if the PRED conveys the same meaning as the GT (semantically equivalent). "
        "If PRED is ambiguous, partially overlapping, adds or removes meaning, or otherwise does not clearly match GT, "
        "set `correct` to false. Optionally include `explain` with a short justification.\n\n"
        f"GT: {gt}\nPRED: {pred}\n\n"
    )

    payload = {"model": model, "messages": [{"role": "user", "content": user_msg}], "max_tokens": 32}
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = resp.read().decode("utf-8")
            try:
                parsed = json.loads(resp_data)
            except Exception:
                # some APIs return text in choices[0].message.content or choices[0].text
                parsed = None
            content = None
            if isinstance(parsed, dict):
                # try to extract message content
                if "choices" in parsed and len(parsed["choices"]) > 0:
                    ch = parsed["choices"][0]
                    if isinstance(ch, dict):
                        if "message" in ch and isinstance(ch["message"], dict):
                            content = ch["message"].get("content")
                        elif "text" in ch:
                            content = ch.get("text")
            if content is None:
                # fallback to raw response string
                content = resp_data

            # find "correct": true/false in content; be strict - require explicit JSON or exact token
            m = re.search(r'"?correct"?\s*:\s*(true|false)', content, re.IGNORECASE)
            if m:
                return m.group(1).lower() == "true"
            # otherwise conservative fallback: return False
            return False
    except urllib.error.HTTPError as e:
        # log and fallback
        print(f"[warn] DeepSeek judge HTTPError: {e.code} {e.reason}")
    except Exception as e:
        print(f"[warn] DeepSeek judge failed: {e}")
    return False


_LOCAL_TOKENIZER = None
_LOCAL_MODEL = None

def llm_judge_correct_local(gt: str, pred: str, model_path: str, max_new_tokens: int = 32, timeout: int = 30) -> bool:
    """
    Local model judge: load a local causal LM and ask it to output a JSON {"correct": true/false}.
    Uses device_map='auto' and torch_dtype=float16 if possible.
    Returns True if judged correct, False otherwise (conservative).
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        print("[warn] transformers not available for local judge")
        return False

    try:
        if _LOCAL_TOKENIZER is None or _LOCAL_MODEL is None:
            # load tokenizer and model lazily
            _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            # prefer float16 on GPU if available
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)

        system = "You are a strict evaluator for CLOSED questions. Return ONLY a single JSON object."
        prompt = (
            f"{system}\nGT: {gt}\nPRED: {pred}\n\n"
            "The JSON must be exactly: {\"correct\": true|false, \"explain\": \"...\"}. "
            "Set correct=true only when PRED and GT are semantically equivalent; otherwise set correct=false. "
            "Do not output any other text."
        )
        inputs = _LOCAL_TOKENIZER(prompt, return_tensors="pt")
        # move inputs to model device(s) if possible
        try:
            for k, v in inputs.items():
                inputs[k] = v.to(next(_LOCAL_MODEL.parameters()).device)
        except Exception:
            pass
        gen = _LOCAL_MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = _LOCAL_TOKENIZER.decode(gen[0], skip_special_tokens=True)
        # try to find JSON substring and extract correct:true/false strictly
        m = re.search(r'\{.*"correct".*?\}', out, re.DOTALL | re.IGNORECASE)
        content = m.group(0) if m else out
        m2 = re.search(r'"?correct"?\s*:\s*(true|false)', content, re.IGNORECASE)
        if m2:
            return m2.group(1).lower() == "true"
        return False
    except Exception as e:
        print(f"[warn] local judge failed: {e}")
    return False


def compute_metrics(baseline: Dict[str, dict], cond: Dict[str, dict], judge_fn=None) -> Tuple[int, int, int, int, float, float]:
    # returns (n_common, closed_total, closed_correct_baseline, open_recall_in_cond, closed_acc_baseline, open_recall_rate)
    common = set(baseline.keys()) & set(cond.keys())
    n = len(common)
    closed_total = 0
    closed_correct_baseline = 0
    open_recall_in_cond = 0
    open_total = 0
    for img in common:
        b = baseline[img]
        c = cond[img]
        gt = (b.get("gt_answer") or b.get("positive_answer") or b.get("answer") or "").strip()
        pred_b = (b.get("answer") or "").strip()
        pred_c = (c.get("answer") or "").strip()
        if is_closed_question(b):
            closed_total += 1
            # If a judge function is provided, use it to decide correctness (useful for fuzzy matching).
            if judge_fn is not None and gt and pred_b:
                try:
                    judged = judge_fn(gt, pred_b)
                    if judged:
                        closed_correct_baseline += 1
                except Exception:
                    # fallback to strict equality if judge errors
                    if gt and pred_b and gt.strip().lower() == pred_b.strip().lower():
                        closed_correct_baseline += 1
            else:
                if gt and pred_b and gt.strip().lower() == pred_b.strip().lower():
                    closed_correct_baseline += 1
        # open recall
        if gt:
            open_total += 1
            if open_recall(gt, pred_c):
                open_recall_in_cond += 1
    closed_acc = closed_correct_baseline / closed_total if closed_total else 0.0
    open_recall_rate = open_recall_in_cond / open_total if open_total else 0.0
    return n, closed_total, closed_correct_baseline, open_recall_in_cond, closed_acc, open_recall_rate


def make_latex_table(results: List[Tuple[str, Tuple]], out_path: str):
    """
    results: list of (cond_name, metrics_tuple)
    metrics_tuple: (n, closed_total, closed_correct_baseline, open_recall_in_cond, closed_acc, open_recall_rate)
    """
    header = r"""\begin{table}[ht]
\centering
\begin{tabular}{lrrrrrr}
\hline
Condition & Samples & Closed Qs & Baseline Acc & Open Qs & Open Recall & Notes \\\\
\hline
"""
    rows = []
    for cond, metrics in results:
        n, closed_total, closed_correct_baseline, open_recall_in_cond, closed_acc, open_recall_rate = metrics
        row = f"{cond} & {n} & {closed_total} & {closed_acc:.3f} & {max(0, (open_recall_in_cond))} & {open_recall_rate:.3f} & \\\\"
        rows.append(row)
    footer = r"""\hline
\end{tabular}
\caption{Comparison of fill/scale/noise conditions on SLAKE lesion-related subset.}
\label{tab:background_randomization}
\end{table}
"""
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write(header)
        for r in rows:
            fw.write(r + "\n")
        fw.write(footer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--conds", nargs="+", required=True, help="cond_name:preds.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--judge", choices=["none", "deepseek", "local"], default="none", help="Use LLM judge for closed QA")
    p.add_argument("--judge_api_key", default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY") or "")
    p.add_argument("--judge_base_url", default=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com")
    p.add_argument("--judge_model", default=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat")
    p.add_argument("--local_judge_model_path", default=os.environ.get("LOCAL_JUDGE_MODEL_PATH") or "")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    baseline = load_jsonl(args.baseline)
    results = []
    for c in args.conds:
        if ":" not in c:
            continue
        name, path = c.split(":", 1)
        preds = load_jsonl(path)
        # choose judge function
        judge_fn = None
        if args.judge == "deepseek":
            def _judge(gt, pred):
                for attempt in range(3):
                    ok = llm_judge_correct_deepseek(gt, pred, args.judge_api_key, args.judge_base_url, args.judge_model)
                    if ok:
                        return True
                    time.sleep(0.5)
                return False
            judge_fn = _judge
        elif args.judge == "local":
            model_path = args.local_judge_model_path or args.judge_api_key
            def _judge(gt, pred):
                # local judge single attempt (heavy); returns boolean
                return llm_judge_correct_local(gt, pred, model_path)
            judge_fn = _judge
        metrics = compute_metrics(baseline, preds, judge_fn=judge_fn)
        results.append((name, metrics))

    # write outputs; if judge used, append suffix to filenames to preserve originals
    suffix = ""
    if args.judge != "none":
        suffix = "_llmjudge"
    tex_path = os.path.join(args.out_dir, f"background_randomization_table{suffix}.tex")
    make_latex_table(results, tex_path)
    # also write CSV summary
    csv_path = os.path.join(args.out_dir, f"background_randomization_summary{suffix}.csv")
    with open(csv_path, "w", encoding="utf-8") as fw:
        fw.write("condition,n,closed_total,closed_correct_baseline,open_recall_in_cond,closed_acc,open_recall_rate\n")
        for name, metrics in results:
            fw.write(",".join([name] + [str(x) for x in metrics]) + "\n")
    msg = f"Wrote LaTeX table to {tex_path} and CSV to {csv_path}"
    print(msg)
    # Also append to a log file in out_dir for callers that expect a logfile
    try:
        log_path = os.path.join(args.out_dir, "evaluate_and_make_latex.log")
        ts = datetime.datetime.now().astimezone().isoformat()
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{ts}] {msg}\n")
            lf.write(f"[{ts}] ALL_DONE\n")
    except Exception as e:
        print(f"[warn] Failed to write evaluate log: {e}")


if __name__ == "__main__":
    main()

