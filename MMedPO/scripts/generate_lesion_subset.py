#!/usr/bin/env python3
"""
Generate a lesion-related question subset from a JSONL dataset.

Reads a JSONL file where each line is a JSON object containing at least a
`question` field and writes a sampled subset of lines whose question text
matches any of the provided keywords.

Usage:
  python generate_lesion_subset.py --input /path/to/slake_dpo_weighted.json \
      --out /path/to/output/questions.jsonl --num 20 --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import random
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Generate lesion-related question subset")
    p.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p.add_argument("--out", "-o", required=True, help="Output JSONL file")
    p.add_argument("--num", "-n", type=int, default=20, help="Number of samples to output")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--keywords",
        "-k",
        type=str,
        default="lesion,mass,nodule,tumor,opacity",
        help="Comma-separated keywords to match (case-insensitive)",
    )
    return p.parse_args()


def matches_keywords(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return True
    return False


def main():
    args = parse_args()
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    matches = []
    read_count = 0
    with open(args.input, "r", encoding="utf-8") as fr:
        first_chars = fr.read(2)
        fr.seek(0)
        if first_chars.lstrip().startswith("["):
            # file is a JSON array
            try:
                all_objs = json.load(fr)
            except Exception as e:
                print(f"[generate_lesion_subset] Failed to parse JSON array: {e}")
                return
            for obj in all_objs:
                read_count += 1
                if not isinstance(obj, dict):
                    continue
                # proceed with same matching logic below
                # Extract question text from common structures:
                # - direct fields: question, question_text, caption, prompt
                # - conversations: list of {from:'human'|'gpt', value: '...'} -> take human value
                qtext = None
                for key in ("question", "question_text", "caption", "prompt"):
                    if key in obj and isinstance(obj[key], str):
                        qtext = obj[key]
                        break
                if not qtext and isinstance(obj.get("conversations"), list):
                    for turn in obj["conversations"]:
                        if isinstance(turn, dict) and turn.get("from") and turn.get("from").lower() == "human" and isinstance(turn.get("value"), str):
                            qtext = turn.get("value")
                            break
                if not qtext:
                    if "question_id" in obj and "question" in obj:
                        qtext = obj.get("question")
                if matches_keywords(qtext or "", keywords):
                    matches.append(obj)
        else:
            # treat as JSONL
            for line in fr:
                read_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue
                # Extract question text from common structures:
                # - direct fields: question, question_text, caption, prompt
                # - conversations: list of {from:'human'|'gpt', value: '...'} -> take human value
                qtext = None
                for key in ("question", "question_text", "caption", "prompt"):
                    if key in obj and isinstance(obj[key], str):
                        qtext = obj[key]
                        break
                if not qtext and isinstance(obj.get("conversations"), list):
                    for turn in obj["conversations"]:
                        if isinstance(turn, dict) and turn.get("from") and turn.get("from").lower() == "human" and isinstance(turn.get("value"), str):
                            qtext = turn.get("value")
                            break
                if not qtext:
                    if "question_id" in obj and "question" in obj:
                        qtext = obj.get("question")
                if matches_keywords(qtext or "", keywords):
                    matches.append(obj)
            # Extract question text from common structures:
            # - direct fields: question, question_text, caption, prompt
            # - conversations: list of {from:'human'|'gpt', value: '...'} -> take human value
            qtext = None
            for key in ("question", "question_text", "caption", "prompt"):
                if key in obj and isinstance(obj[key], str):
                    qtext = obj[key]
                    break
            if not qtext and isinstance(obj.get("conversations"), list):
                for turn in obj["conversations"]:
                    if isinstance(turn, dict) and turn.get("from") and turn.get("from").lower() == "human" and isinstance(turn.get("value"), str):
                        qtext = turn.get("value")
                        break
            # fallback for other nested patterns
            if not qtext:
                if "question_id" in obj and "question" in obj:
                    qtext = obj.get("question")
            if matches_keywords(qtext or "", keywords):
                matches.append(obj)

    if not matches:
        print(f"[generate_lesion_subset] No matches found in {args.input} for keywords={keywords}")
        return

    random.seed(args.seed)
    random.shuffle(matches)
    selected = matches[: args.num]

    with open(args.out, "w", encoding="utf-8") as fw:
        for obj in selected:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[generate_lesion_subset] Read {read_count} lines, found {len(matches)} matches, wrote {len(selected)} to {args.out}")


if __name__ == "__main__":
    main()

