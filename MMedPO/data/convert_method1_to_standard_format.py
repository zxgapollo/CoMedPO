#!/usr/bin/env python3
"""
Convert Method 1 (Visual Indirect) data to standard DPO format.

Supports multiple input schemas:
- Schema A (method1): keys: question, pref{answer,image_path}, disp{answer,image_path}, images{full_image_path,masked_image_path}, weight
- Schema B (DPO-like): keys: question, chosen{answer}, rejected{answer}, image or images{full_image_path}

Outputs a JSON file with entries containing:
- id, image (relative path), conversations (human/gpt), rejected_conversations (human/gpt), weighted_score (optional)
"""

import json
import os
import argparse
from typing import Dict, Any, Optional


def extract_relative_path(image_path: Optional[str], root_hint: Optional[str] = None) -> Optional[str]:
    """Extract a relative path from a full image path.
    If root_hint is provided and is a prefix of image_path, use relpath to root_hint.
    Else, try to split by '/imgs/' and take the tail; otherwise, fallback to last two components.
    """
    if not image_path:
        return None
    try:
        if root_hint and image_path.startswith(root_hint):
            rel = os.path.relpath(image_path, root_hint)
            return rel
        if '/imgs/' in image_path:
            return image_path.split('/imgs/')[-1]
        # Fallback: last two parts
        parts = image_path.strip('/').split('/')
        if len(parts) >= 2:
            return '/'.join(parts[-2:])
        return os.path.basename(image_path)
    except Exception:
        return os.path.basename(image_path)


def generate_rejected_answer_for_method1(pref_answer: str, disp_answer: Optional[str], question: str) -> str:
    """
    Generate a distinct rejected answer for Method 1 data.

    Method 1 compares answers with full image vs background-only image.
    When answers are identical, create visually-limited uncertain responses.
    """
    if disp_answer is None:
        disp_answer = ""

    if pref_answer.strip() == disp_answer.strip():
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['modality', 'imaging', 'scan', 'technique']):
            return "I cannot clearly determine the imaging modality from the background alone."
        elif any(keyword in question_lower for keyword in ['body part', 'part of', 'belong to', 'region', 'anatomy']):
            return "The anatomical region is not clearly identifiable without the main structures visible."
        elif any(keyword in question_lower for keyword in ['contain', 'see', 'visible', 'present']):
            if 'yes' in pref_answer.lower():
                return "I'm not certain, the structure might be present but is not clearly visible in this view."
            else:
                return "I cannot definitively determine if this structure is present without clearer visual details."
        elif any(keyword in question_lower for keyword in ['where', 'location', 'position']):
            return "I cannot accurately determine the location without the main anatomical structures visible."
        elif any(keyword in question_lower for keyword in ['healthy', 'normal', 'abnormal']):
            return "I cannot assess the health status without seeing the complete anatomical structures."
        elif any(keyword in question_lower for keyword in ['disease', 'condition', 'pathology']):
            return "I cannot identify specific diseases or conditions without clear visual details of the main structures."
        elif any(keyword in question_lower for keyword in ['main', 'largest', 'primary', 'organ']):
            return "I cannot clearly identify the main organ without the complete visual information."
        elif any(keyword in question_lower for keyword in ['bigger', 'larger', 'compare', 'size']):
            return "I cannot make accurate size comparisons without seeing the complete structures."
        elif any(keyword in question_lower for keyword in ['weighting', 'weighted', 'mr', 'mri']):
            return "I cannot determine the MR weighting without clear visual details of the main structures."
        else:
            return "I cannot provide a confident answer without seeing the complete anatomical structures."
    else:
        return disp_answer


def _get_nested(d: Dict[str, Any], path: list) -> Optional[Any]:
    cur = d
    try:
        for key in path:
            if isinstance(cur, list) and isinstance(key, int):
                cur = cur[key]
            elif isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
        return cur
    except Exception:
        return None


def convert_method1_to_standard_format(input_file: str, output_file: str, image_root: Optional[str] = None):
    """Convert Method 1/combined data to standard DPO format."""
    converted_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

            # question
            question = data.get('question') or _get_nested(data, ['conversations', 0, 'value'])

            # preferred & dispreferred answers (multi-schema fallback)
            pref_answer = (
                _get_nested(data, ['pref', 'answer']) or
                _get_nested(data, ['chosen', 'answer']) or
                data.get('preferred_answer') or
                _get_nested(data, ['conversations', 1, 'value'])
            )
            disp_answer = (
                _get_nested(data, ['disp', 'answer']) or
                _get_nested(data, ['rejected', 'answer']) or
                data.get('dispreferred_answer') or
                _get_nested(data, ['rejected_conversations', 1, 'value'])
            )

            # image path (prefer full image path)
            image_path_full = (
                _get_nested(data, ['pref', 'image_path']) or
                _get_nested(data, ['images', 'full_image_path']) or
                data.get('image')
            )

            # weight (optional)
            weight = data.get('weight') or data.get('weighted_score')

            # minimal required fields
            if question is None or pref_answer is None:
                print(f"Skip line {line_num}: missing question or preferred answer")
                continue

            rejected_answer = generate_rejected_answer_for_method1(pref_answer, disp_answer, question)
            rounded_weight = round(weight, 2) if isinstance(weight, (int, float)) else None
            image_rel = extract_relative_path(image_path_full, root_hint=image_root) if image_path_full else None

            converted_entry: Dict[str, Any] = {
                "id": data.get('id', line_num - 1),
                "image": image_rel,
                "conversations": [
                    {"from": "human", "value": f"{question}"},
                    {"from": "gpt", "value": pref_answer}
                ],
                "rejected_conversations": [
                    {"from": "human", "value": f"{question}"},
                    {"from": "gpt", "value": rejected_answer}
                ]
            }
            if rounded_weight is not None:
                converted_entry["weighted_score"] = rounded_weight

            converted_data.append(converted_entry)

            if line_num % 100 == 0:
                print(f"Processed {line_num} items...")

    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"\nConversion completed!")
    print(f"Total items processed: {len(converted_data)}")
    print(f"Output saved to: {output_file}")

    # Show sample conversion
    if converted_data:
        sample = converted_data[0]
        print(f"\nSample conversion:")
        print(f"ID: {sample['id']}")
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['conversations'][0]['value']}")
        print(f"Preferred Answer: {sample['conversations'][1]['value']}")
        print(f"Rejected Answer: {sample['rejected_conversations'][1]['value']}")
        print(f"Weighted Score: {sample.get('weighted_score', 'N/A')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Method 1 / combined visual_indirect data to DPO standard format")
    parser.add_argument('--input', '-i', type=str,
                        default="/workspace/MMedPO/MMedPO/outputs/method1/visual_indirect/visual_tie_results.jsonl",
                        help='Input JSONL file path')
    parser.add_argument('--output', '-o', type=str,
                        default="/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method1_slake_aligned.json",
                        help='Output JSON file path')
    parser.add_argument('--image_root', type=str,
                        default="/workspace/MMedPO/datasets/SLAKE/imgs",
                        help='Root directory for images to derive relative paths')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_method1_to_standard_format(args.input, args.output, args.image_root)