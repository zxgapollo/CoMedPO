#!/usr/bin/env python3
"""
Aligned converter for Method 1 (Visual Indirect) to standard DPO format.

Adjustments vs original:
- Image path normalization: always use basename (e.g., synpic29265.jpg) to align
  with Method 2 dataset where image is a filename.
- Keep original script unmodified; this is a new aligned version.
"""

import json
import os
from typing import Dict, Any

def extract_relative_path_aligned(image_path: str) -> str:
    """Return 'subdir/filename' (e.g., CXR1042_IM-0034/0.png).

    Prefer path relative to iu_xray images root; fallback to '<parent>/<filename>'.
    """
    import os
    norm = os.path.normpath(image_path)
    anchors = [
        os.path.join('workspace', 'MMedPO', 'datasets', 'iu_xray', 'images'),
        os.path.join('datasets', 'iu_xray', 'images'),
        '/workspace/MMedPO/datasets/iu_xray/images',
        'datasets/iu_xray/images',
    ]
    for anchor in anchors:
        if anchor in norm:
            suffix = norm.split(anchor, 1)[1].lstrip(os.sep)
            if suffix:
                return suffix
    parent = os.path.basename(os.path.dirname(norm))
    fname = os.path.basename(norm)
    return f"{parent}/{fname}"


def convert_method1_to_standard_format_aligned(input_file: str, output_file: str):
    """Convert Method 1 data to aligned standard DPO format.

    Accepts JSONL built by build_dpo_pairs_visual_indirect.py and maps to aligned SFT/DPO format.
    """
    converted_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if not isinstance(data, dict):
                    continue
                # read unified fields
                question = data.get('question')
                weight = data.get('weight', 1.0)
                pref = data.get('pref', {})
                disp = data.get('disp', {})
                pref_answer = pref.get('answer', '')
                disp_answer = disp.get('answer', '')
                image_path = pref.get('image_path') or disp.get('image_path') or data.get('images', {}).get('full_image_path')

                if image_path is None:
                    # fallback for medgemma tie jsonl that uses 'image' filename
                    image_path = data.get('image')

                # Keep items even if preferred and dispreferred answers are identical

                rounded_weight = round(float(weight), 2)

                qid = data.get('id', line_num - 1)
                converted_entry = {
                    "id": qid,
                    "image": extract_relative_path_aligned(str(image_path)),
                    "conversations": [
                        {"from": "human", "value": f"{question}"},
                        {"from": "gpt", "value": str(pref_answer)}
                    ],
                    "rejected_conversations": [
                        {"from": "human", "value": f"{question}"},
                        {"from": "gpt", "value": str(disp_answer)}
                    ],
                    "weighted_score": rounded_weight
                }
                converted_data.append(converted_entry)
                if line_num % 100 == 0:
                    print(f"Processed {line_num} items...")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"\nConversion completed!")
    print(f"Total items processed: {len(converted_data)}")
    print(f"Output saved to: {output_file}")

    if converted_data:
        sample = converted_data[0]
        print(f"\nSample conversion:")
        print(f"ID: {sample['id']}")
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['conversations'][0]['value']}")
        print(f"Preferred Answer: {sample['conversations'][1]['value']}")
        print(f"Rejected Answer: {sample['rejected_conversations'][1]['value']}")
        print(f"Weighted Score: {sample['weighted_score']}")


if __name__ == "__main__":
    input_file = "/share_docker/workspace/Med/Med-main/Med-main/MMedPO/outputs/combined/dpo_pairs_visual_indirect.jsonl"
    output_file = "/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/tie_dpo_dataset_method1_slake_med_gemma.json"
    convert_method1_to_standard_format_aligned(input_file, output_file)