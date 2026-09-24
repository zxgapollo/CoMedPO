import argparse
import json
import os
import math
import numpy as np

def sigmoid(x, beta=2.0, tau=0.0):
    """Sigmoid function for weight normalization"""
    return 1.0 / (1.0 + math.exp(-beta * (x - tau)))

def normalize_weight(tie_value, w_min=0.05, beta=2.0, tau=0.0):
    """Normalize TIE value to weight using sigmoid function"""
    weight = sigmoid(tie_value, beta, tau)
    return max(weight, w_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tie-results-file', type=str, required=True, help='TIE results JSON file')
    parser.add_argument('--output-pairs-file', type=str, required=True)
    parser.add_argument('--w-min', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--tie-threshold', type=float, default=0.0, help='Minimum TIE value to create pairs')
    args = parser.parse_args()

    # Load TIE results (support both JSON and JSONL formats)
    tie_results = []
    with open(os.path.expanduser(args.tie_results_file), 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON format
            tie_results = json.loads(content)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    tie_results.append(json.loads(line))
    
    print(f"Loaded {len(tie_results)} TIE results")

    out_path = os.path.expanduser(args.output_pairs_file)
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written = 0
    with open(out_path, 'w') as out:
        for tie_result in tie_results:
            # Extract data from TIE result
            qid = tie_result.get('qid')
            question = tie_result.get('question')
            gt_answer = tie_result.get('gt_answer')
            visual_tie = tie_result.get('visual_tie', 0.0)
            
            # Skip if TIE is below threshold
            if visual_tie < args.tie_threshold:
                continue
            
            # Calculate weight from TIE value
            weight = normalize_weight(visual_tie, args.w_min, args.beta, args.tau)
            
            # Build DPO pair according to Visual TIE strategy:
            # Preferred: GT answer with full image
            # Dispreferred: GT answer with background only (simulating degraded performance)
            
            # For visual indirect, we create a pair where:
            # - Preferred uses full image context
            # - Dispreferred uses background only context
            
            full_image_path = tie_result.get('full_image_path')
            masked_image_path = tie_result.get('masked_image_path')
            
            item = {
                'type': 'visual_indirect',
                'id': qid,
                'question': question,
                'pref': { 
                    'answer': gt_answer, 
                    'image_mode': 'full_image',
                    'image_path': full_image_path
                },
                'disp': { 
                    'answer': gt_answer, 
                    'image_mode': 'background_only',
                    'image_path': masked_image_path
                },
                'images': { 
                    'full_image_path': full_image_path, 
                    'masked_image_path': masked_image_path 
                },
                'weight': weight,
                'tie_metrics': {
                    'visual_tie': visual_tie,
                    'll_pref': tie_result.get('ll_pref'),
                    'll_disp': tie_result.get('ll_disp'),
                    'll_full': tie_result.get('ll_full'),
                    'll_bg': tie_result.get('ll_bg'),
                    'tie_positive': tie_result.get('tie_positive', 0.0),
                    'tie_negative': tie_result.get('tie_negative', 0.0),
                    'tie_difference': tie_result.get('tie_difference', visual_tie),
                    # 同步 method1/y_gt 元数据以便下游跟踪
                    'method': tie_result.get('method', 'visual_indirect_method1'),
                    'scoring_target': tie_result.get('scoring_target', 'y_gt'),
                    'tie_formula': tie_result.get('tie_formula', '(y_gt|X ⊕ X_bg) > (y_gt|X_null ⊕ X_bg)'),
                    'pref_condition': tie_result.get('pref_condition', 'full_image'),
                    'disp_condition': tie_result.get('disp_condition', 'background_only')
                }
            }
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            written += 1

    print(f'Wrote {written} visual_indirect pairs to {out_path}')

if __name__ == '__main__':
    main()