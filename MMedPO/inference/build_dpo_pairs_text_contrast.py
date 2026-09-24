import argparse
import json
import os
import math

def load_jsonl(path):
    data = []
    with open(os.path.expanduser(path), 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def sigmoid(x):
    """Sigmoid function for weight normalization"""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def normalize_weight(tie_value, beta=2.0, w_min=0.05):
    """Normalize TIE value to weight using sigmoid function"""
    normalized = sigmoid(beta * tie_value)
    return max(normalized, w_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tie-results-file', type=str, required=True, help='TIE results file from textual TIE calculation')
    parser.add_argument('--output-pairs-file', type=str, required=True)
    parser.add_argument('--tie-threshold', type=float, default=0.0, help='Minimum TIE value to include in pairs')
    parser.add_argument('--w-min', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=2.0)
    args = parser.parse_args()

    # Load TIE results
    tie_results = load_jsonl(args.tie_results_file)

    out_path = os.path.expanduser(args.output_pairs_file)
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    written = 0
    with open(out_path, 'w') as out:
        for result in tie_results:
            # Skip entries with low TIE values
            textual_tie = result.get('textual_tie', 0.0)
            if textual_tie < args.tie_threshold:
                continue
            
            # Calculate weight based on textual TIE
            weight = normalize_weight(textual_tie, args.beta, args.w_min)
            
            # Extract information from TIE result
            qid = result.get('qid') or result.get('id')
            question_with_context = result.get('question_with_context', '')
            question_without_context = result.get('question_without_context', 'Answer based only on visual information.')
            gt_answer = result.get('gt_answer', '')
            image_path = result.get('image_path', '')
            
            # Build DPO pair for Method 3 (Textual TIE)
            # Preferred: GT answer with full context (I, T)
            # Dispreferred: GT answer with null context (I, T_null)
            item = {
                'type': 'text_contrast',
                'id': qid,
                'question_with_context': question_with_context,
                'question_without_context': question_without_context,
                'pref': { 
                    'answer': gt_answer, 
                    'image_mode': 'full_with_context',
                    'context': 'with_text'
                },
                'disp': { 
                    'answer': gt_answer, 
                    'image_mode': 'full_without_context',
                    'context': 'without_text'
                },
                'images': { 
                    'image_path': image_path
                },
                'weight': weight,
                'tie_metrics': {
                    'textual_tie': textual_tie,
                    'tie_positive': result.get('tie_positive', 0.0),
                    'tie_negative': result.get('tie_negative', 0.0),
                    'tie_difference': result.get('tie_difference', 0.0),
                    'll_text': result.get('ll_text', 0.0),
                    'll_null': result.get('ll_null', 0.0)
                }
            }
            out.write(json.dumps(item) + '\n')
            written += 1

    print(f'Wrote {written} text_contrast pairs to {out_path}')

if __name__ == '__main__':
    main()