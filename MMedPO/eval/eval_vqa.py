import re
import argparse
import json
import collections
import csv
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tabulate import tabulate
from collections import defaultdict

# ==============================================================================
#  HELPER FUNCTIONS AND METRIC CALCULATIONS (UNCHANGED AS REQUESTED)
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/path/to/base/dir",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=str,
        default="/path/to/output/eval_results.csv",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    gt = load_jsonl(args.base_dir)

    
    jsonl_files = {
        "answer-file_dataset_test_vqa":{
            "result":
        "/path/to/inference-xxxx.jsonl"
        },
        
    }
    results = []
    for file_name, file_info in jsonl_files.items():
        result_file = file_info["result"]
        pred = load_jsonl(result_file)

        gt_ids = [item["question_id"] for item in gt]
        pred_ids = [item["question_id"] for item in pred]
        num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
        print(f"Evaluating result file: {result_file}")
        print(f"num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}")

        result = evaluate(gt, pred, file_name)
        results.append(result)

    save_results_to_csv(results, args.csv_output_dir)
    print(f"Results saved to {args.csv_output_dir}")

    # Print results in table format
    table_data = []
    for result in results:
        table_data.append([
            result["file_name"],
            f"{result['exact_match_score']:.2f}",
            f"{result['f1_score']:.2f}",
            f"{result['precision']:.2f}",
            f"{result['recall']:.2f}",
            f"{result['bleu_score']:.2f}",
            f"{result['yes_no_accuracy']:.2f}",
        ])

    headers = ["File Name", "Exact Match", "F1 Score", "Precision", "Recall", "BLEU Score", "Yes/No Accuracy"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def load_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)

def load_data(path):
    """Load data from either JSON or JSONL format."""
    try:
        # Try to load as regular JSON first
        return load_json(path)
    except json.JSONDecodeError:
        # If that fails, try JSONL format
        return load_jsonl(path)

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
manual_map = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

def normalize_word(token):
    token = str(token)
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (re.search(comma_strip, token) != None):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)
    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token

def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = str(sentence).lower().strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    if not candidate_words: return 0
    common_word_count = sum(1 for word in reference_words if word in candidate_words)
    return common_word_count / len(candidate_words)

def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    if not candidate_words or not reference_words: return 0, 0, 0
    common_keys = set(candidate_words.keys()) & set(reference_words.keys())
    tp = sum(min(candidate_words[k], reference_words[k]) for k in common_keys)
    if tp == 0: return 0, 0, 0
    precision = tp / sum(candidate_words.values())
    recall = tp / sum(reference_words.values())
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

# ==============================================================================
#  CORE EVALUATION LOGIC (MODIFIED SECTION)
# ==============================================================================

def evaluate(data):
    closed_scores = []
    # Use defaultdict to simplify appending
    open_scores = defaultdict(list)
    smooth_fn = SmoothingFunction().method7

    for item in data:
        pred_answer_str = str(item.get("answer", ""))
        gt_answer_str = str(item.get("answer", ""))  # 使用相同的字段，因为数据文件中answer就是标准答案
        
        # Infer answer type if not present
        inferred_answer_type = "OPEN"
        if normalize_word(gt_answer_str) in ["yes", "no"]:
            inferred_answer_type = "CLOSED"

        # Use the explicit "answer_type" from the file if it exists, otherwise use our inference
        answer_type = item.get("answer_type", inferred_answer_type)

        if answer_type == "CLOSED":
            norm_gt = normalize_word(gt_answer_str)
            norm_pred = normalize_word(pred_answer_str)
            score = 1 if norm_gt in norm_pred else 0
            closed_scores.append(score)
        
        else: # Handle OPEN questions
            # --- START OF FIX: Handle multiple ground truth answers ---
            # 1. Split the ground truth string into a list of individual answers
            gt_answers_list = [ans.strip() for ans in gt_answer_str.split(',')]
            
            # 2. Normalize the prediction once
            pred_norm = normalize_word(pred_answer_str)

            # 3. Keep track of the best score found for this prediction
            max_exact_score = 0
            best_f1_scores = (0, 0, 0) # (f1, precision, recall)
            max_bleu_score = 0

            # 4. Loop through each acceptable ground truth answer
            for single_gt_ans in gt_answers_list:
                gt_norm = normalize_word(single_gt_ans)

                # Calculate metrics using the ORIGINAL functions
                current_exact = calculate_exactmatch(pred_norm, gt_norm)
                current_f1, current_precision, current_recall = calculate_f1score(pred_norm, gt_norm)
                
                pred_tokens = pred_norm.split()
                current_bleu = sentence_bleu(
                    references=[gt_norm.split()],
                    hypothesis=pred_tokens if pred_tokens else [" "],
                    smoothing_function=smooth_fn
                )

                # Update the max score if the current one is better
                if current_exact > max_exact_score:
                    max_exact_score = current_exact
                if current_f1 > best_f1_scores[0]:
                    best_f1_scores = (current_f1, current_precision, current_recall)
                if current_bleu > max_bleu_score:
                    max_bleu_score = current_bleu

            # 5. Append the best scores found to the overall lists
            open_scores["exact_match"].append(max_exact_score)
            open_scores["f1"].append(best_f1_scores[0])
            open_scores["precision"].append(best_f1_scores[1])
            open_scores["recall"].append(best_f1_scores[2])
            open_scores["bleu"].append(max_bleu_score)
            # --- END OF FIX ---

    # --- Aggregate scores (using the original metrics) ---
    final_scores = {}
    if closed_scores:
        final_scores["yes_no_accuracy"] = (sum(closed_scores) / len(closed_scores)) * 100
    else:
        final_scores["yes_no_accuracy"] = 0

    if open_scores["f1"]:
        final_scores["exact_match_score"] = (sum(open_scores["exact_match"]) / len(open_scores["exact_match"])) * 100
        final_scores["f1_score"] = (sum(open_scores["f1"]) / len(open_scores["f1"])) * 100
        final_scores["precision"] = (sum(open_scores["precision"]) / len(open_scores["precision"])) * 100
        final_scores["recall"] = (sum(open_scores["recall"]) / len(open_scores["recall"])) * 100
        final_scores["bleu_score"] = (sum(open_scores["bleu"]) / len(open_scores["bleu"])) * 100
    else:
        final_scores["exact_match_score"] = 0
        final_scores["f1_score"] = 0
        final_scores["precision"] = 0
        final_scores["recall"] = 0
        final_scores["bleu_score"] = 0
        
    return final_scores, len(closed_scores), len(open_scores["f1"])


def save_results_to_csv(result, file_name, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["File Name", "Exact Match", "F1 Score", "Precision", "Recall", "BLEU Score", "Yes/No Accuracy"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "File Name": file_name,
            "Exact Match": f"{result['exact_match_score']:.2f}",
            "F1 Score": f"{result['f1_score']:.2f}",
            "Precision": f"{result['precision']:.2f}",
            "Recall": f"{result['recall']:.2f}",
            "BLEU Score": f"{result['bleu_score']:.2f}",
            "Yes/No Accuracy": f"{result['yes_no_accuracy']:.2f}",
        })

# ==============================================================================
#  MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    args = get_args()
    
    data = load_data(args.input_file)
    
    if data:
        file_name = os.path.basename(args.input_file)
        
        scores, num_closed, num_open = evaluate(data)
        
        save_results_to_csv(scores, file_name, args.csv_output_file)
        print(f"\nResults saved to: {args.csv_output_file}")

        print(f"\nEvaluation Results for: {file_name}")
        print(f"Total Examples: {len(data)}")
        print(f"Open-ended Examples: {num_open}")
        print(f"Closed-ended (Yes/No) Examples: {num_closed}")
        
        headers = ["Metric", "Score (%)"]
        table = [
            ["Exact Match (for OPEN questions)", f"{scores['exact_match_score']:.2f}"],
            ["F1 Score (for OPEN questions)", f"{scores['f1_score']:.2f}"],
            ["Precision (for OPEN questions)", f"{scores['precision']:.2f}"],
            ["Recall (for OPEN questions)", f"{scores['recall']:.2f}"],
            ["BLEU Score (for OPEN questions)", f"{scores['bleu_score']:.2f}"],
            ["Yes/No Accuracy (for CLOSED questions)", f"{scores['yes_no_accuracy']:.2f}"],
        ]
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        print("No data found to evaluate in the input file.")