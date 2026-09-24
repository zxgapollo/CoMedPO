import re
import argparse
import json
import collections
import random
import csv
import os
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate


def load_jsonl(path):
    """Load JSONL format file"""
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def load_json(path):
    """Load JSON format file"""
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def load_data(path):
    """Load data from either JSON or JSONL format"""
    try:
        # First try JSON format
        return load_json(path)
    except json.JSONDecodeError:
        # If JSON fails, try JSONL format
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
    "'ow'sat": "'ow'sat", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

manual_map = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

articles = ["a", "an", "the"]
period_strip = re.compile("(?<!\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def normalize_word(token):
    """Normalize text for comparison"""
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
    """Split sentence into n-grams"""
    words = collections.defaultdict(int)
    tmp_sentence = sentence.lower().strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


# --- begin: EXACT original-style exact match ---
def calculate_exactmatch(candidate, reference):
    """Original exact match: count unique overlapping reference tokens, divide by total candidate token count"""
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0
    else:
        return count / total
# --- end: EXACT original-style exact match ---

# --- begin: EXACT original-style F1 ---
def calculate_f1score(candidate, reference):
    """Original F1: tp sums candidate counts for words present in both; fp and fn sum the extra counts"""
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        if tp == 0 or (precision + recall) == 0:
            return 0, precision, recall
        else:
            return 2 * precision * recall / (precision + recall), precision, recall
# --- end: EXACT original-style F1 ---

# --- begin: EXACT original-style evaluate ---
from nltk.translate.bleu_score import SmoothingFunction

def evaluate(gt, pred, file_name):
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)

    for gt_item, pred_item in zip(gt, pred):
        gt_value = gt_item["positive_answer"].lower()
        pred_value = pred_item["answer"].lower()
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if gt_item["answer_type"] == "CLOSED":
            closed_scores["q_id"].append(pred_item["question_id"])
            if gt_value in pred_value:
                closed_scores["hit"].append(1)
            else:
                closed_scores["hit"].append(0)
        else:
            exact_scores["hit"].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores["q_id"].append(pred_item["question_id"])

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores["f1"].append(f1_score)
            f1_scores["precision"].append(precision)
            f1_scores["recall"].append(recall)
            f1_scores["q_id"].append(pred_item["question_id"])

            smooth_fn = SmoothingFunction().method7
            b_score = sentence_bleu(
                references=[str(gt_value).lower().split()],
                hypothesis=str(pred_value).lower().split(),
                smoothing_function=smooth_fn,
            )
            bleu_scores["bleu_score"].append(b_score)

    exact_score = (sum(exact_scores["hit"]) / len(exact_scores["hit"])) if exact_scores["hit"] else 0.0
    f1_score = (sum(f1_scores["f1"]) / len(f1_scores["f1"])) if f1_scores["f1"] else 0.0
    precision = (sum(f1_scores["precision"]) / len(f1_scores["precision"])) if f1_scores["precision"] else 0.0
    recall = (sum(f1_scores["recall"]) / len(f1_scores["recall"])) if f1_scores["recall"] else 0.0
    bleu_score = (sum(bleu_scores["bleu_score"]) / len(bleu_scores["bleu_score"])) if bleu_scores["bleu_score"] else 0.0
    closed_score = (
        sum(closed_scores["hit"]) / len(closed_scores["hit"]) if len(closed_scores["hit"]) != 0 else 0.0
    )

    return {
        "file_name": file_name,
        "exact_match_score": exact_score * 100,
        "f1_score": f1_score * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "bleu_score": bleu_score * 100,
        "yes_no_accuracy": closed_score * 100,
    }
# --- end: EXACT original-style evaluate ---

# --- begin: align-and-call evaluate_file (IO only) ---
def evaluate_file(gt_file, pred_file):
    """IO glue: load files, align by id, then call original evaluate() on aligned lists."""
    gt_data = load_data(gt_file)
    pred_data = load_data(pred_file)

    gt_by_id = {item["qid"]: item for item in gt_data}
    pred_by_id = {item["question_id"]: item for item in pred_data}

    # Build aligned lists in GT order; skip if prediction missing
    gt_aligned = []
    pred_aligned = []
    for qid in sorted(gt_by_id.keys()):
        if qid in pred_by_id:
            gt_aligned.append(gt_by_id[qid])
            pred_aligned.append(pred_by_id[qid])

    # Call evaluate with original logic
    results = evaluate(gt_aligned, pred_aligned, pred_file.split('/')[-1])

    # Extend with counts for display
    open_count = sum(1 for item in gt_aligned if item.get('answer_type') == 'OPEN')
    closed_count = sum(1 for item in gt_aligned if item.get('answer_type') == 'CLOSED')
    results.update({
        'total_examples': len(gt_data),
        'evaluated_examples': len(gt_aligned),
        'open_ended_count': open_count,
        'closed_ended_count': closed_count,
    })
    return results
# --- end: align-and-call evaluate_file (IO only) ---

# --- begin: Update CSV/print to include BLEU ---
def save_results_to_csv(results, csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'File Name', 'Total Examples', 'Evaluated Examples',
            'Open-ended Count', 'Closed-ended Count',
            'Exact Match', 'F1 Score', 'Precision', 'Recall', 'BLEU Score', 'Yes/No Accuracy'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            'File Name': results['file_name'],
            'Total Examples': results['total_examples'],
            'Evaluated Examples': results['evaluated_examples'],
            'Open-ended Count': results['open_ended_count'],
            'Closed-ended Count': results['closed_ended_count'],
            'Exact Match': f"{results['exact_match_score']:.2f}",
            'F1 Score': f"{results['f1_score']:.2f}",
            'Precision': f"{results['precision']:.2f}",
            'Recall': f"{results['recall']:.2f}",
            'BLEU Score': f"{results['bleu_score']:.2f}",
            'Yes/No Accuracy': f"{results['yes_no_accuracy']:.2f}"
        })

def print_results(results):
    print(f"\nEvaluation Results for: {results['file_name']}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Evaluated Examples: {results['evaluated_examples']}")
    print(f"Open-ended Examples: {results['open_ended_count']}")
    print(f"Closed-ended (Yes/No) Examples: {results['closed_ended_count']}")
    print()

    table_data = [
        ["Exact Match (for OPEN questions)", f"{results['exact_match_score']:.2f}%"],
        ["F1 Score (for OPEN questions)", f"{results['f1_score']:.2f}%"],
        ["Precision (for OPEN questions)", f"{results['precision']:.2f}%"],
        ["Recall (for OPEN questions)", f"{results['recall']:.2f}%"],
        ["BLEU Score (for OPEN questions)", f"{results['bleu_score']:.2f}%"],
        ["Yes/No Accuracy (for CLOSED questions)", f"{results['yes_no_accuracy']:.2f}%"],
    ]
    print(tabulate(table_data, headers=["Metric", "Score (%)"], tablefmt="grid"))
# --- end: Update CSV/print to include BLEU ---


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate VQA results')
    parser.add_argument(
        '--gt-file', 
        type=str, 
        required=True,
        help='Path to ground truth file (JSON or JSONL)'
    )
    parser.add_argument(
        '--pred-file',
        type=str,
        required=True,
        help='Path to prediction file (JSON or JSONL)'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    results = evaluate_file(args.gt_file, args.pred_file)
    
    # Handle csv_output: if it's a directory, generate filename automatically
    csv_output_path = args.csv_output
    if os.path.isdir(csv_output_path):
        csv_filename = "evaluation_results.csv"
        csv_output_path = os.path.join(csv_output_path, csv_filename)
    
    save_results_to_csv(results, csv_output_path)
    print_results(results)