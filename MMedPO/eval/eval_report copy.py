import argparse
import json
import collections
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default="path/to/your/base_dir.jsonl",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=str,
        default="path/to/your/output.csv",
    )
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def save_results_to_csv(results, csv_filename):
    """保存结果到CSV文件"""
    with open(csv_filename, mode="w", newline="") as file:
        fieldnames = [
            "File Name",
            "Exact Match",
            "F1 Score",
            "Precision",
            "Recall",
            "BLEU Score",
            "BLEU Score 1",
            "BLEU Score 2",
            "BLEU Score 3",
            "BLEU Score 4",
            "ROUGE-L",
            "METEOR",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the results for each file
        for file_result in results:
            # print(results)
            writer.writerow(
                {
                    "File Name": file_result["file_name"],
                    "Exact Match": file_result["exact_match_score"],
                    "F1 Score": file_result["f1_score"],
                    "Precision": file_result["precision"],
                    "Recall": file_result["recall"],
                    "BLEU Score": file_result["bleu_score"],
                    "BLEU Score 1": file_result["bleu_score_1"],
                    "BLEU Score 2": file_result["bleu_score_2"],
                    "BLEU Score 3": file_result["bleu_score_3"],
                    "BLEU Score 4": file_result["bleu_score_4"],
                    "ROUGE-L": file_result["rouge_l"],
                    "METEOR": file_result["meteor"],
                }
            )




import re

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
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


import math


def bleu(candidate, references, n, weights):

    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is bigger than n)"
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is smaller than n)"
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return str(bleu_result)


# BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


# Exact match
def calculate_exactmatch(candidate, reference):

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
        return 0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


def similarity_candidate_prediction(candidate_answer, prediction):

    candidate_answer = split_sentence(candidate_answer, 1)

    count = 0
    total = 0
    for word in prediction:
        if word in candidate_answer:
            count += 1

    total = len(candidate_answer)

    if total == 0:
        return 0.0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


def argmax(lst):
    return lst.index(max(lst))


def calculate_appearance_with_normalization(prediction, reference, candidate_set):

    prediction = normalize_word(prediction)
    reference = normalize_word(reference)
    prediction_words = split_sentence(prediction, 1)
    reference_words = split_sentence(reference, 1)

    candidate_set = candidate_set["0"]

    similarity_list = []
    candidate_answer_normalized_list = []
    for candidate_answer in candidate_set:

        if isinstance(candidate_answer, int):
            candidate_answer = str(candidate_answer)

        candidate_answer = normalize_word(candidate_answer)
        candidate_answer_normalized_list.append(candidate_answer)
        similarity_list.append(
            similarity_candidate_prediction(candidate_answer, prediction_words)
        )

    final_prediction = candidate_answer_normalized_list[argmax(similarity_list)]


    if final_prediction == reference:
        return 1.0  #
    else:
        return 0.0


from collections import defaultdict
import re
import math


def brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def modified_precision(candidate, references, n):
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)

    candidate_words = split_sentence(candidate, n)

    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
        min_frequency[word] = min(max_frequency[word], candidate_words[word])
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P


def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


# F1
def calculate_f1score(candidate, reference):

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
        return 0, 0, 0  # "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall

def evaluate(gt, pred, file_name):
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    rouge_l_scores = collections.defaultdict(list)
    meteor_scores = collections.defaultdict(list)

    rouge = Rouge()  # Initialize ROUGE metric

    for gt_item, pred_item in zip(gt, pred):
        gt_value = gt_item["answer"].lower()
        pred_value = pred_item["answer"].lower()
        
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)
        # print(f"pred value:{str(pred_value).lower()}")

        # Exact match
        exact_scores["hit"].append(calculate_exactmatch(pred_value, gt_value))
        exact_scores["q_id"].append(pred_item["question_id"])

        # F1 Score Calculation
        f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
        f1_scores["f1"].append(f1_score)
        f1_scores["precision"].append(precision)
        f1_scores["recall"].append(recall)
        f1_scores["q_id"].append(pred_item["question_id"])
        from nltk.translate.bleu_score import SmoothingFunction

        
        smooth_fn = SmoothingFunction().method7
        # BLEU Score Calculation
        b_score = sentence_bleu(
            references=[str(gt_value).lower().split()],
            hypothesis=str(pred_value).lower().split(),
            smoothing_function=smooth_fn,
        )
        b_score_1 = sentence_bleu(
            references=[str(gt_value).lower().split()],
            hypothesis=str(pred_value).lower().split(),
            weights=(1, 0, 0, 0),
            smoothing_function=smooth_fn,
        )
        b_score_2 = sentence_bleu(
            references=[str(gt_value).lower().split()],
            hypothesis=str(pred_value).lower().split(),
            weights=(0, 1, 0, 0),
            smoothing_function=smooth_fn,
        )
        b_score_3 = sentence_bleu(
            references=[str(gt_value).lower().split()],
            hypothesis=str(pred_value).lower().split(),
            weights=(0, 0, 1, 0),
            smoothing_function=smooth_fn,
        )
        b_score_4 = sentence_bleu(
            references=[str(gt_value).lower().split()],
            hypothesis=str(pred_value).lower().split(),
            weights=(0, 0, 0, 1),
            smoothing_function=smooth_fn,
        )
        
        rouge_l = rouge.get_scores(str(pred_value).lower(), str(gt_value).lower())[0][
            "rouge-l"
        ]["f"]
        rouge_l_scores["rouge_l"].append(rouge_l)

        # Append BLEU Scores
        bleu_scores["q_id"].append(pred_item["question_id"])
        bleu_scores["bleu_score"].append(b_score)
        bleu_scores["bleu_score_1"].append(b_score_1)
        bleu_scores["bleu_score_2"].append(b_score_2)
        bleu_scores["bleu_score_3"].append(b_score_3)
        bleu_scores["bleu_score_4"].append(b_score_4)

        meteor = meteor_score(
            [str(gt_value).lower().split()], str(pred_value).lower().split()
        )
        meteor_scores["meteor"].append(meteor)

    # Final Results Calculation
    exact_score = sum(exact_scores["hit"]) / len(exact_scores["hit"])
    f1_score = sum(f1_scores["f1"]) / len(f1_scores["f1"])
    precision = sum(f1_scores["precision"]) / len(f1_scores["precision"])
    recall = sum(f1_scores["recall"]) / len(f1_scores["recall"])

    bleu_score = sum(bleu_scores["bleu_score"]) / len(bleu_scores["bleu_score"])
    bleu_score_1 = sum(bleu_scores["bleu_score_1"]) / len(bleu_scores["bleu_score_1"])
    bleu_score_2 = sum(bleu_scores["bleu_score_2"]) / len(bleu_scores["bleu_score_2"])
    bleu_score_3 = sum(bleu_scores["bleu_score_3"]) / len(bleu_scores["bleu_score_3"])
    bleu_score_4 = sum(bleu_scores["bleu_score_4"]) / len(bleu_scores["bleu_score_4"])
    rouge_l_score = sum(rouge_l_scores["rouge_l"]) / len(rouge_l_scores["rouge_l"])
    meteor_ss = sum(meteor_scores["meteor"]) / len(meteor_scores["meteor"])

    # Return the evaluation results
    return {
        "file_name": file_name,
        "exact_match_score": exact_score * 100,
        "f1_score": f1_score * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "bleu_score": bleu_score * 100,
        "bleu_score_1": bleu_score_1 * 100,
        "bleu_score_2": bleu_score_2 * 100,
        "bleu_score_3": bleu_score_3 * 100,
        "bleu_score_4": bleu_score_4 * 100,
        "rouge_l": rouge_l_score * 100,
        "meteor": meteor_ss * 100,
    }


if __name__ == "__main__":
    args = get_args()
    gt = load_jsonl(args.base_dir)

    
    jsonl_files = {
        ##
        "answer-file_dataset_test_report":{
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
        print(f"Evaluating result file: {result_file}")
        print(f"num_gt_ids: {len(gt_ids)} || num_pred_ids: {len(pred_ids)}")

        result = evaluate(gt, pred, file_name)
        