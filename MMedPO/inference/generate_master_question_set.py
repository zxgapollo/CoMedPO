import json
import argparse
import os
from tqdm import tqdm

def generate_master_file(full_image_root, masked_image_root, output_file, dpo_data_path=None):
    """
    Generates a master question file containing ONLY DPO data entries that have 
    both positive and negative examples. Original question.json files are used to extract answer_type.
    """
    # --- 1. Validate inputs and convert to absolute paths ---
    full_image_root = os.path.abspath(full_image_root)
    masked_image_root = os.path.abspath(masked_image_root)
    
    if not os.path.isdir(full_image_root):
        print(f"Error: Full image root directory not found at '{full_image_root}'")
        return
    if not os.path.isdir(masked_image_root):
        print(f"Error: Masked image root directory not found at '{masked_image_root}'")
        return

    master_question_list = []
    global_qid = 0
    
    # --- 1.5. Build question to answer_type mapping from original question.json files ---
    print("Building question to answer_type mapping from original SLAKE data...")
    question_to_answer_type = {}
    
    # Scan all case directories to build the mapping
    for case_dir in os.listdir(full_image_root):
        case_path = os.path.join(full_image_root, case_dir)
        if os.path.isdir(case_path):
            question_file = os.path.join(case_path, 'question.json')
            if os.path.exists(question_file):
                try:
                    with open(question_file, 'r', encoding='utf-8') as f:
                        questions = json.load(f)
                    for q in questions:
                        question_text = q.get('question', '').strip()
                        answer_type = q.get('answer_type', 'OPEN')
                        if question_text:
                            question_to_answer_type[question_text] = answer_type
                except Exception as e:
                    print(f"Warning: Could not load questions from {question_file}: {e}")
    
    print(f"Built mapping for {len(question_to_answer_type)} unique questions")
    
    # Load DPO data - this is required for this script
    dpo_data = None
    if not dpo_data_path or not os.path.exists(dpo_data_path):
        print(f"Error: DPO data file is required but not found at: {dpo_data_path}")
        print("This script only processes DPO data with both positive and negative examples.")
        return
    
    try:
        with open(dpo_data_path, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)
        print(f"Loaded DPO data with {len(dpo_data)} entries from: {dpo_data_path}")
    except Exception as e:
        print(f"Error: Could not load DPO data from {dpo_data_path}: {e}")
        return

    # --- 2. Process DPO data with BOTH positive and negative examples only ---
    print("Processing DPO conversations with paired positive and negative examples only...")
    for dpo_entry in tqdm(dpo_data, desc="Processing DPO data"):
        # Check if both conversations and rejected_conversations exist
        has_positive = "conversations" in dpo_entry and len(dpo_entry["conversations"]) >= 2
        has_negative = "rejected_conversations" in dpo_entry and len(dpo_entry["rejected_conversations"]) >= 2
        
        # Only process entries that have BOTH positive and negative examples
        if has_positive and has_negative:
            # Extract positive example
            pos_human_msg = dpo_entry["conversations"][0]
            pos_gpt_msg = dpo_entry["conversations"][1]
            
            # Extract negative example
            neg_human_msg = dpo_entry["rejected_conversations"][0]
            neg_gpt_msg = dpo_entry["rejected_conversations"][1]
            
            if (pos_human_msg.get("from") == "human" and pos_gpt_msg.get("from") == "gpt" and
                neg_human_msg.get("from") == "human" and neg_gpt_msg.get("from") == "gpt"):
                    
                    # Extract question and answers
                    pos_question = pos_human_msg.get("value", "").replace("<image>\n", "").strip()
                    pos_answer = pos_gpt_msg.get("value", "").strip()
                    neg_question = neg_human_msg.get("value", "").replace("<image>\n", "").strip()
                    neg_answer = neg_gpt_msg.get("value", "").strip()
                    
                    # Use positive question as the main question (they should be the same)
                    question_text = pos_question
                    
                    if question_text and pos_answer and neg_answer:
                        image_name = dpo_entry.get("image", "")
                        if image_name:
                            case_id = image_name.split('/')[0] if '/' in image_name else image_name.replace('.jpg', '')
                            full_image_path = os.path.join(full_image_root, case_id, 'source.jpg')
                            masked_image_path = os.path.join(masked_image_root, case_id, 'source_reversed_mask.jpg')
                            
                            if os.path.exists(full_image_path) and os.path.exists(masked_image_path):
                                # Get answer_type from mapping, default to "OPEN" if not found
                                answer_type = question_to_answer_type.get(question_text, "OPEN")
                                
                                # Create combined entry with both positive and negative examples
                                combined_entry = {
                                    "qid": global_qid,
                                    "question": question_text,
                                    "answer_type": answer_type,
                                    "full_image_path": full_image_path,
                                    "masked_image_path": masked_image_path,
                                    "weighted_score": dpo_entry.get("weighted_score", 1.0),
                                    "positive_answer": pos_answer,
                                    "negative_answer": neg_answer,
                                    "has_positive": True,
                                    "has_negative": True
                                }
                                master_question_list.append(combined_entry)
                                global_qid += 1

    # --- 6. Save the consolidated master file ---
    if not master_question_list:
        print("Warning: No questions were found. The output file will be empty.")
    
    # Count examples by type
    dpo_combined = sum(1 for item in master_question_list if "has_positive" in item and "has_negative" in item)
    
    # Count total positive and negative answers
    total_positive_answers = sum(1 for item in master_question_list if item.get("has_positive", False))
    total_negative_answers = sum(1 for item in master_question_list if item.get("has_negative", False))
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(master_question_list, f, indent=3)
        print("\n--- Aggregation Complete ---")
        print(f"Successfully generated master file with {len(master_question_list)} DPO question entries:")
        print(f"  - {dpo_combined} DPO entries with both positive and negative answers")
        print(f"  - Total positive answers: {total_positive_answers}")
        print(f"  - Total negative answers: {total_negative_answers}")
        print(f"Master question file saved to: {output_file}")
    except Exception as e:
        print(f"\nAn error occurred while saving the master file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a master question file containing ONLY DPO data entries that have both positive and negative examples. Original question.json files are ignored."
    )
    parser.add_argument("--full-image-root", required=True, help="Root directory of the original source images (e.g., '.../Slake1.0/imgs').")
    parser.add_argument("--masked-image-root", required=True, help="Root directory of the masked images (e.g., '.../Slake1.0/processed_imgs').")
    parser.add_argument("--output-file", required=True, help="Path to save the new, consolidated master question file.")
    parser.add_argument("--dpo-data-path", help="Path to DPO data file containing rejected conversations (optional).")
    
    args = parser.parse_args()
    generate_master_file(args.full_image_root, args.masked_image_root, args.output_file, args.dpo_data_path)