import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import sys
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import warnings
import numpy as np

# --- Original Python Path Setup ---
llava_code_path = '/workspace/MMedPO/train/dpo'
if llava_code_path not in sys.path:
    sys.path.insert(0, llava_code_path)
import llava.model.language_model.llava_mistral
# --- End Path Setup ---

warnings.simplefilter(action="ignore", category=FutureWarning)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
)
from PIL import Image
import math
from transformers import set_seed, logging
from utils import QuestionDataset, setup, cleanup, tensor_to_serializable

logging.set_verbosity_error()

def stitch_images_side_by_side(image1, image2):
    """
    Stitches two PIL Images together horizontally.
    """
    if image1.height != image2.height:
        new_height = max(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
        image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))

    width1, height1 = image1.size
    width2, height2 = image2.size
    combined_image = Image.new('RGB', (width1 + width2, height1))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))
    return combined_image

def build_prompt_text(question: str) -> str:
    return (
        f"The following image displays a full medical scan on the left and a version with a masked background on the right. "
        f"Based on this composite image, answer the following question.\n"
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}"
    )

def get_log_likelihood_from_scores(scores, generated_tokens, input_length):
    """
    Calculate log likelihood from generation scores and actual generated tokens.
    """
    if scores is None or len(scores) == 0:
        return 0.0
    
    total_log_likelihood = 0.0
    generated_length = generated_tokens.shape[1] - input_length
    
    if generated_length <= 0:
        return 0.0
    
    for i in range(generated_length):
        if i < len(scores):
            # Get the log probabilities for this generation step
            log_probs = torch.log_softmax(scores[i], dim=-1)
            # Get the actual token that was generated
            actual_token = generated_tokens[0, input_length + i].item()
            # Get the log probability of that specific token
            token_log_prob = log_probs[0, actual_token].item()
            total_log_likelihood += token_log_prob
    
    return total_log_likelihood

def compute_log_likelihood_teacher_forcing(model, sequences, input_length, image_tensor, device):
    """
    Compute sequence log-likelihood by teacher forcing over the generated span only.
    - sequences: tensor of shape [1, total_len] that includes prompt + generated tokens
    - input_length: length of the prompt tokens in sequences
    Returns the SUM log-likelihood (float) over generated tokens.
    """
    if sequences is None or sequences.shape[1] <= input_length:
        print(f"LL Debug: sequences is None or total_len (={0 if sequences is None else sequences.shape[1]}) <= input_len (={input_length}). Returning 0.")
        return 0.0

    with torch.inference_mode():
        # Inputs exclude the last token; labels exclude the first token
        inputs = sequences[:, :-1].to(device)
        labels = sequences[:, 1:].clone().to(device)

        # Mask out the prompt portion so loss is only computed on generated tokens
        # Prompt covers positions [0 .. input_length-1] in sequences; in shifted labels these map to [0 .. input_length-2]
        if input_length > 1:
            labels[:, : input_length - 1] = -100
        total_len = sequences.shape[1]
        gen_len = total_len - input_length
        print(f"LL Debug: total_len={total_len}, input_len={input_length}, generated_len={gen_len}")
        print(f"LL Debug: inputs shape={inputs.shape}, labels shape={labels.shape}")

        outputs = model(
            input_ids=inputs,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device),
        )

        logits = outputs.logits  # [1, seq_len-1, vocab]
        vocab_size = logits.size(-1)

        # CrossEntropy over tokens (ignore masked labels)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1)
        ).view(labels.size())  # [1, seq_len-1]

        valid_mask = (labels != -100)
        valid_count = int(valid_mask.sum().item())
        print(f"LL Debug: valid generated tokens={valid_count}")
        if valid_mask.any():
            nll_tensor = per_token_loss[valid_mask]
            nll_sum = nll_tensor.sum().item()
            avg_nll = nll_tensor.mean().item()
            print(f"LL Debug: nll_sum={nll_sum:.6f}, avg_nll={avg_nll:.6f}")
            # Show first few generated token ids and their per-token loss
            # Recover generated token ids from labels where valid
            gen_token_ids = labels[valid_mask].view(-1).tolist()
            preview_k = min(5, len(gen_token_ids))
            if preview_k > 0:
                print(f"LL Debug: first {preview_k} generated token ids={gen_token_ids[:preview_k]}")
            return -nll_sum
        print("LL Debug: No valid generated tokens after masking. Returning 0.")
        return 0.0

def compute_ll_token_by_token(model, prompt_ids, answer_ids, image_tensor, device, log_states=False, tokenizer=None):
    """
    Compute log-likelihood of ground-truth answer by iteratively conditioning on
    prompt + already revealed answer tokens. Avoids alignment issues with vision tokens.
    Returns sum of log probabilities over answer tokens (optionally incl. EOS in answer_ids).
    """
    if answer_ids is None or answer_ids.numel() == 0:
        return 0.0
    total_log_likelihood = 0.0
    state_logs = {"prompt": None, "steps": []} if log_states else None
    with torch.inference_mode():
        for i in range(answer_ids.shape[1]):
            prefix = torch.cat([prompt_ids, answer_ids[:, :i]], dim=1).to(device)
            outputs = model(
                input_ids=prefix,
                images=image_tensor.unsqueeze(0).to(device, dtype=model.dtype),
                use_cache=True,
                output_hidden_states=log_states,
                return_dict=True,
            )
            logits = outputs.logits  # [1, prefix_len, vocab]
            next_logits = logits[:, -1, :]  # last position
            log_probs = torch.log_softmax(next_logits, dim=-1)
            token_id = answer_ids[0, i].item()
            total_log_likelihood += log_probs[0, token_id].item()

            if log_states:
                hidden_norm = None
                if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                    hidden_norm = float(torch.linalg.norm(last_hidden).item())
                topk_vals, topk_ids = torch.topk(log_probs, k=min(5, log_probs.shape[-1]), dim=-1)
                top_list = []
                for j in range(topk_ids.shape[-1]):
                    tid = int(topk_ids[0, j].item())
                    tstr = tokenizer.decode([tid]) if tokenizer is not None else ""
                    top_list.append({"id": tid, "text": tstr, "logprob": float(topk_vals[0, j].item())})
                # Save prompt state for i==0 (before adding any answer token)
                if i == 0:
                    state_logs["prompt"] = {
                        "hidden_norm": hidden_norm,
                        "topk": top_list,
                    }
                state_logs["steps"].append({
                    "step": i,
                    "target_token_id": int(token_id),
                    "target_token_text": tokenizer.decode([token_id]) if tokenizer is not None else "",
                    "hidden_norm": hidden_norm,
                    "topk": top_list,
                })
    return (total_log_likelihood, state_logs) if log_states else total_log_likelihood

def run_inference_with_image(model, tokenizer, image_processor, composite_image, prompt_text, target_answer, args, device):
    """
    Run inference on a composite image and return both text output and log likelihood.
    """
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    if image_processor is not None:
        image_tensor = process_images([composite_image], image_processor, model.config)[0]
    else:
        # Fallback: try to process image without image_processor
        try:
            image_tensor = process_images([composite_image], None, model.config)[0]
        except Exception as e:
            print(f"Warning: Failed to process image: {e}")
            # Create a dummy image tensor as fallback
            image_tensor = torch.zeros((3, 224, 224))  # Dummy tensor
    
    # Ensure image tensor uses bfloat16 precision to match model
    image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        # Generate deterministically to stabilize LL computation
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = eos_id
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device),
            do_sample=False,
            max_new_tokens=getattr(args, 'max_new_tokens', 64),
            min_new_tokens=getattr(args, 'min_new_tokens', 1),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,
            output_scores=False,
        )

    # Prepare ground-truth target sequence: prompt + ground-truth answer
    answer_text = str(target_answer).strip() if target_answer is not None else ""
    if answer_text == "":
        # If no ground truth, we cannot compute LL; still decode model output for record
        try:
            outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
        except Exception:
            outputs = ""
        if getattr(args, 'verbose', False):
            print("LL Debug: Empty ground-truth answer; returning LL=0.")
        input_length = input_ids.shape[1]
        return outputs, 0.0

    # Tokenize target answer without adding extra specials to align with CE
    target_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"].to(device)

    # Optionally append EOS if tokenizer has it
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is not None:
        eos_tensor = torch.tensor([[eos_id]], device=device)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)

    # Build full sequence variants for two LL methods
    full_sequence = torch.cat([input_ids, target_ids], dim=1)

    # Decode a preview of the model's own generated text (optional)
    try:
        gen_only_text = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
    except Exception:
        gen_only_text = ""
    outputs = gen_only_text
    
    # Debug information
    if getattr(args, 'verbose', False):
        print(f"Gen Debug: Generated text preview: {outputs[:120]}...")
        print(f"LL Debug: Input length (prompt tokens): {input_ids.shape[1]}")
        print(f"LL Debug: Target answer length (tokens): {target_ids.shape[1]}")
        print(f"LL Debug: Full sequence length (tokens): {full_sequence.shape[1]}")
    
    # Prefer robust token-by-token LL to avoid vision-token alignment issues
    input_length = input_ids.shape[1]
    try:
        log_likelihood = compute_ll_token_by_token(
            model=model,
            prompt_ids=input_ids,
            answer_ids=target_ids,
            image_tensor=image_tensor,
            device=device,
            log_states=getattr(args, 'log_states', False),
            tokenizer=tokenizer,
        )
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"LL Debug: token-by-token failed: {e}. Falling back to TF method.")
        log_likelihood = compute_log_likelihood_teacher_forcing(
            model=model,
            sequences=full_sequence,
            input_length=input_length,
            image_tensor=image_tensor,
            device=device,
        )
    
    return outputs, log_likelihood

def eval_model(args):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} started")

    set_seed(0)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=f"cuda:{rank}"
    )
    
    # Ensure all model components use consistent precision (bfloat16)
    device = f"cuda:{rank}"
    
    # Force all model parameters to bfloat16 recursively
    def convert_to_bfloat16(module):
        for param in module.parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        for buffer in module.buffers():
            if buffer.dtype != torch.bfloat16 and buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(torch.bfloat16)
        for child in module.children():
            convert_to_bfloat16(child)
    
    # Apply recursive conversion
    convert_to_bfloat16(model)
    
    # Ensure vision tower is properly loaded after type conversion
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print(f"Rank {rank}: Vision tower not loaded, loading now...")
            vision_tower.load_model(device_map=device)
        elif vision_tower is not None and hasattr(vision_tower, 'vision_tower') and vision_tower.vision_tower is None:
            print(f"Rank {rank}: Vision tower attribute is None, reloading...")
            vision_tower.load_model(device_map=device)
    model = model.to(device)
    
    print(f"Rank {rank}: All model parameters forcibly converted to bfloat16")

    if image_processor is None:
        print(f"Rank {rank}: Image processor was not loaded...")
        try:
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
            if hasattr(model, 'get_vision_tower'):
                 model.get_vision_tower().image_processor = image_processor
        except Exception as e:
            print(f"Rank {rank}: Failed to load CLIP image processor: {e}")
            print(f"Rank {rank}: Continuing without image processor...")
            # Create a dummy image processor or use the model's default
            image_processor = None

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    
    dataset = QuestionDataset(questions)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=16, pin_memory=True)
    local_results = []

    processed = 0
    limit = getattr(args, 'limit', None)
    for line in tqdm(dataloader, position=0, file=sys.stdout):
        idx = line.get("qid")[0]
        gt_answer = line.get("answer", [None])[0] 
        answer_type = line.get("answer_type", ["unknown"])[0]
        original_question = line.get("question", [""])[0].strip()
        full_image_path = line.get("full_image_path")[0]
        masked_image_path = line.get("masked_image_path")[0]

        try:
            full_image = Image.open(full_image_path).convert('RGB')
            masked_image = Image.open(masked_image_path).convert('RGB')
            # Create white background image
            white_background = Image.new('RGB', full_image.size, color='white')
        except FileNotFoundError as e:
            tqdm.write(f"Warning: Could not find required image. Skipping. Reason: {e}")
            continue

        # Create composite images
        composite_with_mask = stitch_images_side_by_side(full_image, masked_image)
        composite_with_white = stitch_images_side_by_side(full_image, white_background)

        # Build a single prompt text to ensure strict consistency between both runs
        prompt_text = build_prompt_text(original_question)

        # Run inference on both composite images
        try:
            # Inference with masked background
            output_with_mask, log_likelihood_with_mask = run_inference_with_image(
                model, tokenizer, image_processor, composite_with_mask, prompt_text, gt_answer, args, rank
            )
            
            # Inference with white background
            output_with_white, log_likelihood_with_white = run_inference_with_image(
                model, tokenizer, image_processor, composite_with_white, prompt_text, gt_answer, args, rank
            )
            
            # Calculate difference
            log_likelihood_difference = log_likelihood_with_mask - log_likelihood_with_white
            
        except Exception as e:
            tqdm.write(f"Warning: Error during inference for {idx}. Reason: {e}")
            output_with_mask = "ERROR"
            output_with_white = "ERROR"
            log_likelihood_with_mask = 0.0
            log_likelihood_with_white = 0.0
            log_likelihood_difference = 0.0

        # Save composite images if requested
        if args.output_image_folder and rank == 0:
            try:
                case_id = os.path.basename(os.path.dirname(full_image_path))
                base_name = os.path.splitext(os.path.basename(full_image_path))[0]
                
                # Save masked composite
                output_dir = os.path.join(args.output_image_folder, case_id)
                os.makedirs(output_dir, exist_ok=True)
                masked_output_path = os.path.join(output_dir, f"{base_name}_with_mask.jpg")
                composite_with_mask.save(masked_output_path)
                
                # Save white background composite
                white_output_path = os.path.join(output_dir, f"{base_name}_with_white.jpg")
                composite_with_white.save(white_output_path)
                
            except Exception as e:
                tqdm.write(f"Warning: Could not save composite images for {case_id}. Reason: {e}")

        # Create result entry
        ans_id = shortuuid.uuid()
        case_id = os.path.basename(os.path.dirname(full_image_path))
        # Determine plain values and lengths for reporting
        ll_mask_value = log_likelihood_with_mask[0] if isinstance(log_likelihood_with_mask, tuple) else log_likelihood_with_mask
        ll_white_value = log_likelihood_with_white[0] if isinstance(log_likelihood_with_white, tuple) else log_likelihood_with_white
        answer_text_for_ll = str(gt_answer).strip() if gt_answer is not None else ""
        answer_token_len = 0
        answer_char_len = len(answer_text_for_ll)
        try:
            if answer_text_for_ll != "":
                answer_token_len = int(tokenizer(
                    answer_text_for_ll,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"].shape[1])
        except Exception:
            pass

        # Compute token-avg difference
        ll_token_avg_bg = (ll_mask_value / max(1, answer_token_len)) if answer_token_len > 0 else ll_mask_value
        ll_token_avg_white = (ll_white_value / max(1, answer_token_len)) if answer_token_len > 0 else ll_white_value
        ll_token_avg_diff = ll_token_avg_bg - ll_token_avg_white

        result = {
            "id": idx,
            "case_id": case_id,
            "question": original_question,
            "gt_answer": gt_answer,
            "answer_type": answer_type,
            "output_with_background": output_with_mask,
            "output_with_white_background": output_with_white,
            "log_likelihood_with_background": ll_mask_value,
            "log_likelihood_with_white_background": ll_white_value,
            "log_likelihood_difference": (ll_mask_value - ll_white_value),
            "prompt_text": prompt_text,
            "target_answer_text": answer_text_for_ll,
            "target_answer_token_len": answer_token_len,
            "target_answer_char_len": answer_char_len,
            "ll_token_avg_with_background": ll_token_avg_bg,
            "ll_token_avg_with_white_background": ll_token_avg_white,
            "ll_token_avg_difference": ll_token_avg_diff,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {},
        }

        # Attach optional state logs if enabled and available
        if getattr(args, 'log_states', False):
            # When log_states=True, compute_ll_token_by_token returns (ll, logs)
            if isinstance(log_likelihood_with_mask, tuple):
                ll_mask, logs_mask = log_likelihood_with_mask
                result["state_logs_with_background"] = logs_mask
            if isinstance(log_likelihood_with_white, tuple):
                ll_white, logs_white = log_likelihood_with_white
                result["state_logs_with_white_background"] = logs_white
        
        serial_result = tensor_to_serializable(result)
        local_results.append(serial_result)

        processed += 1
        if limit is not None and processed >= limit:
            break

    dist.barrier()
    print(f"Rank {rank} reached gathering barrier")
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)
    print(f"Rank {rank} finished all_gather_object")

    if rank == 0:
        print(f"Rank {rank} starting to process and write results...")
        all_results = [item for sublist in gathered_results for item in sublist]
        unique_results = []
        seen_ids = set()
        for res in all_results:
            q_id = res["id"]
            if isinstance(q_id, list): q_id = q_id[0]
            if q_id not in seen_ids:
                unique_results.append(res)
                seen_ids.add(q_id)
        
        def get_sort_key(x):
            q_id = x["id"]
            return q_id[0] if isinstance(q_id, list) else q_id
        
        unique_results.sort(key=get_sort_key)
        
        # Save to CSV
        csv_file = os.path.expanduser(args.csv_file)
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # Convert to DataFrame: keep ALL fields (no reduction)
        df_full = pd.DataFrame(unique_results)

        # Ensure we write to an .xlsx file
        if csv_file.lower().endswith(".csv"):
            xlsx_file = os.path.splitext(csv_file)[0] + ".xlsx"
        else:
            xlsx_file = csv_file

        try:
            with pd.ExcelWriter(xlsx_file, engine="xlsxwriter") as writer:
                # Main sheet
                sheet_name = "scores"
                df_full.to_excel(writer, index=False, sheet_name=sheet_name)
                workbook  = writer.book
                worksheet = writer.sheets[sheet_name]

                # Apply conditional formatting: difference < 0 shown in red
                red_format = workbook.add_format({"font_color": "#9C0006"})
                if "log_likelihood_difference" in df_full.columns:
                    # Find the column index for difference (0-based)
                    diff_col_idx = df_full.columns.get_loc("log_likelihood_difference")
                    # Excel columns are letters; build range like C2:C{n}
                    start_row = 2  # 1-based Excel row index, skipping header
                    end_row = len(df_full) + 1
                    # Convert column index to Excel letter(s)
                    def col_to_excel(col_idx):
                        col_str = ""
                        col_idx += 1
                        while col_idx:
                            col_idx, remainder = divmod(col_idx - 1, 26)
                            col_str = chr(65 + remainder) + col_str
                        return col_str
                    diff_col_letter = col_to_excel(diff_col_idx)
                    cell_range = f"{diff_col_letter}{start_row}:{diff_col_letter}{end_row}"
                    worksheet.conditional_format(cell_range, {"type": "cell", "criteria": "<", "value": 0, "format": red_format})

                    # Negatives sheet
                    negatives = df_full[df_full["log_likelihood_difference"] < 0].copy()
                    negatives.to_excel(writer, index=False, sheet_name="negatives")
        except Exception:
            # Fallback to writing without formatting
            with pd.ExcelWriter(xlsx_file) as writer:
                df_full.to_excel(writer, index=False, sheet_name="scores")
                if "log_likelihood_difference" in df_full.columns:
                    df_full[df_full["log_likelihood_difference"] < 0].to_excel(writer, index=False, sheet_name="negatives")
        print(f"Rank {rank} finished writing to Excel file {xlsx_file}")
        print(f"Rank {rank} finished writing to CSV file {args.csv_file}")
        
        # Also save JSON for compatibility
        if args.answers_file:
            answers_file = os.path.expanduser(args.answers_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            with open(answers_file, "w") as ans_file:
                for res in unique_results:
                    ans_file.write(json.dumps(res) + "\n")
            print(f"Rank {rank} finished writing to JSON file {args.answers_file}")

    dist.barrier()
    print(f"Rank {rank} passed final barrier. Preparing to clean up.")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True, help="Path to the self-contained JSON file with full image paths.")
    parser.add_argument("--csv-file", type=str, required=True, help="Path to save the comparison results as CSV.")
    parser.add_argument("--answers-file", type=str, default=None, help="Optional: Path to save results as JSON.")
    parser.add_argument("--output-image-folder", type=str, default=None, help="Optional: Path to save the stitched composite images.")
    
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of samples for debugging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logs")
    parser.add_argument("--log_states", action="store_true", help="Log pre-output and step-wise states (hidden norms, top-k)")
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()
