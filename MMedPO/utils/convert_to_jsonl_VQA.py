import json
import argparse
from tqdm import tqdm

def convert_vqa_to_report_format(input_path, output_path):
    """
    Reads a JSON file containing a list of VQA records, maps the fields
    to a report-generation format, and writes them to a JSON Lines file.

    Args:
        input_path (str): The path to the input JSON file (a list of records).
        output_path (str): The path for the output .jsonl file.
    """
    try:
        print(f"[*] Reading data from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # Load the entire JSON object, which should be a list
            input_data = json.load(f_in)

        # Check if the data is a list
        if not isinstance(input_data, list):
            print(f"[!] Error: The input JSON file is not a direct list of records. Please check the format.")
            return
            
        num_records = len(input_data)
        print(f"[*] Found {num_records} records in the input file.")

        print(f"[*] Writing to new JSONL file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Using tqdm for a progress bar, which is helpful for large files
            for record in tqdm(input_data, desc="Converting records"):
                # Create a new dictionary with the desired structure and field names
                # Use .get() for safety in case a key is missing in some records
                new_record = {
                    # Map 'qid' to 'id'
                    "id": record.get("qid"),
                    
                    # Map 'image_name' to 'image_path' and ensure it's a list,
                    # as the inference script likely expects a list of images.
                    "image_path": [record.get("image_name")],
                    
                    # NOTE: Mapping the 'answer' field to 'report' as a placeholder
                    # for the ground-truth text your inference script will be compared against.
                    "report": record.get("answer"),
                    
                    # It's good practice to keep the original question for context
                    "question": record.get("question")
                }

                # Convert the new dictionary to a JSON string
                json_string = json.dumps(new_record)
                
                # Write the string to the file, followed by a newline character
                f_out.write(json_string + '\n')
        
        print(f"\n[+] Success! Converted {num_records} records to {output_path}")

    except FileNotFoundError:
        print(f"[!] Error: The input file was not found at {input_path}")
    except json.JSONDecodeError:
        print(f"[!] Error: Failed to decode the JSON from {input_path}. Please check if it's a valid JSON file.")
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up argument parser to make the script reusable
    parser = argparse.ArgumentParser(
        description="Convert a VQA-style JSON file (a list of records) to a JSON Lines (.jsonl) file suitable for report-generation inference."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input JSON file (e.g., datasets/vqa_data/dataset.json)"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the new output JSONL file (e.g., datasets/vqa_data/annotation.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Run the conversion function
    convert_vqa_to_report_format(args.input_file, args.output_file)