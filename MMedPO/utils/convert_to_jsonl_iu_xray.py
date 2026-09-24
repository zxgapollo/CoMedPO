import json
import argparse

def convert_json_to_jsonl(input_path, output_path, data_key):
    """
    Reads a JSON file containing a list of records under a specific key,
    and writes them to a JSON Lines (.jsonl) file.

    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path for the output .jsonl file.
        data_key (str): The key in the input JSON that holds the list of records.
    """
    try:
        print(f"[*] Reading data from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # Load the entire JSON object from the file
            data = json.load(f_in)

        # Check if the specified key exists and is a list
        if data_key not in data:
            print(f"[!] Error: The key '{data_key}' was not found in the JSON file.")
            return
        if not isinstance(data[data_key], list):
            print(f"[!] Error: The value under the key '{data_key}' is not a list.")
            return
            
        records = data[data_key]
        num_records = len(records)
        print(f"[*] Found {num_records} records under the key '{data_key}'.")

        print(f"[*] Writing to new JSONL file: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for record in records:
                # Convert each dictionary (record) to a JSON string
                json_string = json.dumps(record)
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
        description="Convert a standard JSON file (with a list of records under a key) to a JSON Lines (.jsonl) file."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input JSON file (e.g., datasets/iu_xray/annotation.json)"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the new output JSONL file (e.g., datasets/iu_xray/annotation.jsonl)"
    )
    parser.add_argument(
        "--key",
        default="train",
        help="The key in the JSON file that contains the list of data records (default: 'train')"
    )
    
    args = parser.parse_args()
    
    # Run the conversion function
    convert_json_to_jsonl(args.input_file, args.output_file, args.key)