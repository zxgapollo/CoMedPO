import json
from PIL import Image, ImageDraw
import argparse
import os

def reverse_lesion_mask(original_image_path, detection_file_path, output_image_path):
    """
    Masks lesions on an image using bounding boxes from a JSON file.
    Assumes JSON format: [{"Organ": [x, y, width, height]}, ...]
    """
    try:
        # Load the source image and detection data
        image = Image.open(original_image_path).convert("RGB")
        with open(detection_file_path, 'r') as f:
            detections = json.load(f)
    except Exception as e:
        print(f"  - ERROR: Could not load files for '{original_image_path}'. Skipping. Reason: {e}")
        return

    # Create a context to draw on the image
    draw = ImageDraw.Draw(image)

    if not isinstance(detections, list):
        print(f"  - WARNING: JSON file '{detection_file_path}' does not contain a list. Skipping.")
        return
        
    # Process each detection in the file
    for detection_dict in detections:
        if not (isinstance(detection_dict, dict) and len(detection_dict) == 1):
            continue

        organ_name, coords_xywh = list(detection_dict.items())[0]

        if not (isinstance(coords_xywh, list) and len(coords_xywh) == 4):
            continue
        
        # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
        x, y, w, h = coords_xywh
        final_box = [x, y, x + w, y + h]
        
        # Draw a filled black rectangle over the area
        draw.rectangle(final_box, outline="black", fill="black")

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_image_path)
        os.makedirs(output_dir, exist_ok=True)
            
        # Save the final image
        image.save(output_image_path)
        
    except Exception as e:
        print(f"  - ERROR: Could not save output image '{output_image_path}'. Reason: {e}")

def process_image_folders(input_folder, output_folder):
    """
    Traverses a directory of specifically named subfolders (e.g., xmlab91),
    finds 'source.jpg' and 'detection.json', and processes them.
    
    Args:
        input_folder (str): The root folder (e.g., './imgs').
        output_folder (str): The root folder for saving results.
    """
    print(f"Starting batch processing for specific structure...")
    print(f"Input Directory: {input_folder}")
    print(f"Output Directory: {output_folder}")
    print("-" * 40)

    # Check if the input directory exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.")
        return

    processed_count = 0
    skipped_count = 0
    
    # List all subdirectories in the input folder
    subfolders = [f.name for f in os.scandir(input_folder) if f.is_dir()]

    for folder_name in subfolders:
        current_folder_path = os.path.join(input_folder, folder_name)
        
        # Define the exact file paths we expect to find
        image_path = os.path.join(current_folder_path, 'source.jpg')
        json_path = os.path.join(current_folder_path, 'detection.json')
        
        print(f"Checking folder: '{current_folder_path}'")
        
        # Check if both required files exist
        if os.path.exists(image_path) and os.path.exists(json_path):
            print(f"  - Found 'source.jpg' and 'detection.json'. Processing...")

            # Define the output path, preserving the subfolder structure
            output_subfolder = os.path.join(output_folder, folder_name)
            output_image_path = os.path.join(output_subfolder, 'source_reversed_mask.jpg')
            
            # Process the files
            reverse_lesion_mask(image_path, json_path, output_image_path)
            
            print(f"  - Success! Saved to: {output_image_path}")
            processed_count += 1
        else:
            # If one or both files are missing, skip this folder
            print(f"  - WARNING: Skipping. Missing 'source.jpg' or 'detection.json'.")
            skipped_count += 1
            
    print("-" * 40)
    print("Batch processing complete.")
    print(f"Successfully processed: {processed_count} folders.")
    print(f"Skipped: {skipped_count} folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process directories with a fixed file structure ('source.jpg', 'detection.json')."
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to the root directory containing subfolders (e.g., './imgs')."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the directory where processed images will be saved, maintaining folder structure."
    )
    
    args = parser.parse_args()
    
    process_image_folders(args.input_folder, args.output_folder)