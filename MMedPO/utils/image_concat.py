#!/usr/bin/env python3
"""
图片拼接脚本：将处理后的遮挡图片与原始图片水平拼接
左边是遮挡图片，右边是原图
"""

import os
import sys
from PIL import Image
import argparse
from pathlib import Path

def concat_images(mask_img_path, original_img_path, output_path):
    """
    将两张图片水平拼接
    
    Args:
        mask_img_path: 遮挡图片路径
        original_img_path: 原始图片路径  
        output_path: 输出图片路径
    """
    try:
        # 打开两张图片
        mask_img = Image.open(mask_img_path)
        original_img = Image.open(original_img_path)
        
        # 获取图片尺寸
        mask_width, mask_height = mask_img.size
        orig_width, orig_height = original_img.size
        
        # 统一高度（取较小的高度）
        target_height = min(mask_height, orig_height)
        
        # 按比例调整宽度
        mask_ratio = target_height / mask_height
        orig_ratio = target_height / orig_height
        
        new_mask_width = int(mask_width * mask_ratio)
        new_orig_width = int(orig_width * orig_ratio)
        
        # 调整图片大小
        mask_img_resized = mask_img.resize((new_mask_width, target_height), Image.Resampling.LANCZOS)
        orig_img_resized = original_img.resize((new_orig_width, target_height), Image.Resampling.LANCZOS)
        
        # 创建新的拼接图片
        total_width = new_mask_width + new_orig_width
        concat_img = Image.new('RGB', (total_width, target_height))
        
        # 粘贴图片：左边是遮挡图片，右边是原图
        concat_img.paste(mask_img_resized, (0, 0))
        concat_img.paste(orig_img_resized, (new_mask_width, 0))
        
        # 保存拼接后的图片
        concat_img.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"Error processing {mask_img_path} and {original_img_path}: {e}")
        return False

def process_all_folders(processed_imgs_dir, imgs_dir):
    """
    批量处理所有xmlab文件夹
    
    Args:
        processed_imgs_dir: processed_imgs目录路径
        imgs_dir: imgs目录路径
    """
    processed_count = 0
    error_count = 0
    
    # 获取所有xmlab文件夹
    xmlab_folders = [f for f in os.listdir(processed_imgs_dir) if f.startswith('xmlab')]
    xmlab_folders.sort()
    
    print(f"Found {len(xmlab_folders)} xmlab folders to process...")
    
    for folder_name in xmlab_folders:
        processed_folder = os.path.join(processed_imgs_dir, folder_name)
        imgs_folder = os.path.join(imgs_dir, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(imgs_folder):
            print(f"Warning: {imgs_folder} does not exist, skipping...")
            error_count += 1
            continue
            
        # 定义文件路径
        mask_img_path = os.path.join(processed_folder, "source_reversed_mask.jpg")
        original_img_path = os.path.join(imgs_folder, "source.jpg")
        output_path = os.path.join(imgs_folder, "source_mask_plus_full.jpg")
        
        # 检查输入文件是否存在
        if not os.path.exists(mask_img_path):
            print(f"Warning: {mask_img_path} does not exist, skipping...")
            error_count += 1
            continue
            
        if not os.path.exists(original_img_path):
            print(f"Warning: {original_img_path} does not exist, skipping...")
            error_count += 1
            continue
        
        # 处理图片拼接
        if concat_images(mask_img_path, original_img_path, output_path):
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} folders...")
        else:
            error_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} folders")
    print(f"Errors encountered: {error_count} folders")
    
    return processed_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Concatenate mask and original images horizontally')
    parser.add_argument('--processed_imgs_dir', 
                       default='/workspace/MMedPO/datasets/SLAKE/processed_imgs',
                       help='Path to processed_imgs directory')
    parser.add_argument('--imgs_dir',
                       default='/workspace/MMedPO/datasets/SLAKE/imgs', 
                       help='Path to imgs directory')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.processed_imgs_dir):
        print(f"Error: {args.processed_imgs_dir} does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.imgs_dir):
        print(f"Error: {args.imgs_dir} does not exist")
        sys.exit(1)
    
    # 开始处理
    process_all_folders(args.processed_imgs_dir, args.imgs_dir)

if __name__ == "__main__":
    main()