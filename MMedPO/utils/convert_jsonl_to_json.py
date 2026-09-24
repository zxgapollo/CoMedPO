#!/usr/bin/env python3
"""
将JSONL格式的DPO数据集转换为JSON数组格式
参考tie_results_with_dpo_weight.json的格式结构
"""

import json
import os
from typing import List, Dict, Any

def convert_jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    """
    将JSONL格式文件转换为JSON数组格式
    
    Args:
        jsonl_path: 输入的JSONL文件路径
        json_path: 输出的JSON文件路径
    """
    print(f"正在读取JSONL文件: {jsonl_path}")
    
    # 读取JSONL文件
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
    
    print(f"成功读取 {len(data_list)} 条记录")
    
    # 写入JSON数组格式
    print(f"正在写入JSON文件: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！输出文件: {json_path}")
    print(f"数据条数: {len(data_list)}")
    
    # 验证输出文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"验证成功: 输出文件包含 {len(loaded_data)} 条记录")
        
        # 显示第一条记录的结构
        if loaded_data:
            print("\n第一条记录的结构:")
            first_record = loaded_data[0]
            for key, value in first_record.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"验证失败: {e}")

def main():
    # 定义文件路径
    input_jsonl = "/workspace/MMedPO/outputs/tie_dpo_dataset_improved.jsonl"
    output_json = "/workspace/MMedPO/outputs/tie_dpo_dataset_improved.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_jsonl):
        print(f"错误: 输入文件不存在: {input_jsonl}")
        return
    
    # 转换格式
    convert_jsonl_to_json(input_jsonl, output_json)
    
    # 同时转换原始数据集（如果存在）
    original_jsonl = "/workspace/MMedPO/data/tie_dpo_dataset.jsonl"
    if os.path.exists(original_jsonl):
        print(f"\n同时转换原始数据集...")
        original_json = "/workspace/MMedPO/data/tie_dpo_dataset.json"
        convert_jsonl_to_json(original_jsonl, original_json)

if __name__ == "__main__":
    main()