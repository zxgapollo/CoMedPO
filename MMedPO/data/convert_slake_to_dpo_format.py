import json
import os
from pathlib import Path

def convert_slake_to_dpo_format():
    """
    将vqa-rad_slake_format.json转换为推理脚本期望的DPO格式
    """
    print("=== 转换slake_format到DPO格式 ===\n")
    
    # 读取slake_format数据
    slake_file = "/workspace/MMedPO/data/vqa-rad_slake_format.json"
    print(f"读取源文件: {slake_file}")
    
    with open(slake_file, 'r', encoding='utf-8') as f:
        slake_data = json.load(f)
    
    print(f"源数据条目数: {len(slake_data)}")
    
    # 转换为DPO格式
    dpo_data = []
    conversion_stats = {
        'total': len(slake_data),
        'converted': 0,
        'skipped': 0,
        'missing_positive': 0,
        'missing_negative': 0,
        'missing_image': 0
    }
    
    for item in slake_data:
        try:
            # 检查必需字段
            if not all(key in item for key in ['qid', 'question', 'positive_answer', 'negative_answer']):
                conversion_stats['skipped'] += 1
                continue
            
            # 检查答案是否存在
            if not item.get('has_positive', False) or not item['positive_answer']:
                conversion_stats['missing_positive'] += 1
                continue
                
            if not item.get('has_negative', False) or not item['negative_answer']:
                conversion_stats['missing_negative'] += 1
                continue
            
            # 提取图像路径
            full_image_path = item.get('full_image_path', '')
            if not full_image_path:
                conversion_stats['missing_image'] += 1
                continue
            
            # 从完整路径中提取图像文件名
            image_filename = os.path.basename(full_image_path)
            
            # 构建DPO格式的数据
            dpo_item = {
                "id": item['qid'],
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": item['positive_answer']
                    }
                ],
                "rejected_conversations": [
                    {
                        "from": "human", 
                        "value": f"<image>\n{item['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": item['negative_answer']
                    }
                ],
                "weighted_score": item.get('weighted_score', 1.0)
            }
            
            dpo_data.append(dpo_item)
            conversion_stats['converted'] += 1
            
        except Exception as e:
            print(f"转换第 {item.get('qid', 'unknown')} 条数据时出错: {e}")
            conversion_stats['skipped'] += 1
    
    # 保存转换后的数据
    output_file = "/workspace/MMedPO/data/vqa-rad_slake_format_dpo.json"
    print(f"\n保存转换后的数据到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    # 打印转换统计
    print(f"\n=== 转换统计 ===")
    print(f"总条目数: {conversion_stats['total']}")
    print(f"成功转换: {conversion_stats['converted']}")
    print(f"跳过条目: {conversion_stats['skipped']}")
    print(f"缺少正例答案: {conversion_stats['missing_positive']}")
    print(f"缺少反例答案: {conversion_stats['missing_negative']}")
    print(f"缺少图像路径: {conversion_stats['missing_image']}")
    
    # 验证转换结果
    print(f"\n=== 验证转换结果 ===")
    print(f"输出文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 检查ID范围
    ids = [item['id'] for item in dpo_data]
    if ids:
        print(f"ID范围: {min(ids)} - {max(ids)}")
        print(f"唯一ID数量: {len(set(ids))}")
        
        # 检查缺失的ID
        expected_ids = set(range(min(ids), max(ids) + 1))
        actual_ids = set(ids)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"缺失ID数量: {len(missing_ids)}")
            print(f"前20个缺失ID: {sorted(list(missing_ids))[:20]}")
        else:
            print("✓ 所有ID连续，无缺失")
    
    # 显示转换后的示例
    if dpo_data:
        print(f"\n=== 转换后数据示例 ===")
        import pprint
        pprint.pprint(dpo_data[0])
    
    return output_file, conversion_stats

if __name__ == "__main__":
    output_file, stats = convert_slake_to_dpo_format()
    print(f"\n✓ 转换完成！输出文件: {output_file}")
    print(f"✓ 成功转换 {stats['converted']} 条数据")
