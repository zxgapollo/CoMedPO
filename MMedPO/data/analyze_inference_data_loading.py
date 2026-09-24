import json
import os
import sys

def analyze_inference_data_loading():
    """
    分析推理脚本的数据加载逻辑，找出为什么只处理了2208条而不是2247条数据
    """
    print("=== 分析推理脚本数据加载逻辑 ===\n")
    
    # 1. 检查slake_format数据文件
    slake_file = "/workspace/MMedPO/data/vqa-rad_slake_format.json"
    print(f"1. 检查数据文件: {slake_file}")
    
    with open(slake_file, 'r', encoding='utf-8') as f:
        slake_data = json.load(f)
    
    print(f"   - 文件中总条目数: {len(slake_data)}")
    
    # 2. 分析数据结构和完整性
    print("\n2. 分析数据结构和完整性:")
    
    valid_entries = 0
    invalid_entries = []
    missing_fields = {}
    
    for i, item in enumerate(slake_data):
        # 检查必需字段
        required_fields = ['id', 'conversations']
        missing_in_item = []
        
        for field in required_fields:
            if field not in item:
                missing_in_item.append(field)
        
        if missing_in_item:
            invalid_entries.append({
                'index': i,
                'id': item.get('id', 'unknown'),
                'missing_fields': missing_in_item
            })
            for field in missing_in_item:
                missing_fields[field] = missing_fields.get(field, 0) + 1
        else:
            # 检查conversations结构
            conversations = item.get('conversations', [])
            if not isinstance(conversations, list) or len(conversations) == 0:
                invalid_entries.append({
                    'index': i,
                    'id': item.get('id', 'unknown'),
                    'issue': 'empty or invalid conversations'
                })
            else:
                # 检查conversations内容
                has_human = False
                has_gpt = False
                for conv in conversations:
                    if isinstance(conv, dict):
                        if conv.get('from') == 'human':
                            has_human = True
                        elif conv.get('from') == 'gpt':
                            has_gpt = True
                
                if not (has_human and has_gpt):
                    invalid_entries.append({
                        'index': i,
                        'id': item.get('id', 'unknown'),
                        'issue': f'incomplete conversations (human: {has_human}, gpt: {has_gpt})'
                    })
                else:
                    valid_entries += 1
    
    print(f"   - 有效条目数: {valid_entries}")
    print(f"   - 无效条目数: {len(invalid_entries)}")
    
    if missing_fields:
        print(f"   - 缺失字段统计: {missing_fields}")
    
    if invalid_entries:
        print(f"\n   前10个无效条目:")
        for entry in invalid_entries[:10]:
            print(f"     索引 {entry['index']}, ID {entry['id']}: {entry.get('missing_fields', entry.get('issue'))}")
    
    # 3. 检查图像文件路径
    print("\n3. 检查图像文件路径:")
    
    image_issues = 0
    for i, item in enumerate(slake_data):
        if 'image' in item:
            image_path = item['image']
            full_path = os.path.join("/workspace/MMedPO/datasets/Slake1.0/imgs", image_path)
            if not os.path.exists(full_path):
                image_issues += 1
                if image_issues <= 5:  # 只显示前5个
                    print(f"     缺失图像: {full_path}")
    
    print(f"   - 缺失图像文件数: {image_issues}")
    
    # 4. 分析推理脚本可能跳过的条目
    print("\n4. 分析推理脚本可能跳过的条目:")
    
    # 根据推理脚本的逻辑，可能跳过的情况：
    # - 图像文件不存在
    # - conversations格式不正确
    # - 缺少必需字段
    
    skipped_count = len(invalid_entries) + image_issues
    processed_count = len(slake_data) - skipped_count
    
    print(f"   - 预计跳过条目数: {skipped_count}")
    print(f"   - 预计处理条目数: {processed_count}")
    print(f"   - 实际处理条目数: 2208")
    print(f"   - 差异: {processed_count - 2208}")
    
    # 5. 检查ID范围和连续性
    print("\n5. 检查ID范围和连续性:")
    
    ids = []
    for item in slake_data:
        if 'id' in item:
            ids.append(item['id'])
    
    if ids:
        ids.sort()
        print(f"   - ID范围: {min(ids)} - {max(ids)}")
        print(f"   - 唯一ID数量: {len(set(ids))}")
        print(f"   - 重复ID数量: {len(ids) - len(set(ids))}")
        
        # 检查缺失的ID
        expected_ids = set(range(min(ids), max(ids) + 1))
        actual_ids = set(ids)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"   - 缺失ID数量: {len(missing_ids)}")
            print(f"   - 前20个缺失ID: {sorted(list(missing_ids))[:20]}")
    
    # 6. 总结
    print("\n=== 总结 ===")
    print(f"数据文件包含 {len(slake_data)} 条记录")
    print(f"推理脚本处理了 2208 条记录")
    print(f"差异: {len(slake_data) - 2208} 条记录")
    
    if len(slake_data) - 2208 == len(missing_ids):
        print("✓ 差异数量与缺失ID数量一致，推理脚本正确跳过了缺失ID的记录")
    else:
        print("⚠ 差异数量与缺失ID数量不一致，需要进一步调查")
    
    return {
        'total_entries': len(slake_data),
        'valid_entries': valid_entries,
        'invalid_entries': len(invalid_entries),
        'image_issues': image_issues,
        'missing_ids': len(missing_ids) if 'missing_ids' in locals() else 0,
        'processed_by_script': 2208
    }

if __name__ == "__main__":
    result = analyze_inference_data_loading()
