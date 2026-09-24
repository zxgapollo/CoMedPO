import json
import os

def create_complete_dpo_dataset():
    """
    合并vqa-rad_dpo_weighted.json和转换后的slake_format数据，
    创建包含完整2247条数据的DPO格式文件
    """
    print("=== 创建完整的DPO数据集 ===\n")
    
    # 读取原始DPO数据
    dpo_weighted_file = "/workspace/MMedPO/data/vqa-rad_dpo_weighted.json"
    print(f"读取原始DPO数据: {dpo_weighted_file}")
    
    with open(dpo_weighted_file, 'r', encoding='utf-8') as f:
        dpo_weighted_data = json.load(f)
    
    print(f"原始DPO数据条目数: {len(dpo_weighted_data)}")
    
    # 读取转换后的slake数据
    slake_dpo_file = "/workspace/MMedPO/data/vqa-rad_slake_format_dpo.json"
    print(f"读取转换后的slake数据: {slake_dpo_file}")
    
    with open(slake_dpo_file, 'r', encoding='utf-8') as f:
        slake_dpo_data = json.load(f)
    
    print(f"转换后的slake数据条目数: {len(slake_dpo_data)}")
    
    # 创建ID到数据的映射
    dpo_weighted_map = {item['id']: item for item in dpo_weighted_data}
    slake_dpo_map = {item['id']: item for item in slake_dpo_data}
    
    print(f"\n原始DPO数据ID范围: {min(dpo_weighted_map.keys())} - {max(dpo_weighted_map.keys())}")
    print(f"转换后slake数据ID范围: {min(slake_dpo_map.keys())} - {max(slake_dpo_map.keys())}")
    
    # 合并数据，优先使用原始DPO数据，缺失的用slake数据补充
    complete_data = []
    stats = {
        'from_dpo_weighted': 0,
        'from_slake': 0,
        'total': 0
    }
    
    # 确定完整的ID范围
    all_ids = set(dpo_weighted_map.keys()) | set(slake_dpo_map.keys())
    max_id = max(all_ids)
    min_id = min(all_ids)
    
    print(f"\n合并后ID范围: {min_id} - {max_id}")
    print(f"预期总数据量: {max_id - min_id + 1}")
    
    # 按ID顺序合并数据
    for id_val in range(min_id, max_id + 1):
        if id_val in dpo_weighted_map:
            # 优先使用原始DPO数据
            complete_data.append(dpo_weighted_map[id_val])
            stats['from_dpo_weighted'] += 1
        elif id_val in slake_dpo_map:
            # 使用转换后的slake数据补充
            complete_data.append(slake_dpo_map[id_val])
            stats['from_slake'] += 1
        else:
            print(f"警告: ID {id_val} 在两个数据源中都不存在")
    
    stats['total'] = len(complete_data)
    
    # 保存完整的数据集
    output_file = "/workspace/MMedPO/data/vqa-rad_complete_dpo.json"
    print(f"\n保存完整数据集到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n=== 合并统计 ===")
    print(f"来自原始DPO数据: {stats['from_dpo_weighted']}")
    print(f"来自转换slake数据: {stats['from_slake']}")
    print(f"总数据量: {stats['total']}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 验证数据完整性
    print(f"\n=== 验证数据完整性 ===")
    ids = [item['id'] for item in complete_data]
    print(f"实际ID数量: {len(ids)}")
    print(f"唯一ID数量: {len(set(ids))}")
    
    if len(ids) == len(set(ids)):
        print("✓ 无重复ID")
    else:
        print("⚠ 存在重复ID")
    
    # 检查ID连续性
    expected_ids = set(range(min(ids), max(ids) + 1))
    actual_ids = set(ids)
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"⚠ 缺失ID数量: {len(missing_ids)}")
        print(f"缺失ID: {sorted(list(missing_ids))}")
    else:
        print("✓ ID完全连续")
    
    # 验证数据格式
    print(f"\n=== 验证数据格式 ===")
    format_valid = True
    required_fields = ['id', 'image', 'conversations', 'rejected_conversations', 'weighted_score']
    
    for i, item in enumerate(complete_data[:10]):  # 检查前10条
        for field in required_fields:
            if field not in item:
                print(f"⚠ 第{i}条数据缺少字段: {field}")
                format_valid = False
                break
    
    if format_valid:
        print("✓ 数据格式验证通过")
    
    # 显示示例数据
    if complete_data:
        print(f"\n=== 完整数据集示例 ===")
        import pprint
        pprint.pprint(complete_data[0])
    
    return output_file, stats

if __name__ == "__main__":
    output_file, stats = create_complete_dpo_dataset()
    print(f"\n✓ 完整DPO数据集创建完成！")
    print(f"✓ 输出文件: {output_file}")
    print(f"✓ 总数据量: {stats['total']} 条")
