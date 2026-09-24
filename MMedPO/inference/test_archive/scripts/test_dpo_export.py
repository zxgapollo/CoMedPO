#!/usr/bin/env python3
"""
测试DPO pairs导出功能和输出格式
"""

import json
import os
import sys
from typing import Dict, List, Any
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_results() -> List[Dict[str, Any]]:
    """创建测试用的结果数据"""
    return [
        {
            "id": "test_1",
            "case_id": "case_001",
            "question": "What is shown in this medical image?",
            "positive_answer": "This shows a normal chest X-ray with clear lung fields.",
            "negative_answer": "This shows pneumonia with consolidation in the right lung.",
            "calculate_tie": True,
            "tie_positive": 1.25,
            "tie_negative": 0.35,
            "tie_pos_token_avg": 1.20,
            "tie_neg_token_avg": 0.30,
            "delta_pos": 1.25,
            "delta_neg": 0.35,
            "m_v": 1.10,
            "m_n": 0.05,
            "gamma": 0.90
        },
        {
            "id": "test_2", 
            "case_id": "case_002",
            "question": "Describe the pathological findings.",
            "positive_answer": "The image shows clear evidence of fracture in the femur.",
            "negative_answer": "The image appears normal with no abnormalities.",
            "calculate_tie": True,
            "tie_positive": 0.85,
            "tie_negative": -0.15,
            "tie_pos_token_avg": 0.80,
            "tie_neg_token_avg": -0.20,
            "delta_pos": 0.85,
            "delta_neg": -0.15,
            "m_v": 0.95,
            "m_n": 0.02,
            "gamma": 1.00
        },
        {
            "id": "test_3",
            "case_id": "case_003", 
            "question": "What is the diagnosis?",
            "positive_answer": "The diagnosis is acute myocardial infarction.",
            "negative_answer": "The diagnosis is normal cardiac function.",
            "calculate_tie": True,
            "tie_positive": 0.05,
            "tie_negative": 0.45,
            "tie_pos_token_avg": 0.02,
            "tie_neg_token_avg": 0.40,
            "delta_pos": 0.05,
            "delta_neg": 0.45,
            "m_v": 0.25,
            "m_n": 0.85,
            "gamma": -0.40
        }
    ]

def create_test_args():
    """创建测试用的参数"""
    class TestArgs:
        def __init__(self):
            # TIE-ANKER参数
            self.enable_tie_anker = True
            self.output_pairs_file = "/workspace/MMedPO/inference/test_dpo_pairs.json"
            
            # TIE-ANKER权重
            self.w_gamma = 1.0
            self.w_v = 0.5
            self.w_n = 0.8
            self.w_s = 0.3
            self.w_o = 0.5
            
            # Sigmoid映射参数
            self.beta = 2.0
            self.tau = 0.0
            self.epsilon = 0.02
            
            # 评分和阈值
            self.token_avg = "true"
            self.use_per_case_zscore = "true"
            self.z_eps = 1e-6
            self.tau_pos = 0.1
            self.tau_gamma_strong = 0.5
            self.tau_gamma_weak = 0.1
            self.tau_v = 0.5
            self.tau_n_percentile = 75
            
            # 权重映射
            self.w_min = 0.05
    
    return TestArgs()

def test_dpo_export_format():
    """测试DPO pairs导出格式"""
    print("=== 测试DPO Pairs导出格式 ===")
    
    # 导入原始实现的函数
    try:
        from inference_dpo_tie_comparison import build_tie_anker_dpo_pairs
        print("✅ 成功导入build_tie_anker_dpo_pairs函数")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 创建测试数据
    test_results = create_test_results()
    test_args = create_test_args()
    
    print(f"📊 测试数据: {len(test_results)}个样本")
    
    # 调用DPO pairs构建函数
    try:
        dpo_pairs = build_tie_anker_dpo_pairs(test_results, test_args)
        print(f"✅ 成功构建DPO pairs: {len(dpo_pairs)}个")
    except Exception as e:
        print(f"❌ DPO pairs构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查输出格式
    if not dpo_pairs:
        print("⚠️  没有生成任何DPO pairs")
        return True
    
    print("\n=== DPO Pairs格式检查 ===")
    
    # 检查第一个pair的格式
    first_pair = dpo_pairs[0]
    required_fields = ["id", "case_id", "question", "image", "chosen", "rejected", "tie_weight", "tie_score", "metadata"]
    
    print("📋 检查必需字段:")
    for field in required_fields:
        if field in first_pair:
            print(f"  ✅ {field}: {type(first_pair[field]).__name__}")
        else:
            print(f"  ❌ 缺失字段: {field}")
    
    # 检查metadata格式
    if "metadata" in first_pair:
        metadata = first_pair["metadata"]
        metadata_fields = ["tie_positive", "tie_negative", "delta_pos", "delta_neg", "m_v", "m_n", "gamma"]
        print("\n📋 检查metadata字段:")
        for field in metadata_fields:
            if field in metadata:
                print(f"  ✅ {field}: {metadata[field]}")
            else:
                print(f"  ❌ 缺失metadata字段: {field}")
    
    # 保存到文件并检查
    try:
        pairs_file = test_args.output_pairs_file
        os.makedirs(os.path.dirname(pairs_file), exist_ok=True)
        
        with open(pairs_file, "w", encoding='utf-8') as f:
            json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ DPO pairs已保存到: {pairs_file}")
        
        # 验证文件可以正确读取
        with open(pairs_file, "r", encoding='utf-8') as f:
            loaded_pairs = json.load(f)
        
        print(f"✅ 文件验证成功: 加载了{len(loaded_pairs)}个pairs")
        
        # 显示第一个pair的内容
        print("\n=== 第一个DPO Pair示例 ===")
        print(json.dumps(loaded_pairs[0], ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ 文件保存/读取失败: {e}")
        return False
    
    return True

def validate_dpo_pair_structure(pair: Dict[str, Any]) -> List[str]:
    """验证单个DPO pair的结构"""
    issues = []
    
    # 检查基本字段
    required_fields = {
        "id": str,
        "case_id": str, 
        "question": str,
        "image": str,
        "chosen": str,
        "rejected": str,
        "tie_weight": (int, float),
        "tie_score": (int, float),
        "metadata": dict
    }
    
    for field, expected_type in required_fields.items():
        if field not in pair:
            issues.append(f"缺失字段: {field}")
        elif not isinstance(pair[field], expected_type):
            issues.append(f"字段类型错误: {field} 应为 {expected_type}, 实际为 {type(pair[field])}")
    
    # 检查metadata结构
    if "metadata" in pair:
        metadata_fields = ["tie_positive", "tie_negative", "delta_pos", "delta_neg", "m_v", "m_n", "gamma"]
        for field in metadata_fields:
            if field not in pair["metadata"]:
                issues.append(f"缺失metadata字段: {field}")
            elif not isinstance(pair["metadata"][field], (int, float)):
                issues.append(f"metadata字段类型错误: {field}")
    
    # 检查逻辑一致性
    if "chosen" in pair and "rejected" in pair:
        if pair["chosen"] == pair["rejected"]:
            issues.append("chosen和rejected答案相同")
    
    return issues

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试DPO pairs导出功能")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    print("🔍 开始测试DPO pairs导出功能...")
    
    success = test_dpo_export_format()
    
    if success:
        print("\n🎉 所有测试通过!")
        
        # 如果测试文件存在，进行额外验证
        test_file = "/workspace/MMedPO/inference/test_dpo_pairs.json"
        if os.path.exists(test_file):
            print("\n=== 额外验证 ===")
            with open(test_file, "r", encoding='utf-8') as f:
                pairs = json.load(f)
            
            total_issues = 0
            for i, pair in enumerate(pairs):
                issues = validate_dpo_pair_structure(pair)
                if issues:
                    print(f"Pair {i+1} 问题:")
                    for issue in issues:
                        print(f"  - {issue}")
                    total_issues += len(issues)
            
            if total_issues == 0:
                print("✅ 所有DPO pairs结构验证通过!")
            else:
                print(f"⚠️  发现 {total_issues} 个结构问题")
    else:
        print("\n❌ 测试失败!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())