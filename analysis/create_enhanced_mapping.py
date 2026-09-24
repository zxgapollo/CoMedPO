#!/usr/bin/env python3
"""
增强版案例图片匹配脚本
通过分析数据集结构更精确地匹配案例与图片
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_slake_dataset():
    """加载SLAKE数据集结构"""
    slake_path = Path("/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE")
    
    dataset_info = {
        'images': {},
        'questions': {},
        'train_data': [],
        'test_data': []
    }
    
    # 加载训练数据
    train_file = slake_path / "train.json"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            dataset_info['train_data'] = json.load(f)
    
    # 加载测试数据
    test_file = slake_path / "test.json"
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            dataset_info['test_data'] = json.load(f)
    
    # 扫描图片文件
    imgs_dir = slake_path / "imgs"
    if imgs_dir.exists():
        for img_file in imgs_dir.rglob("*.jpg"):
            if img_file.is_file():
                relative_path = str(img_file.relative_to(slake_path))
                dataset_info['images'][img_file.stem] = relative_path
    
    return dataset_info

def load_iuxray_dataset():
    """加载IU_XRAY数据集结构"""
    iuxray_path = Path("/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/iu_xray")
    
    dataset_info = {
        'images': {},
        'annotations': {}
    }
    
    # 加载标注数据
    annotation_file = iuxray_path / "annotation.json"
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            dataset_info['annotations'] = json.load(f)
    
    # 扫描图片文件
    images_dir = iuxray_path / "images"
    if images_dir.exists():
        for img_file in images_dir.rglob("*.png"):
            if img_file.is_file():
                relative_path = str(img_file.relative_to(iuxray_path))
                dataset_info['images'][img_file.stem] = relative_path
    
    return dataset_info

def find_matching_images_for_cases():
    """为案例匹配图片"""
    
    # 加载数据集
    print("正在加载数据集...")
    slake_data = load_slake_dataset()
    iuxray_data = load_iuxray_dataset()
    
    print(f"SLAKE: {len(slake_data['images'])} 张图片, {len(slake_data['train_data'])} 训练样本, {len(slake_data['test_data'])} 测试样本")
    print(f"IU_XRAY: {len(iuxray_data['images'])} 张图片")
    
    # 创建问题到图片的映射
    question_to_image = {}
    
    # 处理SLAKE数据
    for data_type in ['train_data', 'test_data']:
        for item in slake_data[data_type]:
            if 'question' in item and 'image' in item:
                question = item['question']
                image_path = item['image']
                question_to_image[question] = {
                    'image_path': f"SLAKE/imgs/{image_path}",
                    'dataset': 'SLAKE',
                    'answer': item.get('answer', '')
                }
    
    # 处理IU_XRAY数据
    for study_id, study_data in iuxray_data['annotations'].items():
        if 'images' in study_data:
            for img_info in study_data['images']:
                if 'id' in img_info:
                    image_id = img_info['id']
                    if image_id in iuxray_data['images']:
                        # 为报告生成创建虚拟问题
                        report_question = f"IU_XRAY report for study {study_id}"
                        question_to_image[report_question] = {
                            'image_path': f"iu_xray/{iuxray_data['images'][image_id]}",
                            'dataset': 'IU_XRAY',
                            'study_id': study_id,
                            'answer': study_data.get('report', '')
                        }
    
    return question_to_image

def enhance_case_mappings():
    """增强案例映射"""
    
    # 获取问题到图片的映射
    question_image_map = find_matching_images_for_cases()
    
    # 处理各个案例文件
    case_files = [
        ('/share_docker/workspace/Med/open_vqa_cases.json', '/share_docker/workspace/Med/open_vqa_cases_enhanced.json'),
        ('/share_docker/workspace/Med/close_vqa_cases.json', '/share_docker/workspace/Med/close_vqa_cases_enhanced.json'),
        ('/share_docker/workspace/Med/report_generation_cases.json', '/share_docker/workspace/Med/report_generation_cases_enhanced.json')
    ]
    
    enhanced_cases = []
    
    for input_file, output_file in case_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cases = data.get('cases', [])
            
            for case in cases:
                question = case.get('question', '')
                
                # 查找匹配的图片
                if question in question_image_map:
                    image_info = question_to_image[question]
                    case['matched_image'] = image_info
                    case['image_found'] = True
                else:
                    # 尝试模糊匹配
                    best_match = find_best_match(question, question_image_map)
                    if best_match:
                        case['matched_image'] = question_image_map[best_match]
                        case['image_found'] = True
                        case['match_method'] = 'fuzzy'
                    else:
                        case['matched_image'] = None
                        case['image_found'] = False
                
                enhanced_cases.append(case)
            
            # 保存增强后的数据
            data['enhanced_cases'] = cases
            data['image_matching_stats'] = {
                'total_cases': len(cases),
                'cases_with_images': sum(1 for case in cases if case.get('image_found', False)),
                'exact_matches': sum(1 for case in cases if case.get('image_found', False) and not case.get('match_method')),
                'fuzzy_matches': sum(1 for case in cases if case.get('match_method') == 'fuzzy')
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"处理完成: {input_file} -> {output_file}")
            
        except Exception as e:
            print(f"处理 {input_file} 时出错: {e}")

def find_best_match(question: str, question_map: dict) -> Optional[str]:
    """查找最佳匹配的问题"""
    question_lower = question.lower()
    
    # 关键词匹配
    keywords = extract_keywords(question_lower)
    
    best_match = None
    best_score = 0
    
    for map_question in question_map.keys():
        map_keywords = extract_keywords(map_question.lower())
        
        # 计算关键词重叠度
        overlap = len(set(keywords) & set(map_keywords))
        score = overlap / max(len(keywords), len(map_keywords)) if max(len(keywords), len(map_keywords)) > 0 else 0
        
        if score > best_score:
            best_score = score
            best_match = map_question
    
    return best_match if best_score > 0.3 else None  # 阈值0.3

def extract_keywords(text: str) -> List[str]:
    """提取关键词"""
    # 医学相关关键词
    medical_keywords = [
        'chest', 'lung', 'heart', 'brain', 'abdomen', 'pelvis', 'head', 'neck',
        'ct', 'mri', 'x-ray', 'xray', 'ultrasound', 'echocardiogram',
        'axial', 'sagittal', 'coronal', 'supine', 'prone', 'upright',
        'normal', 'abnormal', 'clear', 'opacity', 'consolidation', 'effusion',
        'pneumonia', 'cancer', 'tumor', 'mass', 'nodule', 'calcification',
        'cardiomegaly', 'pneumothorax', 'atelectasis', 'fibrosis'
    ]
    
    found_keywords = []
    for keyword in medical_keywords:
        if keyword in text:
            found_keywords.append(keyword)
    
    return found_keywords

def create_enhanced_summary():
    """创建增强版摘要"""
    summary = {
        'datasets_analyzed': {
            'SLAKE': {
                'total_images': 0,
                'total_questions': 0,
                'matched_cases': 0
            },
            'IU_XRAY': {
                'total_images': 0,
                'total_reports': 0,
                'matched_cases': 0
            }
        },
        'matching_quality': {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'unmatched_cases': 0
        },
        'recommendations': [
            '建议在实际应用中使用精确的问题-图片配对数据',
            '对于模糊匹配的案例，建议人工验证匹配准确性',
            '考虑使用更先进的语义匹配算法提高匹配精度'
        ]
    }
    
    with open('/share_docker/workspace/Med/enhanced_mapping_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    print("开始增强案例图片匹配...")
    
    # 执行增强映射
    enhance_case_mappings()
    
    # 创建摘要
    create_enhanced_summary()
    
    print("\n增强映射完成！")
    print("生成的文件:")
    print("- open_vqa_cases_enhanced.json")
    print("- close_vqa_cases_enhanced.json")
    print("- report_generation_cases_enhanced.json")
    print("- enhanced_mapping_summary.json")

if __name__ == "__main__":
    main()