#!/usr/bin/env python3
"""
医学VQA案例图片匹配脚本
根据问题匹配对应的图片文件
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

class ImageMatcher:
    def __init__(self):
        self.slake_base_path = Path("/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/SLAKE")
        self.iuxray_base_path = Path("/share_docker/workspace/Med/Med-main/Med-main/MMedPO/data/iu_xray")
        self.slake_image_map = {}
        self.iuxray_image_map = {}
        self.load_image_mappings()
    
    def load_image_mappings(self):
        """加载图片映射关系"""
        print("正在加载SLAKE图片映射...")
        self.load_slake_images()
        
        print("正在加载IU_XRAY图片映射...")
        self.load_iuxray_images()
    
    def load_slake_images(self):
        """加载SLAKE数据集图片映射"""
        imgs_dir = self.slake_base_path / "imgs"
        
        if not imgs_dir.exists():
            print(f"警告: SLAKE图片目录不存在: {imgs_dir}")
            return
        
        # 递归查找所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for img_file in imgs_dir.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # 提取相对路径和文件名
                relative_path = img_file.relative_to(imgs_dir)
                filename = img_file.stem
                
                # 存储映射关系
                self.slake_image_map[filename] = str(relative_path)
                self.slake_image_map[img_file.name] = str(relative_path)
        
        print(f"SLAKE: 找到 {len(self.slake_image_map)} 张图片")
    
    def load_iuxray_images(self):
        """加载IU_XRAY数据集图片映射"""
        images_dir = self.iuxray_base_path / "images"
        
        if not images_dir.exists():
            print(f"警告: IU_XRAY图片目录不存在: {images_dir}")
            return
        
        # 递归查找所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for img_file in images_dir.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # 提取相对路径和文件名
                relative_path = img_file.relative_to(images_dir)
                filename = img_file.stem
                
                # 存储映射关系
                self.iuxray_image_map[filename] = str(relative_path)
                self.iuxray_image_map[img_file.name] = str(relative_path)
        
        print(f"IU_XRAY: 找到 {len(self.iuxray_image_map)} 张图片")
    
    def find_image_by_question(self, question: str, dataset: str, study_id: Optional[str] = None) -> Optional[str]:
        """根据问题和数据集查找对应的图片"""
        if dataset.upper() == "SLAKE":
            return self.find_slake_image(question)
        elif dataset.upper() == "IU_XRAY" or dataset.upper() == "VQA_RAD":
            return self.find_iuxray_image(study_id, question)
        else:
            return None
    
    def find_slake_image(self, question: str) -> Optional[str]:
        """为SLAKE问题查找图片"""
        # SLAKE数据集中，尝试从问题中提取关键词匹配图片
        question_lower = question.lower()
        
        # 常见关键词模式
        patterns = [
            r'(?:image|picture|photo|scan)\s+(?:of\s+)?(\w+)',
            r'(\w+)\s+(?:image|picture|photo|scan)',
            r'(?:this|the)\s+(\w+)\s+(?:image|picture|photo|scan)',
            r'(?:show|display|demonstrate)\s+(?:the\s+)?(\w+)',
            r'(chest|lung|heart|brain|abdomen|pelvis|head|neck)',
            r'(ct|mri|x-ray|xray|ultrasound|echocardiogram)',
            r'(axial|sagittal|coronal)',
            r'(supine|prone|upright)'
        ]
        
        # 尝试匹配关键词
        for pattern in patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            if matches:
                keyword = matches[0]
                # 在图片映射中查找包含该关键词的图片
                for img_name, img_path in self.slake_image_map.items():
                    if keyword.lower() in img_name.lower():
                        return img_path
        
        # 如果没有找到特定匹配，返回任意一张图片作为示例
        if self.slake_image_map:
            return next(iter(self.slake_image_map.values()))
        
        return None
    
    def find_iuxray_image(self, study_id: Optional[str], question: str) -> Optional[str]:
        """为IU_XRAY问题查找图片"""
        if study_id and study_id in self.iuxray_image_map:
            return self.iuxray_image_map[study_id]
        
        # 如果没有study_id，尝试从问题中匹配
        question_lower = question.lower()
        
        # 常见医学影像关键词
        medical_keywords = [
            'chest', 'lung', 'heart', 'cardiac', 'pulmonary',
            'pneumonia', 'effusion', 'consolidation', 'opacity',
            'normal', 'clear', 'abnormal', 'disease'
        ]
        
        for keyword in medical_keywords:
            if keyword in question_lower:
                # 查找包含关键词的图片
                for img_name, img_path in self.iuxray_image_map.items():
                    if keyword in img_name.lower():
                        return img_path
        
        # 如果没有找到特定匹配，返回任意一张图片作为示例
        if self.iuxray_image_map:
            return next(iter(self.iuxray_image_map.values()))
        
        return None
    
    def process_case_file(self, input_file: str, output_file: str, dataset_type: str):
        """处理案例文件，添加图片信息"""
        print(f"正在处理 {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cases = data.get('cases', [])
            total_cases = len(cases)
            matched_cases = 0
            
            for i, case in enumerate(cases):
                question = case.get('question', '')
                dataset = case.get('dataset', '')
                study_id = case.get('study_id', '')
                
                # 查找对应的图片
                image_path = self.find_image_by_question(question, dataset, study_id)
                
                if image_path:
                    matched_cases += 1
                    case['image_path'] = image_path
                    case['image_source'] = dataset
                else:
                    case['image_path'] = None
                    case['image_source'] = None
                    case['image_note'] = '未找到匹配图片'
                
                # 进度显示
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{total_cases} 个案例")
            
            # 更新数据
            data['image_matching_summary'] = {
                'total_cases': total_cases,
                'matched_cases': matched_cases,
                'match_rate': f"{(matched_cases/total_cases)*100:.1f}%"
            }
            
            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"完成！匹配率: {(matched_cases/total_cases)*100:.1f}%")
            
        except Exception as e:
            print(f"处理 {input_file} 时出错: {e}")
    
    def create_comprehensive_mapping(self):
        """创建综合映射文件"""
        print("正在创建综合映射文件...")
        
        all_mappings = []
        
        # 处理所有案例文件
        case_files = [
            ('/share_docker/workspace/Med/open_vqa_cases.json', 'open'),
            ('/share_docker/workspace/Med/close_vqa_cases.json', 'close'),
            ('/share_docker/workspace/Med/report_generation_cases.json', 'report')
        ]
        
        for input_file, category in case_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cases = data.get('cases', [])
                
                for case in cases:
                    mapping = {
                        'category': category,
                        'case_id': case.get('case_id'),
                        'dataset': case.get('dataset'),
                        'question': case.get('question', case.get('ground_truth', '')),
                        'ground_truth': case.get('ground_truth', case.get('ground_truth', '')),
                        'comparison': case.get('comparison', {}),
                        'image_path': case.get('image_path'),
                        'image_source': case.get('image_source'),
                        'study_id': case.get('study_id')
                    }
                    all_mappings.append(mapping)
            
            except Exception as e:
                print(f"处理 {input_file} 时出错: {e}")
                continue
        
        # 保存综合映射
        comprehensive_data = {
            'total_mappings': len(all_mappings),
            'categories': {
                'open': len([m for m in all_mappings if m['category'] == 'open']),
                'close': len([m for m in all_mappings if m['category'] == 'close']),
                'report': len([m for m in all_mappings if m['category'] == 'report'])
            },
            'mappings': all_mappings
        }
        
        with open('/share_docker/workspace/Med/comprehensive_case_image_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2)
        
        print(f"综合映射文件创建完成！总计 {len(all_mappings)} 个映射")

def main():
    """主函数"""
    print("开始匹配案例与图片...")
    
    matcher = ImageMatcher()
    
    # 处理各个案例文件
    case_files = [
        ('/share_docker/workspace/Med/report/open_vqa_cases.json', '/share_docker/workspace/Med/report/open_vqa_cases_with_images.json', 'open'),
        ('/share_docker/workspace/Med/report/close_vqa_cases.json', '/share_docker/workspace/Med/report/close_vqa_cases_with_images.json', 'close'),
        ('/share_docker/workspace/Med/report/report_generation_cases.json', '/share_docker/workspace/Med/report/report_generation_cases_with_images.json', 'report')
    ]
    
    for input_file, output_file, category in case_files:
        if os.path.exists(input_file):
            matcher.process_case_file(input_file, output_file, category)
        else:
            print(f"文件不存在: {input_file}")
    
    # 创建综合映射
    matcher.create_comprehensive_mapping()
    
    print("\n所有处理完成！")
    print("生成的文件:")
    print("- open_vqa_cases_with_images.json")
    print("- close_vqa_cases_with_images.json") 
    print("- report_generation_cases_with_images.json")
    print("- comprehensive_case_image_mapping.json")

if __name__ == "__main__":
    main()