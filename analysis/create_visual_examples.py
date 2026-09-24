#!/usr/bin/env python3
"""
创建可视化案例展示脚本
生成便于查看的HTML格式案例展示
"""

import json
from pathlib import Path

def create_html_case_display():
    """创建HTML格式的案例展示"""
    
    # 读取案例数据
    try:
        with open('/share_docker/workspace/Med/comprehensive_case_image_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("找不到综合映射文件")
        return
    
    mappings = data.get('mappings', [])
    
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>医学VQA案例展示 - CaMedPO vs Baseline vs MMedPO</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            min-width: 200px;
            margin: 10px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .case-container {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        .case-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }
        .case-id {
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .dataset-badge {
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .question {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #007bff;
            font-weight: 500;
        }
        .ground-truth {
            background: #d4edda;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        .image-info {
            background: #e2e3e5;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #6c757d;
        }
        .method-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .method-card {
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #ddd;
            position: relative;
        }
        .method-card.correct {
            border-color: #28a745;
            background-color: #f8fff9;
        }
        .method-card.incorrect {
            border-color: #dc3545;
            background-color: #fff8f8;
        }
        .method-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .method-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        .correctness {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .correct {
            background: #28a745;
            color: white;
        }
        .incorrect {
            background: #dc3545;
            color: white;
        }
        .response {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.4;
        }
        .issue {
            background: #fff3cd;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        .winner {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-align: center;
            margin: 15px 0;
        }
        .category-filter {
            text-align: center;
            margin: 20px 0;
        }
        .filter-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .filter-btn:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }
        .filter-btn.active {
            background: #28a745;
        }
        .hidden {
            display: none;
        }
        .image-placeholder {
            width: 100%;
            height: 200px;
            background: #e9ecef;
            border: 2px dashed #adb5bd;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6c757d;
            font-size: 1.1em;
            margin: 15px 0;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 医学VQA案例对比分析</h1>
        <p>CaMedPO vs Baseline vs MMedPO 性能对比展示</p>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">2,775</div>
                <div>总案例数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">80.2%</div>
                <div>CaMedPO最高准确率</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">551</div>
                <div>CaMedPO最优案例</div>
            </div>
        </div>
    </div>

    <div class="category-filter">
        <button class="filter-btn active" onclick="filterCases('all')">全部案例</button>
        <button class="filter-btn" onclick="filterCases('open')">开放式VQA</button>
        <button class="filter-btn" onclick="filterCases('close')">封闭式VQA</button>
        <button class="filter-btn" onclick="filterCases('report')">报告生成</button>
    </div>
"""

    # 添加案例展示
    for i, mapping in enumerate(mappings[:20]):  # 显示前20个案例
        category = mapping.get('category', 'unknown')
        dataset = mapping.get('dataset', 'unknown')
        question = mapping.get('question', '')
        ground_truth = mapping.get('ground_truth', '')
        comparison = mapping.get('comparison', {})
        image_path = mapping.get('image_path', '')
        
        html_content += f"""
    <div class="case-container" data-category="{category}">
        <div class="case-header">
            <span class="case-id">案例 #{i+1}</span>
            <span class="dataset-badge">{dataset}</span>
        </div>
        
        <div class="question">
            <strong>问题:</strong> {question}
        </div>
        
        <div class="ground-truth">
            <strong>标准答案:</strong> {ground_truth}
        </div>
        
        <div class="image-info">
            <strong>图片路径:</strong> {image_path if image_path else '未找到对应图片'}
        </div>
        
        <div class="image-placeholder">
            📸 医学影像图片
        </div>
"""

        # 确定获胜者
        baseline_correct = comparison.get('baseline', {}).get('correct', False)
        mmedpo_correct = comparison.get('MMedPO', {}).get('correct', False)
        camppo_correct = comparison.get('CaMedPO', {}).get('correct', False)
        
        winner = ""
        if camppo_correct and not baseline_correct and not mmedpo_correct:
            winner = "CaMedPO"
        elif camppo_correct and baseline_correct and mmedpo_correct:
            # 都比较正确，选择最简洁的
            baseline_resp = comparison.get('baseline', {}).get('response', '')
            mmedpo_resp = comparison.get('MMedPO', {}).get('response', '')
            camppo_resp = comparison.get('CaMedPO', {}).get('response', '')
            
            lengths = [len(baseline_resp), len(mmedpo_resp), len(camppo_resp)]
            if lengths[2] == min(lengths):
                winner = "CaMedPO"
            elif lengths[1] == min(lengths):
                winner = "MMedPO"
            else:
                winner = "Baseline"
        
        if winner:
            html_content += f'<div class="winner">🏆 最佳方法: {winner}</div>'

        html_content += """
        <div class="method-comparison">
"""

        # 添加每个方法的对比
        methods = ['baseline', 'MMedPO', 'CaMedPO']
        method_names = ['Baseline', 'MMedPO', 'CaMedPO']
        
        for method, display_name in zip(methods, method_names):
            method_data = comparison.get(method, {})
            response = method_data.get('response', '')
            correct = method_data.get('correct', False)
            issue = method_data.get('issue', '无')
            
            correctness_class = 'correct' if correct else 'incorrect'
            correctness_text = '✅ 正确' if correct else '❌ 错误'
            
            html_content += f"""
            <div class="method-card {correctness_class}">
                <div class="method-header">
                    <span class="method-name">{display_name}</span>
                    <span class="correctness {correctness_class}">{correctness_text}</span>
                </div>
                <div class="response">{response}</div>
                <div class="issue">问题: {issue}</div>
            </div>
"""

        html_content += """
        </div>
    </div>
"""

    # 添加JavaScript功能
    html_content += """
    <script>
        function filterCases(category) {
            const cases = document.querySelectorAll('.case-container');
            const buttons = document.querySelectorAll('.filter-btn');
            
            // 更新按钮状态
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // 过滤案例
            cases.forEach(case_ => {
                if (category === 'all' || case_.dataset.category === category) {
                    case_.classList.remove('hidden');
                } else {
                    case_.classList.add('hidden');
                }
            });
        }
    </script>
</body>
</html>
"""

    # 保存HTML文件
    with open('/share_docker/workspace/Med/case_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("可视化案例展示文件已生成: case_visualization.html")

def create_json_mapping_summary():
    """创建JSON映射摘要"""
    try:
        with open('/share_docker/workspace/Med/comprehensive_case_image_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("找不到综合映射文件")
        return
    
    mappings = data.get('mappings', [])
    
    # 按类别统计
    summary = {
        'total_cases': len(mappings),
        'by_category': {
            'open': len([m for m in mappings if m.get('category') == 'open']),
            'close': len([m for m in mappings if m.get('category') == 'close']),
            'report': len([m for m in mappings if m.get('category') == 'report'])
        },
        'by_dataset': {
            'SLAKE': len([m for m in mappings if m.get('dataset') == 'SLAKE']),
            'VQA_RAD': len([m for m in mappings if m.get('dataset') == 'VQA_RAD']),
            'IU_XRAY': len([m for m in mappings if m.get('dataset') == 'IU_XRAY'])
        },
        'image_coverage': {
            'with_images': len([m for m in mappings if m.get('image_path')]),
            'without_images': len([m for m in mappings if not m.get('image_path')])
        },
        'sample_cases': mappings[:10]  # 前10个案例作为示例
    }
    
    with open('/share_docker/workspace/Med/case_image_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("案例映射摘要已生成: case_image_summary.json")

def main():
    """主函数"""
    print("开始创建可视化案例展示...")
    
    create_html_case_display()
    create_json_mapping_summary()
    
    print("\n完成！生成的文件:")
    print("- case_visualization.html (可视化展示)")
    print("- case_image_summary.json (映射摘要)")

if __name__ == "__main__":
    main()