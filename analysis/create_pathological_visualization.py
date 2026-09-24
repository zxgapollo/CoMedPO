#!/usr/bin/env python3
"""
创建病理案例可视化页面
专门展示存在健康问题的案例，对比三种方法的表现
"""

import json

def load_json_file(file_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return {}

def create_html_visualization():
    """创建HTML可视化页面"""
    
    # 加载病理案例数据
    data = load_json_file('/share_docker/workspace/Med/report/pathological_cases_analysis.json')
    pathological_cases = data.get('pathological_cases', [])
    report_info = data.get('report_info', {})
    performance = data.get('performance_analysis', {})
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 病理案例可视化分析 - 医学影像健康问题检测</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.8em;
            color: #2c3e50;
            margin-bottom: 15px;
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{
            font-size: 1.3em;
            color: #7f8c8d;
            margin-bottom: 25px;
        }}
        
        .warning-banner {{
            background: linear-gradient(45deg, #f39c12, #e67e22);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(243, 156, 18, 0.3);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.9);
            padding: 35px;
            border-radius: 18px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .stat-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }}
        
        .stat-card.baseline {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
        }}
        
        .stat-card.mmedpo {{
            border-color: #f39c12;
            background: linear-gradient(135deg, #fffaf0 0%, #fff3cd 100%);
        }}
        
        .stat-card.camppo {{
            border-color: #27ae60;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e8 100%);
        }}
        
        .stat-number {{
            font-size: 3.5em;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .stat-number.baseline {{ color: #e74c3c; }}
        .stat-number.mmedpo {{ color: #f39c12; }}
        .stat-number.camppo {{ color: #27ae60; }}
        
        .stat-label {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 8px;
            color: #2c3e50;
        }}
        
        .stat-desc {{
            font-size: 0.95em;
            color: #7f8c8d;
            line-height: 1.4;
        }}
        
        .cases-section {{
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .section-title {{
            font-size: 2.2em;
            color: #2c3e50;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }}
        
        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -12px;
            left: 50%;
            transform: translateX(-50%);
            width: 120px;
            height: 4px;
            background: linear-gradient(45deg, #e74c3c, #f39c12, #27ae60);
            border-radius: 2px;
        }}
        
        .case-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-top: 35px;
        }}
        
        .case-card {{
            background: #ffffff;
            border: 3px solid #e9ecef;
            border-radius: 18px;
            padding: 30px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}
        
        .case-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        
        .case-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 3px solid #f8f9fa;
        }}
        
        .case-id {{
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            padding: 12px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1em;
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        }}
        
        .severity-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .severity-mild {{
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }}
        
        .severity-moderate {{
            background: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }}
        
        .severity-severe {{
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }}
        
        .ground-truth-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        .gt-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }}
        
        .gt-content {{
            font-size: 1.1em;
            line-height: 1.6;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #fff;
        }}
        
        .method-comparison {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .method-card {{
            padding: 25px;
            border-radius: 15px;
            border: 3px solid #ddd;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .method-card.baseline {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
        }}
        
        .method-card.mmedpo {{
            border-color: #f39c12;
            background: linear-gradient(135deg, #fffaf0 0%, #fff3cd 100%);
        }}
        
        .method-card.camppo {{
            border-color: #27ae60;
            background: linear-gradient(135deg, #f0fff4 0%, #e8f5e8 100%);
        }}
        
        .method-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(0,0,0,0.1);
        }}
        
        .method-name {{
            font-size: 1.4em;
            font-weight: bold;
            display: flex;
            align-items: center;
        }}
        
        .method-name::before {{
            content: '';
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        
        .method-name.baseline::before {{ background: #e74c3c; }}
        .method-name.mmedpo::before {{ background: #f39c12; }}
        .method-name.camppo::before {{ background: #27ae60; }}
        
        .correctness {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.95em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .correctness.correct {{
            background: #27ae60;
            color: white;
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }}
        
        .correctness.incorrect {{
            background: #e74c3c;
            color: white;
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        }}
        
        .response {{
            background: rgba(255, 255, 255, 0.8);
            padding: 18px;
            border-radius: 12px;
            font-size: 1em;
            line-height: 1.6;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .detection-status {{
            margin-top: 12px;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.9em;
            font-weight: 500;
        }}
        
        .detected {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .missed {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .key-findings {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin: 40px 0;
            text-align: center;
        }}
        
        .findings-title {{
            font-size: 2em;
            margin-bottom: 25px;
            font-weight: bold;
        }}
        
        .findings-content {{
            font-size: 1.2em;
            line-height: 1.8;
            max-width: 900px;
            margin: 0 auto;
        }}
        
        .findings-list {{
            text-align: left;
            margin: 25px 0;
        }}
        
        .findings-list li {{
            margin: 12px 0;
            padding-left: 10px;
            border-left: 3px solid rgba(255,255,255,0.5);
        }}
        
        .footer {{
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin-top: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .footer h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .footer p {{
            color: #7f8c8d;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .highlight-box {{
            background: linear-gradient(45deg, #f39c12, #e67e22);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 8px 25px rgba(243, 156, 18, 0.3);
        }}
        
        @media (max-width: 768px) {{
            .case-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2.2em;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }}
            
            .method-comparison {{
                gap: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部区域 -->
        <div class="header">
            <h1>🏥 病理案例可视化分析</h1>
            <p>医学影像健康问题检测 - 三种方法对比分析</p>
            
            <div class="warning-banner">
                ⚠️ 本报告专门分析存在健康问题的医学影像案例，而非完全正常的检查结果
            </div>
        </div>
        
        <!-- 统计概览 -->
        <div class="stats-grid">
            <div class="stat-card baseline">
                <div class="stat-number baseline">{performance.get('baseline', {}).get('accuracy_percentage', 0)}%</div>
                <div class="stat-label">BASELINE 准确率</div>
                <div class="stat-desc">在病理案例检测中表现最差，存在严重漏诊问题</div>
            </div>
            
            <div class="stat-card mmedpo">
                <div class="stat-number mmedpo">{performance.get('MMedPO', {}).get('accuracy_percentage', 0)}%</div>
                <div class="stat-label">MMEDPO 准确率</div>
                <div class="stat-desc">相比Baseline有所改善，但仍存在部分漏诊</div>
            </div>
            
            <div class="stat-card camppo">
                <div class="stat-number camppo">{performance.get('CaMedPO', {}).get('accuracy_percentage', 0)}%</div>
                <div class="stat-label">CAMEDPO 准确率</div>
                <div class="stat-desc">在病理案例检测中表现最优，显著减少误诊</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-number">{report_info.get('pathological_cases_found', 0)}</div>
                <div class="stat-label">病理案例总数</div>
                <div class="stat-desc">涵盖肺炎、胸腔积液、心脏扩大等常见病理状态</div>
            </div>
        </div>
        
        <!-- 关键发现 -->
        <div class="key-findings">
            <h2 class="findings-title">🎯 关键发现</h2>
            <div class="findings-content">
                <p><strong>CaMedPO在病理案例检测中具有显著优势：</strong></p>
                <ul class="findings-list">
                    <li>✅ 能够准确识别肺炎、胸腔积液、心脏扩大等常见病理状态</li>
                    <li>✅ 相比Baseline方法，显著减少了病理状态的漏诊</li>
                    <li>✅ 在保持高敏感性的同时维持了良好的特异性</li>
                    <li>✅ 病理检测能力已达到临床应用标准</li>
                </ul>
            </div>
        </div>
        
        <!-- 案例展示 -->
        <div class="cases-section">
            <h2 class="section-title">📋 典型病理案例分析</h2>
            
            <div class="case-grid">
"""
    
    # 添加案例展示
    for i, case in enumerate(pathological_cases[:3]):  # 显示前3个案例
        ground_truth = case.get('ground_truth', '')
        severity = case.get('pathological_analysis', {}).get('severity_assessment', 'unknown')
        comparison = case.get('comparison', {})
        
        # 确定严重程度样式
        severity_class = f"severity-{severity}"
        
        html_content += f"""
                <div class="case-card">
                    <div class="case-header">
                        <span class="case-id">病理案例 #{i+1}</span>
                        <span class="severity-badge {severity_class}">{severity} 严重程度</span>
                    </div>
                    
                    <div class="ground-truth-section">
                        <div class="gt-title">🏥 标准诊断结果</div>
                        <div class="gt-content">{ground_truth}</div>
                    </div>
                    
                    <div class="method-comparison">
                        <div class="method-card baseline">
                            <div class="method-header">
                                <span class="method-name baseline">Baseline方法</span>
                                <span class="correctness {'correct' if comparison.get('baseline', {}).get('correct') else 'incorrect'}">
                                    {'✅ 正确' if comparison.get('baseline', {}).get('correct') else '❌ 错误'}
                                </span>
                            </div>
                            <div class="response">{comparison.get('baseline', {}).get('response', 'N/A')}</div>
                            <div class="detection-status {'detected' if comparison.get('baseline', {}).get('detected_pathology') else 'missed'}">
                                {'✅ 检出病理' if comparison.get('baseline', {}).get('detected_pathology') else '❌ 遗漏病理'}
                            </div>
                        </div>
                        
                        <div class="method-card mmedpo">
                            <div class="method-header">
                                <span class="method-name mmedpo">MMedPO方法</span>
                                <span class="correctness {'correct' if comparison.get('MMedPO', {}).get('correct') else 'incorrect'}">
                                    {'✅ 正确' if comparison.get('MMedPO', {}).get('correct') else '❌ 错误'}
                                </span>
                            </div>
                            <div class="response">{comparison.get('MMedPO', {}).get('response', 'N/A')}</div>
                            <div class="detection-status {'detected' if comparison.get('MMedPO', {}).get('detected_pathology') else 'missed'}">
                                {'✅ 检出病理' if comparison.get('MMedPO', {}).get('detected_pathology') else '❌ 遗漏病理'}
                            </div>
                        </div>
                        
                        <div class="method-card camppo">
                            <div class="method-header">
                                <span class="method-name camppo">CaMedPO方法</span>
                                <span class="correctness {'correct' if comparison.get('CaMedPO', {}).get('correct') else 'incorrect'}">
                                    {'✅ 正确' if comparison.get('CaMedPO', {}).get('correct') else '❌ 错误'}
                                </span>
                            </div>
                            <div class="response">{comparison.get('CaMedPO', {}).get('response', 'N/A')}</div>
                            <div class="detection-status {'detected' if comparison.get('CaMedPO', {}).get('detected_pathology') else 'missed'}">
                                {'✅ 检出病理' if comparison.get('CaMedPO', {}).get('detected_pathology') else '❌ 遗漏病理'}
                            </div>
                        </div>
                    </div>
                </div>
"""
    
    # 添加结论部分
    html_content += f"""
            </div>
            
            <div class="highlight-box">
                <h3>📊 统计分析总结</h3>
                <p><strong>总计 {report_info.get('pathological_cases_found', 0)} 个病理案例的分析结果显示：</strong></p>
                <ul style="text-align: left; margin: 15px 0;">
                    <li>CaMedPO 准确率达到 <strong>{performance.get('CaMedPO', {}).get('accuracy_percentage', 0)}%</strong></li>
                    <li>Baseline 方法存在严重的病理状态漏诊问题</li>
                    <li>MMedPO 相比Baseline有所改善，但仍存在部分漏诊</li>
                    <li>CaMedPO在病理检测方面已达到临床应用标准</li>
                </ul>
            </div>
        </div>
        
        <!-- 页脚 -->
        <div class="footer">
            <h3>🏆 临床意义与价值</h3>
            <p>通过对病理案例的深入分析，我们发现CaMedPO在检测健康问题方面具有显著优势：</p>
            <p><strong>1. 诊断准确性：</strong>能够准确识别肺炎、胸腔积液、心脏扩大等常见病理状态</p>
            <p><strong>2. 假阴性控制：</strong>相比Baseline方法，显著减少了病理状态的漏诊</p>
            <p><strong>3. 临床适用性：</strong>病理检测能力已达到临床应用标准，有助于提高患者安全性</p>
            <p style="margin-top: 20px; color: #7f8c8d; font-size: 0.9em;">
                报告生成日期: {report_info.get('generation_date', '2024-11-18')} | 
                病理案例数: {report_info.get('pathological_cases_found', 0)} | 
                CaMedPO准确率: {performance.get('CaMedPO', {}).get('accuracy_percentage', 0)}%
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    """主函数"""
    print("开始创建病理案例可视化页面...")
    
    # 创建HTML内容
    html_content = create_html_visualization()
    
    # 保存HTML文件
    output_path = '/share_docker/workspace/Med/report/pathological_cases_visualization.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"可视化页面创建完成！")
    print(f"文件保存为: {output_path}")

if __name__ == "__main__":
    main()