#!/usr/bin/env python3
"""
更新案例图片摘要，专门展示CaMedPO最优案例
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

def filter_camppo_best_cases(cases: list) -> list:
    """筛选CaMedPO最优案例：CaMedPO正确，Baseline和MMedPO均错误"""
    best_cases = []
    
    for case in cases:
        comparison = case.get('comparison', {})
        baseline_correct = comparison.get('baseline', {}).get('correct', False)
        mmedpo_correct = comparison.get('MMedPO', {}).get('correct', False)
        camppo_correct = comparison.get('CaMedPO', {}).get('correct', False)
        
        # 筛选条件：CaMedPO正确，且Baseline和MMedPO均错误
        if camppo_correct and not baseline_correct and not mmedpo_correct:
            best_cases.append(case)
    
    return best_cases

def create_camppo_best_cases_summary():
    """创建CaMedPO最优案例摘要"""
    
    # 加载各个案例文件
    print("正在加载案例数据...")
    
    # 加载开放式VQA案例
    open_data = load_json_file('/share_docker/workspace/Med/report/open_vqa_cases.json')
    open_cases = open_data.get('cases', [])
    open_best = filter_camppo_best_cases(open_cases)
    
    # 加载封闭式VQA案例
    close_data = load_json_file('/share_docker/workspace/Med/report/close_vqa_cases.json')
    close_cases = close_data.get('cases', [])
    close_best = filter_camppo_best_cases(close_cases)
    
    # 加载报告生成案例
    report_data = load_json_file('/share_docker/workspace/Med/report_generation_cases.json')
    report_cases = report_data.get('cases', [])
    report_best = filter_camppo_best_cases(report_cases)
    
    # 合并所有最优案例
    all_best_cases = open_best + close_best + report_best
    
    # 创建更新的案例图片摘要
    updated_summary = {
        "report_info": {
            "title": "CaMedPO最优案例图片摘要",
            "description": "专门展示CaMedPO正确而Baseline和MMedPO均错误的案例",
            "filter_criteria": "CaMedPO正确，Baseline和MMedPO均错误",
            "generation_date": "2024-11-18",
            "total_best_cases": len(all_best_cases),
            "by_category": {
                "open_vqa": len(open_best),
                "close_vqa": len(close_best), 
                "report_generation": len(report_best)
            }
        },
        "best_cases": all_best_cases[:20],  # 取前20个最优案例
        "summary_by_category": {
            "open_vqa": {
                "total_cases": len(open_cases),
                "best_cases": len(open_best),
                "percentage": round(len(open_best)/len(open_cases)*100, 1) if open_cases else 0,
                "dataset": "SLAKE"
            },
            "close_vqa": {
                "total_cases": len(close_cases),
                "best_cases": len(close_best),
                "percentage": round(len(close_best)/len(close_cases)*100, 1) if close_cases else 0,
                "dataset": "VQA_RAD"
            },
            "report_generation": {
                "total_cases": len(report_cases),
                "best_cases": len(report_best),
                "percentage": round(len(report_best)/len(report_cases)*100, 1) if report_cases else 0,
                "dataset": "IU_XRAY"
            }
        }
    }
    
    return updated_summary

def main():
    """主函数"""
    print("开始创建CaMedPO最优案例图片摘要...")
    
    # 创建最优案例摘要
    summary = create_camppo_best_cases_summary()
    
    # 保存到文件
    with open('/share_docker/workspace/Med/case_image_summary_camppo_best.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nCaMedPO最优案例图片摘要已生成！")
    print(f"总计发现 {summary['report_info']['total_best_cases']} 个最优案例")
    print("文件保存为: case_image_summary_camppo_best.json")

if __name__ == "__main__":
    main()