# analysis — Experiment Analysis and Visualization

This directory holds scripts for analyzing experiment results: comparing **baseline / MMedPO / CaMedPO (CaMPPO)** on medical VQA and report generation, organizing cases by category, and visualizing representative examples.


## Script categories

### 1. Result comparison

| Script | Description |
|--------|-------------|
| `analyze_medical_results.py` | Main analysis script: compares the three methods on VQA and report generation |
| `analyze_pathological_cases.py` | Focused analysis of pathological cases |
| `enhance_report_cases.py` | Adds / enriches analysis dimensions for report generation cases |

### 2. Case categorization and matching

| Script | Description |
|--------|-------------|
| `categorize_cases.py` | Organizes cases into `open` (open-ended), `close` (closed-ended) and `report` categories |
| `match_cases_with_images.py` | Matches each question with its corresponding image file |
| `filter_camppo_superior_cases.py` | Filters cases where CaMedPO is clearly better |

### 3. Visualization and report generation

| Script | Description |
|--------|-------------|
| `create_camppo_best_visualization.py` | Builds a visualization page for the best CaMedPO cases |
| `create_camppo_best_cases_report.py` | Generates the best-case report |
| `create_pathological_visualization.py` | Visualization of pathological cases |
| `create_enhanced_pathological_cases.py` | Enhanced pathological case collection |
| `create_visual_examples.py` | General visual example generation |
| `create_detailed_mapping_report.py` | Generates a detailed mapping report |
| `create_enhanced_mapping.py` | Builds enhanced mappings |
| `update_case_image_summary.py` | Updates the case image summary |
| `update_report_with_images.py` | Fills images back into the report |

## Outputs

Running the scripts produces analysis artifacts in the same directory:

- `close_vqa_cases.json`, `open_vqa_cases.json` — VQA cases split by type
- `camppo_superior_cases.json` + `camppo_superior_analysis.md` — cases where CaMedPO wins, and the analysis
- `enhanced_pathological_cases.json` + `enhanced_pathological_cases_analysis.md` — pathological case analysis
- `pathological_cases_analysis.json` / `.md`
- `pathological_cases_visualization.html` — open directly in a browser
- `summary_by_category.json`, `report_generation_cases.json` — per-category summary and report generation cases

## Usage

Each script is standalone; edit the data paths inside and run:

```bash
python analysis/analyze_medical_results.py
python analysis/categorize_cases.py
python analysis/create_camppo_best_visualization.py
```

> The scripts consume evaluation outputs from MMedPO / MedEvalKit — run the [`../MedEvalKit`](../MedEvalKit/Readme.md) evaluation pipeline first.
