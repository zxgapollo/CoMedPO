# evaluation — Evaluation Tools

This directory contains automatic evaluation tools commonly used in medical imaging (especially radiology report generation), complementing the metrics provided by MedEvalKit.


| Subdirectory | Purpose | Main metric |
|--------------|---------|-------------|
| `radgraph/` | Radiology report entity-relation graph extraction | F1-RadGraph |
| `CheXbert/` | Automated labeling of 14 chest X-ray observations | F1-CheXbert, RadGraph-XL support |
| `medgemma/` | Official Google MedGemma notebooks | Baseline model reference |

## radgraph / RadGraph-XL

Parses radiology reports into entity-relation graphs to measure factual consistency between generated and reference reports.

```bash
pip install -e evaluation/radgraph
```

See [`radgraph/README.md`](radgraph/README.md) for details (RadGraph-XL and F1-RadGraph computation).

## CheXbert

Automatically labels 14 observations in chest X-ray reports (Fracture, Consolidation, Cardiomegaly, Edema, Pleural Effusion, etc.); paper: [EMNLP 2020](https://arxiv.org/abs/2004.09167).

```bash
cd evaluation/CheXbert
pip install -r requirements.txt
# Labeling
python src/label.py --reports_path <report file> --output_path <output>
```

> **Model weights must be downloaded separately**:
> - `BiomedVLP-CXR-BERT-general/` (1.7 GB) — follow the official instructions
> - `radgraph-xl.tar.gz` (397 MB) — download and extract before use

## medgemma

Google's medical multimodal models built on Gemma 3, available in 4B / 27B variants with text and image understanding capabilities. The official notebooks are kept here as a baseline model and usage reference; see [`medgemma/README.md`](medgemma/README.md).
