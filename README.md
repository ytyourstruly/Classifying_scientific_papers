# Classifying Scientific Papers — Usage

Overview

This repository contains scripts to classify scientific texts (from the `osf` or `texts` folders) and produce CSV outputs with classification results and evaluation metrics.

Prerequisites

- Python 3.8+ (recommend 3.10+)

Setup

- Ensure you have a `csv_files` folder at the project root to collect CSV outputs. 
- Install required packages:


```bash
python3 -m pip install -r requirements.txt
```

Quick start

1. Create the output folder:

```bash
mkdir -p csv_files
```

2. Run a classifier or batch job (examples below).

Primary scripts and flags

- [baseline_classifying_chatgpt.py](baseline_classifying_chatgpt.py)
	- Flags: `--input` (input text file), `--output` (single-row CSV output)
- [extract_classification.py](extract_classification.py)
	- Flags: `--input` (directory or comma-separated dirs), `--output-dir`
- [extract_classification_v2.py](extract_classification_v2.py)
	- Same interface as above (v2)
- [compare_models.py](compare_models.py)
	- Flags: `--gpt-file`, `--human-file`, `--out-chunks`, `--out-metrics`
- [calculate_metrics.py](calculate_metrics.py)
	- Flags: `--input` (file, glob, or directory), `--output`, `--truth-col`, `--pred-col`, `--pos-label`

Usage examples

```bash
# Classify a single file (baseline)
python3 baseline_classifying_chatgpt.py --input texts/sample.txt --output csv_files/baseline_sample.csv

# Batch classify a year's texts
python3 extract_classification.py --input texts/2020 --output-dir csv_files

# Compare GPT vs Human and save chunk results + metrics
python3 compare_models.py --gpt-file GPT_texts.txt --human-file Human_text.txt \
	--out-chunks csv_files/chunk_classification_results.csv \
	--out-metrics csv_files/model_metrics.csv

# Calculate metrics for evaluation CSVs
python3 calculate_metrics.py --input "csv_files/evaluation_*.csv" --output csv_files/all_metrics.csv
```

Outputs

- Classification CSVs: `csv_files/*.csv`
- Chunk-level classifications: example `csv_files/chunk_classification_results.csv`
- Aggregated metrics: example `csv_files/model_metrics.csv`

Notes

- `calculate_metrics.py` expects by default columns named `truth` and `prediction`; change with `--truth-col` / `--pred-col`.
- `torch` installation may require following PyTorch platform-specific instructions (CUDA vs CPU). See https://pytorch.org/get-started/locally/ for details.

Files

- Requirements: [requirements.txt](requirements.txt)
- Main scripts: [baseline_classifying_chatgpt.py](baseline_classifying_chatgpt.py), [extract_classification.py](extract_classification.py), [extract_classification_v2.py](extract_classification_v2.py), [compare_models.py](compare_models.py), [calculate_metrics.py](calculate_metrics.py)

