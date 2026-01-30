# Classifying Scientific Papers — Usage

Overview

This repository contains scripts to classify scientific texts (from the `osf` or `texts` folders) and produce CSV outputs with classification results and evaluation metrics.

Prerequisites

- Python 3.8+ (recommend 3.10+)
- Install dependencies (if any) used by the project. If there's a `requirements.txt`, install with pip.

Setup

- Ensure you have a `csv_files` folder at the project root to collect CSV outputs. Create it if missing:

```bash

# Classifying Scientific Papers — Usage

Overview

This repository contains scripts to classify scientific texts (from the `osf` or `texts` folders) and produce CSV outputs with classification results and evaluation metrics. The main scripts now accept CLI flags `--input` and `--output` (or `--output-dir`) and write CSVs to `csv_files/` by default.

Prerequisites

- Python 3.8+ (recommend 3.10+)
- Install dependencies (if any). If there's a `requirements.txt`, install with:

```bash
python3 -m pip install -r requirements.txt
```

Quick setup

- Create the CSV output folder if it doesn't exist:

```bash
mkdir -p csv_files
```

Entry-point scripts (with CLI)

- [baseline_classifying_chatgpt.py](baseline_classifying_chatgpt.py) — baseline classifier; flags: `--input`, `--output`
- [extract_classification.py](extract_classification.py) — batch classify texts; flags: `--input`, `--output-dir`
- [extract_classification_v2.py](extract_classification_v2.py) — same as above (v2)
- [compare_models.py](compare_models.py) — compares GPT vs Human texts and saves chunk results and metrics; flags: `--gpt-file`, `--human-file`, `--out-chunks`, `--out-metrics`
- [calculate_metrics.py](calculate_metrics.py) — computes metrics from one or more CSVs; flags: `--input`, `--output`, `--truth-col`, `--pred-col`, `--pos-label`

Notes on defaults

- All scripts default to writing outputs under `csv_files/` when an output path is not provided.
- `calculate_metrics.py` expects columns named `truth` and `prediction` by default; change with `--truth-col` / `--pred-col`.

Usage examples

Classify a single text file with the baseline classifier (writes a single-row CSV):

```bash
python3 baseline_classifying_chatgpt.py --input texts/sample.txt --output csv_files/baseline_sample.csv
```

Batch-classify a year's texts and write per-year CSVs:

```bash
python3 extract_classification.py --input texts/2020 --output-dir csv_files
```

Batch-classify using v2 (same interface):

```bash
python3 extract_classification_v2.py --input "texts/2020,texts/2021" --output-dir csv_files
```

Compare models on the standard GPT/Human test files and save chunk-level classifications and aggregated metrics:

```bash
python3 compare_models.py --gpt-file GPT_texts.txt --human-file Human_text.txt \
	--out-chunks csv_files/chunk_classification_results.csv \
	--out-metrics csv_files/model_metrics.csv
```

Calculate metrics for one or many evaluation CSVs (supports glob or directory):

```bash
# single file
python3 calculate_metrics.py --input csv_files/evaluation_fakespot.csv --output csv_files/fakespot_metrics.csv

# multiple files via glob (aggregates and prints each)
python3 calculate_metrics.py --input "csv_files/evaluation_*.csv" --output csv_files/all_metrics.csv

# change truth/pred column names
python3 calculate_metrics.py --input csv_files/evaluation_openai-detector.csv \
	--truth-col truth --pred-col prediction --pos-label Human
```

Outputs

- Classification CSVs: `csv_files/*.csv`
- Chunk classification: `csv_files/chunk_classification_results.csv` (example)
- Aggregated metrics: `csv_files/model_metrics.csv` or any filename passed to `--output`

Troubleshooting

- If a script fails due to missing dependencies, install packages listed in `requirements.txt` or inspect the top of the script for imports.
- If your CSV columns differ from the defaults, use the column flags on `calculate_metrics.py`.

Next steps I can take for you

- Move existing CSV files in the repo root into `csv_files/` (I can do this now).
- Remove or archive any files you consider unnecessary — tell me which ones to delete and I'll remove them.
- Add or adjust argparse options further if you'd like different defaults.


