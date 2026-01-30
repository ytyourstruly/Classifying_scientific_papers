import argparse
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import re
import pandas as pd
from html import unescape
from sklearn.metrics import accuracy_score, precision_score, f1_score
import os

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
MAX_TOKENS = 512

# Load models
model1 = pipeline("text-classification", model="fakespot-ai/roberta-base-ai-text-detection-v1")
model2 = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
model3 = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")

# Cleaning functions
def clean_text(t):
    t = clean_markdown(t)
    t = re.sub(r"[\n\t\r]+", " ", t)
    t = re.sub(" +", " ", t)
    return t.strip()

def clean_markdown(md_text):
    md_text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
    md_text = re.sub(r'`[^`]*`', '', md_text)
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    md_text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', md_text)
    md_text = re.sub(r'(\*\*|__|\*|_)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'#+ ', '', md_text)
    md_text = re.sub(r'^(>.*$)', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^(\s*[-*+]|\d+\.)\s+', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'<.*?>', '', md_text)
    md_text = unescape(md_text)
    return md_text.strip()

# Chunk splitting
def split_into_chunks(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk, current_tokens = "", 0

    for sentence in sentences:
        tokenized = tokenizer.tokenize(sentence)
        if current_tokens + len(tokenized) <= MAX_TOKENS:
            current_chunk += " " + sentence
            current_tokens += len(tokenized)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = len(tokenized)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Classify chunks
def classify_chunks(chunks, true_label):
    results = []
    print(f"\nClassifying {len(chunks)} chunks (true label: {true_label})...\n")
    for chunk in tqdm(chunks, desc="Processing", unit="chunk"):
        result1 = model1(chunk, truncation=True, max_length=MAX_TOKENS)[0]
        result2 = model2(chunk, truncation=True, max_length=MAX_TOKENS)[0]
        result3 = model3(chunk, truncation=True, max_length=MAX_TOKENS)[0]

        results.append({
            "chunk": chunk,
            "true_label": true_label,
            "model1_pred": result1["label"],
            "model2_pred": result2["label"],
            "model3_pred": result3["label"],
        })
    return results

# Main execution
def main():
    parser = argparse.ArgumentParser(description='Compare models on GPT / Human texts and produce CSV outputs')
    parser.add_argument('--gpt-file', default='GPT_texts.txt', help='Path to GPT texts file')
    parser.add_argument('--human-file', default='Human_text.txt', help='Path to Human texts file')
    parser.add_argument('--out-chunks', default='csv_files/chunk_classification_results.csv', help='Output CSV for chunk classifications')
    parser.add_argument('--out-metrics', default='csv_files/model_metrics.csv', help='Output CSV for aggregated model metrics')
    args = parser.parse_args()

    # Read and clean
    with open(args.gpt_file, 'r', encoding='utf-8') as f:
        gpt_text = clean_text(f.read())
    with open(args.human_file, 'r', encoding='utf-8') as f:
        human_text = clean_text(f.read())

    # Split into chunks
    gpt_chunks = split_into_chunks(gpt_text)
    human_chunks = split_into_chunks(human_text)

    # Make chunks equal in number
    n_chunks = min(len(gpt_chunks), len(human_chunks))
    gpt_chunks = gpt_chunks[:n_chunks]
    human_chunks = human_chunks[:n_chunks]

    # Classify
    classified = classify_chunks(gpt_chunks, "AI") + classify_chunks(human_chunks, "Human")

    # Ensure output directory exists
    out_chunks_dir = os.path.dirname(args.out_chunks) or '.'
    os.makedirs(out_chunks_dir, exist_ok=True)

    # Save chunk classifications
    df = pd.DataFrame(classified)
    df.to_csv(args.out_chunks, index=False)

    # Calculate metrics
    y_true = df["true_label"].map({"Human": 0, "AI": 1}).values
    metrics = {}

    for model in ["model1_pred", "model2_pred", "model3_pred"]:
        if model == "model1_pred":
            y_pred = df[model].map({"Human": 0, "AI": 1}).values
        elif model == "model2_pred":
            y_pred = df[model].map({"Human": 0, "ChatGPT": 1}).values
        elif model == "model3_pred":
            y_pred = df[model].map({"Real": 0, "Fake": 1}).values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics[model] = {"accuracy": round(acc, 4), "precision": round(prec, 4), "f1_score": round(f1, 4)}

    # Ensure metrics output dir exists
    out_metrics_dir = os.path.dirname(args.out_metrics) or '.'
    os.makedirs(out_metrics_dir, exist_ok=True)

    # Save metrics
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index")
    df_metrics.index = ["Model1_fakespot", "Model2_simpleAI", "Model3_openai"]
    df_metrics.to_csv(args.out_metrics)

    print(f"\nSaved chunk classification results to '{args.out_chunks}'")
    print(f"Saved model metrics to '{args.out_metrics}'")


if __name__ == '__main__':
    main()
