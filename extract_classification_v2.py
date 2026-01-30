import os
import re
import pandas as pd
# import nltk
from html import unescape
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import pipeline
from tqdm import tqdm
# === SETUP === #


MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 512
base_dirs = ["texts/2020", "texts/2021", "texts/2022", "texts/2023", "texts/2024"]

# === CLEANING === #
def clean_text(t):
    t = clean_markdown(t)
    t = t.replace("\n", " ").replace("\t", " ").replace("^M", " ").replace("\r", " ")
    t = t.replace(" ,", ",")
    t = re.sub(" +", " ", t)
    return t

def clean_markdown(text):
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*[-*+]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|.*?\|', '', text)
    text = re.sub(r'<.*?>', '', text)
    return unescape(text)

# === CHUNKING === #
def split_into_chunks(text):
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences

    chunks = []
    current_chunk, current_tokens = "", 0

    for sentence in sentences:
        # Tokenize the sentence using the classifier's tokenizer
        tokenized = tokenizer.encode(sentence, add_special_tokens=True)
        token_count = len(tokenized)

        # If the sentence is too long, split it into smaller parts
        if token_count > MAX_TOKENS:
            # Split the sentence into smaller chunks
            sub_sentences = [sentence[i:i + MAX_TOKENS] for i in range(0, len(sentence), MAX_TOKENS)]
            for sub_sentence in sub_sentences:
                tokenized_sub = tokenizer.encode(sentence, add_special_tokens=True)
                token_count_sub = len(tokenized_sub)

                # Add the sub-sentence to the current chunk if space allows
                if current_tokens + token_count_sub <= MAX_TOKENS:
                    current_chunk += " " + sub_sentence
                    current_tokens += token_count_sub
                else:
                    if current_chunk:
                        chunks.append((current_chunk.strip(), current_tokens))
                    current_chunk = sub_sentence
                    current_tokens = token_count_sub
        else:
            # If the sentence is short enough, process as usual
            if current_tokens + token_count <= MAX_TOKENS:
                current_chunk += " " + sentence
                current_tokens += token_count
            else:
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_tokens))
                current_chunk = sentence
                current_tokens = token_count

    if current_chunk:
        chunks.append((current_chunk.strip(), current_tokens))

    return chunks

from collections import defaultdict

def confidence_weighted_voting(predictions):
    label_confidence_sum = defaultdict(float)

    for p in predictions:
        for label, confidence in p["scores"].items():
            label_confidence_sum[label] += confidence * p["weight"]  # Now weighted by token count

    best_label, total_confidence = max(label_confidence_sum.items(), key=lambda x: x[1])
    total_sum = sum(label_confidence_sum.values())
    confidence_pct = total_confidence / total_sum * 100 if total_sum else 0.0
    return best_label, round(confidence_pct.item(), 2)





# classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=tokenizer, device=0)

def classify_text(text):
    if not text.strip():
        return "Unknown", 0.0, 0.0

    chunks = split_into_chunks(clean_text(text))
    if not chunks:
        return "Unknown", 0.0, 0.0

    chunk_texts = [chunk[0] for chunk in chunks]
    token_counts = [chunk[1] for chunk in chunks]  # token count per chunk

    # Convert to HF Dataset and tokenize for batching
    dataset = Dataset.from_dict({"text": chunk_texts})
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=MAX_TOKENS),
        batched=True
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(
        list(zip(tokenized["input_ids"], tokenized["attention_mask"], token_counts)),
        batch_size=16
    )

    model.to(device)
    model.eval()
    model_preds = []

    with torch.no_grad():
        for input_ids_batch, attention_mask_batch, token_counts_batch in loader:
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            probs = F.softmax(outputs.logits, dim=1)

            for prob, weight in zip(probs, token_counts_batch):
                top_score, top_label = torch.max(prob, dim=0)
                label_str = model.config.id2label[top_label.item()]
                model_preds.append({
                    "scores": {label_str: top_score.item()},
                    "weight": weight
                })

    # Aggregated label and confidence score
    label, avg_score = confidence_weighted_voting(model_preds)

    # === New metric: high-confidence AI chunk ratio ===
    total_tokens = sum(p["weight"] for p in model_preds)
    high_conf_ai_tokens = sum(
        p["weight"]
        for p in model_preds
        if list(p["scores"].keys())[0] == "AI" and list(p["scores"].values())[0] >= 0.9
    )
    ai_token_ratio = (high_conf_ai_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    print(label)
    print(round(ai_token_ratio.item()))
    return label, round(avg_score * 100, 2), round(ai_token_ratio.item(), 2)




# === PROCESS FILE === #
def process_file(file_path):
    paper_id = os.path.splitext(os.path.basename(file_path))[0]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        label, confidence, ai_ratio = classify_text(text)
        return {
            "paper_id": paper_id,
            "Classification": label,
            "Confidence (%)": confidence,
            "AI Chunk Percentage (%)": ai_ratio
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            "paper_id": paper_id,
            "Classification": "Error",
            "Confidence (%)": 0.0,
            "AI Chunk Percentage (%)": 0.0
        }



# === LOOP THROUGH FILES (No threading) === #



def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Classify texts under given directories and write CSVs (v2)')
    parser.add_argument('--input', '-i', default=None,
                        help='Input directory (e.g. texts/2020) or comma-separated list of dirs. If omitted, defaults in-script years are used')
    parser.add_argument('--output-dir', '-o', default='csv_files', help='Directory to write CSV outputs')
    args = parser.parse_args()

    if args.input:
        base_dirs_local = [p.strip() for p in args.input.split(',') if p.strip()]
    else:
        base_dirs_local = base_dirs

    os.makedirs(args.output_dir, exist_ok=True)

    for year_dir in base_dirs_local:
        year = os.path.basename(year_dir.rstrip('/'))
        matched = glob.glob(os.path.join(year_dir, '*.txt')) if os.path.isdir(year_dir) or '*' in year_dir else glob.glob(year_dir)
        txt_files = [p for p in matched if p.endswith('.txt')]

        year_data = []
        for file_path in txt_files:
            year_data.append(process_file(file_path))

        out_path = os.path.join(args.output_dir, f"classification_{year}.csv")
        df = pd.DataFrame(year_data)
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()

