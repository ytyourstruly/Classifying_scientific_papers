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
from collections import defaultdict
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
        tokenized = classifier.tokenizer.tokenize(sentence)
        token_count = len(tokenized)

        # If the sentence is too long, split it into smaller parts
        if token_count > MAX_TOKENS:
            # Split the sentence into smaller chunks
            sub_sentences = [sentence[i:i + MAX_TOKENS] for i in range(0, len(sentence), MAX_TOKENS)]
            for sub_sentence in sub_sentences:
                tokenized_sub = classifier.tokenizer.tokenize(sub_sentence)
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

def confidence_weighted_voting(predictions):
    label_confidence_sum = {}
    for p in predictions:
        for label, confidence in p["scores"].items():
            label_confidence_sum[label] = label_confidence_sum.get(label, 0.0) + confidence * p["weight"]  # weighted by token count

    best_label = None
    total_confidence = -1.0

    for label, confidence in label_confidence_sum.items():
        if confidence > total_confidence:
            best_label = label
            total_confidence = confidence
        elif confidence == total_confidence:
            # Tie-breaker: prefer 'Human'
            if best_label != 'Human' and label == 'Human':
                best_label = 'Human'
                total_confidence = confidence
    total_sum = sum(label_confidence_sum.values())
    confidence_pct = total_confidence / total_sum * 100 if total_sum else 0.0
    return best_label, round(confidence_pct, 2)





classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=tokenizer, device=0)

def classify_text(text):
    if not text.strip():
        return "Unknown", 0.0

    chunks = split_into_chunks(clean_text(text))
    if not chunks:
        return "Unknown", 0.0

    print(f"\nClassifying {len(chunks)} chunks...\n")
    model_preds = []

    for i, (chunk_text, weight) in enumerate(tqdm(chunks, desc="Processing", unit="chunk")):
        result = classifier(chunk_text, truncation=True, max_length=MAX_TOKENS)[0]

        model_preds.append({
            "scores": {result["label"]: result["score"]},
            "weight": weight
        })
    

    # Aggregated label and confidence score
    label, avg_score = confidence_weighted_voting(model_preds)

    # === New metric: high-confidence AI chunk ratio ===
    total_tokens = sum(p["weight"] for p in model_preds)
    high_conf_ai_tokens = 0
    for p in model_preds:
        for label, score in p["scores"].items():
            if label == "AI" and score >= 0.9:
                high_conf_ai_tokens += p["weight"]
    ai_token_ratio = (high_conf_ai_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    print(label)
    print(round(ai_token_ratio))
    return label, avg_score, round(ai_token_ratio, 2)




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



for year_dir in base_dirs:
    year = os.path.basename(year_dir)
    txt_files = [
        os.path.join(year_dir, f)
        for f in os.listdir(year_dir)
        if f.endswith(".txt")
    ]

    # Sequential processing instead of threading
    year_data = []
    for file_path in txt_files:
        year_data.append(process_file(file_path))

    df = pd.DataFrame(year_data)
    df.to_csv(f"classification_{year}.csv", index=False)
    print(f"Saved classification_{year}.csv")

