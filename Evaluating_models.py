# Final version will include chunk-wise classification and metric analysis for three models
# using Human_text.txt as the ground truth.

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

# nltk.download('punkt', quiet=True)

# === BASE SETUP === #
MODELS = {
    "fakespot": "fakespot-ai/roberta-base-ai-text-detection-v1",
    "openai-detector": "roberta-base-openai-detector",
    "hellosimpleai-detector": "Hello-SimpleAI/chatgpt-detector-roberta"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

MAX_TOKENS = 510
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def clean_text(t):
    t = re.sub(r'\s+', ' ', t.replace("\n", " "))
    return unescape(t.strip())

def split_into_chunks(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk, current_tokens = "", 0
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_tokens + len(tokens) <= MAX_TOKENS:
            current_chunk += " " + sentence
            current_tokens += len(tokens)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = len(tokens)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def classify_chunks(chunks, model, tokenizer):
    cleaned = [clean_text(chunk) for chunk in chunks]
    tokenized = tokenizer(cleaned, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    dataset = DataLoader(list(zip(tokenized['input_ids'], tokenized['attention_mask'])), batch_size=16)

    model.eval()
    results = []
    with torch.no_grad():
        for input_ids_batch, mask_batch in dataset:
            input_ids_batch = input_ids_batch.to(device)
            mask_batch = mask_batch.to(device)
            outputs = model(input_ids=input_ids_batch, attention_mask=mask_batch)
            probs = F.softmax(outputs.logits, dim=-1)
            for prob in probs:
                score, label = torch.max(prob, dim=0)
                label_str = model.config.id2label[label.item()]
                results.append(label_str)
    return results

def evaluate_model(model_name, human_chunks, machine_chunks, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    all_chunks = [(c, "Human") for c in human_chunks] + [(c, "AI") for c in machine_chunks]
    texts, truths = zip(*all_chunks)
    predictions = classify_chunks(texts, model, tokenizer)

    results = []
    TP = TN = FP = FN = 0

        results.append({
            "chunk_index": i,
            "prediction": pred,
            "truth": true,
            "classification": category
        })


    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

# === LOAD AND CHUNK TEXTS === #
with open("Human_text.txt", "r", encoding="utf-8") as f:
    human_chunks = split_into_chunks(f.read())

with open("GPT_texts.txt", "r", encoding="utf-8") as f:
    machine_chunks = split_into_chunks(f.read())

# === EVALUATE EACH MODEL === #
for key, model_name in MODELS.items():
    print(f"Evaluating {key}...")
    evaluate_model(model_name, human_chunks, machine_chunks, f"evaluation_{key}.csv")
