from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, pipeline
import os
import re
import pandas as pd
# import nltk
from html import unescape


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
MAX_TOKENS = 512

model1_classifier = pipeline(
    "text-classification",
    model="fakespot-ai/roberta-base-ai-text-detection-v1"
)
model2_classifier = pipeline(
    "text-classification",
    model="Hello-SimpleAI/chatgpt-detector-roberta"
)
model3_classifier = pipeline(
    "text-classification",
    model="Hello-SimpleAI/chatgpt-detector-roberta"
)
def split_into_chunks(text, max_tokens):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
def clean_text(t):
    t = clean_markdown(t)
    t = t.replace("\n"," ")
    t = t.replace("\t"," ")
    t = t.replace("^M"," ")
    t = t.replace("\r"," ")
    t = t.replace(" ,", ",")
    t = re.sub(" +", " ", t)
    return t
def clean_markdown(md_text):
    # Remove code blocks
    md_text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
    # Remove inline code
    md_text = re.sub(r'`[^`]*`', '', md_text)
    # Remove images
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    # Remove links but keep link text
    md_text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', md_text)
    # Remove bold and italic (groups of *, _)
    md_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'(\*|_)(.*?)\1', r'\2', md_text)
    # Remove headings
    md_text = re.sub(r'#+ ', '', md_text)
    # Remove blockquotes
    md_text = re.sub(r'^>.*$', '', md_text, flags=re.MULTILINE)
    # Remove list markers
    md_text = re.sub(r'^(\s*[-*+]|\d+\.)\s+', '', md_text, flags=re.MULTILINE)
    # Remove horizontal rules
    md_text = re.sub(r'^\s*[-*_]{3,}\s*$', '', md_text, flags=re.MULTILINE)
    # Remove tables
    md_text = re.sub(r'\|.*?\|', '', md_text)
    # Remove raw HTML tags
    md_text = re.sub(r'<.*?>', '', md_text)
    # Decode HTML entities
    md_text = unescape(md_text)
    return md_text
def weighted_average(predictions):
    total_weight = sum(p['weight'] for p in predictions)
    label_scores = {'AI': 0.0, 'Human': 0.0}

    for p in predictions:
        weight = p['weight']
        label, score = list(p['scores'].items())[0]

        # Add the given label
        label_scores[label] += score * weight

        # Add the complementary label
        other_label = 'Human' if label == 'AI' else 'AI'
        label_scores[other_label] += (1 - score) * weight

    avg_scores = {label: score_sum / total_weight for label, score_sum in label_scores.items()}
    max_score = max(avg_scores.values())
    top_labels = [label for label, score in avg_scores.items() if score == max_score]
    best_label = 'Human' if 'Human' in top_labels else top_labels[0]
    return best_label, max_score 

def classify_text(text):
    chunks = split_into_chunks(text, MAX_TOKENS)
    
    model1_preds, model2_preds = [], []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        truncated_chunk = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        token_count = len(inputs["input_ids"][0])
        weight = token_count
        cleaned = clean_text(truncated_chunk)
        print(cleaned)
        result1 = model1_classifier(cleaned)[0]
        scores1 = {result1['label']: result1['score']}
        print(scores1)
        model1_preds.append({'scores': scores1, 'weight': weight})

        # result2 = client.text_classification(cleaned, model="roberta-base-openai-detector")
        # scores2 = {r['label']: r['score'] for r in result2}
        # result2 = client.text_classification(cleaned, model="Hello-SimpleAI/chatgpt-detector-roberta")
        # scores2 = {r['label']: r['score'] for r in result1}
        # model2_preds.append({'scores': scores2, 'weight': weight})

    label1, avg_score1 = weighted_average(model1_preds)
    # label2, avg_score2 = weighted_average(model2_preds)

    # Decision logic
    # if avg_score1 >= 0.99:
    #     final_label = label1
    #     final_score = avg_score1
    # else:
    #     if avg_score2 >= 0.5:
    #         final_label = "Human" if label2.lower() == "real" else "ChatGPT"
    #         final_score = avg_score2
    #     else:
    #         final_label = label1
    #         final_score = avg_score1

    return {
        "First Classification": (label1, round(avg_score1 * 100, 2)),
        # "Second Classification": (label1, round(avg_score1 * 100, 2))
    }
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Classify a text file with baseline classifier')
    parser.add_argument('--input', '-i', default='GPT_texts.txt', help='Input text file to classify')
    parser.add_argument('--output', '-o', default='csv_files/baseline_classification.csv', help='Output CSV file to write results')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        full_text = f.read()

    result = classify_text(full_text)
    print(f"First: {result['First Classification'][0]} ({result['First Classification'][1]}%)")

    # Save to CSV (single-row)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df = pd.DataFrame([{
        'file': os.path.basename(args.input),
        'classification': result['First Classification'][0],
        'score_pct': result['First Classification'][1]
    }])
    df.to_csv(args.output, index=False)
    print(f"Saved baseline result to {args.output}")


if __name__ == '__main__':
    main()
    # print(f"Model 2: {result['Second Classification'][0]} ({result['Second Classification'][1]}%)")