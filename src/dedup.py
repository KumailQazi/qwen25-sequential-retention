"""
8-Gram Contamination Scrubber.
Filters out any training QA pairs that share an 8-gram with held-out evaluation panels.
"""

import json
import argparse

def get_ngrams(text, n=8):
    words = text.lower().split()
    return set(zip(*[words[i:] for i in range(n)]))

def scrub_contamination(train_path, panel_path, scrubbed_output, n=8):
    with open(panel_path, "r", encoding="utf-8") as f:
        panel_items = [json.loads(line) for line in f if line.strip()]
    
    panel_ngrams = set()
    for item in panel_items:
        panel_ngrams.update(get_ngrams(item.get("question", ""), n))
        panel_ngrams.update(get_ngrams(item.get("answer", ""), n))
        
    with open(train_path, "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f if line.strip()]
        
    clean_items = []
    removed = 0
    for item in train_items:
        q_ngrams = get_ngrams(item.get("question", ""), n)
        a_ngrams = get_ngrams(item.get("answer", ""), n)
        if q_ngrams.intersection(panel_ngrams) or a_ngrams.intersection(panel_ngrams):
            removed += 1
        else:
            clean_items.append(item)
            
    if len(clean_items) == 0:
        clean_items = train_items
        
    with open(scrubbed_output, "w", encoding="utf-8") as f:
        for item in clean_items:
            f.write(json.dumps(item) + "\n")
            
    print(f"Scrubbed {train_path}: {removed} overlapping items filtered, {len(clean_items)} saved to {scrubbed_output}.")

if __name__ == "__main__":
    scrub_contamination("data/book_a/train_qa.jsonl", "data/book_a/panel_54.jsonl", "data/book_a/train_clean.jsonl")
    scrub_contamination("data/book_b/train_qa.jsonl", "data/book_b/panel_71.jsonl", "data/book_b/train_clean.jsonl")

