"""
Paired Item-Level Bootstrap 95% Confidence Interval Calculator.
Resamples identical question IDs on both candidate configurations simultaneously.
"""

import json
import random
import argparse

def compute_paired_bootstrap(scores_prot, scores_naive, n_resamples=5000, seed=42):
    random.seed(seed)
    n = len(scores_prot)
    diffs = []
    for _ in range(n_resamples):
        idx = [random.randrange(n) for _ in range(n)]
        mean_prot = sum(scores_prot[i] for i in idx) / n
        mean_naive = sum(scores_naive[i] for i in idx) / n
        diffs.append((mean_prot - mean_naive) * 100)
        
    diffs.sort()
    ci_low = diffs[int(0.025 * len(diffs))]
    ci_high = diffs[int(0.975 * len(diffs))]
    mean_diff = sum(diffs) / len(diffs)
    return mean_diff, ci_low, ci_high

def load_scores(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line)["correct"] for line in f if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prot", default="outputs/judgments_m2_prot_book_a.jsonl")
    parser.add_argument("--naive", default="outputs/judgments_m2_naive_book_a.jsonl")
    args = parser.parse_args()
    
    prot_scores = load_scores(args.prot)
    naive_scores = load_scores(args.naive)
    
    mean_adv, low, high = compute_paired_bootstrap(prot_scores, naive_scores)
    print(f"Sample Size: N = {len(prot_scores)} items")
    print(f"Mean Advantage: {mean_adv:+.1f}%")
    print(f"95% Paired Bootstrap Confidence Interval: [{low:+.1f}%, {high:+.1f}%]")

