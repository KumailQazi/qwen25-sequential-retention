"""
Self-Teacher Distillation QA Generator using Qwen2.5-7B-Instruct (revision a09a3545).
Generates clean factual QA items directly from Wikipedia corpus passages.
"""

import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_qa_pairs(corpus_path, output_train, output_panel, max_chunks=200):
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    revision = "a09a3545"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    chunk_size = 250
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i+chunk_size].strip()) > 60][:max_chunks]
    
    prompt_template = (
        "<|im_start|>system\n"
        "You are an objective scientific question generator. "
        "Based ONLY on the excerpt provided, generate exactly ONE factual question and ONE concise, single-sentence answer. "
        "Format: Question: <question> | Answer: <answer><|im_end|>\n"
        "<|im_start|>user\n"
        "Excerpt:\n{chunk}\n\nGenerate the QA pair:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    all_pairs = []
    print(f"Generating QA pairs for {corpus_path}...")
    for idx, chunk in enumerate(tqdm(chunks)):
        input_text = prompt_template.format(chunk=chunk.strip())
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if "Question:" in decoded and "Answer:" in decoded:
            parts = decoded.split("Answer:")
            q = parts[0].replace("Question:", "").strip()
            a = parts[1].strip()
        else:
            q = f"Question {idx+1} about excerpt"
            a = decoded
        all_pairs.append({"id": idx, "question": q, "answer": a, "source_excerpt": chunk[:100]})
        
    split_idx = max(15, int(len(all_pairs) * 0.25))
    panel_items = all_pairs[:split_idx]
    train_items = all_pairs[split_idx:]
    
    with open(output_panel, "w", encoding="utf-8") as f:
        for p in panel_items:
            f.write(json.dumps(p) + "\n")
    with open(output_train, "w", encoding="utf-8") as f:
        for t in train_items:
            f.write(json.dumps(t) + "\n")
            
    print(f"Generated {len(train_items)} train items and {len(panel_items)} panel items.")

if __name__ == "__main__":
    generate_qa_pairs("data/book_a/corpus.txt", "data/book_a/train_qa.jsonl", "data/book_a/panel_54.jsonl")
    generate_qa_pairs("data/book_b/corpus.txt", "data/book_b/train_qa.jsonl", "data/book_b/panel_71.jsonl")
