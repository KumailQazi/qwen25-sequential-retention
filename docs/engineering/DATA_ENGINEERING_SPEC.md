# LLM_DALI Data Engineering Specification: Ingestion, Curation & Masking

## 1. Data Pipeline Architecture & Design Principles

The data engineering pipeline provides an end-to-end lineage from public Wikipedia raw text to PyTorch-ready tensor batches. The pipeline adheres to four non-negotiable data quality principles:
1. **Zero External API Contamination**: Exclusively self-distilled via `Qwen2.5-7B-Instruct` ($0 API cost, zero proprietary model text).
2. **Absolute Determinism & Reproducibility**: Cryptographic SHA-256 validation on all source articles and random seed pinning.
3. **Strict Decontamination**: Zero 8-gram overlap permitted between training sets and test evaluation panels.
4. **Answer-Only Gradient Attribution**: Prompt tokens are strictly masked out (`label = -100`) to guarantee gradient attribution is confined to answer tokens.

---

## 2. Ingestion Contracts & Data Schemas

### 2.1 Manifest Schema (`data/manifest.json`)
The manifest acts as the tamper-evident provenance log:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CorpusManifest",
  "type": "object",
  "properties": {
    "book_a": {
      "type": "object",
      "properties": {
        "domain": { "type": "string", "enum": ["Quantum Physics & Condensed Matter"] },
        "articles": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "title": { "type": "string" },
              "url": { "type": "string", "format": "uri" },
              "license": { "type": "string", "enum": ["CC BY-SA 4.0"] },
              "sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
              "char_count": { "type": "integer", "minimum": 1000 }
            },
            "required": ["title", "url", "license", "sha256", "char_count"]
          }
        }
      },
      "required": ["domain", "articles"]
    },
    "book_b": {
      "type": "object",
      "properties": {
        "domain": { "type": "string", "enum": ["Molecular Biology & Epigenetics"] },
        "articles": { "type": "array" }
      },
      "required": ["domain", "articles"]
    }
  },
  "required": ["book_a", "book_b"]
}
```

---

### 2.2 Evaluation Panel Schema (`data/book_{x}/panel.jsonl`)
Each line in `panel.jsonl` contains an immutable test item:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EvaluationPanelItem",
  "type": "object",
  "properties": {
    "id": { "type": "string", "description": "Deterministic hash ID e.g. b1_0042" },
    "question": { "type": "string", "minLength": 10 },
    "answer": { "type": "string", "minLength": 1, "maxLength": 200 },
    "keywords": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "source_article": { "type": "string" },
    "chunk_id": { "type": "integer" }
  },
  "required": ["id", "question", "answer", "keywords", "source_article"]
}
```

---

### 2.3 Evaluation Judgment Schema (`outputs/judgments_{checkpoint}_{book}.jsonl`)
Every evaluated generation is logged to an auditable judgment ledger:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EvaluationJudgmentRecord",
  "type": "object",
  "properties": {
    "item_id": { "type": "string" },
    "checkpoint": { "type": "string", "enum": ["m0", "m1", "m2_naive", "m2_kl", "m2_prot"] },
    "book": { "type": "string", "enum": ["book_a", "book_b"] },
    "prompt": { "type": "string" },
    "gold_answer": { "type": "string" },
    "predicted_answer": { "type": "string" },
    "strict_match": { "type": "boolean" },
    "lenient_match": { "type": "boolean" },
    "token_f1": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "latency_ms": { "type": "number", "minimum": 0.0 }
  },
  "required": ["item_id", "checkpoint", "book", "gold_answer", "predicted_answer", "strict_match", "lenient_match"]
}
```

---

## 3. Chunking & Ingestion Engineering

```
Raw Text Stream: [ ... w_1, w_2, w_3, ... w_n ... ]
│
├── Window 0: [ Char 000 ──────── Char 500 ] (Aligned to last whitespace before char 500)
│   Stride: 250 Chars
├── Window 1: [ Char 250 ──────── Char 750 ] (Aligned to last whitespace before char 750)
│   Stride: 250 Chars
└── Window 2: [ Char 500 ──────── Char 1000]
```

### Parameter Invariants:
* $\text{Window Size} = 500\text{ characters}$ ($\approx 80\text{ words}$).
* $\text{Stride Size} = 250\text{ characters}$ ($\approx 40\text{ words}$, yielding $50\%$ semantic overlap).
* $\text{Max Chunks per Corpus} = 1,200\text{ chunks}$.
* $\text{Boundary Condition}$: Chunks never terminate mid-word. Regex `\s` backward scanning determines cut boundaries.

---

## 4. Decontamination: 8-Gram Shingle Algorithm

To prevent test-set memorization or data leakage, candidate training QA pairs are audited against the test panel shingles:

```python
def extract_8grams(text: str) -> set:
    tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    if len(tokens) < 8:
        return set()
    return {" ".join(tokens[i:i+8]) for i in range(len(tokens) - 7)}

def audit_dataset_cleanliness(training_pool, panel_items):
    # Construct panel test shingle index
    panel_shingles = set()
    for item in panel_items:
        panel_shingles.update(extract_8grams(item["question"]))
        panel_shingles.update(extract_8grams(item["answer"]))
        
    clean_pool = []
    contaminated_count = 0
    
    for candidate in training_pool:
        cand_shingles = extract_8grams(candidate["question"]) | extract_8grams(candidate["answer"])
        if cand_shingles & panel_shingles:
            contaminated_count += 1
            continue  # Discard contaminated candidate
        clean_pool.append(candidate)
        
    return clean_pool, contaminated_count
```

---

## 5. Tokenization & Answer-Only Masking Algorithm

PyTorch `CrossEntropyLoss` ignores label indices set to `-100`. The tokenization pipeline formats inputs as ChatML and sets all prompt tokens to `-100`:

```python
def encode_qa_pair(tokenizer, question: str, answer: str, max_length: int = 512):
    # Delimit prompt structure
    prompt_str = (
        "<|im_start|>system\n"
        "You are a factual assistant. Answer the question directly and concisely.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = f"{answer}<|im_end|>"
    
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    
    # Truncate and pad
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
```
This guarantees that backpropagation is strictly localized to the answer token sequence.
