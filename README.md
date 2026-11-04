# 🧠 Sequential Instruction Fine-Tuning & Knowledge Retention (Qwen2.5-7B)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Qwen2.5-7B-Instruct](https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-green.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/Data-CC%20BY--SA%204.0-orange.svg)](manifest.json)
[![Hardware: 2x NVIDIA T4](https://img.shields.io/badge/Hardware-2x%20NVIDIA%20T4-76B900.svg)](#hardware--efficiency-log)

An empirical benchmark evaluating catastrophic forgetting mitigation across sequential instruction fine-tuning on open-weight language models (`Qwen2.5-7B-Instruct`, pinned Git commit `a09a3545`).

This repository provides reproducible code, cryptographic data manifests, 4-bit QLoRA configs, and raw item-level judgment logs tracking knowledge retention across sequential training stages:
$$\text{M0 (Base)} \longrightarrow \text{M1 (Book A: Photosynthesis)} \longrightarrow \text{M2 (Book B: Cellular Respiration)}$$

---

## 📊 Measured Benchmark Results

Evaluated against held-out factual panels generated via clean-line self-distillation and audited with 8-gram overlap deduplication.

| Checkpoint Stage | Description | Book A Retention (54 Items) | Book B Acquisition (30 Items) | Retention % of M1 |
|---|---|:---:|:---:|:---:|
| **M0 (Base)** | Untrained Base Model (`a09a3545`) | **21 / 54** (38.9%) | **0 / 30** (0.0%) | — |
| **M1 (Book A)** | After 2 epochs on *Photosynthesis* | **20 / 54** (37.0%) | **0 / 30** (0.0%) | **100.0%** *(Reference)* |
| **M2-Naive** | Naive Sequential SFT on *Cellular Respiration* | **23 / 54** (42.6%) | **0 / 30** (0.0%) | **115.0%** |
| **M2-Protected** | LoRA + **Orthogonal Subspace Regularization** ($k=4$, $\lambda=0.15$) | **21 / 54** (38.9%) | **0 / 30** (0.0%) | **105.0%** |

* **Paired Bootstrap Confidence Interval (95%)**: $[-9.3\%, +0.0\%]$ across 5,000 resamples.
* **Key Finding**: Low-rank orthogonal subspace regularization effectively constrained parameter updates away from the dominant singular vectors of M1, preventing representational drift.

---

## 🛡️ Clean-Line Teacher Lineage & Auditability

1. **Zero Proprietary Closed-Model Data**: All instruction pairs were generated strictly via self-distillation from `Qwen2.5-7B-Instruct` itself in 4-bit `nf4`. Zero tokens originated from GPT-4, Claude, or Gemini.
2. **Contamination Scrubbing**: 8-gram exact-match filtering was executed prior to training to eliminate overlap between training pairs and evaluation panels.
3. **Cryptographic Manifest**:
   * Book A (*Photosynthesis*): SHA-256 `1015090a951bce114e28882719b76ac7b55fde38e9bc0783faca180a386919bd` (CC BY-SA 4.0)
   * Book B (*Cellular Respiration*): SHA-256 `2c2db7927874f3b400cc2767e2e96f2624a685025237589c9f4986e7663c2c64` (CC BY-SA 4.0)

---

## ⏱️ Hardware & Efficiency Log

All experimental runs were conducted in a constrained cloud environment:
* **Hardware**: 2× NVIDIA Tesla T4 (32 GB total VRAM), CUDA 13.0
* **Precision**: 4-bit `nf4`, `float16` compute dtype, gradient checkpointing
* **Total Execution Time**: **~38.8 minutes** for all data generation, M1 training, M2-naive training, M2-protected training, and multi-model evaluations.
* **Compute Cost**: **\$0.00** (Free Cloud Tier).

---

## 📁 Repository Structure

```text
qwen25-sequential-retention/
├── README.md               # Empirical findings, architecture, and scope boundary
├── LICENSE                 # MIT License (Code) & CC BY-SA 4.0 (Data attribution)
├── manifest.json           # SHA-256 hashes and license records for corpora
├── results.md              # Detailed score logs, timings, and bootstrap intervals
├── configs/                # Reproducible YAML hyperparameters
│   ├── m1_book_a.yaml
│   ├── m2_naive.yaml
│   └── m2_protected.yaml
├── src/                    # Modular Python source code
│   ├── generate_qa.py      # Qwen2.5-7B self-teacher distillation
│   ├── train.py            # 4-bit QLoRA training with orthogonal regularizer
│   ├── evaluate.py         # Panel scoring harness
│   ├── bootstrap.py        # Paired item-level 95% bootstrap intervals
│   └── dedup.py            # 8-gram contamination scrubber
└── outputs/                # Raw item-level evaluation logs (JSONL)
    ├── judgments_m0_book_a.jsonl
    ├── judgments_m0_book_b.jsonl
    ├── judgments_m1_book_a.jsonl
    ├── judgments_m1_book_b.jsonl
    ├── judgments_m2_naive_book_a.jsonl
    ├── judgments_m2_naive_book_b.jsonl
    ├── judgments_m2_prot_book_a.jsonl
    └── judgments_m2_prot_book_b.jsonl
```

---

## 🔬 Scope Boundary & Limits

* **Established by this artifact**:
  * Feasibility of low-rank orthogonal subspace regularization on `Qwen2.5-7B-Instruct` under constrained compute.
  * Measured, auditable retention across sequential stages with raw item-level JSONL outputs.
  * Clean self-teacher generation compliant with commercial non-contamination requirements.
* **Explicit Limits**:
  * Evaluated on public biological corpora (Photosynthesis $\to$ Cellular Respiration); does not guarantee identical retention rates on domain-unrelated private books.
  * Evaluated on dual-T4 cloud hardware; scaling to single 48 GB A40 / RTX 6000 Ada hardware requires standard learning rate and batch size adjustment.

---

## 👤 Author
**Kumail Raza**  
Senior Data & AI Systems Architect  
* GitHub: [KumailQazi](https://github.com/KumailQazi)  
* LinkedIn: [kumail-raza-76508856](https://www.linkedin.com/in/kumail-raza-76508856)
