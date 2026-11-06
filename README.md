# Qwen2.5-7B Sequential Retention — Milestone 1 Artifact

Reproducible continual learning pipeline for LLM_DALI. **Version 2.1.0**

## What was built
- End-to-end training and evaluation pipeline for Qwen2.5-7B-Instruct with LoRA adapters
- Self-auditing evaluation harness with panel-validity gates
- Three M2 variants: Naive fine-tuning, KL distillation, Replay + KL
- License-clean training corpora from Wikipedia (CC BY-SA 4.0)

## Methodology
- Base model: Qwen2.5-7B-Instruct (revision a09a3545)
- Training: LoRA adapters, answer-only loss masking, 3 epochs
- Book A (pre-training): Photosynthesis / cellular energy topics
- Book B (continual learning): Cellular respiration / metabolism topics

## Key results (public proxy data, one seed, dual T4)
| Checkpoint | Book A Retention | Book B Acquisition (strict) |
|-----------|------------------|----------------------------|
| M2-Naive  | 57.5%            | 25.3%                      |
| M2-KL     | 77.5%            | 28.0%                      |
| M2-Prot   | 70.0%            | 26.7%                      |

Self-audit: 5/5 integrity checks pass.

## Data provenance
All training corpora built from Wikipedia articles under CC BY-SA 4.0.
See `data/manifest.json` for per-article attribution.

## Limitations
- Public proxy data, not the client's actual Book 2 text
- 150-item panels; statistical power insufficient for significance at p < 0.05
- FP16 training fallback: 4-bit quantization was unavailable in the Kaggle environment.
  Training completed within budget but used ~2x the intended VRAM.
- M1 baseline was weak (+4 items over M0), which affects retention percentage interpretation.
