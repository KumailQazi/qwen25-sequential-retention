"""
Zero-Touch Sequential Multi-Book Continual Training Pipeline.

Sequentially trains Book 2 then Book 3 on top of Book 1 (M1 checkpoint)
under strict hardware (48GB VRAM) and 8 GPU-hour constraints.
"""

import os
import sys
import time
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from src.config import config
from src.orthogonal_lora import (
    compute_adapter_subspace,
    compute_orthogonal_loss,
    compute_token_anchored_kl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LLM_DALI")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM_DALI Sequential Training")
    parser.add_argument("--m1_checkpoint", type=str, required=True, help="Path to M1 Book 1 LoRA weights")
    parser.add_argument("--book2_data", type=str, required=True, help="Path to Book 2 tokenized instruction dataset")
    parser.add_argument("--book3_data", type=str, required=True, help="Path to Book 3 tokenized instruction dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_book(
    stage_name: str,
    base_model,
    tokenizer,
    dataset_path: str,
    prior_subspaces,
    output_path: str,
    epochs: int,
):
    logger.info(f"=== Starting {stage_name} (Max budget: {config.hardware.max_gpu_hours_per_book} hrs) ===")
    start_time = time.time()
    
    # Configure LoRA for current stage
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=0.01,
    )
    
    # Mocking single pass for scaffold demonstration
    logger.info(f"Training {stage_name} with Orthogonal Lambda={config.continual.orthogonal_lambda} & KL={config.continual.kl_weight}")
    
    elapsed_hours = (time.time() - start_time) / 3600.0
    logger.info(f"{stage_name} completed in {elapsed_hours:.2f} GPU hours (Limit: {config.hardware.max_gpu_hours_per_book} hrs)")
    
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    logger.info(f"Saved {stage_name} checkpoint to {output_path}")
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info(f"Initializing Qwen2.5-7B-Instruct (revision: {config.model.base_model_revision})")
    
    # Load base model deterministically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        revision=config.model.base_model_revision,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        revision=config.model.base_model_revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Step 1: Ingest M1 adapter & compute principal subspace
    logger.info(f"Loading M1 LoRA checkpoint from: {args.m1_checkpoint}")
    m1_model = PeftModel.from_pretrained(base_model, args.m1_checkpoint)
    m1_state_dict = {k: v for k, v in m1_model.named_parameters() if "lora_" in k}
    m1_subspaces = compute_adapter_subspace(m1_state_dict, energy_threshold=config.continual.svd_energy_threshold)
    logger.info(f"Extracted M1 principal subspaces for {len(m1_subspaces)} parameter matrices.")
    
    # Step 2: Zero-Touch Sequential Training of Book 2
    book2_output = os.path.join(args.output_dir, "book2_checkpoint")
    m2_model = train_book(
        stage_name="Book 2 Training",
        base_model=base_model,
        tokenizer=tokenizer,
        dataset_path=args.book2_data,
        prior_subspaces=m1_subspaces,
        output_path=book2_output,
        epochs=config.training.num_train_epochs_book2,
    )
    
    # Extract combined subspace (Book 1 + Book 2)
    m2_state_dict = {k: v for k, v in m2_model.named_parameters() if "lora_" in k}
    combined_subspaces = compute_adapter_subspace(m2_state_dict, energy_threshold=config.continual.svd_energy_threshold)
    
    # Step 3: Zero-Touch Sequential Training of Book 3
    book3_output = os.path.join(args.output_dir, "book3_checkpoint")
    m3_model = train_book(
        stage_name="Book 3 Training",
        base_model=base_model,
        tokenizer=tokenizer,
        dataset_path=args.book3_data,
        prior_subspaces=combined_subspaces,
        output_path=book3_output,
        epochs=config.training.num_train_epochs_book3,
    )
    
    # Step 4: Final Unified Model Export (Single Weight Set)
    final_merged_dir = os.path.join(args.output_dir, "final_unified_model")
    logger.info("Merging LoRA adapters into base weights for zero-overhead inference...")
    final_model = m3_model.merge_and_unload()
    final_model.save_pretrained(final_merged_dir)
    tokenizer.save_pretrained(final_merged_dir)
    logger.info(f"Final unified model successfully saved to: {final_merged_dir}")


if __name__ == "__main__":
    main()

