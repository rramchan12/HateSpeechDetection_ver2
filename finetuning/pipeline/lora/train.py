#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Hate Speech Detection

Based on lora_ft_approach.md specifications with QLoRA (4-bit quantization).

Usage:
    # Using config file (recommended)
    accelerate launch --num_processes 4 \\
        -m finetuning.pipeline.lora.train \\
        --config_file ./finetuning/pipeline/lora/config.json
    
    # With command-line overrides
    accelerate launch --num_processes 4 \\
        -m finetuning.pipeline.lora.train \\
        --config_file ./finetuning/pipeline/lora/config.json \\
        --learning_rate 3e-4 \\
        --num_train_epochs 5
    
    # Direct command-line (no config file)
    accelerate launch --num_processes 4 \\
        -m finetuning.pipeline.lora.train \\
        --model_name_or_path openai/gpt-oss-20b \\
        --train_file ./finetuning/data/ft_prompts/train.jsonl \\
        --validation_file ./finetuning/data/ft_prompts/validation.jsonl \\
        --output_dir ./finetuning/models/lora_checkpoints \\
        --learning_rate 2e-4 \\
        --num_train_epochs 3

Monitor training:
    tail -f finetuning/models/lora_checkpoints/training.log
    tensorboard --logdir finetuning/models/lora_checkpoints/logs
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="openai/gpt-oss-20b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default="./data/models",
        metadata={"help": "Directory to cache downloaded models"}
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load model in 4-bit quantization (QLoRA)"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type: nf4 (NormalFloat4) or fp4"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit base models: bfloat16, float16, or float32"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_file: str = field(
        default="finetuning/data/ft_prompts/train.jsonl",
        metadata={"help": "Path to training data (JSONL format with 'messages' field)"}
    )
    validation_file: str = field(
        default="finetuning/data/ft_prompts/validation.jsonl",
        metadata={"help": "Path to validation data (JSONL format with 'messages' field)"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length (tokens)"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration"""
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA attention dimension (rank). Higher = more capacity, more parameters"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling parameter. Typically set equal to lora_r"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability for regularization"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA (e.g., 'q_proj,v_proj')"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias training strategy: 'none', 'all', or 'lora_only'"}
    )


def load_config_from_file(config_file: str) -> dict:
    """Load configuration from JSON file and convert to flat dict"""
    logger.info(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Convert list fields to comma-separated strings for argparse
    if 'lora_target_modules' in config and isinstance(config['lora_target_modules'], list):
        config['lora_target_modules'] = ','.join(config['lora_target_modules'])
    
    # Convert report_to to string if it's a list
    if 'report_to' in config and isinstance(config['report_to'], list):
        config['report_to'] = config['report_to'][0] if config['report_to'] else 'none'
    
    return config


def prepare_model_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """
    Load and prepare model with LoRA adapters and 4-bit quantization
    
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info("="*70)
    logger.info("MODEL LOADING")
    logger.info("="*70)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"4-bit quantization: {model_args.load_in_4bit}")
    logger.info(f"Quantization type: {model_args.bnb_4bit_quant_type}")
    logger.info(f"Compute dtype: {model_args.bnb_4bit_compute_dtype}")
    
    # Configure 4-bit quantization (QLoRA)
    # Note: openai/gpt-oss-20b is pre-quantized with Mxfp4, so skip BitsAndBytes config
    if model_args.load_in_4bit:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
        )
        logger.info("✓ 4-bit quantization configured (QLoRA)")
    else:
        bnb_config = None
        logger.info("✓ Full precision (no quantization)")
    
    # Load base model
    logger.info("Loading base model...")
    try:
        # Try loading with quantization config first
        # Note: Don't use device_map when using Accelerate for distributed training
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )
    except ValueError as e:
        if "quantized with" in str(e) and "but you are passing" in str(e):
            # Model is pre-quantized, load without additional quantization config
            logger.warning(f"Model is pre-quantized, loading without BitsAndBytes config")
            logger.warning(f"Original error: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=model_args.cache_dir,
            )
            logger.info("✓ Base model loaded (using model's native quantization)")
            bnb_config = None  # Clear config since we're not using BitsAndBytes
        else:
            raise
    else:
        logger.info("✓ Base model loaded")
    
    # Prepare model for k-bit training (only if using BitsAndBytes quantization)
    if model_args.load_in_4bit and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
        logger.info("✓ Model prepared for k-bit training")
    else:
        # For pre-quantized models or full precision, enable gradient checkpointing manually
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        logger.info("✓ Input gradients enabled for training")
    
    # Configure LoRA
    logger.info("\nLoRA Configuration:")
    logger.info(f"  Rank (r): {lora_args.lora_r}")
    logger.info(f"  Alpha: {lora_args.lora_alpha}")
    logger.info(f"  Dropout: {lora_args.lora_dropout}")
    logger.info(f"  Target modules: {lora_args.lora_target_modules}")
    logger.info(f"  Bias: {lora_args.lora_bias}")
    
    # Parse target modules
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(',')]
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",  # Causal language modeling
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    logger.info("✓ LoRA adapters applied")
    
    # Enable gradient checkpointing if requested
    # This must be done after applying LoRA
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled on model")
    
    # Explicitly enable gradients for trainable parameters
    # This is crucial for pre-quantized models that get dequantized
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = True  # Ensure it's truly enabled
    logger.info("✓ Gradients explicitly enabled for trainable parameters")
    
    # Print trainable parameters
    logger.info("\nTrainable Parameters:")
    model.print_trainable_parameters()
    
    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✓ Padding token set to EOS token: {tokenizer.eos_token}")
    
    logger.info("✓ Tokenizer loaded")
    logger.info("="*70)
    
    return model, tokenizer


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess training examples from JSONL format
    
    Expected format:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    """
    texts = []
    
    for messages in examples["messages"]:
        # Format messages into chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,  # Don't return tensors yet (HF datasets will handle this)
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    """Main training function"""
    
    # Parse command-line arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    
    # Check for config file
    if "--config_file" in sys.argv:
        config_idx = sys.argv.index("--config_file")
        config_file = sys.argv[config_idx + 1]
        
        # Load config from file
        config = load_config_from_file(config_file)
        
        # Convert config to command-line style args
        args_list = []
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{key}")
            elif isinstance(value, list):
                # For lists, add each item separately
                args_list.append(f"--{key}")
                args_list.extend([str(v) for v in value])
            else:
                args_list.extend([f"--{key}", str(value)])
        
        # Add remaining command-line args (these override config file)
        remaining_args = sys.argv[1:]
        # Remove --config_file and its value
        if "--config_file" in remaining_args:
            config_idx = remaining_args.index("--config_file")
            remaining_args.pop(config_idx)  # Remove --config_file
            remaining_args.pop(config_idx)  # Remove config file path
        
        args_list.extend(remaining_args)
        
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses(args_list)
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Fix DDP + gradient checkpointing + LoRA compatibility issue
    # The combination of DDP + gradient checkpointing + LoRA causes "parameter marked ready twice" error
    # Use static graph optimization which is more efficient and avoids the issue
    if accelerator.num_processes > 1 and training_args.gradient_checkpointing:
        # Disable find_unused_parameters (incompatible with static graph)
        training_args.ddp_find_unused_parameters = False
        # Enable static graph (tells DDP the computation graph doesn't change)
        # This is the recommended fix for gradient checkpointing + DDP
        # See: https://pytorch.org/docs/stable/notes/ddp.html#ddp-static-graph
        if hasattr(training_args, 'ddp_static_graph'):
            training_args.ddp_static_graph = True
            logger.info("✓ DDP configured with static_graph=True for gradient checkpointing compatibility")
        else:
            # Fallback for older transformers versions
            training_args.ddp_find_unused_parameters = False
            logger.info("✓ DDP configured with find_unused_parameters=False for gradient checkpointing")
    elif accelerator.num_processes > 1:
        training_args.ddp_find_unused_parameters = False
        logger.info("✓ DDP configured with find_unused_parameters=False")

    
    # Setup output directory and logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_file = os.path.join(training_args.output_dir, "training.log")
    
    # Add file handler for logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Print configuration summary
    logger.info("\n" + "="*70)
    logger.info("LORA FINE-TUNING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Validation file: {data_args.validation_file}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"\nTraining Hyperparameters:")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Warmup steps: {training_args.warmup_steps}")
    logger.info(f"  Weight decay: {training_args.weight_decay}")
    logger.info(f"  Max gradient norm: {training_args.max_grad_norm}")
    logger.info(f"  LR scheduler: {training_args.lr_scheduler_type}")
    logger.info(f"\nLoRA Configuration:")
    logger.info(f"  Rank (r): {lora_args.lora_r}")
    logger.info(f"  Alpha: {lora_args.lora_alpha}")
    logger.info(f"  Dropout: {lora_args.lora_dropout}")
    logger.info(f"  Target modules: {lora_args.lora_target_modules}")
    logger.info(f"\nCompute Configuration:")
    logger.info(f"  BF16: {training_args.bf16}")
    logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  4-bit quantization: {model_args.load_in_4bit}")
    logger.info(f"  Num processes: {accelerator.num_processes}")
    logger.info("="*70 + "\n")
    
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_args, lora_args)
    
    # Load datasets
    logger.info("="*70)
    logger.info("LOADING DATASETS")
    logger.info("="*70)
    logger.info(f"Train file: {data_args.train_file}")
    logger.info(f"Validation file: {data_args.validation_file}")
    
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    
    try:
        raw_datasets = load_dataset("json", data_files=data_files)
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        logger.error("Make sure the JSONL files exist and are properly formatted")
        raise
    
    logger.info(f"✓ Train samples: {len(raw_datasets['train'])}")
    logger.info(f"✓ Validation samples: {len(raw_datasets['validation'])}")
    logger.info("="*70 + "\n")
    
    # Preprocess datasets
    logger.info("Preprocessing datasets (tokenization)...")
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )
    logger.info("✓ Tokenization complete\n")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Calculate training statistics
    total_train_samples = len(tokenized_datasets["train"])
    effective_batch_size = (
        training_args.per_device_train_batch_size * 
        training_args.gradient_accumulation_steps * 
        accelerator.num_processes
    )
    steps_per_epoch = total_train_samples // effective_batch_size
    total_steps = int(steps_per_epoch * training_args.num_train_epochs)
    
    logger.info("="*70)
    logger.info("TRAINING STATISTICS")
    logger.info("="*70)
    logger.info(f"Total train samples: {total_train_samples}")
    logger.info(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Num processes: {accelerator.num_processes}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {training_args.warmup_steps} ({training_args.warmup_steps/total_steps*100:.1f}%)")
    logger.info("="*70 + "\n")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    
    # Train
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    logger.info("Monitor progress:")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  TensorBoard: tensorboard --logdir {training_args.logging_dir}")
    logger.info("="*70 + "\n")
    
    try:
        train_result = trainer.train()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Save final model
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    logger.info(f"Output directory: {training_args.output_dir}")
    trainer.save_model()
    logger.info("✓ Model saved")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("✓ Training metrics saved")
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"Final validation loss: {eval_metrics['eval_loss']:.4f}")
    logger.info("✓ Evaluation metrics saved")
    logger.info("="*70)
    
    # Print completion summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Model checkpoint: {training_args.output_dir}")
    logger.info(f"Training log: {log_file}")
    logger.info(f"TensorBoard logs: {training_args.logging_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Validate model with finetuning/pipeline/baseline/runner.py")
    logger.info("2. Compare post-FT performance vs. baseline (F1 >= 0.615)")
    logger.info("3. Optionally merge adapters with: python -m finetuning.pipeline.lora.merge")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
