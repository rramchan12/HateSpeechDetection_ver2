#!/usr/bin/env python3
"""
Merge LoRA adapters with base model for deployment

This script merges the trained LoRA adapters back into the base model,
creating a standalone model that can be deployed without PEFT.

Usage:
    # Merge LoRA adapters
    python -m finetuning.pipeline.lora.merge \\
        --base_model openai/gpt-oss-20b \\
        --adapter_path ./finetuning/models/lora_checkpoints \\
        --output_dir ./finetuning/models/merged_model
    
    # With custom cache directory
    python -m finetuning.pipeline.lora.merge \\
        --base_model openai/gpt-oss-20b \\
        --adapter_path ./finetuning/models/lora_checkpoints \\
        --output_dir ./finetuning/models/merged_model \\
        --cache_dir ./data/models

Note:
    - Merged model will be ~78GB (full model size)
    - Alternative: Keep adapters separate (~32MB) and load with PEFT library
    - Merging is optional - adapters can be loaded dynamically during inference
"""

import argparse
import logging
import torch
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora_adapters(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    cache_dir: str = "./data/models",
    device_map: str = "auto",
):
    """
    Merge LoRA adapters into base model
    
    Args:
        base_model_path: Path or HuggingFace ID of base model
        adapter_path: Path to LoRA checkpoint directory
        output_dir: Where to save merged model
        cache_dir: Directory to cache downloaded models
        device_map: Device mapping strategy ('auto', 'cpu', etc.)
    """
    logger.info("="*70)
    logger.info("LORA ADAPTER MERGE")
    logger.info("="*70)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70 + "\n")
    
    # Verify adapter path exists
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Check for adapter config
    adapter_config_path = adapter_path_obj / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            "Make sure this is a valid LoRA checkpoint directory."
        )
    
    logger.info("Step 1: Loading base model...")
    logger.info(f"  Model: {base_model_path}")
    logger.info(f"  Device map: {device_map}")
    logger.info(f"  Dtype: bfloat16")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        logger.info("✓ Base model loaded\n")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise
    
    logger.info("Step 2: Loading LoRA adapters...")
    logger.info(f"  Adapter path: {adapter_path}")
    
    try:
        # Load PEFT config first to verify
        peft_config = PeftConfig.from_pretrained(adapter_path)
        logger.info(f"  LoRA rank: {peft_config.r}")
        logger.info(f"  LoRA alpha: {peft_config.lora_alpha}")
        logger.info(f"  Target modules: {peft_config.target_modules}")
        
        # Load model with adapters
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("✓ LoRA adapters loaded\n")
    except Exception as e:
        logger.error(f"Failed to load LoRA adapters: {e}")
        raise
    
    logger.info("Step 3: Merging adapters with base model...")
    logger.info("  This may take a few minutes...")
    
    try:
        merged_model = model.merge_and_unload()
        logger.info("✓ Adapters merged successfully\n")
    except Exception as e:
        logger.error(f"Failed to merge adapters: {e}")
        raise
    
    logger.info("Step 4: Saving merged model...")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save merged model
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # Use safetensors format
        )
        logger.info("✓ Merged model saved")
        
        # Save tokenizer
        logger.info("\nStep 5: Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        tokenizer.save_pretrained(output_dir)
        logger.info("✓ Tokenizer saved\n")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
    
    # Print summary
    logger.info("="*70)
    logger.info("MERGE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Merged model saved to: {output_dir}")
    logger.info("\nModel files:")
    for file in sorted(output_path.glob("*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  {file.name}: {size_mb:.2f} MB")
    
    logger.info("\nUsage:")
    logger.info("  # Load merged model")
    logger.info(f"  from transformers import AutoModelForCausalLM")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    logger.info("\n  # Or use with baseline runner")
    logger.info(f"  python -m finetuning.pipeline.baseline.runner \\")
    logger.info(f"      --model_name {output_dir} \\")
    logger.info(f"      --data_file unified \\")
    logger.info(f"      --max_samples 100")
    logger.info("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model (e.g., 'openai/gpt-oss-20b')"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., './finetuning/models/lora_checkpoints')"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save merged model (e.g., './finetuning/models/merged_model')"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/models",
        help="Directory to cache downloaded models (default: ./data/models)"
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy: 'auto', 'cpu', or specific device (default: auto)"
    )
    
    args = parser.parse_args()
    
    try:
        merge_lora_adapters(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
        )
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("MERGE FAILED")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}")
        logger.error(f"{'='*70}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
