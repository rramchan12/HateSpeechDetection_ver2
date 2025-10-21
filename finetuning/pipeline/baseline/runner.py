#!/usr/bin/env python3
"""
Baseline Validation Pipeline CLI

Run baseline inference on GPT-OSS model to establish performance baseline.

Usage:
    python -m finetuning.pipeline.baseline.runner \
        --model_name gpt-oss-20b \
        --data_file ./data/validation.jsonl \
        --output_dir ./outputs

Example:
    # Quick test with 50 samples
    python -m finetuning.pipeline.baseline.runner --max_samples 50
    
    # Full validation
    python -m finetuning.pipeline.baseline.runner
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from finetuning.pipeline.baseline.model_loader import load_model
from finetuning.pipeline.baseline.inference import run_inference
from finetuning.pipeline.baseline.metrics import calculate_metrics
from prompt_engineering.metrics.persistence_helper import PersistenceHelper


def test_connection(args):
    """
    Test model connection with a simple prompt
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        0 if successful, 1 if failed
    """
    print(f"\n{'='*60}")
    print("CONNECTION TEST MODE")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    try:
        model, tokenizer = load_model(args.model_name)
    except Exception as e:
        print(f"\n[FAILED] Could not load model: {e}")
        return 1
    
    # Test prompt
    test_prompt = "Classify this text as 'hate' or 'not hate': Hello, how are you?\n### Classification:"
    
    print(f"\nTest prompt: {test_prompt}")
    print(f"\nGenerating response...")
    
    try:
        # Tokenize
        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n{'='*60}")
        print("RESPONSE:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
        
        print(f"\n[SUCCESS] Connection test passed!")
        print(f"Model is loaded and responding correctly.")
        return 0
        
    except Exception as e:
        print(f"\n[FAILED] Error during inference: {e}")
        return 1


def main(args):
    """
    Execute baseline validation pipeline
    
    Args:
        args: Parsed command line arguments
    """
    
    # Print configuration
    print(f"\n{'='*60}")
    print("BASELINE VALIDATION PIPELINE")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model_name)
    
    # Run inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"baseline_results_{timestamp}.json"
    
    print(f"\nRunning inference (output: {results_file})...")
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        data_file=args.data_file,
        output_file=str(results_file),
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Calculate metrics (returns PerformanceMetrics object)
    print(f"\nCalculating metrics...")
    metrics_obj = calculate_metrics(results, strategy_name="baseline")
    
    if metrics_obj:
        # Use PersistenceHelper to save results (same as prompt_engineering)
        persistence = PersistenceHelper(output_dir)
        
        # Save metrics using the standard format
        metrics_dict = {
            'strategy': metrics_obj.strategy,
            'accuracy': metrics_obj.accuracy,
            'precision': metrics_obj.precision,
            'recall': metrics_obj.recall,
            'f1_score': metrics_obj.f1_score,
            'false_positive_rate': metrics_obj.false_positive / (metrics_obj.false_positive + metrics_obj.true_negative) 
                                   if (metrics_obj.false_positive + metrics_obj.true_negative) > 0 else 0,
            'false_negative_rate': metrics_obj.false_negative / (metrics_obj.false_negative + metrics_obj.true_positive) 
                                   if (metrics_obj.false_negative + metrics_obj.true_positive) > 0 else 0,
            'confusion_matrix': {
                'true_positive': metrics_obj.true_positive,
                'false_positive': metrics_obj.false_positive,
                'true_negative': metrics_obj.true_negative,
                'false_negative': metrics_obj.false_negative
            }
        }
        
        metrics_file = output_dir / f"baseline_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"[OK] Metrics saved to: {metrics_file}")
        
        print(f"\n[SUCCESS] Baseline validation complete!")
        return 0
    else:
        print(f"\n[FAILED] Baseline validation failed!")
        return 1


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Baseline Validation Pipeline for GPT-OSS Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-oss-20b",
        help="HuggingFace model identifier (default: %(default)s)"
    )
    
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/validation.jsonl",
        help="Path to validation data JSONL file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results (default: %(default)s)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (None for all, default: %(default)s)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum input token length (default: %(default)s)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum generated tokens (default: %(default)s)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation (default: %(default)s)"
    )
    
    parser.add_argument(
        "--test_connection",
        action="store_true",
        help="Test model connection with a simple prompt and exit"
    )
    
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    # Run connection test if requested
    if args.test_connection:
        exit(test_connection(args))
    else:
        exit(main(args))
