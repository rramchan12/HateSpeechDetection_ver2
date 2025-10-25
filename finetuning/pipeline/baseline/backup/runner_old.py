#!/usr/bin/env python3
"""
Baseline Validation Pipeline CLI

Run baseline inference on GPT-OSS model to establish performance baseline.

Usage:
    python -m finetuning.pipeline.baseline.runner \
        --model_name openai/gpt-oss-20b \
        --data_file ./finetuning/data/prepared/validation.jsonl \
        --output_dir ./outputs

    # Using prompt template from prompt_engineering
    python -m finetuning.pipeline.baseline.runner \
        --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
        --strategy combined_optimized

Example:
    # Quick test with 50 samples
    python -m finetuning.pipeline.baseline.runner --max_samples 50
    
    # Full validation with prompt template
    python -m finetuning.pipeline.baseline.runner \
        --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
        --strategy combined_optimized
    
    # With HuggingFace token for private models
    HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner
"""

import argparse
import json
import sys
import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from finetuning.pipeline.baseline.model_loader import load_model
from prompt_engineering.metrics import EvaluationMetrics


def load_prompt_template(template_file: str, strategy_name: str = "combined_optimized") -> Dict[str, Any]:
    """
    Load prompt template from JSON file.
    
    Args:
        template_file: Path to JSON template file
        strategy_name: Name of strategy to load
        
    Returns:
        Dictionary with system_prompt, user_template, and parameters
    """
    with open(template_file, 'r') as f:
        data = json.load(f)
    
    if 'strategies' in data and strategy_name in data['strategies']:
        strategy = data['strategies'][strategy_name]
        return {
            'strategy_name': strategy_name,
            'system_prompt': strategy['system_prompt'],
            'user_template': strategy['user_template'],
            'parameters': strategy.get('parameters', {
                'max_tokens': 512,
                'temperature': 0.1,
                'top_p': 1.0
            })
        }
    else:
        raise ValueError(f"Strategy '{strategy_name}' not found in template file")


def load_validation_data(data_file: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load validation data from JSONL file.
    
    Args:
        data_file: Path to validation JSONL file
        max_samples: Maximum samples to load (None for all)
        
    Returns:
        List of dicts with 'text' and 'label' keys
    """
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Validation data file not found: {data_file}")
    
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            
            data = json.loads(line.strip())
            messages = data.get('messages', [])
            
            # Extract text from user message
            text = None
            for msg in messages:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if 'Text: ' in content:
                        text = content.split('Text: ', 1)[1].strip().strip('"')
                    break
            
            # Extract label from assistant message
            label = None
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    try:
                        response = json.loads(content)
                        classification = response.get('classification', '').lower()
                        if 'hate' in classification and 'not' not in classification:
                            label = 'hate'
                        else:
                            label = 'normal'
                    except json.JSONDecodeError:
                        pass
                    break
            
            if text and label:
                samples.append({'text': text, 'label': label})
    
    return samples


def run_inference_with_prompt(
    model,
    tokenizer,
    dataset: List[Dict],
    prompt_config: Dict[str, Any]
) -> List[Dict]:
    """
    Run inference on dataset using prompt template.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        dataset: List of samples with 'text' and 'label' keys
        prompt_config: Prompt configuration dict
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Processing")):
        try:
            # Create messages
            messages = [
                {"role": "system", "content": prompt_config["system_prompt"]},
                {"role": "user", "content": prompt_config["user_template"].format(text=sample['text'])}
            ]
            
            # Format as chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=prompt_config["parameters"]["max_tokens"],
                    temperature=prompt_config["parameters"]["temperature"],
                    top_p=prompt_config["parameters"].get("top_p", 1.0),
                    do_sample=True if prompt_config["parameters"]["temperature"] > 0 else False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response
            try:
                result = json.loads(response.strip())
                classification = result.get("classification", "").lower()
                rationale = result.get("rationale", "")
                
                if "hate" in classification and "not" not in classification:
                    pred_label = "hate"
                elif "normal" in classification or "not" in classification:
                    pred_label = "normal"
                else:
                    pred_label = "unknown"
            except json.JSONDecodeError:
                # Fallback text parsing
                response_lower = response.lower()
                if "hate" in response_lower and "not" not in response_lower:
                    pred_label = "hate"
                elif "normal" in response_lower or "not" in response_lower:
                    pred_label = "normal"
                else:
                    pred_label = "unknown"
                rationale = "JSON parse failed"
            
            results.append({
                'sample_id': i,
                'text': sample['text'],
                'true_label': sample['label'],
                'prediction': pred_label,
                'rationale': rationale,
                'raw_response': response,
                'strategy': prompt_config['strategy_name']
            })
            
        except Exception as e:
            results.append({
                'sample_id': i,
                'text': sample['text'],
                'true_label': sample['label'],
                'prediction': 'error',
                'rationale': f"Error: {str(e)}",
                'raw_response': "",
                'strategy': prompt_config['strategy_name']
            })
    
    return results


def save_results(output_dir: Path, timestamp: str, results: List[Dict], metrics: Any, dataset: List[Dict], prompt_config: Dict):
    """Save all results files."""
    import csv
    
    # 1. Evaluation report
    report_file = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("HATE SPEECH DETECTION - BASELINE EVALUATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Strategy: {prompt_config['strategy_name']}\n")
        f.write(f"Data Source: validation\n")
        f.write(f"Total samples tested: {len(dataset)}\n\n")
        
        f.write("STRATEGY PERFORMANCE:\n")
        f.write("-"*25 + "\n\n")
        f.write(f"{prompt_config['strategy_name'].upper()} Strategy:\n")
        f.write(f"  Accuracy:  {metrics.accuracy:.3f}\n")
        f.write(f"  Precision: {metrics.precision:.3f}\n")
        f.write(f"  Recall:    {metrics.recall:.3f}\n")
        f.write(f"  F1-Score:  {metrics.f1_score:.3f}\n")
        f.write(f"  Confusion Matrix: TP={metrics.true_positive}, TN={metrics.true_negative}, ")
        f.write(f"FP={metrics.false_positive}, FN={metrics.false_negative}\n")
    
    print(f"[OK] Saved evaluation report: {report_file}")
    
    # 2. Performance metrics CSV
    metrics_file = output_dir / f"performance_metrics_{timestamp}.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['strategy', 'accuracy', 'precision', 'recall', 'f1_score',
                        'true_positive', 'false_positive', 'true_negative', 'false_negative'])
        writer.writerow([
            prompt_config['strategy_name'],
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1_score:.4f}",
            metrics.true_positive,
            metrics.false_positive,
            metrics.true_negative,
            metrics.false_negative
        ])
    
    print(f"[OK] Saved performance metrics: {metrics_file}")
    
    # 3. Detailed results CSV
    results_file = output_dir / f"strategy_results_{timestamp}.csv"
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'text', 'true_label', 'prediction',
                                               'rationale', 'strategy'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'text': r['text'],
                'true_label': r['true_label'],
                'prediction': r['prediction'],
                'rationale': r['rationale'],
                'strategy': r['strategy']
            })
    
    print(f"[OK] Saved detailed results: {results_file}")
    
    # 4. Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy: {prompt_config['strategy_name']}")
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1-Score:  {metrics.f1_score:.3f}")
    print(f"{'='*60}\n")


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
    
    # Load model with caching
    try:
        model, tokenizer = load_model(args.model_name, cache_dir=args.cache_dir)
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
    if args.prompt_template:
        print(f"Prompt template: {args.prompt_template}")
        print(f"Strategy: {args.strategy}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}\n")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompt template if provided
    if args.prompt_template:
        print(f"Loading prompt template: {args.prompt_template}")
        prompt_config = load_prompt_template(args.prompt_template, args.strategy)
        print(f"[OK] Loaded strategy: {prompt_config['strategy_name']}\n")
    else:
        # Use default simple prompt
        prompt_config = {
            'strategy_name': 'baseline',
            'system_prompt': "You are a hate speech detection assistant. Classify posts as 'hate' or 'normal'.",
            'user_template': "Text: {text}\n\nClassify as hate or normal:",
            'parameters': {
                'max_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'top_p': 1.0
            }
        }
        print(f"[OK] Using default baseline prompt\n")
    
    # Load model with caching
    model, tokenizer = load_model(args.model_name, cache_dir=args.cache_dir)
    
    # Load validation data
    print(f"Loading validation data from: {args.data_file}")
    dataset = load_validation_data(args.data_file, args.max_samples)
    print(f"[OK] Loaded {len(dataset)} samples\n")
    
    # Run inference
    print(f"Running inference...")
    results = run_inference_with_prompt(model, tokenizer, dataset, prompt_config)
    
    # Calculate metrics
    print(f"\n\nCalculating metrics...")
    valid_results = [r for r in results if r['prediction'] not in ['unknown', 'error']]
    
    if len(valid_results) == 0:
        print("[FAILED] No valid predictions!")
        return 1
    
    true_labels = [r['true_label'] for r in valid_results]
    pred_labels = [r['prediction'] for r in valid_results]
    
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_comprehensive_metrics(
        y_true=true_labels,
        y_pred=pred_labels,
        strategy_name=prompt_config['strategy_name']
    )
    
    # Save results
    save_results(output_dir, timestamp, results, metrics, dataset, prompt_config)
    
    print(f"\n[SUCCESS] Baseline validation complete!")
    print(f"Results saved to: {output_dir}")
    return 0


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
        default="openai/gpt-oss-20b",
        help="HuggingFace model identifier (default: %(default)s)"
    )
    
    parser.add_argument(
        "--data_file",
        type=str,
        default="./finetuning/data/prepared/validation.jsonl",
        help="Path to validation data JSONL file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/baseline",
        help="Directory to save results (default: %(default)s)"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache models (default: data/models)"
    )
    
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Path to prompt template JSON file (e.g., prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="combined_optimized",
        help="Strategy name from prompt template (default: %(default)s)"
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
