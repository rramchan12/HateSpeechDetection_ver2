#!/usr/bin/env python3
"""
Baseline Validation Pipeline CLI

Run baseline inference on GPT-OSS model to establish performance baseline.
Supports both single-GPU and multi-GPU execution via Accelerate.

Usage:
    # Single GPU (default)
    python -m finetuning.pipeline.baseline.runner \
        --model_name openai/gpt-oss-20b \
        --data_file ./finetuning/data/prepared/validation.jsonl \
        --output_dir ./outputs

    # Multi-GPU with Accelerate (automatic distribution)
    accelerate launch --num_processes 4 -m finetuning.pipeline.baseline.runner \
        --use_accelerate \
        --model_name openai/gpt-oss-20b \
        --data_file unified \
        --max_samples 100

    # Using prompt template from prompt_engineering
    python -m finetuning.pipeline.baseline.runner \
        --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
        --strategy combined_optimized

Example:
    # Quick test with canned dataset (single GPU)
    python -m finetuning.pipeline.baseline.runner --data_file canned_50_quick --max_samples 5
    
    # Use unified test dataset (single GPU)
    python -m finetuning.pipeline.baseline.runner --data_file unified --max_samples 50
    
    # Multi-GPU with Accelerate (4 GPUs)
    accelerate launch --num_processes 4 -m finetuning.pipeline.baseline.runner \
        --use_accelerate \
        --data_file unified \
        --max_samples 100
    
    # Full validation with prompt template and canned data
    python -m finetuning.pipeline.baseline.runner \
        --data_file canned_100_stratified \
        --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
        --strategy combined_optimized
    
    # Use fine-tuning validation data (JSONL)
    python -m finetuning.pipeline.baseline.runner \
        --data_file ./finetuning/data/prepared/validation.jsonl \
        --max_samples 10
    
    # With HuggingFace token for private models
    HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner
"""

import argparse
import json
import sys
import os
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from finetuning.pipeline.baseline.model_loader import load_model
from prompt_engineering.metrics import EvaluationMetrics, PersistenceHelper, ValidationResult
from prompt_engineering.loaders import load_dataset_by_filename, load_dataset, DatasetType, StrategyTemplatesLoader

# Import AccelerateConnector for optional multi-GPU support
try:
    from finetuning.pipeline.baseline.connector.accelerate_connector import AccelerateConnector
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def load_strategy_config(strategy_loader: StrategyTemplatesLoader, strategy_name: str) -> Dict[str, Any]:
    """
    Load strategy configuration using StrategyTemplatesLoader.
    
    Args:
        strategy_loader: Initialized StrategyTemplatesLoader instance
        strategy_name: Name of strategy to load
        
    Returns:
        Dictionary with keys:
            - strategy_name: Name of the loaded strategy
            - system_prompt: System message to set model context
            - user_template: Template string with {text} placeholder
            - parameters: Dict with max_tokens, temperature, top_p settings
            
    Raises:
        ValueError: If strategy name not found
        
    Example:
        >>> loader = StrategyTemplatesLoader('./prompts/baseline_v1.json')
        >>> config = load_strategy_config(loader, 'baseline')
    """
    strategy = strategy_loader.get_strategy(strategy_name)
    
    if strategy is None:
        available = strategy_loader.get_available_strategy_names()
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available strategies: {', '.join(available)}"
        )
    
    return {
        'strategy_name': strategy.name,
        'system_prompt': strategy.template.system_prompt,
        'user_template': strategy.template.user_template,
        'parameters': strategy.parameters
    }


def load_validation_data(data_file: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load validation data from JSONL file with support for multiple formats.
    
    This unified function handles:
    - 'unified' keyword for unified test dataset (comprehensive test set)
    - Canned dataset names (e.g., 'canned_50_quick', 'canned_100_stratified')
    - JSONL files with 'messages' format (fine-tuning data)
    - Direct JSON with 'text' and 'label' fields
    
    Args:
        data_file: Data source identifier:
                  - 'unified': Load from unified test dataset
                  - 'canned_*': Load specific canned dataset
                  - Path: Load JSONL file from path
        max_samples: Maximum samples to load (None for all)
        
    Returns:
        List of dicts with 'text' and 'label' keys
        
    Example:
        >>> samples = load_validation_data('unified', max_samples=50)
        >>> samples = load_validation_data('canned_50_quick', max_samples=5)
        >>> samples = load_validation_data('./finetuning/data/prepared/validation.jsonl', max_samples=5)
    """
    # Check if this is the unified dataset (special keyword)
    if data_file.lower() == "unified":
        unified_samples = load_dataset(
            DatasetType.UNIFIED,
            num_samples=max_samples if max_samples else "all"
        )
        # Convert to expected format
        return [
            {
                'text': s['text'],
                'label': s['label_binary'],
                'target_group': s.get('target_group_norm', 'unknown'),
                'source': s.get('source_dataset', 'unified')
            }
            for s in unified_samples
        ]
    
    # Check if this is a canned dataset name (no path separators, not .jsonl)
    if '/' not in data_file and '\\' not in data_file and not data_file.endswith('.jsonl'):
        try:
            # Use unified dataset loader for canned datasets
            canned_samples = load_dataset_by_filename(
                data_file, 
                num_samples=max_samples if max_samples else "all"
            )
            # Convert to expected format (already has 'text', need to map 'label_binary' to 'label')
            return [
                {
                    'text': s['text'],
                    'label': s['label_binary'],
                    'target_group': s.get('target_group_norm', 'unknown'),
                    'source': s.get('source_dataset', 'canned')
                }
                for s in canned_samples
            ]
        except FileNotFoundError:
            # Not a canned dataset, try as regular file
            pass
    
    # Load from JSONL file
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
                    # Handle different prompt formats
                    if 'Text: ' in content:
                        text = content.split('Text: ', 1)[1].strip().strip('"')
                    elif 'Classify the following text' in content:
                        # Alternative format: "Classify the following text as...\n\nActual text here"
                        parts = content.split('\n\n', 1)
                        if len(parts) > 1:
                            text = parts[1].strip()
                    else:
                        text = content.strip()
                    break
            
            # Extract label from assistant message
            label = None
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    try:
                        # Handle escaped JSON format
                        cleaned_content = content.replace('{{', '{').replace('}}', '}')
                        response = json.loads(cleaned_content)
                        classification = response.get('classification', '').lower()
                        if 'hate' in classification and 'not' not in classification:
                            label = 'hate'
                        else:
                            label = 'normal'
                    except json.JSONDecodeError:
                        # Fallback text parsing
                        if 'hate' in content.lower() and 'not' not in content.lower():
                            label = 'hate'
                        else:
                            label = 'normal'
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
    
    This function processes each sample through the model with the specified
    prompt configuration, handling tokenization, generation, and response parsing.
    
    Args:
        model: Loaded HuggingFace model instance
        tokenizer: Loaded tokenizer instance
        dataset: List of samples with 'text' and 'label' keys
        prompt_config: Prompt configuration dict with keys:
            - system_prompt: System message for the model
            - user_template: Template string with {text} placeholder
            - parameters: Dict with max_tokens, temperature, top_p
        
    Returns:
        List of result dictionaries with keys:
            - sample_id, text, true_label, prediction, rationale, 
              raw_response, strategy
    """
    import logging
    logger = logging.getLogger(__name__)
    results = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Processing", disable=True)):
        try:
            # Log progress every 10 samples
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
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
                'response': response,  # For ValidationResult
                'strategy': prompt_config['strategy_name'],
                'target_group': sample.get('target_group', 'unknown'),
                'source': sample.get('source', 'validation')
            })
            
        except Exception as e:
            results.append({
                'sample_id': i,
                'text': sample['text'],
                'true_label': sample['label'],
                'prediction': 'error',
                'rationale': f"Error: {str(e)}",
                'raw_response': "",
                'response': "",  # For ValidationResult
                'strategy': prompt_config['strategy_name'],
                'target_group': sample.get('target_group', 'unknown'),
                'source': sample.get('source', 'validation')
            })
    
    return results


def run_inference_with_accelerate(
    connector: 'AccelerateConnector',
    dataset: List[Dict],
    prompt_config: Dict[str, Any],
    output_dir: Path = None,
    timestamp: str = None,
    batch_size: int = 4
) -> List[Dict]:
    """
    Run inference on dataset using Accelerate for multi-GPU distribution.
    
    This function uses the AccelerateConnector to automatically split the dataset
    across available GPUs and process samples in parallel with batching.
    
    Args:
        connector: AccelerateConnector instance (already initialized with model)
        dataset: List of samples with 'text' and 'label' keys
        prompt_config: Prompt configuration dict with keys:
            - system_prompt: System message for the model
            - user_template: Template string with {text} placeholder
            - parameters: Dict with max_tokens, temperature, top_p
        output_dir: Output directory for intermediate saves (optional)
        timestamp: Timestamp for file naming (optional)
        batch_size: Number of samples to process in each batch (default: 4)
        
    Returns:
        List of result dictionaries from all GPUs (main process only)
    """
    logger = logging.getLogger(__name__)
    
    # Split dataset across GPUs automatically
    process_dataset = connector.split_dataset(dataset)
    
    logger.info(f"Process {connector.process_index}: Processing {len(process_dataset)} samples with batch_size={batch_size}")
    
    results = []
    intermediate_save_file = None
    
    # Setup intermediate save file (only main process)
    if connector.is_main_process and output_dir and timestamp:
        intermediate_save_file = output_dir / f"intermediate_results_{timestamp}.jsonl"
        logger.info(f"Intermediate results will be saved to: {intermediate_save_file}")
    
    # Process samples in batches
    for batch_start in tqdm(range(0, len(process_dataset), batch_size), 
                           desc=f"GPU {connector.process_index}", 
                           disable=not connector.is_main_process):
        batch_end = min(batch_start + batch_size, len(process_dataset))
        batch_samples = process_dataset[batch_start:batch_end]
        
        try:
            # Create messages for all samples in batch
            messages_batch = [
                [
                    {"role": "system", "content": prompt_config["system_prompt"]},
                    {"role": "user", "content": prompt_config["user_template"].format(text=sample['text'])}
                ]
                for sample in batch_samples
            ]
            
            # Get responses for entire batch
            responses = connector.complete_batch(messages_batch, **prompt_config["parameters"])
            
            # Parse each response in the batch
            for i, (sample, response) in enumerate(zip(batch_samples, responses)):
                sample_idx = batch_start + i
                response_text = response.choices[0].message.content
                
                # Parse response
                try:
                    result = json.loads(response_text.strip())
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
                    response_lower = response_text.lower()
                    if "hate" in response_lower and "not" not in response_lower:
                        pred_label = "hate"
                    elif "normal" in response_lower or "not" in response_lower:
                        pred_label = "normal"
                    else:
                        pred_label = "unknown"
                    rationale = "JSON parse failed"
                
                result_dict = {
                    'sample_id': sample_idx,
                    'text': sample['text'],
                    'true_label': sample['label'],
                    'prediction': pred_label,
                    'rationale': rationale,
                    'raw_response': response_text,
                    'response': response_text,
                    'strategy': prompt_config['strategy_name'],
                    'target_group': sample.get('target_group', 'unknown'),
                    'source': sample.get('source', 'validation')
                }
                results.append(result_dict)
                
                # Save intermediate result (only main process)
                if connector.is_main_process and intermediate_save_file:
                    with open(intermediate_save_file, 'a') as f:
                        f.write(json.dumps(result_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Error processing batch starting at {batch_start}: {e}")
            # Add error entries for all samples in failed batch
            for i, sample in enumerate(batch_samples):
                sample_idx = batch_start + i
                error_dict = {
                    'sample_id': sample_idx,
                    'text': sample['text'],
                    'true_label': sample['label'],
                    'prediction': 'error',
                    'rationale': f"Error: {str(e)}",
                    'raw_response': "",
                    'response': "",
                    'strategy': prompt_config['strategy_name'],
                    'target_group': sample.get('target_group', 'unknown'),
                    'source': sample.get('source', 'validation')
                }
                results.append(error_dict)
                
                # Save error to intermediate file
                if connector.is_main_process and intermediate_save_file:
                    with open(intermediate_save_file, 'a') as f:
                        f.write(json.dumps(error_dict) + '\n')
    
    # Wait for all GPUs to finish
    connector.wait_for_everyone()
    
    # Gather results from all GPUs (only main process gets complete results)
    all_results = connector.gather_results(results)
    
    if connector.is_main_process:
        logger.info(f"Gathered {len(all_results)} total results from all GPUs")
        return all_results
    else:
        return []


def save_results(output_dir: Path, timestamp: str, results: List[Dict], dataset: List[Dict], 
                prompt_config: Dict, command_line: str = "Unknown"):
    """
    Save comprehensive validation results using PersistenceHelper and EvaluationMetrics.
    
    Creates all output files matching prompt_engineering format:
    1. evaluation_report_*.txt - Comprehensive report with samples and bias metrics
    2. strategy_unified_results_*.csv - Full inference results  
    3. performance_metrics_*.csv - Performance metrics by strategy
    4. bias_metrics_*.csv - Bias analysis by persona/target group
    5. test_samples_*.csv - Original test samples used
    
    Args:
        output_dir: Path to output directory for this run
        timestamp: Timestamp string for file naming
        results: List of inference result dictionaries
        dataset: Original dataset used for validation
        prompt_config: Prompt configuration dictionary
        command_line: Command line used to run validation
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialize PersistenceHelper and EvaluationMetrics
    persistence = PersistenceHelper(output_dir.parent)  # Parent since output_dir is run_xxx
    evaluator = EvaluationMetrics()
    
    # Convert results to ValidationResult format for prompt_engineering compatibility
    validation_results = []
    for r in results:
        val_result = ValidationResult(
            strategy_name=r['strategy'],
            input_text=r['text'],
            true_label=r['true_label'],
            predicted_label=r['prediction'],
            response_text=r.get('response', ''),  # Full model response
            response_time=0.0,  # Not tracked in current implementation
            metrics={
                'target_group': r.get('target_group', 'unknown'),
                'source': r.get('source', 'validation')
            },
            rationale=r['rationale']
        )
        validation_results.append(val_result)
    
    # Convert results to dict format for detailed CSV
    detailed_results = [
        {
            'strategy': r.strategy_name,
            'input_text': r.input_text,
            'true_label': r.true_label,
            'predicted_label': r.predicted_label,
            'response_time': r.response_time,
            'rationale': r.rationale,
            'target_group_norm': r.metrics.get('target_group', 'unknown'),
            'persona_tag': r.metrics.get('target_group', 'unknown').upper() if r.metrics.get('target_group') else 'UNKNOWN',
            'source_dataset': r.metrics.get('source', 'validation'),
            'status': 'success'
        }
        for r in validation_results
    ]
    
    # Save incremental files (strategy results and test samples)
    persistence.initialize_incremental_storage(timestamp, output_dir)
    
    # Save detailed results incrementally
    for result_dict in detailed_results:
        persistence.save_result_incrementally(result_dict)
    
    # Save test samples incrementally
    for sample in dataset:
        sample_dict = {
            'text': sample['text'],
            'label_binary': sample['label'],
            'target_group': sample.get('target_group', 'unknown'),
            'source': sample.get('source', 'validation')
        }
        persistence.save_sample_incrementally(sample_dict)
    
    # Finalize incremental storage
    persistence.finalize_incremental_storage()
    
    # Use EvaluationMetrics.calculate_metrics_from_runid() to calculate and save everything
    # This handles: performance metrics, bias metrics, and evaluation report
    run_id = f"run_{timestamp}"
    
    try:
        metrics_result = evaluator.calculate_metrics_from_runid(
            run_id,
            str(output_dir.parent),  # Base output directory
            prompt_config.get('model_name', 'openai/gpt-oss-20b'),
            prompt_config.get('template_file', 'default'),
            prompt_config.get('data_source', 'validation'),
            command_line
        )
        
        logger.info("All metrics calculated and saved:")
        if 'performance_metrics' in metrics_result:
            logger.info("  ✓ Performance metrics")
        if 'bias_metrics' in metrics_result:
            logger.info("  ✓ Bias metrics")
        if 'evaluation_report' in metrics_result:
            logger.info("  ✓ Evaluation report")
            
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
    
    # Calculate summary metrics for logging
    all_results = {prompt_config['strategy_name']: validation_results}
    performance_metrics = evaluator.calculate_metrics_for_all_strategies(all_results, dataset)
    
    logger.info("="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Strategy: {prompt_config['strategy_name']}")
    if performance_metrics:
        first_metric = performance_metrics[0]
        logger.info(f"Accuracy:  {first_metric.accuracy:.3f}")
        logger.info(f"Precision: {first_metric.precision:.3f}")
        logger.info(f"Recall:    {first_metric.recall:.3f}")
        logger.info(f"F1-Score:  {first_metric.f1_score:.3f}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)


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


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration with file output only (no console output).
    
    Args:
        debug (bool): Enable debug level logging
        log_file (Optional[str]): Path to log file for file output
    """
    import logging
    
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers - only file handler, no console output
    handlers = []
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    if handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override any existing configuration
        )
    else:
        # If no file handler, configure minimal logging to avoid output
        logging.basicConfig(
            level=logging.CRITICAL,  # Only show critical errors
            handlers=[logging.NullHandler()],
            force=True
        )


def main(args):
    """
    Execute baseline validation pipeline with comprehensive logging.
    
    This is the main entry point for running validation. It orchestrates:
    1. Configuration setup and validation
    2. Model loading with caching
    3. Dataset loading (supports JSONL and canned datasets)
    4. Inference execution with progress tracking
    5. Metrics calculation and result saving
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Create output directory with timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file (no console output to keep prints clean)
    log_file = output_dir / f"validation_log_{timestamp}.log"
    setup_logging(args.debug, str(log_file))
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting baseline validation pipeline")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Output directory: {output_dir}")
    if args.prompt_template:
        logger.info(f"Prompt template: {args.prompt_template}")
        logger.info(f"Strategy: {args.strategy}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    else:
        logger.info(f"Max samples: ALL")
    
    # Print ONLY Run ID to console
    print(f"\n{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    # Initialize strategy loader and load prompt template if provided
    if args.prompt_template:
        logger.info(f"Loading prompt template: {args.prompt_template}")
        strategy_loader = StrategyTemplatesLoader(args.prompt_template)
        prompt_config = load_strategy_config(strategy_loader, args.strategy)
        logger.info(f"Loaded strategy: {prompt_config['strategy_name']}")
    else:
        # Use default simple prompt (no template file needed)
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
        logger.info("Using default baseline prompt")
    
    # Check if using Accelerate for multi-GPU
    if args.use_accelerate:
        if not ACCELERATE_AVAILABLE:
            logger.error("Accelerate not available. Install with: pip install accelerate")
            print("[ERROR] Accelerate not installed. Run: pip install accelerate")
            return 1
        
        logger.info("Using Accelerate for multi-GPU inference")
        
        # Initialize AccelerateConnector
        connector = AccelerateConnector(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            batch_size=1,  # Single sample processing for now
            mixed_precision='bf16'
        )
        
        # Load model through connector
        connector.load_model_once()
        
        # Load validation data
        logger.info(f"Loading validation data from: {args.data_file}")
        dataset = load_validation_data(args.data_file, args.max_samples)
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Run inference with Accelerate (automatic multi-GPU with batching)
        logger.info("Running inference with Accelerate (batch processing)...")
        results = run_inference_with_accelerate(
            connector, 
            dataset, 
            prompt_config,
            output_dir=output_dir,
            timestamp=timestamp,
            batch_size=4  # Process 4 samples per batch for faster inference
        )
        
        # Only main process continues to save results
        if not connector.is_main_process:
            logger.info(f"GPU {connector.process_index} finished processing")
            return 0
            
    else:
        # Standard single-GPU inference
        logger.info("Loading model...")
        model, tokenizer = load_model(args.model_name, cache_dir=args.cache_dir)
        
        # Load validation data
        logger.info(f"Loading validation data from: {args.data_file}")
        dataset = load_validation_data(args.data_file, args.max_samples)
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Run inference
        logger.info("Running inference...")
        results = run_inference_with_prompt(model, tokenizer, dataset, prompt_config)
    
    # Add metadata to prompt_config for reporting
    prompt_config['model_name'] = args.model_name
    prompt_config['data_source'] = args.data_file
    prompt_config['template_file'] = args.prompt_template if args.prompt_template else 'default'
    
    # Capture command line for reporting
    # Check if running under accelerate (RANK env var is set by accelerate)
    if args.use_accelerate and 'RANK' in os.environ:
        # Reconstruct accelerate command from environment
        num_processes = os.environ.get('WORLD_SIZE', '1')
        command_line = f"accelerate launch --num_processes {num_processes} -m finetuning.pipeline.baseline.runner " + " ".join(sys.argv[1:])
    else:
        command_line = " ".join(sys.argv)
    
    # Save comprehensive results using PersistenceHelper
    logger.info("Saving results...")
    save_results(output_dir, timestamp, results, dataset, prompt_config, command_line)
    
    logger.info("Baseline validation complete!")
    print(f"Results saved to: {output_dir}\n")
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
        default="canned_50_quick",
        help="Data source: 'unified' for unified test dataset, canned dataset name (e.g., 'canned_50_quick', 'canned_100_stratified'), or path to JSONL file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuning/outputs/baseline",
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
        help="Path to prompt template JSON file (e.g., prompt_engineering/prompt_templates/baseline_v1.json). If not provided, uses default baseline prompt."
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline_conservative",
        help="Strategy name from prompt template (default: %(default)s). Only used when --prompt_template is provided."
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,  # Default to 5 samples for quick testing
        help="Maximum samples to process (default: %(default)s for quick validation)"
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
        default=100,  # Increased default for JSON responses
        help="Maximum generated tokens (default: %(default)s)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation (default: %(default)s)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to file"
    )
    
    parser.add_argument(
        "--use_accelerate",
        action="store_true",
        help="Use Accelerate for multi-GPU inference (requires: accelerate launch)"
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
