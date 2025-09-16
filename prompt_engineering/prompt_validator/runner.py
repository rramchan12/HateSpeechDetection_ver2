"""
Command-line runner for GPT-OSS-20B prompt validation.

This module provides a comprehensive CLI interface for testing and evaluating 
different prompt strategies for hate speech detection using GPT-OSS-20B.

Main functionalities:
- Connection testing to Azure AI endpoint
- Single strategy testing with canned samples
- Comprehensive strategy evaluation with unified datasets
- Performance metrics and result file generation

Usage:
    python runner.py                          # Run default comprehensive test
    python runner.py --test-connection        # Test endpoint connection only
    python runner.py --test-prompt baseline   # Test specific strategy with canned samples
    python runner.py --test-strategy all      # Test all strategies with unified data
"""

import argparse
import logging
import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Local imports
from core_validator import PromptValidator
from strategy_templates import load_strategy_templates


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_strategies() -> List[str]:
    """
    Get list of available strategies from JSON configuration file.
    
    Returns:
        List[str]: Available strategy names
    """
    try:
        templates = load_strategy_templates()
        return list(templates.keys())
    except Exception as e:
        print(f"Warning: Could not load strategies from JSON: {e}")
        return ["baseline", "policy", "persona", "combined"]  # fallback


def get_strategy_choices() -> List[str]:
    """
    Get strategy choices including 'all' option for CLI argument parsing.
    
    Returns:
        List[str]: Strategy names plus 'all' option
    """
    strategies = get_available_strategies()
    return strategies + ["all"]


def setup_logging() -> None:
    """
    Configure logging for the application.
    Sets up INFO level logging with timestamp and module information.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ============================================================================
# CONNECTION TESTING
# ============================================================================

def test_connection() -> bool:
    """
    Test connection to GPT-OSS-20B endpoint with simple validation request.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\nTESTING GPT-OSS-20B CONNECTION")
    print("=" * 35)
    
    # Validate environment variables
    endpoint = os.getenv('AZURE_AI_ENDPOINT')
    key = os.getenv('AZURE_AI_KEY')
    
    if not endpoint or not key:
        print("ERROR: AZURE_AI_ENDPOINT or AZURE_AI_KEY not set in environment")
        print("Please set these environment variables and try again")
        return False
    
    print(f"Environment variables configured")
    print(f"Endpoint: {endpoint}")
    
    # Initialize validator and test with simple prompt
    try:
        validator = PromptValidator()
        print("Validator initialized successfully")
        
        # Use canned sample for connection test
        test_text = "This is a test message to verify the connection"
        result = validator.test_single_strategy("baseline", test_text, "normal")
        
        if result and hasattr(result, 'predicted_label'):
            print(f"Connection successful! Model responded with: {result.predicted_label}")
            print(f"Response time: {result.response_time:.3f}s")
            if hasattr(result, 'rationale') and result.rationale:
                print(f"Rationale: {result.rationale}")
            return True
        else:
            print("ERROR: Connection failed: No valid response from model")
            return False
            
    except Exception as e:
        print(f"ERROR: Connection failed: {str(e)}")
        return False


# ============================================================================
# PROMPT STRATEGY TESTING
# ============================================================================

def test_prompt_strategy(strategy: str) -> bool:
    """
    Test a specific prompt strategy using canned samples from JSON file.
    
    This function demonstrates prompt format and validates basic functionality
    without running comprehensive evaluation metrics.
    
    Args:
        strategy (str): Strategy name to test
        
    Returns:
        bool: True if test successful, False otherwise
    """
    print(f"\nTESTING PROMPT STRATEGY: {strategy.upper()}")
    print("=" * 50)
    
    # Load canned samples for testing
    canned_samples_path = Path(__file__).parent / "canned_samples.json"
    if not canned_samples_path.exists():
        print(f"ERROR: Canned samples file not found: {canned_samples_path}")
        return False
    
    try:
        with open(canned_samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} canned samples")
    except Exception as e:
        print(f"ERROR: Error loading canned samples: {e}")
        return False
    
    # Initialize validator
    try:
        validator = PromptValidator()
        print("Validator initialized")
    except Exception as e:
        print(f"ERROR: Error initializing validator: {e}")
        return False
    
    # Test strategy on first sample
    test_sample = samples[0]
    text = test_sample['text']
    expected_label = test_sample['label_binary']
    
    print(f"\nTest Sample:")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Expected: {expected_label}")
    
    try:
        result = validator.test_single_strategy(strategy, text, expected_label)
        if result:
            print(f"Strategy '{strategy}' completed successfully")
            print(f"  Prediction: {result.predicted_label}")
            print(f"  Response time: {result.response_time:.3f}s")
            if hasattr(result, 'rationale') and result.rationale:
                print(f"  Rationale: {result.rationale}")
            return True
        else:
            print(f"ERROR: Strategy '{strategy}' failed: No result returned")
            return False
            
    except Exception as e:
        print(f"ERROR: Error testing strategy '{strategy}': {e}")
        return False


# ============================================================================
# COMPREHENSIVE STRATEGY EVALUATION
# ============================================================================

def test_strategies(strategies: List[str], sample_size: int = 5) -> bool:
    """
    Run comprehensive evaluation of specified strategies using unified dataset.
    
    This function loads data, runs validation on multiple samples, calculates
    performance metrics, and saves detailed results to output files.
    
    Args:
        strategies (List[str]): List of strategy names to evaluate
        sample_size (int): Number of samples to test per strategy
        
    Returns:
        bool: True if evaluation completed successfully
    """
    print(f"\nCOMPREHENSIVE STRATEGY EVALUATION")
    print("=" * 50)
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Sample size: {sample_size}")
    
    # Load test data
    canned_samples_path = Path(__file__).parent / "canned_samples.json"
    if not canned_samples_path.exists():
        print(f"ERROR: Canned samples file not found: {canned_samples_path}")
        return False
    
    try:
        with open(canned_samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # Limit to requested sample size
        if len(samples) > sample_size:
            samples = samples[:sample_size]
        
        print(f"Loaded {len(samples)} test samples")
    except Exception as e:
        print(f"ERROR: Error loading test data: {e}")
        return False
    
    # Initialize validator
    try:
        validator = PromptValidator()
        print("Validator initialized")
    except Exception as e:
        print(f"ERROR: Error initializing validator: {e}")
        return False
    
    # Prepare output directory
    output_dir = Path(__file__).parent / "validation_outputs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Storage for all results
    all_results = {}
    detailed_results = []
    
    # Run evaluation for each strategy
    for strategy in strategies:
        print(f"\n--- Testing Strategy: {strategy.upper()} ---")
        strategy_results = []
        
        for i, sample in enumerate(samples, 1):
            text = sample['text']
            true_label = sample['label_binary']
            
            print(f"Sample {i}/{len(samples)}: Processing...")
            
            try:
                result = validator.test_single_strategy(strategy, text, true_label)
                if result:
                    # Store detailed result
                    detailed_result = {
                        'strategy': strategy,
                        'sample_id': i,
                        'text': text,
                        'true_label': true_label,
                        'predicted_label': result.predicted_label,
                        'response_time': result.response_time,
                        'rationale': getattr(result, 'rationale', '') or ''
                    }
                    detailed_results.append(detailed_result)
                    strategy_results.append(result)
                    
                    print(f"  Predicted: {result.predicted_label} (time: {result.response_time:.3f}s)")
                else:
                    print(f"  ERROR: Failed to get prediction")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
        
        all_results[strategy] = strategy_results
        print(f"Completed {strategy}: {len(strategy_results)}/{len(samples)} successful predictions")
    
    # Generate and save performance metrics
    return _save_evaluation_results(all_results, detailed_results, samples, timestamp, output_dir)


def _save_evaluation_results(all_results: Dict[str, List], detailed_results: List[Dict], 
                           samples: List[Dict], timestamp: str, output_dir: Path) -> bool:
    """
    Calculate metrics and save all evaluation results to files.
    
    Args:
        all_results: Results grouped by strategy
        detailed_results: Detailed prediction results
        samples: Original test samples
        timestamp: Timestamp for file naming
        output_dir: Directory to save output files
        
    Returns:
        bool: True if saving completed successfully
    """
    from evaluation_metrics import EvaluationMetrics
    from sklearn.metrics import confusion_matrix
    
    try:
        # Initialize evaluation metrics calculator
        evaluator = EvaluationMetrics()
        # Save detailed results to CSV
        detailed_csv_path = output_dir / f"strategy_unified_results_{timestamp}.csv"
        with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
            if detailed_results:
                fieldnames = detailed_results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detailed_results)
        print(f"Detailed results saved to: {detailed_csv_path}")
        
        # Save test samples
        samples_csv_path = output_dir / f"test_samples_{timestamp}.csv"
        with open(samples_csv_path, 'w', newline='', encoding='utf-8') as f:
            if samples:
                fieldnames = samples[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(samples)
        print(f"Test samples saved to: {samples_csv_path}")
        
        # Calculate and save performance metrics
        performance_data = []
        report_lines = []
        
        report_lines.append("HATE SPEECH DETECTION - STRATEGY EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total samples tested: {len(samples)}")
        report_lines.append("")
        
        # Add test samples to report
        report_lines.append("TEST SAMPLES:")
        report_lines.append("-" * 20)
        for i, sample in enumerate(samples, 1):
            report_lines.append(f"{i}. Text: {sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}")
            report_lines.append(f"   Label: {sample['label_binary']}")
            report_lines.append("")
        
        report_lines.append("\nSTRATEGY PERFORMANCE:")
        report_lines.append("-" * 25)
        
        for strategy, results in all_results.items():
            if not results:
                continue
                
            # Extract predictions and true labels
            y_true = [sample['label_binary'] for sample in samples[:len(results)]]
            y_pred = [result.predicted_label for result in results]
            
            # Calculate metrics using sklearn
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            try:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, pos_label='hate', average='binary', zero_division=0)
                recall = recall_score(y_true, y_pred, pos_label='hate', average='binary', zero_division=0)
                f1 = f1_score(y_true, y_pred, pos_label='hate', average='binary', zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            except Exception as e:
                print(f"Warning: Error calculating metrics for {strategy}: {e}")
                metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['hate', 'normal'])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
            
            # Store performance data
            performance_row = {
                'strategy': strategy,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            }
            performance_data.append(performance_row)
            
            # Add to report
            report_lines.append(f"\n{strategy.upper()} Strategy:")
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.3f}")
            report_lines.append(f"  Precision: {metrics['precision']:.3f}")
            report_lines.append(f"  Recall:    {metrics['recall']:.3f}")
            report_lines.append(f"  F1-Score:  {metrics['f1_score']:.3f}")
            report_lines.append(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Save performance metrics
        performance_csv_path = output_dir / f"performance_metrics_{timestamp}.csv"
        if performance_data:
            with open(performance_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = performance_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(performance_data)
        print(f"Performance metrics saved to: {performance_csv_path}")
        
        # Save human-readable report
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"Evaluation report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error saving results: {e}")
        return False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="GPT-OSS-20B Prompt Validation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                           # Run comprehensive evaluation (default)
  python runner.py --test-connection         # Test endpoint connection only
  python runner.py --test-prompt baseline    # Test baseline strategy with canned samples
  python runner.py --test-strategy all       # Test all strategies with evaluation metrics
  python runner.py --test-strategy baseline persona --sample-size 3
        """
    )
    
    # Connection testing
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test connection to GPT-OSS-20B endpoint'
    )
    
    # Prompt strategy testing (quick validation with canned samples)
    parser.add_argument(
        '--test-prompt',
        choices=get_available_strategies(),
        help='Test specific prompt strategy with canned samples (quick validation)'
    )
    
    # Strategy evaluation (comprehensive with metrics)
    parser.add_argument(
        '--test-strategy',
        nargs='+',
        choices=get_strategy_choices(),
        help='Run comprehensive strategy evaluation with metrics and output files'
    )
    
    # Sample size control
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5,
        help='Number of samples to use for strategy testing (default: 5)'
    )
    
    return parser


def main():
    """
    Main entry point for the CLI application.
    
    Handles argument parsing, validates environment setup, and dispatches
    to appropriate testing/evaluation functions. If no arguments provided,
    runs comprehensive evaluation of all strategies.
    """
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()
    
    # If no arguments provided, run comprehensive test as default
    if len(sys.argv) == 1:
        print("No arguments provided. Running comprehensive strategy evaluation...")
        args.test_strategy = ["all"]
        args.sample_size = 20
    
    success = True
    
    # Handle connection testing
    if args.test_connection:
        success = test_connection()
    
    # Handle prompt strategy testing (quick validation)
    elif args.test_prompt:
        success = test_prompt_strategy(args.test_prompt)
    
    # Handle comprehensive strategy evaluation
    elif args.test_strategy:
        strategies = args.test_strategy
        if "all" in strategies:
            strategies = get_available_strategies()
        success = test_strategies(strategies, args.sample_size)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()