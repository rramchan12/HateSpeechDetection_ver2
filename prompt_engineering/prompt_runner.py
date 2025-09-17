"""
Command-line runner for GPT-OSS-20B prompt validation.

This module provides a comprehensive CLI interface for testing and evaluating 
different prompt strategies for hate speech detection using GPT-OSS-20B.

Main functionalities:
- Connection testing to Azure AI endpoint
- Unified dataset loading with flexible data source selection
- Performance metrics and result file generation

Data Sources:
- Canned: Small curated set of test samples for quick validation
- Unified: Large comprehensive test dataset from unified_test.json

Usage:
    # Connection testing
    python prompt_runner.py --test-connection                  # Test endpoint connection only
    
    # Unified dataset approach
    python prompt_runner.py --dataset-type canned --num-samples 5 --strategy baseline
    python prompt_runner.py --dataset-type unified --num-samples 100 --strategy policy persona
    python prompt_runner.py --dataset-type unified --num-samples all --strategy all
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
from prompts_validator import PromptValidator
from strategy_templates_loader import load_strategy_templates
from unified_dataset_loader import load_dataset, get_dataset_info, DatasetType


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
    endpoint = os.getenv('AZURE_INFERENCE_SDK_ENDPOINT')
    key = os.getenv('AZURE_INFERENCE_SDK_KEY')
    
    if not endpoint or not key:
        print("ERROR: AZURE_INFERENCE_SDK_ENDPOINT or AZURE_INFERENCE_SDK_KEY not set in environment")
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
# UNIFIED DATASET TESTING
# ============================================================================

def test_unified_strategies(dataset_type: str, num_samples: str, strategies: List[str]) -> bool:
    """
    Test strategies using the unified dataset loader with flexible data source selection.
    
    This function uses the unified dataset loader to load samples from either canned
    or unified datasets and runs comprehensive evaluation with metrics generation.
    
    Args:
        dataset_type (str): Type of dataset to use ("canned" or "unified")
        num_samples (str): Number of samples ("all" or integer as string)
        strategies (List[str]): List of strategy names to evaluate
        
    Returns:
        bool: True if testing completed successfully, False otherwise
    """
    print(f"\nUNIFIED STRATEGY EVALUATION")
    print("=" * 50)
    print(f"Dataset type: {dataset_type}")
    print(f"Strategies: {', '.join(strategies)}")
    
    try:
        # Convert num_samples to appropriate type
        if num_samples.lower() == "all":
            sample_count = "all"
        else:
            try:
                sample_count = int(num_samples)
                if sample_count <= 0:
                    raise ValueError("Number of samples must be positive")
            except ValueError:
                print(f"ERROR: Invalid num-samples value: {num_samples}")
                return False
        
        print(f"Sample count: {sample_count}")
        
        # Load samples using unified dataset loader
        print(f"Loading samples from {dataset_type} dataset...")
        samples = load_dataset(dataset_type, sample_count, random_seed=42)
        print(f"Loaded {len(samples)} test samples")
        
        # Validate sample structure
        from unified_dataset_loader import UnifiedDatasetLoader
        loader = UnifiedDatasetLoader()
        loader.validate_sample_structure(samples)
        
        # Initialize validator
        validator = PromptValidator()
        if not validator.client:
            print("ERROR: Failed to initialize validator")
            return False
        
        print("Validator initialized")
        
        # Process each strategy
        all_results = {}
        detailed_results = []
        
        for strategy in strategies:
            print(f"\n--- Testing Strategy: {strategy.upper()} ---")
            strategy_results = []
            
            for i, sample in enumerate(samples, 1):
                print(f"Sample {i}/{len(samples)}: Processing...")
                
                # Run validation
                result = validator.test_single_strategy(
                    strategy_name=strategy,
                    text=sample['text'],
                    true_label=sample.get('label_binary')
                )
                
                strategy_results.append(result)
                print(f"  Predicted: {result.predicted_label} (time: {result.response_time:.3f}s)")
                
                # Add to detailed results for CSV export
                detailed_result = {
                    'strategy': strategy,
                    'input_text': sample['text'],
                    'true_label': sample.get('label_binary'),
                    'predicted_label': result.predicted_label,
                    'response_time': result.response_time,
                    'rationale': result.rationale,
                    'target_group_norm': sample.get('target_group_norm'),
                    'persona_tag': sample.get('persona_tag'),
                    'source_dataset': sample.get('source_dataset')
                }
                detailed_results.append(detailed_result)
            
            all_results[strategy] = strategy_results
            successful = len([r for r in strategy_results if r.predicted_label])
            print(f"Completed {strategy}: {successful}/{len(strategy_results)} successful predictions")
        
        # Save comprehensive results using persistence helper
        from persistence_helper import PersistenceHelper
        persistence = PersistenceHelper()
        timestamp = persistence.generate_timestamp()
        
        # Save all results and generate metrics
        output_paths = persistence.calculate_and_save_comprehensive_results(
            all_results, detailed_results, samples, timestamp
        )
        
        # Display output file information
        print(f"\nResults saved to:")
        for output_type, path in output_paths.items():
            if output_type == 'detailed_results':
                print(f"Detailed results saved to: {path}")
            elif output_type == 'test_samples':
                print(f"Test samples saved to: {path}")
            elif output_type == 'performance_metrics':
                print(f"Performance metrics saved to: {path}")
            elif output_type == 'evaluation_report':
                print(f"Evaluation report saved to: {path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error in unified strategy testing: {e}")
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
  python prompt_runner.py                                    # Run comprehensive evaluation (default)
  python prompt_runner.py --test-connection                  # Test endpoint connection only
  python prompt_runner.py --dataset-type canned --num-samples 3 --strategy baseline
  python prompt_runner.py --dataset-type unified --num-samples 100 --strategy policy persona
  python prompt_runner.py --dataset-type unified --num-samples all --strategy all
        """
    )
    
    # Connection testing
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test connection to GPT-OSS-20B endpoint'
    )
    
    # Unified dataset arguments
    parser.add_argument(
        '--dataset-type',
        choices=['canned', 'unified'],
        default='canned',
        help='Type of dataset to use: canned (small curated set) or unified (large comprehensive set)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=str,
        default='all',
        help='Number of samples to use: "all" or a positive integer'
    )
    
    parser.add_argument(
        '--strategy',
        nargs='+',
        choices=get_available_strategies() + ['all'],
        help='Strategy(ies) to test: policy, persona, combined, baseline, or all'
    )
    
    return parser


def main():
    """
    Main entry point for the CLI application.
    
    Handles argument parsing, validates environment setup, and dispatches
    to appropriate testing/evaluation functions. Supports both legacy and 
    new unified dataset loading approaches.
    """
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()
    
    success = True
    
    # If no arguments provided, run comprehensive test as default
    if len(sys.argv) == 1:
        print("No arguments provided. Running comprehensive strategy evaluation...")
        strategies = get_available_strategies()
        success = test_unified_strategies('canned', 'all', strategies)
    
    # Handle connection testing
    elif args.test_connection:
        success = test_connection()
    
    # Handle unified dataset approach
    elif args.strategy:
        strategies = args.strategy
        if "all" in strategies:
            strategies = get_available_strategies()
        success = test_unified_strategies(args.dataset_type, args.num_samples, strategies)
    
    # If no specific action but unified dataset args provided, use them
    elif hasattr(args, 'dataset_type') and (args.dataset_type != 'canned' or args.num_samples != 'all'):
        print("Using unified dataset approach with default strategies...")
        strategies = get_available_strategies()
        success = test_unified_strategies(args.dataset_type, args.num_samples, strategies)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
