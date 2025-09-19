"""
Main command-line runner for hate speech detection prompt validation.

This module provides a comprehensive CLI interface for testing and evaluating 
different prompt strategies for hate speech detection using Azure AI models.

Main functionalities:
- Model configuration via YAML with multi-model support (GPT-OSS-20B, GPT-5)
- Unified dataset loading with flexible data source selection
- Performance metrics and comprehensive result file generation
- Connection testing and debugging capabilities

Data Sources:
- Unified: Large comprehensive test dataset from unified_test.json
- Canned files: Specific curated datasets by name (e.g., canned_basic_all, canned_100_all)

Usage Examples:
    # Basic validation with default settings
    python prompt_runner.py --data-source unified --strategies baseline combined
    
    # Use specific canned dataset
    python prompt_runner.py --data-source canned_100_all --strategies policy persona
    
    # Run all strategies on a specific canned file
    python prompt_runner.py --data-source canned_basic_all --strategies all
    
    # Test specific model with unified dataset
    python prompt_runner.py --model gpt-5 --data-source unified --strategies policy persona
    
    # Connection testing
    python prompt_runner.py --test-connection
    
    # Metrics only calculation
    python prompt_runner.py --metrics-only --data-source unified
    
    # Custom configuration and output
    python prompt_runner.py --config custom_model.yaml --output-dir results --strategies all
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
from azure.ai.inference.models import SystemMessage, UserMessage

# Local imports
from azureai_mi_connector_wrapper import AzureAIConnector
from strategy_templates_loader import StrategyTemplatesLoader
from unified_dataset_loader import load_dataset, load_dataset_by_filename, get_dataset_info, DatasetType
from evaluation_metrics_calc import EvaluationMetrics, ValidationResult
from persistence_helper import PersistenceHelper


# ============================================================================
# PROMPT RUNNER CLASS
# ============================================================================

class PromptRunner:
    """
    Main class for orchestrating hate speech detection prompt validation.
    
    This class encapsulates all the functionality for running prompt validation
    workflows, including strategy loading, model connection, and result generation.
    
    Attributes:
        model_id (str): Model identifier for configuration lookup
        config_path (Optional[str]): Path to YAML configuration file
        strategy_loader (StrategyTemplatesLoader): Strategy templates loader instance
        connector (Optional[AzureAIConnector]): Azure AI connector instance
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, model_id: str = "gpt-oss-20b", config_path: Optional[str] = None):
        """
        Initialize the PromptRunner.
        
        Args:
            model_id (str): Model identifier for configuration lookup
            config_path (Optional[str]): Path to YAML configuration file
        """
        self.model_id = model_id
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy loader
        self.strategy_loader = StrategyTemplatesLoader()
        
        # Initialize connector (will be created when needed)
        self.connector: Optional[AzureAIConnector] = None
    
    def get_connector(self) -> AzureAIConnector:
        """
        Get or create the Azure AI connector instance.
        
        Returns:
            AzureAIConnector: Initialized connector instance
        """
        if self.connector is None:
            self.connector = AzureAIConnector(
                model_id=self.model_id,
                config_path=self.config_path
            )
            self.logger.info(f"Azure AI connector initialized for model '{self.model_id}'")
        
        return self.connector
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategy names.
        
        Returns:
            List[str]: Available strategy names
        """
        return self.strategy_loader.get_available_strategy_names()
    
    def validate_strategies(self, strategy_names: List[str]) -> List[str]:
        """
        Validate strategy names and return any invalid ones.
        
        Args:
            strategy_names (List[str]): Strategy names to validate
            
        Returns:
            List[str]: Invalid strategy names (empty if all valid)
        """
        return self.strategy_loader.validate_strategies(strategy_names)
    
    def test_connection(self) -> bool:
        """
        Test the connection to Azure AI endpoint.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        print(f"\nTesting connection to Azure AI for model: {self.model_id}")
        print("=" * 50)
        
        try:
            # Get connector
            connector = self.get_connector()
            print("Azure AI connector initialized successfully")
            
            # Get baseline strategy for testing
            baseline_strategy = self.strategy_loader.get_strategy('baseline')
            if baseline_strategy is None:
                print("ERROR: Baseline strategy not found in templates")
                return False
            
            # Prepare test messages
            test_text = "This is a test message to verify the connection"
            user_prompt = baseline_strategy.format_prompt(test_text, target_group="general")
            system_prompt = baseline_strategy.template.system_prompt
            
            # Test the connection with a simple completion
            import time
            start_time = time.time()
            
            # Create messages for the completion using actual strategy prompts
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ]
            
            response = connector.complete(
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                response_format="json_object"
            )
            
            response_time = time.time() - start_time
            
            if response and response.choices:
                content = response.choices[0].message.content
                print(f"Connection successful! Model responded with content")
                print(f"Response time: {response_time:.3f}s")
                print(f"Content preview: {content[:100]}...")
                return True
            else:
                print("ERROR: Connection failed: No valid response from model")
                return False
                
        except Exception as e:
            print(f"ERROR: Connection failed: {str(e)}")
            return False
    
    def run_validation(self, strategies: List[str], data_source: str, output_dir: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run validation for specified strategies.
        
        Args:
            strategies: List of strategy names to validate
            data_source: Data source type ("unified" or "canned")
            output_dir: Directory for output files
            sample_size: Optional number of samples to use (extracts from unified dataset if specified)
            
        Returns:
            Dict[str, Any]: Validation results with summary information
        """
        # Use the class method for validation
        return self.run_strategy_validation(
            strategies=strategies,
            data_source=data_source,
            output_dir=output_dir,
            sample_size=sample_size
        )
    
    def calculate_metrics_only(self, run_id: str, output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Calculate metrics from previously saved validation results in a specific run folder.
        
        Args:
            run_id: The run ID (folder name) containing the validation results
            output_dir: Base output directory containing run folders (default: outputs)
            
        Returns:
            Dict[str, Any]: Recalculated validation metrics from saved results
        """
        # Use the standalone function with the new signature
        return calculate_metrics_only(run_id=run_id, output_dir=output_dir)
    
    def run_strategy_validation(self, strategies: List[str], data_source: str, output_dir: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run validation for specified strategies using incremental storage approach.
        This method saves results immediately to reduce memory pressure for large datasets.
        
        Args:
            strategies: List of strategy names to validate
            data_source: Data source type ("unified" or "canned")
            output_dir: Directory for output files
            sample_size: Optional number of samples to use (extracts from unified dataset if specified)
            
        Returns:
            Dict containing validation results and summary
        """
        import json
        import time
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize Azure AI connector with YAML config only
            connector = AzureAIConnector(
                model_id=self.model_id,
                config_path=self.config_path
            )
            logger.info(f"Azure AI connector initialized for model '{self.model_id}'")
            
            # Load strategy templates using StrategyTemplatesLoader
            strategy_templates = self.strategy_loader.get_strategies()
            logger.info(f"Loaded {len(strategy_templates)} strategy templates")
            
            # Validate requested strategies
            invalid_strategies = self.strategy_loader.validate_strategies(strategies)
            if invalid_strategies:
                raise ValueError(f"Invalid strategies: {invalid_strategies}")
            
            # Load test data using unified dataset loader
            try:
                if data_source == "unified":
                    if sample_size is not None:
                        # Load specified number of samples from unified dataset
                        samples = load_dataset("unified", sample_size)
                        logger.info(f"Extracting {sample_size} samples from unified dataset")
                    else:
                        samples = load_dataset("unified", "all")
                elif data_source.startswith("canned_"):
                    # Load specific canned file by name with optional sampling
                    if sample_size is not None:
                        samples = load_dataset_by_filename(data_source, sample_size)
                        logger.info(f"Loaded {sample_size} samples from canned file: {data_source}.json")
                    else:
                        samples = load_dataset_by_filename(data_source)
                        logger.info(f"Loaded samples from canned file: {data_source}.json")
                else:
                    # Default fallback to basic canned samples with optional sampling
                    if sample_size is not None:
                        samples = load_dataset("canned", sample_size)
                        logger.info(f"Loaded {sample_size} samples from default canned dataset")
                    else:
                        samples = load_dataset("canned", "all")
                        logger.info(f"Loaded samples from default canned dataset")
                
                logger.info(f"Loaded {len(samples)} samples from {data_source} dataset")
                
                if len(samples) == 0:
                    error_msg = f"CRITICAL ERROR: No test data available in {data_source} dataset. Cannot proceed with validation."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Failed to load {data_source} dataset: {e}. Cannot proceed with validation."
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Create unique run ID and output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"
            
            # Initialize persistence helper with incremental storage
            persistence = PersistenceHelper(output_dir)
            runid_dir = persistence.create_runid_directory(timestamp)
            persistence.initialize_incremental_storage(timestamp, runid_dir)
            
            logger.info(f"Created run directory: {runid_dir}")
            logger.info(f"Initialized incremental storage for runId: {run_id}")
            
            # Save samples incrementally (once at the beginning)
            for sample in samples:
                sample_data = {
                    'text': sample['text'],
                    'label_binary': sample.get('label_binary', 'unknown'),
                    'target_group_norm': sample.get('target_group_norm', 'unknown'),
                    'persona_tag': sample.get('persona_tag', 'unknown')
                }
                persistence.save_sample_incrementally(sample_data)
            
            # Run validation for each strategy with incremental storage
            total_predictions = 0
            
            logger.info(f"Validating strategies {strategies} with {len(samples)} samples")
            logger.info(f"Running batch validation: {len(samples)} texts, {len(strategies)} strategies")
            
            for strategy_name in strategies:
                strategy = strategy_templates[strategy_name]
                logger.info(f"Processing strategy: {strategy_name}")
                
                for i, sample in enumerate(samples):
                    try:
                        # Format prompt using strategy
                        user_prompt = strategy.format_prompt(
                            sample['text'], 
                            target_group=sample.get('target_group_norm', 'general')
                        )
                        
                        # Get system prompt from strategy template
                        system_prompt = strategy.template.system_prompt
                        
                        # Prepare model parameters
                        params = strategy.get_model_parameters()
                        
                        # Create completion
                        start_time = time.time()
                        
                        # Create messages for the completion using actual strategy prompts
                        messages = [
                            SystemMessage(content=system_prompt),
                            UserMessage(content=user_prompt)
                        ]
                        
                        response = connector.complete(
                            messages=messages,
                            **params
                        )
                        response_time = time.time() - start_time
                        
                        # Parse response
                        if response and response.choices:
                            content = response.choices[0].message.content
                            predicted_label, rationale = parse_model_response(content)
                            response_text = content
                        else:
                            predicted_label, rationale = "unknown", "No response received"
                            response_text = ""
                        
                        # Save result incrementally (no memory accumulation)
                        result_data = {
                            'strategy': strategy_name,
                            'input_text': sample['text'],
                            'true_label': sample.get('label_binary', 'unknown'),
                            'predicted_label': predicted_label,
                            'response_time': response_time,
                            'rationale': rationale,
                            'target_group_norm': sample.get('target_group_norm', 'unknown'),
                            'persona_tag': sample.get('persona_tag', 'unknown'),
                            'source_dataset': data_source
                        }
                        persistence.save_result_incrementally(result_data)
                        total_predictions += 1
                        
                        # Log progress for large datasets
                        if (i + 1) % 100 == 0:
                            logger.info(f"Processed {i + 1}/{len(samples)} samples for strategy {strategy_name}")
                        
                    except Exception as e:
                        logger.error(f"Error processing sample {i} with strategy {strategy_name}: {e}")
                        # Save error result incrementally
                        error_result = {
                            'strategy': strategy_name,
                            'input_text': sample['text'],
                            'true_label': sample.get('label_binary', 'unknown'),
                            'predicted_label': "error",
                            'response_time': 0.0,
                            'rationale': f"Error: {str(e)}",
                            'target_group_norm': sample.get('target_group_norm', 'unknown'),
                            'persona_tag': sample.get('persona_tag', 'unknown'),
                            'source_dataset': data_source
                        }
                        persistence.save_result_incrementally(error_result)
                        total_predictions += 1
            
            # Finalize incremental storage
            output_paths = persistence.finalize_incremental_storage()
            logger.info(f"Finalized incremental storage. Saved {total_predictions} predictions.")
            
            # Calculate metrics from stored results (memory-efficient)
            try:
                # Use evaluation_metrics_calc to calculate metrics from the runId
                evaluator = EvaluationMetrics()
                metrics_result = evaluator.calculate_metrics_from_runid(run_id, output_dir)
                
                # Add metrics paths to output_paths
                if 'performance_metrics' in metrics_result:
                    output_paths['performance_metrics'] = metrics_result['performance_metrics']
                if 'evaluation_report' in metrics_result:
                    output_paths['evaluation_report'] = metrics_result['evaluation_report']
                
                logger.info(f"Calculated and saved metrics for runId: {run_id}")
            except Exception as e:
                logger.error(f"Error calculating metrics for runId {run_id}: {e}")
                # Don't fail the entire validation if metrics calculation fails
            
            logger.info(f"Results saved to {len(output_paths)} files in {runid_dir}/")
            
            return {
                "run_id": run_id,
                "output_directory": str(runid_dir),
                "summary": {
                    "total_strategies": len(strategies),
                    "total_samples": len(samples),
                    "total_predictions": total_predictions,
                    "data_source": data_source,
                    "sample_size": sample_size if sample_size is not None else len(samples),
                    "output_files": len(output_paths),
                    "run_id": run_id,
                    "incremental_storage": True,
                    "memory_efficient": True
                }
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_metrics_only(run_id: str, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Calculate validation metrics from previously saved results in a specific run folder.
    This function delegates metrics calculation to evaluation_metrics_calc.py for proper separation of concerns.
    
    Args:
        run_id: The run ID (folder name) containing the validation results
        output_dir: Base output directory containing run folders (default: outputs)
        
    Returns:
        Dict containing recalculated validation metrics
    """
    import os
    import pandas as pd
    from datetime import datetime
    logger = logging.getLogger(__name__)
    
    try:
        # Construct path to run folder
        run_folder = os.path.join(output_dir, run_id)
        
        if not os.path.exists(run_folder):
            return {"error": f"Run folder '{run_folder}' not found", "summary": {"total_samples": 0}}
        
        logger.info(f"Recalculating metrics from run folder: {run_folder}")
        
        # Look for the strategy results CSV file
        results_files = [f for f in os.listdir(run_folder) if f.startswith('strategy_unified_results_') and f.endswith('.csv')]
        
        if not results_files:
            return {"error": f"No strategy results file found in {run_folder}", "summary": {"total_samples": 0}}
        
        results_file = os.path.join(run_folder, results_files[0])
        logger.info(f"Loading results from: {results_file}")
        
        # Use evaluation_metrics_calc for proper separation of concerns
        try:
            evaluator = EvaluationMetrics()
            metrics_result = evaluator.calculate_metrics_from_runid(run_id, output_dir)
            
            # Load basic stats for summary
            df = pd.read_csv(results_file)
            
            logger.info(f"Recalculated metrics for {len(df)} samples from run {run_id}")
            
            return {
                "run_info": {
                    "run_id": run_id,
                    "run_folder": run_folder,
                    "results_file": results_file,
                    "recalculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "metrics_files": metrics_result,
                "summary": {
                    "total_samples": metrics_result.get('total_samples', len(df)),
                    "strategies_count": len(metrics_result.get('strategies_tested', [])),
                    "overall_accuracy": 0.0,  # Will be calculated by metrics module
                    "run_id": run_id
                }
            }
            
        except Exception as metrics_error:
            # Fallback to basic calculation if advanced metrics fail
            logger.warning(f"Advanced metrics calculation failed: {metrics_error}")
            df = pd.read_csv(results_file)
            
            total_samples = len(df)
            total_correct = (df['predicted_label'] == df['true_label']).sum()
            overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0
            
            return {
                "run_info": {
                    "run_id": run_id,
                    "run_folder": run_folder,
                    "results_file": results_file,
                    "recalculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "summary": {
                    "total_samples": total_samples,
                    "strategies_count": len(df['strategy'].unique()),
                    "overall_accuracy": round(overall_accuracy, 4),
                    "run_id": run_id
                }
            }
        
    except Exception as e:
        logger.error(f"Error recalculating metrics from run {run_id}: {e}")
        return {"error": str(e), "summary": {"total_samples": 0}}

def parse_model_response(response_text: str) -> tuple[str, str]:
    """
    Parse model response to extract prediction and rationale from JSON format.
    
    Args:
        response_text: Raw model response (expected JSON format)
        
    Returns:
        Tuple[str, str]: (prediction, rationale) where prediction is 'hate' or 'normal'
    """
    if not response_text or response_text.strip() == "":
        return "normal", "No response received"
    
    # Try to parse as JSON first
    try:
        import json
        # Clean up response text
        cleaned_text = response_text.strip()
        
        # Handle common JSON formatting issues
        if not cleaned_text.startswith('{'):
            # Look for JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
            else:
                raise ValueError("No JSON content found")
        
        # Parse JSON
        parsed_response = json.loads(cleaned_text)
        
        # Extract classification and rationale
        classification = parsed_response.get('classification', '').lower()
        rationale = parsed_response.get('rationale', 'No rationale provided')
        
        # Normalize classification
        if classification in ['hate', 'hateful']:
            prediction = 'hate'
        elif classification in ['normal', 'not hate', 'not hateful']:
            prediction = 'normal'
        else:
            prediction = 'normal'  # Default to normal for unclear responses
            
        return prediction, rationale
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fallback: simple text analysis
        text_lower = response_text.lower()
        if any(word in text_lower for word in ['hate', 'hateful', 'offensive']):
            return 'hate', f"Text analysis fallback: {response_text[:100]}"
        else:
            return 'normal', f"Text analysis fallback: {response_text[:100]}"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        debug (bool): Enable debug level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the comprehensive argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all options
    """
    parser = argparse.ArgumentParser(
        description="Hate Speech Detection Prompt Validation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default settings
  python prompt_runner.py --data-source unified --strategies baseline combined
  
  # Use specific canned dataset with sampling
  python prompt_runner.py --data-source canned_100_all --strategies policy persona --sample-size 10
  
  # Run all strategies on a specific canned file with limited samples
  python prompt_runner.py --data-source canned_basic_all --strategies all --sample-size 3
  
  # Test specific model with unified dataset  
  python prompt_runner.py --model gpt-5 --data-source unified --strategies policy persona
  
  # Extract 50 samples from unified dataset for validation
  python prompt_runner.py --data-source unified --sample-size 50 --strategies baseline
  
  # Connection testing
  python prompt_runner.py --test-connection
  
  # Metrics recalculation from previous run
  python prompt_runner.py --metrics-only --run-id run_20250920_003554
  
  # Custom configuration and output
  python prompt_runner.py --config custom_model.yaml --output-dir results --strategies all
        """
    )
    
    # Model and configuration
    parser.add_argument("--model", "-m", default="gpt-oss-20b", 
                       help="Model to use from YAML config (default: gpt-oss-20b)")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to YAML model configuration file (auto-detected if not specified)")
    
    # Data source and strategy selection
    parser.add_argument("--data-source", "-d", default="unified", 
                       help="Data source: 'unified' for full dataset, or specific canned file name (e.g., 'canned_basic_all', 'canned_100_all')")
    
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of samples to extract from the dataset (applies to all data sources)")
    
    parser.add_argument("--strategies", "-s", nargs="+", 
                       default=["baseline"],
                       choices=["policy", "persona", "combined", "baseline", "all"],
                       help="Strategy(ies) to validate (default: baseline). Use 'all' to run all available strategies")
    
    # Output and debugging
    parser.add_argument("--output-dir", "-o", default="outputs",
                       help="Directory for output files (default: outputs)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--metrics-only", action="store_true",
                       help="Calculate metrics from a previous run (requires --run-id)")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Run ID to recalculate metrics from (used with --metrics-only)")
    
    # Connection testing
    parser.add_argument("--test-connection", action="store_true",
                       help="Test connection to Azure AI endpoint")
    
    return parser


def main():
    """
    Main entry point for the hate speech detection prompt validation CLI.
    
    Handles comprehensive argument parsing, validates environment setup, and dispatches
    to appropriate testing/evaluation functions using direct dependencies.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle "all" strategies
        if "all" in args.strategies:
            runner = PromptRunner(model_id=args.model, config_path=args.config)
            args.strategies = runner.get_available_strategies()
        
        logger.info(f"Starting prompt validation for model '{args.model}'")
        
        # Auto-detect config file if not specified
        config_path = args.config
        if config_path is None:
            # Look for model_connection.yaml in current directory or prompt_engineering directory
            current_dir = Path(".")
            prompt_eng_dir = Path("prompt_engineering")
            
            for potential_path in [current_dir / "model_connection.yaml", 
                                 prompt_eng_dir / "model_connection.yaml"]:
                if potential_path.exists():
                    config_path = str(potential_path)
                    logger.info(f"Auto-detected config file: {config_path}")
                    break
        
        # Handle connection testing
        if args.test_connection:
            logger.info("Testing connection to Azure AI endpoint")
            runner = PromptRunner(model_id=args.model, config_path=config_path)
            success = runner.test_connection()
            sys.exit(0 if success else 1)
        
        # Run validation based on arguments
        if args.metrics_only:
            if not args.run_id:
                logger.error("--run-id is required when using --metrics-only")
                sys.exit(1)
            
            logger.info(f"Recalculating metrics from run: {args.run_id}")
            runner = PromptRunner(model_id=args.model, config_path=config_path)
            results = runner.calculate_metrics_only(run_id=args.run_id, output_dir=args.output_dir)
        else:
            logger.info(f"Running validation with strategies: {args.strategies}")
            if args.sample_size is not None:
                logger.info(f"Using sample size: {args.sample_size}")
            runner = PromptRunner(model_id=args.model, config_path=config_path)
            results = runner.run_validation(
                strategies=args.strategies,
                data_source=args.data_source,
                output_dir=args.output_dir,
                sample_size=args.sample_size
            )
        
        # Log summary
        if isinstance(results, dict) and 'summary' in results:
            logger.info("Validation completed successfully")
            if 'run_id' in results:
                logger.info(f"  run_id: {results['run_id']}")
            if 'output_directory' in results:
                logger.info(f"  output_directory: {results['output_directory']}")
            for key, value in results['summary'].items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("Validation completed")
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
