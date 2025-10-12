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
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from azure.ai.inference.models import SystemMessage, UserMessage

# Local imports
from connector import AzureAIConnector
from loaders import StrategyTemplatesLoader, load_dataset, load_dataset_by_filename, get_dataset_info, DatasetType
from metrics import EvaluationMetrics, ValidationResult, calculate_metrics_from_runid, PersistenceHelper


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
    
    def __init__(self, model_id: str = "gpt-oss-120b", config_path: Optional[str] = None, 
                 prompt_template_file: str = "all_combined.json"):
        """
        Initialize the PromptRunner.
        
        Args:
            model_id (str): Model identifier for configuration lookup
            config_path (Optional[str]): Path to YAML configuration file
            prompt_template_file (str): Prompt template file name in prompt_templates folder
        """
        self.model_id = model_id
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy loader with the specified template file
        template_path = os.path.join(os.path.dirname(__file__), "prompt_templates", prompt_template_file)
        try:
            self.strategy_loader = StrategyTemplatesLoader(template_path)
        except FileNotFoundError:
            self.logger.error(f"Prompt template file not found: {template_path}")
            self.logger.error(f"Available template files in prompt_templates directory:")
            templates_dir = os.path.join(os.path.dirname(__file__), "prompt_templates")
            if os.path.exists(templates_dir):
                for file in os.listdir(templates_dir):
                    if file.endswith('.json'):
                        self.logger.error(f"  - {file}")
            else:
                self.logger.error(f"  Templates directory not found: {templates_dir}")
            raise FileNotFoundError(f"Template file '{prompt_template_file}' not found in prompt_templates directory")
        
        # Store metadata for reporting
        self.prompt_template_file = prompt_template_file
        self.command_line = "Unknown"
        self.data_source = "Unknown"
        
        # Initialize connector (will be created when needed)
        self.connector: Optional[AzureAIConnector] = None
    
    def set_execution_metadata(self, command_line: str, data_source: str):
        """
        Set execution metadata for reporting purposes.
        
        Args:
            command_line (str): The command line used to run the validation
            data_source (str): The data source name used
        """
        self.command_line = command_line
        self.data_source = data_source
    
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
    
    def process_sample_concurrent(self, args: tuple) -> Dict[str, Any]:
        """
        Process a single sample with a specific strategy (for concurrent execution).
        
        Args:
            args: Tuple containing (connector, strategy, sample, strategy_name, data_source)
        
        Returns:
            Dict[str, Any]: Result data for the processed sample
        """
        connector, strategy, sample, strategy_name, data_source = args
        
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
            
            # Create messages for the completion using actual strategy prompts
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ]
            
            # Make request with intelligent retry and throttling
            response, response_time = self._make_request_with_retry(connector, messages, params)
            
            # Parse response
            if response and response.choices:
                content = response.choices[0].message.content
                predicted_label, rationale = parse_model_response(content)
                response_text = content
            else:
                predicted_label, rationale = "unknown", "No response received"
                response_text = ""
            
            # Return result data
            return {
                'strategy': strategy_name,
                'input_text': sample['text'],
                'true_label': sample.get('label_binary', 'unknown'),
                'predicted_label': predicted_label,
                'response_time': response_time,
                'rationale': rationale,
                'target_group_norm': sample.get('target_group_norm', 'unknown'),
                'persona_tag': sample.get('persona_tag', 'unknown'),
                'source_dataset': data_source,
                'status': 'success'
            }
            
        except Exception as e:
            # Return error result
            return {
                'strategy': strategy_name,
                'input_text': sample['text'],
                'true_label': sample.get('label_binary', 'unknown'),
                'predicted_label': "error",
                'response_time': 0.0,
                'rationale': f"Error: {str(e)}",
                'target_group_norm': sample.get('target_group_norm', 'unknown'),
                'persona_tag': sample.get('persona_tag', 'unknown'),
                'source_dataset': data_source,
                'status': 'error'
            }

    def _make_request_with_retry(self, connector, messages, params, max_retries: int = 3) -> tuple:
        """
        Make API request with intelligent retry logic and rate limiting detection.
        
        Args:
            connector: Azure AI connector
            messages: List of messages for the API
            params: Model parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            tuple: (response, response_time)
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Add progressive delay to reduce load on API
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = min(30, (2 ** attempt) + random.uniform(0, 1))
                    self.logger.warning(f"Retrying request after {delay:.1f}s delay (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                
                # Add small random delay to avoid thundering herd
                time.sleep(random.uniform(0.1, 0.3))
                
                # Make the actual request
                start_time = time.time()
                response = connector.complete(messages=messages, **params)
                response_time = time.time() - start_time
                
                # Parse rate limit headers if available
                rate_limit_info = self._parse_rate_limit_headers(response)
                if rate_limit_info:
                    remaining_requests = rate_limit_info.get('remaining_requests', 0)
                    remaining_tokens = rate_limit_info.get('remaining_tokens', 0)
                    
                    # Log rate limit status periodically
                    if remaining_requests is not None and remaining_requests < 100:
                        self.logger.warning(f"Low remaining requests: {remaining_requests}")
                    if remaining_tokens is not None and remaining_tokens < 10000:
                        self.logger.warning(f"Low remaining tokens: {remaining_tokens}")
                
                # Log slow responses as potential throttling indicators
                if response_time > 5.0:
                    self.logger.warning(f"Slow response detected: {response_time:.1f}s (possible throttling)")
                
                return response, response_time
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check for rate limiting indicators
                if any(indicator in error_msg for indicator in ['rate limit', 'throttled', 'too many requests', '429']):
                    self.logger.warning(f"Rate limiting detected: {e}")
                    if attempt < max_retries:
                        # Longer delay for rate limiting
                        delay = min(60, (3 ** attempt) + random.uniform(5, 15))
                        self.logger.info(f"Rate limited - backing off for {delay:.1f}s")
                        time.sleep(delay)
                        continue
                
                # Check for timeout indicators  
                elif any(indicator in error_msg for indicator in ['timeout', 'connection', 'network']):
                    self.logger.warning(f"Network/timeout error: {e}")
                    if attempt < max_retries:
                        delay = min(30, (2 ** attempt) + random.uniform(1, 3))
                        time.sleep(delay)
                        continue
                
                # For other errors, retry with shorter delay
                elif attempt < max_retries:
                    delay = min(10, (1.5 ** attempt) + random.uniform(0.5, 1.5))
                    time.sleep(delay)
                    continue
                    
                # If we've exhausted retries, raise the last exception
                self.logger.error(f"Request failed after {max_retries + 1} attempts: {e}")
                raise last_exception
        
        # Should never reach here, but just in case
        raise last_exception if last_exception else Exception("Request failed for unknown reason")
    
    def _parse_rate_limit_headers(self, response) -> dict:
        """
        Parse rate limit headers from Azure AI response.
        
        Args:
            response: Azure AI response object
            
        Returns:
            dict: Parsed rate limit information
        """
        try:
            # Access response headers if available
            if hasattr(response, 'http_response') and hasattr(response.http_response, 'headers'):
                headers = response.http_response.headers
                
                rate_limit_info = {}
                
                # Parse remaining requests
                if 'x-ratelimit-remaining-requests' in headers:
                    rate_limit_info['remaining_requests'] = int(headers['x-ratelimit-remaining-requests'])
                
                # Parse remaining tokens  
                if 'x-ratelimit-remaining-tokens' in headers:
                    rate_limit_info['remaining_tokens'] = int(headers['x-ratelimit-remaining-tokens'])
                
                # Parse request limits
                if 'x-ratelimit-limit-requests' in headers:
                    rate_limit_info['limit_requests'] = int(headers['x-ratelimit-limit-requests'])
                    
                # Parse token limits
                if 'x-ratelimit-limit-tokens' in headers:
                    rate_limit_info['limit_tokens'] = int(headers['x-ratelimit-limit-tokens'])
                
                return rate_limit_info
                
        except (AttributeError, ValueError, KeyError) as e:
            # If we can't parse headers, just return empty dict
            pass
        
        return {}

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
    
    def run_validation(self, strategies: List[str], data_source: str, output_dir: str, 
                     sample_size: Optional[int] = None, use_concurrent: bool = True, 
                     max_workers: int = 5, batch_size: int = 10) -> Dict[str, Any]:
        """
        Run validation for specified strategies.
        
        Args:
            strategies: List of strategy names to validate
            data_source: Data source type ("unified" or "canned")
            output_dir: Directory for output files
            sample_size: Optional number of samples to use (extracts from unified dataset if specified)
            use_concurrent: Whether to use concurrent processing (default: True for better performance)
            max_workers: Maximum number of concurrent threads (default: 5)
            batch_size: Number of samples to process in each batch (default: 10)
            
        Returns:
            Dict[str, Any]: Validation results with summary information
        """
        if use_concurrent:
            # Use concurrent processing for better performance
            return self.run_strategy_validation_concurrent(
                strategies=strategies,
                data_source=data_source,
                output_dir=output_dir,
                sample_size=sample_size,
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            # Use sequential processing (original method)
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
    
    def run_strategy_validation_concurrent(self, strategies: List[str], data_source: str, output_dir: str, 
                                        sample_size: Optional[int] = None, max_workers: int = 5, 
                                        batch_size: int = 10) -> Dict[str, Any]:
        """
        Run validation for specified strategies using concurrent processing for better performance.
        This method processes multiple samples simultaneously using ThreadPoolExecutor.
        
        Args:
            strategies: List of strategy names to validate
            data_source: Data source type ("unified" or "canned")
            output_dir: Directory for output files
            sample_size: Optional number of samples to use
            max_workers: Maximum number of concurrent threads (default: 5)
            batch_size: Number of samples to process in each batch (default: 10)
            
        Returns:
            Dict containing validation results and summary
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Get connector
            connector = self.get_connector()
            
            # Load strategy templates
            try:
                strategy_templates = self.strategy_loader.get_strategies()
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Failed to load strategy templates: {e}. Cannot proceed."
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Filter requested strategies
            available_strategies = list(strategy_templates.keys())
            valid_strategies = [s for s in strategies if s in available_strategies]
            
            if not valid_strategies:
                error_msg = f"CRITICAL ERROR: No valid strategies found. Available: {available_strategies}, Requested: {strategies}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Load dataset samples with error handling
            try:
                # Convert None sample_size to "all" for compatibility
                num_samples = sample_size if sample_size is not None else "all"
                
                if data_source == "unified":
                    # Use proper unified dataset loader
                    samples = load_dataset(DatasetType.UNIFIED, num_samples=num_samples, random_seed=42)
                else:
                    # Use canned file loader for specific canned datasets
                    samples = load_dataset_by_filename(data_source, random_seed=42, num_samples=num_samples)
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
            
            # Print runID to console for tracking
            print(f"Run ID: {run_id}")
            
            logger.info(f"Created run directory: {runid_dir}")
            logger.info(f"Initialized incremental storage for runId: {run_id}")
            logger.info(f"Using concurrent processing: {max_workers} workers, batch size: {batch_size}")
            
            # Save samples incrementally (once at the beginning)
            for sample in samples:
                sample_data = {
                    'text': sample['text'],
                    'label_binary': sample.get('label_binary', 'unknown'),
                    'target_group_norm': sample.get('target_group_norm', 'unknown'),
                    'persona_tag': sample.get('persona_tag', 'unknown')
                }
                persistence.save_sample_incrementally(sample_data)
            
            # Run validation for each strategy with concurrent processing
            total_predictions = 0
            total_start_time = time.time()
            
            logger.info(f"Validating strategies {valid_strategies} with {len(samples)} samples")
            logger.info(f"Running concurrent validation: {len(samples)} texts, {len(valid_strategies)} strategies")
            
            for strategy_name in valid_strategies:
                strategy = strategy_templates[strategy_name]
                logger.info(f"Processing strategy: {strategy_name} (concurrent)")
                
                # Prepare tasks for concurrent execution
                tasks = []
                for sample in samples:
                    task_args = (connector, strategy, sample, strategy_name, data_source)
                    tasks.append(task_args)
                
                # Process tasks in batches with concurrent execution
                processed_count = 0
                strategy_start_time = time.time()
                
                # Process tasks with a SINGLE ThreadPoolExecutor for the entire strategy
                # This prevents connection pool exhaustion from recreating executors
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Process in batches to control memory usage
                    for i in range(0, len(tasks), batch_size):
                        batch_tasks = tasks[i:i + batch_size]
                        
                        # Submit all tasks in the batch
                        future_to_task = {
                            executor.submit(self.process_sample_concurrent, task): task 
                            for task in batch_tasks
                        }
                        
                        # Collect results as they complete
                        batch_results = []
                        for future in as_completed(future_to_task):
                            try:
                                result_data = future.result()
                                batch_results.append(result_data)
                                total_predictions += 1
                                processed_count += 1
                                
                            except Exception as e:
                                logger.error(f"Error processing task in batch: {e}")
                        
                        # Save all batch results at once for better I/O efficiency
                        for result_data in batch_results:
                            persistence.save_result_incrementally(result_data)
                        
                        # Log progress periodically
                        if processed_count % 50 == 0:
                            elapsed = time.time() - strategy_start_time
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            logger.info(f"Strategy {strategy_name}: {processed_count}/{len(samples)} samples processed ({rate:.1f}/sec)")
                        
                        # Adaptive delay between batches based on processing rate
                        if i + batch_size < len(tasks):  # Not the last batch
                            # Check if we're processing too fast (potential rate limiting)
                            elapsed = time.time() - strategy_start_time
                            current_rate = processed_count / elapsed if elapsed > 0 else 0
                            
                            # If processing very fast (>20/sec), add small delay to respect rate limits
                            if current_rate > 20:
                                adaptive_delay = 0.5
                                logger.debug(f"High processing rate ({current_rate:.1f}/sec), adding {adaptive_delay}s delay")
                                time.sleep(adaptive_delay)
                            # Normal processing, minimal delay
                            else:
                                time.sleep(0.1)  # Minimal delay to prevent thundering herd
                
                strategy_elapsed = time.time() - strategy_start_time
                strategy_rate = len(samples) / strategy_elapsed if strategy_elapsed > 0 else 0
                logger.info(f"Strategy {strategy_name} completed: {len(samples)} samples in {strategy_elapsed:.1f}s ({strategy_rate:.1f}/sec)")
            
            total_elapsed = time.time() - total_start_time
            total_rate = total_predictions / total_elapsed if total_elapsed > 0 else 0
            
            logger.info(f"Concurrent validation completed: {total_predictions} predictions in {total_elapsed:.1f}s ({total_rate:.1f}/sec)")
            
            # Finalize results files
            persistence.finalize_incremental_storage()
            
            # Generate final metrics from results CSV using the separate module
            metrics_start_time = time.time()
            metrics_result = calculate_metrics_from_runid(run_id, output_dir, 
                                                        self.model_id, 
                                                        self.prompt_template_file,
                                                        self.data_source,
                                                        self.command_line)
            metrics_elapsed = time.time() - metrics_start_time
            
            logger.info(f"Metrics calculation completed in {metrics_elapsed:.2f}s")
            
            # Return result summary
            result = {
                'run_id': run_id,
                'output_directory': str(runid_dir),
                'strategies_processed': valid_strategies,
                'total_samples': len(samples),
                'total_predictions': total_predictions,
                'processing_time': total_elapsed,
                'processing_rate': total_rate,
                'metrics': metrics_result.get('metrics', {}),
                'config': {
                    'data_source': data_source,
                    'sample_size': sample_size,
                    'max_workers': max_workers,
                    'batch_size': batch_size,
                    'model_id': self.model_id
                }
            }
            
            logger.info(f"Validation completed successfully. Results saved to: {runid_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

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
                        samples = load_dataset(DatasetType.UNIFIED, num_samples=sample_size, random_seed=42)
                        logger.info(f"Extracting {sample_size} samples from unified dataset")
                    else:
                        samples = load_dataset(DatasetType.UNIFIED)
                elif data_source.startswith("canned_"):
                    # Load specific canned file by name with optional sampling
                    if sample_size is not None:
                        samples = load_dataset_by_filename(data_source, num_samples=sample_size, random_seed=42)
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
            
            # Print runID to console for tracking
            print(f"Run ID: {run_id}")
            
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
                metrics_result = evaluator.calculate_metrics_from_runid(run_id, output_dir,
                                                                      self.model_id, 
                                                                      self.prompt_template_file,
                                                                      self.data_source,
                                                                      self.command_line)
                
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
    from pathlib import Path
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
        
        # Try to extract metadata from original log file for better context
        original_model = "Unknown (recalc)"
        original_template = "Unknown (recalc)" 
        original_data_source = "Unknown (recalc)"
        
        # Look for validation log to extract original metadata
        log_file = Path(run_folder) / f"validation_log_{run_id[4:]}.log"  # Remove 'run_' prefix
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    # Extract model info from log
                    if "Model:" in log_content:
                        for line in log_content.split('\n'):
                            if "Model:" in line:
                                original_model = line.split("Model:")[-1].strip()
                                break
                    # Extract other metadata if available
                    if "prompt_template_file" in log_content:
                        for line in log_content.split('\n'):
                            if "prompt_template_file" in line and "=" in line:
                                original_template = line.split("=")[-1].strip().strip("'\"")
                                break
            except Exception as e:
                logger.warning(f"Could not extract original metadata from log: {e}")
        
        # Use evaluation_metrics_calc for proper separation of concerns
        try:
            evaluator = EvaluationMetrics()
            metrics_result = evaluator.calculate_metrics_from_runid(run_id, output_dir,
                                                                  original_model, 
                                                                  original_template,
                                                                  original_data_source,
                                                                  f"Recalculation from run: {run_id}")
            
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

def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration with file output only (no console output).
    
    Args:
        debug (bool): Enable debug level logging
        log_file (Optional[str]): Path to log file for file output
    """
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
            level=logging.CRITICAL,  # Only critical errors
            handlers=[logging.NullHandler()],
            force=True
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
  
  # Use custom prompt template file
  python prompt_runner.py --prompt-template-file custom_prompts.json --strategies baseline policy
  
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
    parser.add_argument("--model", "-m", default="gpt-oss-120b", 
                       help="Model to use from YAML config (default: gpt-oss-120b)")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to YAML model configuration file (auto-detected if not specified)")
    
    # Data source and strategy selection
    parser.add_argument("--data-source", "-d", default="unified", 
                       help="Data source: 'unified' for full dataset, or specific canned file name (e.g., 'canned_basic_all', 'canned_100_all')")
    
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of samples to extract from the dataset (applies to all data sources)")
    
    parser.add_argument("--strategies", "-s", nargs="+", 
                       default=["baseline"],
                       help="Strategy(ies) to validate (default: baseline). Use 'all' to run all available strategies. Available strategies depend on the loaded template file.")
    
    parser.add_argument("--prompt-template-file", "-p", default="all_combined.json",
                       help="Prompt template file to load from prompt_templates folder (default: all_combined.json)")
    
    # Output and debugging
    parser.add_argument("--output-dir", "-o", default="outputs",
                       help="Directory for output files (default: outputs)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--metrics-only", action="store_true",
                       help="Calculate metrics from a previous run (requires --run-id)")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Run ID to recalculate metrics from (used with --metrics-only)")
    
    # Performance options
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential processing instead of concurrent (slower but more reliable)")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="Maximum number of concurrent threads (default: 5)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of samples to process in each batch (default: 10)")
    
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
    
    # Capture the command line for logging
    import sys
    command_line = " ".join(sys.argv)
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Create runner to validate strategies
        runner = PromptRunner(model_id=args.model, config_path=args.config, 
                            prompt_template_file=args.prompt_template_file)
        runner.set_execution_metadata(command_line, "strategy_validation")
        
        # Handle "all" strategies
        if "all" in args.strategies:
            args.strategies = runner.get_available_strategies()
        else:
            # Validate individual strategies
            invalid_strategies = runner.validate_strategies(args.strategies)
            if invalid_strategies:
                available_strategies = runner.get_available_strategies()
                logger.error(f"Invalid strategy names: {invalid_strategies}")
                logger.error(f"Available strategies in '{args.prompt_template_file}': {available_strategies}")
                sys.exit(1)
        
        logger.info(f"Starting prompt validation for model '{args.model}'")
        
        # Auto-detect config file if not specified
        config_path = args.config
        if config_path is None:
            # Look for model_connection.yaml in new connector package location and legacy locations
            current_dir = Path(".")
            prompt_eng_dir = Path("prompt_engineering")
            connector_dir = Path(__file__).parent / "connector"
            
            for potential_path in [connector_dir / "model_connection.yaml",
                                 current_dir / "model_connection.yaml", 
                                 prompt_eng_dir / "model_connection.yaml"]:
                if potential_path.exists():
                    config_path = str(potential_path)
                    logger.info(f"Auto-detected config file: {config_path}")
                    break
        
        # Handle connection testing
        if args.test_connection:
            logger.info("Testing connection to Azure AI endpoint")
            runner = PromptRunner(model_id=args.model, config_path=config_path, 
                                prompt_template_file=args.prompt_template_file)
            runner.set_execution_metadata(command_line, "connection_test")
            success = runner.test_connection()
            sys.exit(0 if success else 1)
        
        # Run validation based on arguments
        if args.metrics_only:
            if not args.run_id:
                logger.error("--run-id is required when using --metrics-only")
                sys.exit(1)
            
            logger.info(f"Recalculating metrics from run: {args.run_id}")
            runner = PromptRunner(model_id=args.model, config_path=config_path, 
                                prompt_template_file=args.prompt_template_file)
            runner.set_execution_metadata(command_line, f"Previous run: {args.run_id}")
            results = runner.calculate_metrics_only(run_id=args.run_id, output_dir=args.output_dir)
        else:
            logger.info(f"Running validation with strategies: {args.strategies}")
            if args.sample_size is not None:
                logger.info(f"Using sample size: {args.sample_size}")
            
            # Generate timestamp for logging and create early log file in runID folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"
            log_file = os.path.join(args.output_dir, run_id, f"validation_log_{timestamp}.log")
            
            # Reconfigure logging to include file output
            setup_logging(args.debug, log_file)
            logger = logging.getLogger(__name__)
            logger.info(f"Enhanced logging enabled - writing to: {log_file}")
            
            runner = PromptRunner(model_id=args.model, config_path=config_path, 
                                prompt_template_file=args.prompt_template_file)
            runner.set_execution_metadata(command_line, args.data_source)
            results = runner.run_validation(
                strategies=args.strategies,
                data_source=args.data_source,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                use_concurrent=not args.sequential,
                max_workers=args.max_workers,
                batch_size=args.batch_size
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
            
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
