"""
Core validation engine for testing GPT-OSS-20B connection and prompt strategies.
Refactored to use the new Azure AI connector wrapper.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Local imports
from azureai_mi_connector_wrapper import AzureAIConnector
from strategy_templates_loader import PromptStrategy, PromptTemplate, load_strategy_templates, format_prompt_with_context
from evaluation_metrics_calc import EvaluationMetrics, ValidationResult


def load_unified_dataset():
    """
    Load the unified dataset from processed files
    
    Returns:
        Dictionary containing train_data, val_data, test_data
    """
    import json
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "unified"
    
    data = {}
    
    # Load each split
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f"unified_{split}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data[f"{split}_data"] = json.load(f)
        else:
            data[f"{split}_data"] = []
    
    return data


def load_canned_dataset():
    """
    Load the canned dataset from data_samples directory
    
    Returns:
        Dictionary containing train_data, val_data, test_data
    """
    import json
    from pathlib import Path
    
    # Try to load canned_basic_all.json from data_samples directory
    data_samples_dir = Path(__file__).parent / "data_samples"
    canned_file = data_samples_dir / "canned_basic_all.json"
    
    data = {}
    
    if canned_file.exists():
        with open(canned_file, 'r', encoding='utf-8') as f:
            canned_data = json.load(f)
            # Split into train/val/test (simple approach: 70/15/15 split)
            total_samples = len(canned_data)
            train_end = int(0.7 * total_samples)
            val_end = int(0.85 * total_samples)
            
            data["train_data"] = canned_data[:train_end]
            data["val_data"] = canned_data[train_end:val_end]
            data["test_data"] = canned_data[val_end:]
    else:
        # Fallback to empty data
        data = {"train_data": [], "val_data": [], "test_data": []}
    
    return data


class PromptValidator:
    """
    Prompt validation engine using the Azure AI connector wrapper with YAML configuration.
    """
    
    def __init__(self, model_id: str = None, config_path: str = None, 
                 endpoint: str = None, key: str = None, model_name: str = None):
        """
        Initialize the prompt validator with YAML configuration support.
        
        Args:
            model_id: Model identifier from YAML config (e.g., 'gpt-oss-20b', 'gpt-5')
            config_path: Path to model_connection.yaml file
            endpoint: Azure AI endpoint URL (legacy compatibility - overrides YAML)
            key: Azure AI key (legacy compatibility - overrides YAML)
            model_name: Model deployment name (legacy compatibility - overrides YAML)
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Check for legacy usage (backward compatibility)
        if endpoint is not None or key is not None or model_name is not None:
            self.logger.warning("Using legacy parameters. Consider migrating to YAML configuration.")
            # Create legacy connector (if this path is still needed for backward compatibility)
            # For now, we'll use the new system but log a warning
            if model_id is None:
                model_id = "gpt-oss-20b"  # Default fallback
        
        # Initialize Azure AI connector with YAML configuration
        self.connector = AzureAIConnector(model_id=model_id, config_path=config_path)
        
        # Store model information
        self.model_info = self.connector.get_model_info()
        
        # Legacy compatibility properties
        self.endpoint = self.connector.endpoint
        self.key = self.connector.key
        self.deployment_name = self.connector.model_name
        self.client = self.connector.client
        self.is_connected = self.connector.is_connected
        
        self.logger.info(f"PromptValidator initialized for model '{self.model_info['model_id']}' ({self.model_info['model_name']})")
    
    def validate_connection(self):
        """
        Test connection to GPT-OSS-20B using the connector.
        
        Returns:
            bool: True if connection successful
        """
        success = self.connector.test_connection()
        self.is_connected = success
        return success
    
    def test_single_strategy(self, strategy_name: str, text: str, true_label: str = None) -> ValidationResult:
        """
        Test a single prompt strategy with GPT-OSS-20B.
        
        Args:
            strategy_name: Name of strategy to test
            text: Input text
            true_label: Optional true label for accuracy calculation
            
        Returns:
            ValidationResult: Test results with prediction and metrics
        """
        if not self.client:
            self.logger.error("Client not initialized")
            return ValidationResult(
                strategy_name=strategy_name,
                input_text=text,
                predicted_label=None,
                true_label=true_label,
                response_text="Error: Client not initialized",
                response_time=0.0,
                metrics={"error": "No client"}
            )
        
        try:
            templates = load_strategy_templates()
            if strategy_name not in templates:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            strategy = templates[strategy_name]
            
            # Format prompts
            system_prompt = strategy.template.system_prompt
            user_prompt = strategy.template.user_template.format(text=text)
            
            # Make API call with parameters from strategy template
            start_time = time.time()
            
            # Get all parameters from strategy template with defaults
            strategy_params = strategy.parameters or {}
            api_params = {
                "max_tokens": strategy_params.get("max_tokens", 512),
                "temperature": strategy_params.get("temperature", 0.1),
                "top_p": strategy_params.get("top_p", 1.0),
                "frequency_penalty": strategy_params.get("frequency_penalty", 0.0),
                "presence_penalty": strategy_params.get("presence_penalty", 0.0)
            }
            
            # Try to add response_format if specified in strategy parameters
            response_format = strategy_params.get("response_format")
            if response_format:
                api_params["response_format"] = response_format
                self.logger.info(f"Attempting to use response_format: {response_format}")
            
            self.logger.info(f"Using strategy parameters: {api_params}")
            
            # Use the connector's simple_completion method
            response_content = self.connector.simple_completion(
                system_message=system_prompt,
                user_message=user_prompt,
                **api_params
            )
            response_time = time.time() - start_time
            
            if response_content:
                response_text = response_content.strip()
                
                # Parse prediction and rationale
                predicted_label, rationale = self._parse_prediction(response_text)
                
                # Calculate basic metrics
                metrics = {}
                if true_label:
                    metrics["correct"] = predicted_label == true_label
                metrics["response_time"] = response_time
                
                return ValidationResult(
                    strategy_name=strategy_name,
                    input_text=text,
                    predicted_label=predicted_label,
                    true_label=true_label,
                    response_text=response_text,
                    response_time=response_time,
                    metrics=metrics,
                    rationale=rationale  # Added rationale
                )
            else:
                return ValidationResult(
                    strategy_name=strategy_name,
                    input_text=text,
                    predicted_label=None,
                    true_label=true_label,
                    response_text="No response received",
                    response_time=response_time,
                    metrics={"error": "No response"},
                    rationale="No response received from model"
                )
                
        except Exception as e:
            self.logger.error(f"Error testing strategy {strategy_name}: {e}")
            return ValidationResult(
                strategy_name=strategy_name,
                input_text=text,
                predicted_label=None,
                true_label=true_label,
                response_text=f"Error: {str(e)}",
                response_time=0.0,
                metrics={"error": str(e)},
                rationale=f"Error occurred: {str(e)}"
            )
    
    def batch_validate(self, texts: List[str], strategies: List[str] = None, true_labels: List[str] = None) -> Dict:
        """
        Batch validation across multiple texts and strategies.
        
        Args:
            texts: List of texts to validate
            strategies: List of strategies to test (default: all strategies)
            true_labels: Optional list of true labels
            
        Returns:
            Dict: Comprehensive validation results
        """
        strategies = strategies or ["baseline", "policy", "persona", "combined"]
        true_labels = true_labels or [None] * len(texts)
        
        self.logger.info(f"Running batch validation: {len(texts)} texts, {len(strategies)} strategies")
        
        results = []
        strategy_summaries = {}
        
        for strategy in strategies:
            strategy_results = []
            
            for i, text in enumerate(texts):
                true_label = true_labels[i] if i < len(true_labels) else None
                result = self.test_single_strategy(strategy, text, true_label)
                results.append(result)
                strategy_results.append(result)
                
                # Brief pause between API calls
                time.sleep(0.5)
            
            # Calculate strategy summary
            successful = [r for r in strategy_results if r.predicted_label is not None]
            with_labels = [r for r in successful if r.true_label is not None]
            correct = [r for r in with_labels if r.metrics.get("correct", False)]
            
            strategy_summaries[strategy] = {
                "total_tests": len(strategy_results),
                "successful": len(successful),
                "success_rate": len(successful) / len(strategy_results) if strategy_results else 0,
                "accuracy": len(correct) / len(with_labels) if with_labels else 0,
                "avg_response_time": sum(r.response_time for r in successful) / len(successful) if successful else 0
            }
        
        return {
            "results": results,
            "strategy_summaries": strategy_summaries,
            "total_tests": len(results),
            "strategies_tested": strategies
        }
    
    def validate_strategies(self, strategies: List[str] = None, data_source: str = "unified", 
                          output_dir: str = "outputs", sample_size: int = 25) -> Dict:
        """
        Validate specified prompt strategies using test dataset.
        
        Args:
            strategies: List of strategy names to test (default: all)
            data_source: Data source ('unified' or 'canned')
            output_dir: Output directory for results
            sample_size: Number of samples to test from dataset
            
        Returns:
            Dict: Comprehensive validation results
        """
        try:
            # Load test data based on source
            if data_source == "unified":
                data = load_unified_dataset()
                test_data = data.get('test_data', [])
            else:  # canned
                data = load_canned_dataset()
                test_data = data.get('test_data', [])
            
            if not test_data:
                self.logger.warning("No test data available, using sample size of 5")
                sample_size = min(sample_size, 5)
                texts = [f"Sample text {i}" for i in range(sample_size)]
                labels = ["unknown"] * sample_size
            else:
                # Sample from test data
                import random
                random.seed(42)  # For reproducibility
                samples = random.sample(test_data, min(sample_size, len(test_data)))
                texts = [sample["text"] for sample in samples]
                labels = [sample.get("label_binary", "unknown") for sample in samples]
            
            self.logger.info(f"Validating strategies {strategies} with {len(texts)} samples")
            results = self.batch_validate(texts, strategies=strategies, true_labels=labels)
            
            # Generate output files if validation was successful
            if results and 'results' in results:
                try:
                    from persistence_helper import PersistenceHelper
                    import datetime
                    
                    # Generate timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Initialize persistence helper with output directory
                    persistence = PersistenceHelper(output_dir)
                    
                    # Convert ValidationResult objects to dictionaries for persistence
                    detailed_results = []
                    for result in results['results']:
                        detailed_results.append({
                            'strategy': result.strategy_name,
                            'input_text': result.input_text,
                            'true_label': result.true_label,
                            'predicted_label': result.predicted_label,
                            'response_time': result.response_time,
                            'rationale': result.rationale,
                            'target_group_norm': getattr(result, 'target_group_norm', 'unknown'),
                            'persona_tag': getattr(result, 'persona_tag', 'unknown'),
                            'source_dataset': data_source
                        })
                    
                    # Group ValidationResult objects (not dictionaries) by strategy for metrics calculation
                    strategy_results = {}
                    for strategy in strategies:
                        strategy_results[strategy] = [r for r in results['results'] if r.strategy_name == strategy]
                    
                    # Create samples list from the test data
                    samples = []
                    for i, text in enumerate(texts):
                        samples.append({
                            'text': text,
                            'label_binary': labels[i] if i < len(labels) else 'unknown',
                            'target_group_norm': 'unknown',  # This could be enhanced if available
                            'persona_tag': 'unknown'  # This could be enhanced if available
                        })
                    
                    # Save comprehensive results (CSV files and evaluation report)
                    output_paths = persistence.calculate_and_save_comprehensive_results(
                        strategy_results, detailed_results, samples, timestamp
                    )
                    
                    # Add output file information to results
                    results['output_files'] = output_paths
                    results['timestamp'] = timestamp
                    
                    self.logger.info(f"Results saved to {len(output_paths)} files in {output_dir}/")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save output files: {e}")
                    # Continue execution even if file saving fails
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in strategy validation: {e}")
            return {"error": str(e)}
    
    def calculate_metrics_only(self, data_source: str = "unified") -> Dict:
        """
        Calculate dataset metrics without running model validation.
        
        Args:
            data_source: Data source ('unified' or 'canned')
            
        Returns:
            Dict: Dataset metrics and summary
        """
        try:
            # Load data based on source
            if data_source == "unified":
                data = load_unified_dataset()
            else:  # canned
                data = load_canned_dataset()
            
            # Get test data for metrics
            test_data = data.get('test_data', [])
            
            if not test_data:
                return {
                    "error": "No test data available for metrics calculation",
                    "summary": {
                        "total_samples": 0,
                        "hate_samples": 0,
                        "normal_samples": 0,
                        "data_source": data_source
                    }
                }
            
            # Calculate basic metrics
            labels = [item.get('label_binary', 'unknown') for item in test_data]
            hate_count = sum(1 for label in labels if label == 'hate')
            normal_count = sum(1 for label in labels if label == 'normal')
            
            metrics = {
                "dataset_info": {
                    "source": data_source,
                    "total_samples": len(test_data),
                    "hate_samples": hate_count,
                    "normal_samples": normal_count,
                    "hate_percentage": round((hate_count / len(test_data)) * 100, 2) if len(test_data) > 0 else 0,
                    "normal_percentage": round((normal_count / len(test_data)) * 100, 2) if len(test_data) > 0 else 0
                },
                "sample_texts": test_data[:3],  # Show first 3 samples
                "summary": {
                    "total_samples": len(test_data),
                    "hate_samples": hate_count,
                    "normal_samples": normal_count,
                    "data_source": data_source
                }
            }
            
            self.logger.info(f"Calculated metrics for {len(test_data)} samples from {data_source} dataset")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e), "summary": {"total_samples": 0}}
    
    def _parse_prediction(self, response_text: str) -> Tuple[str, str]:
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
                json_match = re.search(r'\{[^}]*"classification"[^}]*\}', cleaned_text)
                if json_match:
                    cleaned_text = json_match.group(0)
                else:
                    # If no JSON found, fall back to text parsing
                    prediction = self._parse_text_prediction(cleaned_text)
                    return prediction, "Fallback text parsing used"
            
            # Parse JSON
            parsed_response = json.loads(cleaned_text)
            classification = parsed_response.get('classification', '').lower().strip()
            rationale = parsed_response.get('rationale', 'No rationale provided').strip()
            
            # Normalize to expected labels
            normalized_prediction = self._normalize_prediction(classification)
            return normalized_prediction, rationale
            
        except (json.JSONDecodeError, KeyError, AttributeError):
            # Fall back to text parsing if JSON parsing fails
            prediction = self._parse_text_prediction(response_text)
            return prediction, "Fallback due to JSON parsing error"
    
    def _normalize_prediction(self, prediction: str) -> str:
        """
        Normalize model predictions to binary labels.
        
        Args:
            prediction: Raw prediction string
            
        Returns:
            str: Standardized prediction ('hate' or 'normal')
        """
        if not prediction:
            return "normal"
        
        prediction = prediction.lower().strip()
        
        # Map variations to standard labels
        hate_variants = ['hate', 'hate_speech', 'hateful', 'toxic', '1']
        normal_variants = ['normal', 'not_hate', 'non_hate', 'no_hate', 'clean', '0', 'not_hateful']
        
        if prediction in hate_variants or any(variant in prediction for variant in hate_variants):
            return "hate"
        elif prediction in normal_variants or any(variant in prediction for variant in normal_variants):
            return "normal"
        else:
            # Default to normal for unclear cases
            return "normal"
    
    def _parse_text_prediction(self, response_text: str) -> str:
        """
        Parse non-JSON response text for prediction.
        
        Args:
            response_text: Raw model response text
            
        Returns:
            str: Parsed prediction ('hate' or 'normal')
        """
        if not response_text:
            return "normal"
        
        response_lower = response_text.lower()
        
        # Look for explicit labels
        if 'hate' in response_lower and ('not hate' not in response_lower and 'non-hate' not in response_lower):
            return "hate"
        elif 'normal' in response_lower or 'not hate' in response_lower or 'non-hate' in response_lower:
            return "normal"
        else:
            return "normal"  # Default to normal


def main():
    """
    Main entry point for prompt validation with flexible configuration.
    
    Environment Variables:
        AZURE_AI_ENDPOINT: Azure AI endpoint (if YAML not used)
        AZURE_AI_KEY: Azure AI key (if YAML not used)
        MODEL_CONNECTION_PATH: Path to model_connection.yaml (optional)
    """
    parser = argparse.ArgumentParser(description="Validate prompts using Azure AI")
    parser.add_argument("--model", "-m", default="gpt-oss-20b", 
                      help="Model ID from YAML config (default: gpt-oss-20b)")
    parser.add_argument("--config", "-c", default=None,
                      help="Path to model_connection.yaml (default: auto-detect)")
    parser.add_argument("--data-source", "-d", default="unified", 
                      choices=["unified", "canned"],
                      help="Data source to use (default: unified)")
    parser.add_argument("--strategies", "-s", nargs="+", 
                      default=["policy", "persona", "combined", "baseline"],
                      choices=["policy", "persona", "combined", "baseline"],
                      help="Prompt strategies to validate (default: all)")
    parser.add_argument("--output-dir", "-o", default="outputs",
                      help="Output directory for results (default: outputs)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument("--metrics-only", action="store_true",
                      help="Generate metrics without running prompts")
    
    # Legacy compatibility arguments (deprecated)
    parser.add_argument("--endpoint", help="Azure AI endpoint (deprecated - use YAML config)")
    parser.add_argument("--key", help="Azure AI key (deprecated - use YAML config)")
    parser.add_argument("--model-name", help="Model deployment name (deprecated - use YAML config)")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting prompt validation for model '{args.model}'")
        
        # Auto-detect config path if not provided
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
        
        # Warn about legacy arguments
        if args.endpoint or args.key or args.model_name:
            logger.warning("Legacy arguments detected. Consider migrating to YAML configuration.")
        
        # Initialize validator with new interface
        validator = PromptValidator(
            model_id=args.model,
            config_path=config_path,
            endpoint=args.endpoint,
            key=args.key,
            model_name=args.model_name
        )
        
        # Run validation based on arguments
        if args.metrics_only:
            logger.info("Running metrics calculation only")
            results = validator.calculate_metrics_only(data_source=args.data_source)
        else:
            logger.info(f"Running validation with strategies: {args.strategies}")
            results = validator.validate_strategies(
                strategies=args.strategies,
                data_source=args.data_source,
                output_dir=args.output_dir
            )
        
        # Log summary
        if isinstance(results, dict) and 'summary' in results:
            logger.info("Validation completed successfully")
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