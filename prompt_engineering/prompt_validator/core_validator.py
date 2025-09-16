"""
Core validation engine for testing GPT-OSS-20B connection and prompt strategies.
Simplified version focusing on connection testing with scaffolding for future features.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
import time

# Azure AI Inference SDK
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Local imports
from strategy_templates import PromptStrategy, PromptTemplate, create_strategy_templates, format_prompt_with_context
from evaluation_metrics import EvaluationMetrics, ValidationResult


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


class PromptValidator:
    """
    Connection-focused validator for GPT-OSS-20B with scaffolding for prompt strategies.
    """
    
    def __init__(self, endpoint: str = None, key: str = None, deployment_name: str = "gpt-oss-120b"):
        """
        Initialize the prompt validator.
        
        Args:
            endpoint: Azure AI endpoint URL
            key: Azure AI key
            deployment_name: Model deployment name
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Connection settings
        self.endpoint = endpoint or os.getenv('AZURE_AI_ENDPOINT')
        self.key = key or os.getenv('AZURE_AI_KEY')
        self.deployment_name = deployment_name
        
        # Validation state
        self.is_connected = False
        self.client = None
        
        # Initialize client
        if self.endpoint and self.key:
            self._initialize_client()
        else:
            self.logger.error("Missing Azure AI credentials. Set AZURE_AI_ENDPOINT and AZURE_AI_KEY environment variables.")
    
    def _initialize_client(self):
        """Initialize Azure AI client"""
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            self.logger.info(f"PromptValidator initialized with endpoint: {self.endpoint}")
        except Exception as e:
            self.logger.error(f"Failed to initialize client: {e}")
    
    def validate_connection(self):
        """
        Test connection to GPT-OSS-20B.
        
        Returns:
            bool: True if connection successful
        """
        if not self.client:
            self.logger.error("Client not initialized")
            return False
        
        self.logger.info("Validating connection to GPT-OSS-20B...")
        
        try:
            # Test connection
            test_response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Respond with 'Connection successful' only.")
                ],
                max_tokens=10,
                temperature=0.0,
                model=self.deployment_name
            )
            
            if test_response and test_response.choices:
                response_content = test_response.choices[0].message.content
                response_text = response_content.strip() if response_content else "Empty response"
                self.logger.info(f"Connection validated. Response: {response_text}")
                self.is_connected = True
                return True
            else:
                self.logger.error("Connection failed: No response received")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
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
            templates = create_strategy_templates()
            if strategy_name not in templates:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            strategy = templates[strategy_name]
            
            # Format prompts
            system_prompt = strategy.template.system_prompt
            user_prompt = strategy.template.user_template.format(text=text)
            
            # Make API call with optional response format from strategy parameters
            start_time = time.time()
            
            # Prepare API call parameters
            api_params = {
                "messages": [
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt)
                ],
                "max_tokens": 300,  # Increased for rationale field
                "temperature": strategy.parameters.get("temperature", 0.1),
                "model": self.deployment_name
            }
            
            # Try to add response_format if specified in strategy parameters
            response_format = strategy.parameters.get("response_format")
            if response_format:
                try:
                    # Try with response_format first
                    api_params["response_format"] = response_format
                    self.logger.info(f"Attempting to use response_format: {response_format}")
                    response = self.client.complete(**api_params)
                except Exception as e:
                    # If response_format fails, retry without it
                    self.logger.warning(f"response_format not supported ({e}), falling back to prompt engineering")
                    api_params.pop("response_format", None)
                    response = self.client.complete(**api_params)
            else:
                # No response_format specified, use standard call
                response = self.client.complete(**api_params)
            response_time = time.time() - start_time
            
            if response and response.choices:
                response_content = response.choices[0].message.content
                response_text = response_content.strip() if response_content else "Empty response"
                
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
    
    def validate_strategies(self, sample_size: int = 25) -> Dict:
        """
        Validate all prompt strategies using test dataset.
        
        Args:
            sample_size: Number of samples to test from dataset
            
        Returns:
            Dict: Comprehensive validation results
        """
        try:
            # Load test data
            data = load_unified_dataset()
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
            
            self.logger.info(f"Validating strategies with {len(texts)} samples")
            return self.batch_validate(texts, true_labels=labels)
            
        except Exception as e:
            self.logger.error(f"Error in strategy validation: {e}")
            return {"error": str(e)}
    
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