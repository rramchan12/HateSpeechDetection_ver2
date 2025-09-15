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
            
            # Make API call
            start_time = time.time()
            response = self.client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt)
                ],
                max_tokens=200,
                temperature=strategy.parameters.get("temperature", 0.1),
                model=self.deployment_name
            )
            response_time = time.time() - start_time
            
            if response and response.choices:
                response_content = response.choices[0].message.content
                response_text = response_content.strip() if response_content else "Empty response"
                
                # Parse prediction
                predicted_label = self._parse_prediction(response_text)
                
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
                    metrics=metrics
                )
            else:
                return ValidationResult(
                    strategy_name=strategy_name,
                    input_text=text,
                    predicted_label=None,
                    true_label=true_label,
                    response_text="No response received",
                    response_time=response_time,
                    metrics={"error": "No response"}
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
                metrics={"error": str(e)}
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
    
    def _parse_prediction(self, response_text: str) -> str:
        """
        Parse model response to extract prediction.
        
        Args:
            response_text: Raw model response
            
        Returns:
            str: Parsed prediction ('hate', 'not_hate', or 'unknown')
        """
        if not response_text or response_text.strip() == "":
            return "unknown"
        
        response_lower = response_text.lower()
        if "hate" in response_lower and "not hate" not in response_lower and "normal" not in response_lower:
            return "hate"
        elif "normal" in response_lower or "not hate" not in response_lower:
            return "not_hate"
        else:
            return "unknown"