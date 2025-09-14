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
    
    # ============================================================================
    # SCAFFOLDING: Future prompt strategy validation methods
    # ============================================================================
    
    def validate_strategies(self, sample_size: int = 25) -> Dict:
        """
        SCAFFOLDING: Validate all prompt strategies.
        
        Args:
            sample_size: Number of samples to test
            
        Returns:
            Dict: Validation results (placeholder)
        """
        # TODO: Implement strategy validation logic
        self.logger.info(f"SCAFFOLDING: Strategy validation for {sample_size} samples")
        return {"status": "scaffolding", "message": "Strategy validation not implemented yet"}
    
    def test_single_strategy(self, strategy_name: str, text: str) -> Dict:
        """
        SCAFFOLDING: Test a single prompt strategy.
        
        Args:
            strategy_name: Name of strategy to test
            text: Input text
            
        Returns:
            Dict: Test results (placeholder)
        """
        # TODO: Implement single strategy testing
        self.logger.info(f"SCAFFOLDING: Testing {strategy_name} strategy")
        return {"status": "scaffolding", "strategy": strategy_name, "text": text}
    
    def batch_validate(self, texts: List[str], strategies: List[str] = None) -> Dict:
        """
        SCAFFOLDING: Batch validation across multiple texts and strategies.
        
        Args:
            texts: List of texts to validate
            strategies: List of strategies to test
            
        Returns:
            Dict: Batch results (placeholder)
        """
        # TODO: Implement batch validation
        strategies = strategies or ["policy", "persona", "combined", "baseline"]
        self.logger.info(f"SCAFFOLDING: Batch validation for {len(texts)} texts, {len(strategies)} strategies")
        return {"status": "scaffolding", "text_count": len(texts), "strategy_count": len(strategies)}