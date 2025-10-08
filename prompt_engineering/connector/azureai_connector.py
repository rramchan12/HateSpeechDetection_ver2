"""
Azure AI Inference SDK Connector Wrapper

This module provides a clean wrapper around the Azure AI Inference SDK
for connecting to multiple AI models (GPT-OSS-20B, GPT-5, etc.) using 
configuration from model_connection.yaml.

Features:
- YAML-based model configuration
- Multiple model support (GPT-OSS-20B, GPT-5)
- Environment variable substitution
- JSON response format support
- Error handling and logging
- Flexible message handling
"""

import os
import logging
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import HttpLoggingPolicy


class ModelConfigLoader:
    """Helper class to load and process model configuration from YAML"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader with path to YAML file"""
        self.logger = logging.getLogger(__name__)
        if config_path is None:
            # Default to model_connection.yaml in same directory as this script
            config_path = Path(__file__).parent / "model_connection.yaml"
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution"""
        if self._config is not None:
            return self._config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Substitute environment variables (${VAR_NAME} format)
            config_content = self._substitute_env_vars(config_content)
            
            # Parse YAML
            self._config = yaml.safe_load(config_content)
            self.logger.info(f"Loaded model configuration from {self.config_path}")
            return self._config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        def replace_env_var(match):
            var_name = match.group(1)
            var_value = os.environ.get(var_name)
            if var_value is None:
                self.logger.warning(f"Environment variable '{var_name}' not found, keeping placeholder")
                return match.group(0)  # Return original ${VAR_NAME} if not found
            return var_value
        
        # Replace ${VAR_NAME} with environment variable values
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        config = self.load_config()
        
        if model_id not in config.get('models', {}):
            available_models = list(config.get('models', {}).keys())
            raise ValueError(f"Model '{model_id}' not found in configuration. Available: {available_models}")
        
        return config['models'][model_id]
    
    def get_default_model_id(self) -> str:
        """Get the default model ID from configuration"""
        config = self.load_config()
        return config.get('default_model', 'gpt-oss-20b')
    
    def list_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        config = self.load_config()
        return list(config.get('models', {}).keys())


class AzureAIConnector:
    """
    Wrapper class for Azure AI Inference SDK connections with YAML configuration support.
    
    Provides a simplified interface for connecting to multiple Azure AI models
    with configuration loaded from model_connection.yaml file.
    """
    
    def __init__(self, 
                 model_id: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Azure AI connector with YAML configuration.
        
        Args:
            model_id: Model identifier (e.g., 'gpt-oss-20b', 'gpt-5'). 
                     If None, uses default from config.
            config_path: Path to model_connection.yaml file. If None, 
                        uses default location.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_loader = ModelConfigLoader(config_path)
        
        # Determine model to use
        if model_id is None:
            model_id = self.config_loader.get_default_model_id()
        
        self.model_id = model_id
        self.model_config = self.config_loader.get_model_config(model_id)
        
        # Extract connection details
        self.endpoint = self.model_config.get('endpoint')
        self.key = self.model_config.get('api_key')
        self.model_name = self.model_config.get('model_deployment')
        self.default_parameters = self.model_config.get('default_parameters', {})
        
        # Connection state
        self.client = None
        self.is_connected = False
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Azure AI client."""
        if not self.endpoint or not self.key:
            self.logger.error(f"Missing Azure AI credentials for model '{self.model_id}':")
            self.logger.error(f"- endpoint: {'✓' if self.endpoint else '✗ (check environment variables in config)'}")
            self.logger.error(f"- api_key: {'✓' if self.key else '✗ (check environment variables in config)'}")
            self.logger.error("Please ensure environment variables referenced in model_connection.yaml are set")
            return
        
        try:
            # Configure logging policy to show rate limit headers
            logging_policy = HttpLoggingPolicy()
            # Allow rate limiting and token usage headers to be shown in logs
            logging_policy.allowed_header_names.update([
                'x-ratelimit-remaining-requests',
                'x-ratelimit-remaining-tokens', 
                'x-ratelimit-limit-requests',
                'x-ratelimit-limit-tokens',
                'prompt_token_len',
                'sampling_token_len',
                'apim-request-id',
                'x-ms-region'
            ])
            
            # Get connection settings from config
            connection_config = self.config_loader.load_config().get('connection', {})
            timeout = connection_config.get('timeout', 30)
            
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
                logging_policy=logging_policy,
                connection_timeout=timeout,
                read_timeout=timeout
            )
            self.logger.info(f"Azure AI client initialized for '{self.model_id}' with endpoint: {self.endpoint}")
            self.logger.debug("Configured logging policy to show rate limit headers")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure AI client for '{self.model_id}': {e}")
    
    def test_connection(self) -> bool:
        """
        Test connection to the Azure AI endpoint.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.client:
            self.logger.error("Client not initialized")
            return False
        
        try:
            self.logger.info("Testing Azure AI connection...")
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content="Respond with 'Connection successful' only.")
                ],
                max_tokens=10,
                temperature=0.0,
                model=self.model_name
            )
            
            if response and response.choices:
                self.is_connected = True
                self.logger.info("Connection test successful")
                return True
            else:
                self.logger.error("Connection test failed: No response")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test error: {e}")
            return False
    
    def complete(self, 
                 messages: List[Union[SystemMessage, UserMessage]],
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 response_format: Optional[str] = None,
                 **kwargs) -> Optional[Any]:
        """
        Make a completion request to the Azure AI model.
        
        Args:
            messages: List of SystemMessage and UserMessage objects
            max_tokens: Maximum tokens in response (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            frequency_penalty: Frequency penalty (uses config default if None)
            presence_penalty: Presence penalty (uses config default if None)
            response_format: Response format (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            API response object or None if error
        """
        if not self.client:
            self.logger.error("Client not initialized")
            return None
        
        try:
            # Use configuration defaults if parameters not provided
            defaults = self.default_parameters
            
            # Build request parameters with config defaults
            params = {
                "messages": messages,
                "max_tokens": max_tokens if max_tokens is not None else defaults.get("max_tokens", 2048),
                "temperature": temperature if temperature is not None else defaults.get("temperature", 1.0),
                "top_p": top_p if top_p is not None else defaults.get("top_p", 1.0),
                "frequency_penalty": frequency_penalty if frequency_penalty is not None else defaults.get("frequency_penalty", 0.0),
                "presence_penalty": presence_penalty if presence_penalty is not None else defaults.get("presence_penalty", 0.0),
                "model": self.model_name,
                **kwargs
            }
            
            # Add response_format if specified or use default
            final_response_format = response_format if response_format is not None else defaults.get("response_format")
            if final_response_format:
                params["response_format"] = final_response_format
            
            # Make API call
            response = self.client.complete(**params)
            return response
            
        except Exception as e:
            self.logger.error(f"API completion error: {e}")
            return None
    
    def simple_completion(self, 
                         system_message: str,
                         user_message: str,
                         response_format: Optional[str] = None,
                         **kwargs) -> Optional[str]:
        """
        Simple completion with string messages (convenience method).
        
        Args:
            system_message: System message content
            user_message: User message content
            response_format: Optional response format
            **kwargs: Additional parameters
            
        Returns:
            Response text or None if error
        """
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=user_message)
        ]
        
        response = self.complete(messages=messages, response_format=response_format, **kwargs)
        
        if response and response.choices:
            return response.choices[0].message.content
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model configuration.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_config.get("name", "Unknown"),
            "endpoint": self.endpoint,
            "model_deployment": self.model_name,
            "provider": self.model_config.get("provider", "Unknown"),
            "description": self.model_config.get("description", "No description"),
            "default_parameters": self.default_parameters,
            "is_connected": self.is_connected,
            "client_initialized": self.client is not None
        }
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get basic connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "model_deployment": self.model_name,
            "is_connected": self.is_connected,
            "client_initialized": self.client is not None
        }
    
    def list_available_models(self) -> List[str]:
        """
        Get list of available models from configuration.
        
        Returns:
            List of available model IDs
        """
        return self.config_loader.list_available_models()
    
    def switch_model(self, model_id: str) -> bool:
        """
        Switch to a different model configuration.
        
        Args:
            model_id: Model identifier to switch to
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            # Load new model configuration
            new_config = self.config_loader.get_model_config(model_id)
            
            # Update configuration
            self.model_id = model_id
            self.model_config = new_config
            self.endpoint = new_config.get('endpoint')
            self.key = new_config.get('api_key')
            self.model_name = new_config.get('model_deployment')
            self.default_parameters = new_config.get('default_parameters', {})
            
            # Reset connection state
            self.client = None
            self.is_connected = False
            
            # Reinitialize client
            self._initialize_client()
            
            self.logger.info(f"Switched to model: {model_id}")
            return self.client is not None
            
        except Exception as e:
            self.logger.error(f"Failed to switch to model '{model_id}': {e}")
            return False


def create_connector(model_id: str = None, 
                    config_path: str = None) -> AzureAIConnector:
    """
    Factory function to create an Azure AI connector with YAML configuration.
    
    Args:
        model_id: Model identifier (e.g., 'gpt-oss-20b', 'gpt-5')
        config_path: Path to model_connection.yaml file
        
    Returns:
        AzureAIConnector instance
    """
    return AzureAIConnector(model_id=model_id, config_path=config_path)


def test_connection_simple() -> bool:
    """
    Simple connection test using environment variables.
    
    Returns:
        bool: True if connection successful
    """
    connector = create_connector()
    return connector.test_connection()


# Example usage and testing
if __name__ == "__main__":
    """
    Test the Azure AI connector wrapper with YAML configuration.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Azure AI Connector with YAML Configuration ===")
    
    # Test 1: Create connector with default model (from config)
    print("\n1. Creating connector with default model...")
    connector = create_connector()
    model_info = connector.get_model_info()
    print(f"   Model: {model_info['model_id']} ({model_info['model_name']})")
    print(f"   Description: {model_info['description']}")
    print(f"   Default parameters: {model_info['default_parameters']}")
    
    # Test 2: List available models
    print("\n2. Available models in configuration:")
    available_models = connector.list_available_models()
    for model_id in available_models:
        print(f"   - {model_id}")
    
    # Test 3: Test connection
    print("\n3. Testing connection...")
    if connector.test_connection():
        print("   ✅ Connection successful!")
        
        # Test simple completion with config defaults
        print("\n4. Testing completion with config defaults...")
        response = connector.simple_completion(
            system_message="You are a helpful assistant.",
            user_message="Respond with just 'Hello from {model_name}' where {model_name} is your model name.",
        )
        
        if response:
            print(f"   ✅ Response: {response}")
        else:
            print("   ❌ No response received")
            
        # Test 5: Switch model (if GPT-5 is available)
        if "gpt-5" in available_models:
            print(f"\n5. Switching to GPT-5...")
            if connector.switch_model("gpt-5"):
                print("   ✅ Successfully switched to GPT-5")
                new_info = connector.get_model_info()
                print(f"   New model: {new_info['model_id']} ({new_info['model_name']})")
            else:
                print("   ❌ Failed to switch to GPT-5")
        else:
            print(f"\n5. GPT-5 not configured, skipping model switch test")
            
    else:
        print("   ❌ Connection failed!")
        print("   Check your environment variables and configuration")
    
    print("\n=== Test Complete ===")