"""
Azure AI Inference SDK Connector Wrapper

This module provides a clean wrapper around the Azure AI Inference SDK
for connecting to GPT-OSS-120B and other Azure AI models.

Features:
- Simple connection management
- Environment variable configuration
- JSON response format support
- Error handling and logging
- Flexible message handling
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


class AzureAIConnector:
    """
    Wrapper class for Azure AI Inference SDK connections.
    
    Provides a simplified interface for connecting to Azure AI models
    with support for environment variable configuration and JSON responses.
    """
    
    def __init__(self, 
                 endpoint: Optional[str] = None, 
                 key: Optional[str] = None, 
                 model_name: str = "gpt-oss-120b"):
        """

        
        Initialize the Azure AI connector.
        
        Args:
            endpoint: Azure AI endpoint URL (defaults to AZURE_INFERENCE_SDK_ENDPOINT env variable)
            key: Azure AI API key (defaults to AZURE_INFERENCE_SDK_KEY env variable)
            model_name: Model deployment name
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration - Use environment variables if not provided
        self.endpoint = endpoint or os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
        self.key = key or os.getenv("AZURE_INFERENCE_SDK_KEY")
        self.model_name = model_name
        
        # Connection state
        self.client = None
        self.is_connected = False
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Azure AI client."""
        if not self.endpoint or not self.key:
            self.logger.error("Missing Azure AI credentials. Please set environment variables:")
            self.logger.error("- AZURE_INFERENCE_SDK_ENDPOINT")
            self.logger.error("- AZURE_INFERENCE_SDK_KEY")
            return
        
        try:
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            self.logger.info(f"Azure AI client initialized with endpoint: {self.endpoint}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure AI client: {e}")
    
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
                 max_tokens: int = 2048,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 response_format: Optional[str] = None,
                 **kwargs) -> Optional[Any]:
        """
        Make a completion request to the Azure AI model.
        
        Args:
            messages: List of SystemMessage and UserMessage objects
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            response_format: Response format ("json_object" or None)
            **kwargs: Additional parameters
            
        Returns:
            API response object or None if error
        """
        if not self.client:
            self.logger.error("Client not initialized")
            return None
        
        try:
            # Build request parameters
            params = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "model": self.model_name,
                **kwargs
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
            
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
        Simple completion with string messages.
        
        Args:
            system_message: System message content
            user_message: User message content
            response_format: Response format ("json_object" or None)
            **kwargs: Additional parameters for complete()
            
        Returns:
            Response content string or None if error
        """
        messages = [
            SystemMessage(content=system_message),
            UserMessage(content=user_message)
        ]
        
        response = self.complete(
            messages=messages,
            response_format=response_format,
            **kwargs
        )
        
        if response and response.choices:
            return response.choices[0].message.content
        return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "is_connected": self.is_connected,
            "client_initialized": self.client is not None
        }


def create_connector(endpoint: str = None, 
                    key: str = None, 
                    model_name: str = "gpt-oss-120b") -> AzureAIConnector:
    """
    Factory function to create an Azure AI connector.
    
    Args:
        endpoint: Azure AI endpoint URL
        key: Azure AI API key
        model_name: Model deployment name
        
    Returns:
        AzureAIConnector instance
    """
    return AzureAIConnector(endpoint=endpoint, key=key, model_name=model_name)


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
    Test the Azure AI connector wrapper.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create connector
    connector = create_connector()
    
    print("Testing Azure AI Connector...")
    print(f"Connection info: {connector.get_connection_info()}")
    print("-" * 50)
    
    # Test connection
    if connector.test_connection():
        print("✅ Connection successful!")
        
        # Test simple completion
        print("\nTesting simple completion...")
        response = connector.simple_completion(
            system_message="You are a helpful assistant.",
            user_message="What are 3 things to visit in Seattle? Please respond in JSON format.",
            response_format="json_object"
        )
        
        if response:
            print("✅ Simple completion successful!")
            print(f"Response: {response}")
        else:
            print("❌ Simple completion failed")
    else:
        print("❌ Connection failed")