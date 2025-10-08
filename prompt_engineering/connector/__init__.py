"""
Connector package for Azure AI model connections.

This package provides Azure AI Inference SDK wrapper functionality
for connecting to multiple AI models with YAML-based configuration.
"""

from .azureai_connector import AzureAIConnector, ModelConfigLoader, create_connector, test_connection_simple

__all__ = [
    'AzureAIConnector',
    'ModelConfigLoader', 
    'create_connector',
    'test_connection_simple'
]