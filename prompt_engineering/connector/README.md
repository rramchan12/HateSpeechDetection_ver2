# Azure AI Connector Package

This package provides Azure AI Inference SDK wrapper functionality for connecting to multiple AI models with YAML-based configuration.

## Structure

```
connector/
├── __init__.py                 # Package initialization and exports
├── azureai_connector.py        # Main AzureAIConnector and ModelConfigLoader classes
└── model_connection.yaml       # YAML configuration file for models
```

## Classes

### AzureAIConnector
Main wrapper class for Azure AI Inference SDK connections with YAML configuration support.

**Key Features:**
- YAML-based model configuration
- Multiple model support (GPT-OSS-20B, GPT-5, etc.)
- Environment variable substitution
- JSON response format support
- Error handling and logging
- Model switching capabilities

### ModelConfigLoader
Helper class to load and process model configuration from YAML files.

**Key Features:**
- Environment variable substitution (${VAR_NAME} format)
- Configuration caching
- Model validation
- Default model handling

## Usage

```python
from connector import AzureAIConnector, ModelConfigLoader

# Create connector with default model
connector = AzureAIConnector()

# Create connector with specific model
connector = AzureAIConnector(model_id="gpt-5")

# Create connector with custom config path
connector = AzureAIConnector(
    model_id="gpt-oss-20b",
    config_path="path/to/custom/model_connection.yaml"
)

# Test connection
success = connector.test_connection()

# Make completion request
response = connector.complete(messages=[...])
```

## Configuration

The `model_connection.yaml` file contains:
- Model definitions with endpoints, API keys, and parameters
- Environment variable references (${VAR_NAME})
- Default model specification
- Connection settings (timeouts, retry logic)

## Environment Variables

Required environment variables (referenced in YAML):
- `AZURE_INFERENCE_SDK_ENDPOINT`: Azure AI endpoint URL
- `AZURE_INFERENCE_SDK_KEY`: Azure AI API key

## Migration from Legacy Files



# New import  
from connector import AzureAIConnector
```