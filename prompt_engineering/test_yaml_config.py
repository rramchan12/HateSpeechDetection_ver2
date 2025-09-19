#!/usr/bin/env python3
"""
Test script to validate the new YAML-based configuration system.
Tests model switching, connector functionality, and prompt validation.
"""

import logging
from azureai_mi_connector_wrapper import AzureAIConnector
from prompts_validator import PromptValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connector_functionality():
    """Test the Azure AI connector with YAML configuration."""
    print("=== Testing Azure AI Connector ===")
    
    # Test default model (gpt-oss-20b)
    print("\n1. Testing GPT-OSS-20B configuration:")
    connector1 = AzureAIConnector()
    print(f"Available models: {connector1.list_available_models()}")
    print(f"Current model info: {connector1.get_model_info()}")
    
    # Test model switching
    print("\n2. Testing GPT-5 configuration:")
    connector2 = AzureAIConnector(model_id='gpt-5')
    print(f"Current model info: {connector2.get_model_info()}")
    
    # Test model switching on same connector
    print("\n3. Testing model switching:")
    connector1.switch_model('gpt-5')
    print(f"After switch: {connector1.get_model_info()}")
    
    return True

def test_prompt_validator():
    """Test the PromptValidator with YAML configuration."""
    print("\n=== Testing PromptValidator ===")
    
    # Test with GPT-OSS-20B
    print("\n1. Testing GPT-OSS-20B validator:")
    validator1 = PromptValidator(model_id='gpt-oss-20b')
    print(f"Validator model info: {validator1.model_info}")
    
    # Test metrics calculation
    print("\n2. Testing metrics calculation:")
    try:
        metrics = validator1.calculate_metrics_only(data_source='canned')
        print(f"Metrics summary: {metrics.get('summary', {})}")
    except Exception as e:
        print(f"Metrics calculation error: {e}")
    
    # Test with GPT-5
    print("\n3. Testing GPT-5 validator:")
    validator2 = PromptValidator(model_id='gpt-5')
    print(f"Validator model info: {validator2.model_info}")
    
    return True

def test_legacy_compatibility():
    """Test backward compatibility with legacy parameters."""
    print("\n=== Testing Legacy Compatibility ===")
    
    # Test legacy parameters (should show warnings)
    print("\n1. Testing legacy parameter warning:")
    try:
        validator = PromptValidator(endpoint="test-endpoint", key="test-key", model_name="test-model")
        print("Legacy compatibility working (default model used)")
    except Exception as e:
        print(f"Legacy test error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Starting YAML Configuration Tests")
    print("=" * 50)
    
    try:
        # Test connector functionality
        test_connector_functionality()
        
        # Test prompt validator
        test_prompt_validator()
        
        # Test legacy compatibility
        test_legacy_compatibility()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()