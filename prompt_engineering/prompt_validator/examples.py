"""
Example usage of the Prompt Validator package

This script demonstrates how to use the prompt validator
to test different prompting strategies.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from prompt_engineering.prompt_validator import PromptValidator, PromptStrategy


def example_basic_validation():
    """Example: Basic validation with all strategies"""
    print("EXAMPLE: Basic Validation")
    print("=" * 30)
    
    # Initialize validator
    validator = PromptValidator(output_dir="example_outputs")
    
    # Test connection
    print("Testing connection...")
    if not validator.validate_connection():
        print("Connection failed. Check Azure credentials.")
        return
    
    print("Connection successful!")
    
    # Run validation with small sample
    print("\nRunning validation with 10 samples...")
    results = validator.run_full_validation(sample_size=10)
    
    if results:
        print("\nValidation completed. Files saved:")
        for file_type, path in results.items():
            print(f"- {file_type}: {path}")
    else:
        print("Validation failed.")


def example_single_strategy():
    """Example: Test a single strategy"""
    print("\nEXAMPLE: Single Strategy Test")
    print("=" * 35)
    
    # Initialize validator  
    validator = PromptValidator()
    
    if not validator.validate_connection():
        print("Connection failed.")
        return
    
    # Load a small test sample manually
    try:
        from data_preparation.data_unification import load_unified_dataset
        data = load_unified_dataset()
        test_samples = data['test_data'][:5]  # Just 5 samples
        
        print(f"Testing POLICY strategy with {len(test_samples)} samples...")
        
        # Test just the policy strategy
        results = validator.validate_strategy(PromptStrategy.POLICY, test_samples)
        
        print(f"\nResults summary:")
        valid_results = [r for r in results if r.error_message is None]
        print(f"Valid predictions: {len(valid_results)}/{len(results)}")
        
        if valid_results:
            accuracy = sum(1 for r in valid_results if r.predicted_label == r.true_label) / len(valid_results)
            avg_time = sum(r.response_time for r in valid_results) / len(valid_results)
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Average response time: {avg_time:.2f}s")
            
            # Show some examples
            print(f"\nExample predictions:")
            for i, result in enumerate(valid_results[:3]):
                print(f"{i+1}. True: {result.true_label}, Predicted: {result.predicted_label}")
                print(f"   Text: {result.input_text[:100]}...")
                print(f"   Response: {result.raw_response[:100]}...")
                print()
        
    except Exception as e:
        print(f"Error: {e}")


def example_custom_sample():
    """Example: Test with custom sample data"""
    print("\nEXAMPLE: Custom Sample Test")
    print("=" * 30)
    
    # Create a custom test sample matching the expected format
    custom_sample = {
        "text": "it is impossible for me to follow the rules on twitter you can not even call girls who shave the sides of their heads ugly dykes anymore",
        "label_binary": "hate",  # This should be classified as hate speech
        "label_multiclass": "hatespeech",
        "target_group_norm": "lgbtq",
        "persona_tag": "homosexual", 
        "source_dataset": "hatexplain",
        "is_synthetic": False,
        "rationale_text": "ugly dykes",
        "original_id": "example_001",
        "split": "test"
    }
    
    # Initialize validator
    validator = PromptValidator()
    
    if not validator.validate_connection():
        print("Connection failed.")
        return
    
    print("Testing all strategies with custom sample...")
    print(f"Sample text: {custom_sample['text'][:100]}...")
    print(f"True label: {custom_sample['label_binary']}")
    print()
    
    # Test each strategy
    for strategy in PromptStrategy:
        print(f"Testing {strategy.value} strategy...")
        
        results = validator.validate_strategy(strategy, [custom_sample])
        
        if results and not results[0].error_message:
            result = results[0]
            correct = result.predicted_label == result.true_label
            status = "CORRECT" if correct else "INCORRECT"
            
            print(f"  Predicted: {result.predicted_label} ({status})")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Response time: {result.response_time:.2f}s")
            print(f"  Raw response: {result.raw_response[:150]}...")
        else:
            print(f"  ERROR: {results[0].error_message if results else 'No results'}")
        print()


def main():
    """Run examples"""
    print("PROMPT VALIDATOR EXAMPLES")
    print("=" * 40)
    print("This script demonstrates different ways to use the prompt validator.")
    print()
    
    # Check environment
    required_vars = ["AZURE_INFERENCE_SDK_KEY", "AZURE_INFERENCE_SDK_ENDPOINT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these before running examples.")
        return
    
    print("Environment check passed.")
    print()
    
    # Run examples
    try:
        # Basic validation example
        example_basic_validation()
        
        # Single strategy example  
        example_single_strategy()
        
        # Custom sample example
        example_custom_sample()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nExample error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()