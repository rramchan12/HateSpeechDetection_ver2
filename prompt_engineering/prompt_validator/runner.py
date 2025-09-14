"""
Command-line runner for prompt validator.
Simplified version focusing on connection testing.
"""

import argparse
import logging
import os
import sys

from core_validator import PromptValidator


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_connection():
    """Test connection to GPT-OSS-20B"""
    print("\nTESTING GPT-OSS-20B CONNECTION")
    print("=" * 35)
    
    # Check environment variables
    endpoint = os.getenv('AZURE_AI_ENDPOINT')
    key = os.getenv('AZURE_AI_KEY')
    
    if not endpoint or not key:
        print("ERROR: Missing environment variables")
        print("Please set:")
        print("  AZURE_AI_ENDPOINT=<your_azure_ai_endpoint>")
        print("  AZURE_AI_KEY=<your_azure_ai_key>")
        return False
    
    # Initialize validator and test connection
    validator = PromptValidator(endpoint=endpoint, key=key)
    
    if validator.validate_connection():
        print("Connection successful! Ready for validation.")
        return True
    else:
        print("Connection failed. Check credentials and endpoint.")
        return False


def show_data_format():
    """Show expected data format for validation"""
    print("\nEXPECTED DATA FORMAT")
    print("=" * 20)
    print("The unified dataset should contain JSON objects with:")
    print("  - text: Input text to classify")
    print("  - label_binary: Ground truth label (hate/not_hate)")
    print("  - target_group_norm: Normalized target group")
    print("  - source_dataset: Original dataset (toxigen/hatexplain)")
    print("\nExample:")
    print('{')
    print('  "text": "This is example text",')
    print('  "label_binary": "not_hate",')
    print('  "target_group_norm": "lgbtq",')
    print('  "source_dataset": "toxigen"')
    print('}')


def run_scaffolding_demo(sample_size: int = 5):
    """Run scaffolding demonstration"""
    print(f"\nSCAFFOLDING DEMO (Sample Size: {sample_size})")
    print("=" * 35)
    
    # Test connection first
    if not test_connection():
        return False
    
    # Initialize validator
    validator = PromptValidator()
    
    # Demo scaffolding methods
    print("\n1. Testing strategy validation scaffolding...")
    result1 = validator.validate_strategies(sample_size=sample_size)
    print(f"   Result: {result1}")
    
    print("\n2. Testing single strategy scaffolding...")
    result2 = validator.test_single_strategy("policy", "This is a test text")
    print(f"   Result: {result2}")
    
    print("\n3. Testing batch validation scaffolding...")
    test_texts = ["Text 1", "Text 2", "Text 3"]
    result3 = validator.batch_validate(test_texts)
    print(f"   Result: {result3}")
    
    print("\nSCAFFOLDING DEMO COMPLETE")
    print("All methods are placeholder implementations ready for development.")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prompt Strategy Validator")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test connection only")
    parser.add_argument("--show-format", action="store_true",
                       help="Show expected data format")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of samples for scaffolding demo (default: 5)")
    parser.add_argument("--demo", action="store_true",
                       help="Run scaffolding demonstration")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Handle different modes
    if args.test_connection:
        success = test_connection()
        sys.exit(0 if success else 1)
    
    elif args.show_format:
        show_data_format()
        sys.exit(0)
    
    elif args.demo:
        success = run_scaffolding_demo(args.sample_size)
        sys.exit(0 if success else 1)
    
    else:
        # Default: run connection test and scaffolding demo
        print("PROMPT VALIDATOR - SIMPLIFIED VERSION")
        print("=" * 40)
        print("This version focuses on connection testing with scaffolding for future features.")
        print()
        
        # Test connection
        if test_connection():
            print()
            run_scaffolding_demo(args.sample_size)
        
        sys.exit(0)


if __name__ == "__main__":
    main()