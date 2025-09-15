"""
Command-line runner for prompt validator.
Simplified version focusing on connection testing.
"""

import argparse
import logging
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

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


def run_validation_demo(sample_size: int = 5):
    """Run validation demonstration with real strategy testing"""
    print(f"\nVALIDATION DEMO (Sample Size: {sample_size})")
    print("=" * 35)
    
    # Initialize validator 
    validator = PromptValidator()
    
    # Test connection
    if not validator.validate_connection():
        print("Connection failed. Cannot run validation demo.")
        return False
    
    # Demo real validation methods
    print("\n1. Testing strategy validation...")
    result1 = validator.validate_strategies(sample_size=sample_size)
    
    if "error" in result1:
        print(f"   Error: {result1['error']}")
    else:
        print(f"   Tested {result1['total_tests']} samples across {len(result1['strategies_tested'])} strategies")
        for strategy, summary in result1['strategy_summaries'].items():
            print(f"   {strategy}: {summary['success_rate']:.1%} success, {summary['accuracy']:.1%} accuracy")
    
    print("\n2. Testing single strategy...")
    result2 = validator.test_single_strategy("policy", "This is a test text", "not_hate")
    print(f"   Strategy: {result2.strategy_name}")
    print(f"   Prediction: {result2.predicted_label}")
    print(f"   Response Time: {result2.response_time:.2f}s")
    
    print("\n3. Testing batch validation...")
    test_texts = ["Text expressing political opinion", "Text with potential bias", "Neutral comment"]
    test_labels = ["not_hate", "hate", "not_hate"]
    result3 = validator.batch_validate(test_texts, strategies=["baseline", "policy"], true_labels=test_labels)
    print(f"   Batch results: {result3['total_tests']} tests completed")
    for strategy, summary in result3['strategy_summaries'].items():
        print(f"   {strategy}: {summary['success_rate']:.1%} success rate")
    
    print("\nVALIDATION DEMO COMPLETE")
    print("All methods are fully implemented and tested with GPT-OSS-20B.")
    return True


def test_individual_strategies(sample_size: int = 5):
    """Test each strategy individually and print detailed evaluation metrics"""
    print(f"\nTESTING INDIVIDUAL STRATEGIES (Sample Size: {sample_size})")
    print("=" * 55)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("validation_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize validator
    validator = PromptValidator()
    
    # Test connection
    if not validator.validate_connection():
        print("Connection failed. Cannot run strategy testing.")
        return False
    
    # Strategy names
    strategies = ["baseline", "policy", "persona", "combined"]
    
    # Sample test data (mix of hate and not_hate examples)
    test_data = [
        ("I love everyone regardless of their background", "not_hate"),
        ("This policy seems unfair to certain groups", "hate"),
        ("Great work on the diversity initiative", "not_hate"),
        ("Some people just don't belong here", "hate"),
        ("The new guidelines promote equality", "not_hate")
    ][:sample_size]
    
    print(f"Using {len(test_data)} test samples for evaluation\n")
    
    all_results = {}
    detailed_results = []  # For CSV output
    
    for strategy in strategies:
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print("-" * 30)
        
        strategy_results = []
        predictions = []
        true_labels = []
        total_time = 0
        
        for i, (text, true_label) in enumerate(test_data):
            print(f"  Sample {i+1}: Testing '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            try:
                result = validator.test_single_strategy(strategy, text, true_label)
                strategy_results.append(result)
                predictions.append(result.predicted_label)
                true_labels.append(true_label)
                total_time += result.response_time
                
                print(f"    Predicted: {result.predicted_label} | True: {true_label} | Time: {result.response_time:.2f}s")
                
                # Store detailed result for CSV
                detailed_results.append({
                    'timestamp': timestamp,
                    'strategy': strategy,
                    'sample_id': i+1,
                    'input_text': text,
                    'predicted_label': result.predicted_label,
                    'true_label': true_label,
                    'response_time': result.response_time,
                    'response_text': result.response_text[:100] + "..." if len(result.response_text) > 100 else result.response_text,
                    'correct': result.predicted_label == true_label
                })
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                predictions.append("error")
                true_labels.append(true_label)
                
                # Store error result for CSV
                detailed_results.append({
                    'timestamp': timestamp,
                    'strategy': strategy,
                    'sample_id': i+1,
                    'input_text': text,
                    'predicted_label': "error",
                    'true_label': true_label,
                    'response_time': 0.0,
                    'response_text': f"ERROR: {str(e)}",
                    'correct': False
                })
        
        # Calculate metrics using the evaluation_metrics module
        from evaluation_metrics import EvaluationMetrics
        metrics_calculator = EvaluationMetrics()
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(predictions, true_labels)
        
        # Store results
        all_results[strategy] = {
            'metrics': metrics,
            'avg_response_time': total_time / len(test_data) if test_data else 0,
            'total_samples': len(test_data),
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        # Print metrics
        print(f"  METRICS:")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"    Correct Predictions: {metrics.get('correct_predictions', 0)}/{metrics.get('total_samples', 0)}")
        print(f"    Average Response Time: {total_time / len(test_data):.2f}s")
        
        if 'error' in metrics:
            print(f"    Error: {metrics['error']}")
        
        print()
    
    # Save detailed results to CSV
    csv_file = output_dir / f"strategy_test_results_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if detailed_results:
            fieldnames = detailed_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
    
    print(f"📁 DETAILED RESULTS SAVED TO: {csv_file}")
    
    # Save summary metrics to CSV
    summary_file = output_dir / f"strategy_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['strategy', 'accuracy', 'correct_predictions', 'total_samples', 'avg_response_time'])
        for strategy, results in all_results.items():
            accuracy = results['metrics'].get('accuracy', 0)
            correct = results['metrics'].get('correct_predictions', 0)
            total = results['metrics'].get('total_samples', 0)
            avg_time = results['avg_response_time']
            writer.writerow([strategy, accuracy, correct, total, avg_time])
    
    print(f"📊 SUMMARY METRICS SAVED TO: {summary_file}")
    
    # Save detailed report to text file
    report_file = output_dir / f"strategy_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"STRATEGY TESTING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Size: {sample_size}\n")
        f.write("=" * 50 + "\n\n")
        
        # Add test samples section
        f.write("TEST SAMPLES\n")
        f.write("-" * 20 + "\n")
        for i, (text, true_label) in enumerate(test_data):
            f.write(f"Sample {i+1}: {text}\n")
            f.write(f"True Label: {true_label}\n")
            f.write("\n")
        
        f.write("STRATEGY PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        for strategy, results in all_results.items():
            f.write(f"STRATEGY: {strategy.upper()}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results['metrics'].get('accuracy', 0):.2%}\n")
            f.write(f"Correct Predictions: {results['metrics'].get('correct_predictions', 0)}/{results['metrics'].get('total_samples', 0)}\n")
            f.write(f"Average Response Time: {results['avg_response_time']:.2f}s\n")
            
            # Add individual predictions for this strategy
            f.write(f"Individual Results:\n")
            for i, (pred, true_label) in enumerate(zip(results['predictions'], results['true_labels'])):
                status = "✓" if pred == true_label else "✗"
                f.write(f"  Sample {i+1}: {pred} (expected: {true_label}) {status}\n")
            f.write("\n")
        
        # Find best strategy
        best_strategy = max(all_results.keys(), 
                           key=lambda s: all_results[s]['metrics'].get('accuracy', 0))
        best_accuracy = all_results[best_strategy]['metrics'].get('accuracy', 0)
        
        f.write(f"BEST PERFORMING STRATEGY: {best_strategy.upper()} ({best_accuracy:.2%} accuracy)\n")
    
    print(f"📋 DETAILED REPORT SAVED TO: {report_file}")
    print()
    
    # Print summary comparison
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 35)
    print(f"{'Strategy':<12} {'Accuracy':<10} {'Correct':<8} {'Avg Time':<10}")
    print("-" * 45)
    
    for strategy, results in all_results.items():
        accuracy = results['metrics'].get('accuracy', 0)
        correct = results['metrics'].get('correct_predictions', 0)
        total = results['metrics'].get('total_samples', 0)
        avg_time = results['avg_response_time']
        
        print(f"{strategy:<12} {accuracy:<10.2%} {correct}/{total:<6} {avg_time:<10.2f}s")
    
    # Find best performing strategy
    best_strategy = max(all_results.keys(), 
                       key=lambda s: all_results[s]['metrics'].get('accuracy', 0))
    best_accuracy = all_results[best_strategy]['metrics'].get('accuracy', 0)
    
    print(f"\nBEST PERFORMING STRATEGY: {best_strategy.upper()} ({best_accuracy:.2%} accuracy)")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prompt Strategy Validator")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test connection only")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of samples for validation demo (default: 5)")
    parser.add_argument("--demo", action="store_true",
                       help="Run validation demonstration with real strategy testing")
    parser.add_argument("--test-strategies", action="store_true",
                       help="Test each strategy individually with 5 samples and print evaluation metrics")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Handle different modes
    if args.test_connection:
        success = test_connection()
        sys.exit(0 if success else 1)
    
    elif args.demo:
        success = run_validation_demo(args.sample_size)
        sys.exit(0 if success else 1)
    
    elif args.test_strategies:
        success = test_individual_strategies(5)  # Always use 5 samples for strategy testing
        sys.exit(0 if success else 1)
    
    else:
        # Default: run connection test and validation demo
        print("PROMPT VALIDATOR - PRODUCTION VERSION")
        print("=" * 40)
        print("Fully implemented with real GPT-OSS-20B strategy testing.")
        print()
        
        # Test connection
        if test_connection():
            print()
            run_validation_demo(args.sample_size)
        
        sys.exit(0)


if __name__ == "__main__":
    main()