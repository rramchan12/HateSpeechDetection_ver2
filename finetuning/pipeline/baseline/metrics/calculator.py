"""
Metrics calculation for baseline validation.

Reuses code from prompt_engineering/metrics for consistency.
"""

import sys
from pathlib import Path

# Add project root to path to import from prompt_engineering
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from prompt_engineering.metrics.evaluation_metrics_calc import (
    EvaluationMetrics,
    PerformanceMetrics,
    ValidationResult
)
from typing import List, Dict, Any


def calculate_metrics(results: List[Dict[str, Any]], strategy_name: str = "baseline") -> PerformanceMetrics:
    """
    Calculate baseline performance metrics using sklearn.
    Reuses EvaluationMetrics from prompt_engineering.
    
    Args:
        results: List of inference results with predictions and true labels
        strategy_name: Name of the strategy (default: "baseline")
        
    Returns:
        PerformanceMetrics: Calculated performance metrics
    """
    # Filter out unknown and error predictions
    valid_results = [
        r for r in results
        if r.get('prediction') not in ['unknown', 'error', None]
    ]
    
    print(f"\n{'='*60}")
    print("BASELINE VALIDATION METRICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {len(valid_results)} ({100*len(valid_results)/len(results):.1f}%)")
    print(f"Invalid predictions: {len(results) - len(valid_results)}")
    
    if len(valid_results) == 0:
        print("[ERROR] No valid predictions!")
        return PerformanceMetrics(strategy_name, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    
    # Extract labels and normalize
    true_labels = []
    pred_labels = []
    
    for r in valid_results:
        true_label = r['true_label']
        pred_label = r['prediction']
        
        # Normalize to 'hate' or 'normal'
        if true_label.lower() in ['hate', 'hateful']:
            true_labels.append('hate')
        else:
            true_labels.append('normal')
            
        if pred_label.lower() in ['hate', 'hateful']:
            pred_labels.append('hate')
        else:
            pred_labels.append('normal')
    
    # Use EvaluationMetrics from prompt_engineering
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_comprehensive_metrics(
        y_true=true_labels,
        y_pred=pred_labels,
        strategy_name=strategy_name
    )
    
    # Print results
    print(f"\n{'Metric':<20} {'Value':<15}")
    print(f"{'-'*35}")
    print(f"{'Accuracy':<20} {metrics.accuracy:.4f}")
    print(f"{'Precision':<20} {metrics.precision:.4f}")
    print(f"{'Recall':<20} {metrics.recall:.4f}")
    print(f"{'F1-Score':<20} {metrics.f1_score:.4f}")
    
    # Calculate FPR/FNR
    fpr = metrics.false_positive / (metrics.false_positive + metrics.true_negative) \
          if (metrics.false_positive + metrics.true_negative) > 0 else 0
    fnr = metrics.false_negative / (metrics.false_negative + metrics.true_positive) \
          if (metrics.false_negative + metrics.true_positive) > 0 else 0
    
    print(f"{'FPR':<20} {fpr:.4f}")
    print(f"{'FNR':<20} {fnr:.4f}")
    
    # Confusion matrix
    print(f"\n{'Confusion Matrix':<20}")
    print(f"{'-'*35}")
    print(f"{'True Positives':<20} {metrics.true_positive}")
    print(f"{'False Positives':<20} {metrics.false_positive}")
    print(f"{'True Negatives':<20} {metrics.true_negative}")
    print(f"{'False Negatives':<20} {metrics.false_negative}")
    
    print(f"\n{'='*60}")
    print(f"BASELINE F1-SCORE: {metrics.f1_score:.4f}")
    print(f"TARGET TO BEAT: 0.620 (fine-tuning goal)")
    print(f"{'='*60}\n")
    
    return metrics
