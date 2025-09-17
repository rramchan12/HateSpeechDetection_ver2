"""
Evaluation metrics calculation for prompt strategy validation.

This module provides comprehensive evaluation metrics calculation including
accuracy, precision, recall, F1-score, and confusion matrix generation.
It handles the computation of performance metrics for hate speech detection models.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@dataclass
class ValidationResult:
    """
    Validation result structure for storing individual prediction results.
    
    Attributes:
        strategy_name (str): Name of the strategy used for prediction
        input_text (str): The input text that was classified
        predicted_label (Optional[str]): Model's predicted label
        true_label (Optional[str]): Ground truth label
        response_text (str): Full model response text
        response_time (float): Time taken for prediction in seconds
        metrics (Dict): Additional metrics or metadata
        rationale (Optional[str]): Model's reasoning for the prediction
    """
    strategy_name: str
    input_text: str
    predicted_label: Optional[str]
    true_label: Optional[str]
    response_text: str
    response_time: float
    metrics: Dict
    rationale: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """
    Structure for storing calculated performance metrics.
    
    Attributes:
        strategy (str): Strategy name
        accuracy (float): Overall accuracy score
        precision (float): Precision score for hate detection
        recall (float): Recall score for hate detection
        f1_score (float): F1-score for hate detection
        true_positive (int): True positive count
        true_negative (int): True negative count
        false_positive (int): False positive count
        false_negative (int): False negative count
    """
    strategy: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics calculator for hate speech detection.
    
    This class provides methods to calculate various performance metrics
    including accuracy, precision, recall, F1-score, and confusion matrix
    for binary classification tasks.
    """
    
    def __init__(self):
        """Initialize metrics calculator with logging"""
        self.logger = logging.getLogger(__name__)
        self.supported_metrics = ["accuracy", "precision", "recall", "f1_score"]
    
    def parse_prediction(self, response_text: str) -> str:
        """
        Parse model response to extract prediction.
        
        Args:
            response_text (str): Raw model response
            
        Returns:
            str: Parsed prediction ('hate', 'normal', or 'uncertain')
        """
        if not response_text or response_text.strip() == "":
            return "uncertain"
        
        # Simple keyword-based parsing
        response_lower = response_text.lower()
        if "hate" in response_lower or "toxic" in response_lower:
            return "hate"
        elif "normal" in response_lower or "not hate" in response_lower or "clean" in response_lower:
            return "normal"
        else:
            return "uncertain"
    
    def calculate_basic_metrics(self, predictions: List[str], true_labels: List[str]) -> Dict:
        """
        Calculate basic evaluation metrics (legacy method for backward compatibility).
        
        Args:
            predictions (List[str]): List of predicted labels
            true_labels (List[str]): List of ground truth labels
            
        Returns:
            Dict: Metrics dictionary with accuracy and counts
        """
        if not predictions or not true_labels or len(predictions) != len(true_labels):
            return {"error": "Invalid input data", "accuracy": 0.0}
        
        # Basic accuracy calculation
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return {
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct_predictions": correct
        }
    
    def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str], 
                                      strategy_name: str) -> PerformanceMetrics:
        """
        Calculate comprehensive evaluation metrics using scikit-learn.
        
        Args:
            y_true (List[str]): Ground truth labels
            y_pred (List[str]): Predicted labels
            strategy_name (str): Name of the strategy being evaluated
            
        Returns:
            PerformanceMetrics: Comprehensive performance metrics
            
        Raises:
            ValueError: If input lists are empty or mismatched in length
        """
        if not y_true or not y_pred:
            raise ValueError("Input lists cannot be empty")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Mismatched lengths: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        try:
            # Filter out None predictions and corresponding true labels
            filtered_pairs = [(true, pred) for true, pred in zip(y_true, y_pred) if pred is not None]
            
            if not filtered_pairs:
                # If all predictions are None, return zero metrics
                return PerformanceMetrics(
                    strategy=strategy_name,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    true_positive=0,
                    true_negative=0,
                    false_positive=0,
                    false_negative=0
                )
            
            # Unzip the filtered pairs
            filtered_y_true, filtered_y_pred = zip(*filtered_pairs)
            
            # Calculate metrics using sklearn
            accuracy = accuracy_score(filtered_y_true, filtered_y_pred)
            precision = precision_score(filtered_y_true, filtered_y_pred, pos_label='hate', average='binary', zero_division=0)
            recall = recall_score(filtered_y_true, filtered_y_pred, pos_label='hate', average='binary', zero_division=0)
            f1 = f1_score(filtered_y_true, filtered_y_pred, pos_label='hate', average='binary', zero_division=0)
            
            # Create confusion matrix
            cm = confusion_matrix(filtered_y_true, filtered_y_pred, labels=['hate', 'normal'])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
            
            return PerformanceMetrics(
                strategy=strategy_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                true_positive=int(tp),
                true_negative=int(tn),
                false_positive=int(fp),
                false_negative=int(fn)
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics for {strategy_name}: {e}")
            return PerformanceMetrics(
                strategy=strategy_name,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                true_positive=0,
                true_negative=0,
                false_positive=0,
                false_negative=0
            )
    
    def calculate_metrics_for_all_strategies(self, all_results: Dict[str, List[ValidationResult]], 
                                           samples: List[Dict]) -> List[PerformanceMetrics]:
        """
        Calculate metrics for all strategies in the results.
        
        Args:
            all_results (Dict[str, List[ValidationResult]]): Results grouped by strategy
            samples (List[Dict]): Original test samples with ground truth
            
        Returns:
            List[PerformanceMetrics]: List of performance metrics for each strategy
        """
        performance_metrics = []
        
        for strategy, results in all_results.items():
            if not results:
                continue
                
            # Extract predictions and true labels
            y_true = [sample['label_binary'] for sample in samples[:len(results)]]
            y_pred = [result.predicted_label for result in results]
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_true, y_pred, strategy)
            performance_metrics.append(metrics)
        
        return performance_metrics
    
    def generate_performance_report_lines(self, performance_metrics: List[PerformanceMetrics], 
                                        samples: List[Dict]) -> List[str]:
        """
        Generate human-readable performance report lines.
        
        Args:
            performance_metrics (List[PerformanceMetrics]): Calculated performance metrics
            samples (List[Dict]): Original test samples
            
        Returns:
            List[str]: List of report lines for text output
        """
        from datetime import datetime
        
        report_lines = []
        
        # Build report header
        report_lines.append("HATE SPEECH DETECTION - STRATEGY EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total samples tested: {len(samples)}")
        report_lines.append("")
        
        # Add test samples to report
        report_lines.append("TEST SAMPLES:")
        report_lines.append("-" * 20)
        for i, sample in enumerate(samples, 1):
            text_preview = sample['text'][:100] + ('...' if len(sample['text']) > 100 else '')
            report_lines.append(f"{i}. Text: {text_preview}")
            report_lines.append(f"   Label: {sample['label_binary']}")
            report_lines.append("")
        
        # Add strategy performance
        report_lines.append("\nSTRATEGY PERFORMANCE:")
        report_lines.append("-" * 25)
        
        for metrics in performance_metrics:
            report_lines.append(f"\n{metrics.strategy.upper()} Strategy:")
            report_lines.append(f"  Accuracy:  {metrics.accuracy:.3f}")
            report_lines.append(f"  Precision: {metrics.precision:.3f}")
            report_lines.append(f"  Recall:    {metrics.recall:.3f}")
            report_lines.append(f"  F1-Score:  {metrics.f1_score:.3f}")
            report_lines.append(f"  Confusion Matrix: TP={metrics.true_positive}, "
                              f"TN={metrics.true_negative}, FP={metrics.false_positive}, "
                              f"FN={metrics.false_negative}")
        
        return report_lines
    
    def performance_metrics_to_dict_list(self, performance_metrics: List[PerformanceMetrics]) -> List[Dict]:
        """
        Convert PerformanceMetrics objects to dictionary list for CSV export.
        
        Args:
            performance_metrics (List[PerformanceMetrics]): Performance metrics objects
            
        Returns:
            List[Dict]: List of dictionaries suitable for CSV writing
        """
        return [
            {
                'strategy': metrics.strategy,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'true_positive': metrics.true_positive,
                'true_negative': metrics.true_negative,
                'false_positive': metrics.false_positive,
                'false_negative': metrics.false_negative
            }
            for metrics in performance_metrics
        ]
    
    # Legacy method alias for backward compatibility
    def calculate_metrics(self, predictions: List[str], true_labels: List[str]) -> Dict:
        """Legacy method - use calculate_basic_metrics instead"""
        return self.calculate_basic_metrics(predictions, true_labels)


def create_evaluation_metrics() -> EvaluationMetrics:
    """
    Factory function to create an evaluation metrics calculator.
    
    Returns:
        EvaluationMetrics: Configured evaluation metrics instance
    """
    return EvaluationMetrics()