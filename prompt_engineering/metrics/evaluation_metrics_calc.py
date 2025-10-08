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
            # Filter out None and "unknown" predictions and corresponding true labels
            filtered_pairs = [(true, pred) for true, pred in zip(y_true, y_pred) 
                             if pred is not None and pred != "unknown"]
            
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
            # Confusion matrix structure for labels=['hate', 'normal']:
            # [[tp, fn],   <- hate true: tp predicted hate, fn predicted normal
            #  [fp, tn]]   <- normal true: fp predicted hate, tn predicted normal
            tp, fn, fp, tn = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
            
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
                
            # Extract predictions and true labels from the results themselves
            # (don't use samples array as it might not be in the same order)
            y_true = [result.true_label for result in results]
            y_pred = [result.predicted_label for result in results]
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_true, y_pred, strategy)
            performance_metrics.append(metrics)
        
        return performance_metrics
    
    def generate_performance_report_lines(self, performance_metrics: List[PerformanceMetrics], 
                                        samples: List[Dict], 
                                        model_name: str = "Unknown", 
                                        prompt_template_file: str = "Unknown",
                                        data_source: str = "Unknown",
                                        command_line: str = "Unknown") -> List[str]:
        """
        Generate human-readable performance report lines.
        
        Args:
            performance_metrics (List[PerformanceMetrics]): Calculated performance metrics
            samples (List[Dict]): Original test samples
            model_name (str): Name of the model used for evaluation
            prompt_template_file (str): Name of the prompt template file used
            data_source (str): Name of the data source/file used
            command_line (str): Command line that was used to run the evaluation
            
        Returns:
            List[str]: List of report lines for text output
        """
        from datetime import datetime
        
        report_lines = []
        
        # Build report header
        report_lines.append("HATE SPEECH DETECTION - STRATEGY EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {model_name}")
        report_lines.append(f"Prompt Template File: {prompt_template_file}")
        report_lines.append(f"Data Source: {data_source}")
        report_lines.append(f"Total samples tested: {len(samples)}")
        report_lines.append("")
        report_lines.append("EXECUTION DETAILS:")
        report_lines.append("-" * 20)
        report_lines.append(f"Command Line: {command_line}")
        report_lines.append("")
        
        # Add sample test samples to report (show only a few important ones)
        report_lines.append("SAMPLE TEST DATA:")
        report_lines.append("-" * 20)
        
        # Show first 3 samples
        for i in range(min(3, len(samples))):
            sample = samples[i]
            text_preview = sample['text'][:100] + ('...' if len(sample['text']) > 100 else '')
            report_lines.append(f"{i+1}. Text: {text_preview}")
            report_lines.append(f"   Label: {sample['label_binary']}")
            report_lines.append("")
        
        # Show a few samples from the middle if we have more than 10 samples
        if len(samples) > 10:
            report_lines.append("... (middle samples) ...")
            mid_start = len(samples) // 2 - 1
            for i in range(mid_start, min(mid_start + 2, len(samples))):
                sample = samples[i]
                text_preview = sample['text'][:100] + ('...' if len(sample['text']) > 100 else '')
                report_lines.append(f"{i+1}. Text: {text_preview}")
                report_lines.append(f"   Label: {sample['label_binary']}")
                report_lines.append("")
        
        # Show last 2 samples if we have more than 5 samples
        if len(samples) > 5:
            report_lines.append("... (final samples) ...")
            for i in range(max(len(samples) - 2, 3), len(samples)):
                sample = samples[i]
                text_preview = sample['text'][:100] + ('...' if len(sample['text']) > 100 else '')
                report_lines.append(f"{i+1}. Text: {text_preview}")
                report_lines.append(f"   Label: {sample['label_binary']}")
                report_lines.append("")
        
        report_lines.append(f"[Showing sample of {min(7, len(samples))} out of {len(samples)} total test samples]")
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
    
    def calculate_metrics_from_runid(self, runid: str, output_dir: str = "outputs",
                                    model_name: str = "Unknown", 
                                    prompt_template_file: str = "Unknown",
                                    data_source: str = "Unknown",
                                    command_line: str = "Unknown") -> Dict[str, any]:
        """
        Calculate comprehensive metrics from stored results in a runId folder.
        
        Args:
            runid: Run identifier (e.g., "run_20250920_003815")
            output_dir: Base output directory containing run folders
            model_name: Name of the model used
            prompt_template_file: Name of the prompt template file used
            data_source: Name of the data source used
            command_line: Command line that was used
            
        Returns:
            Dict[str, Path]: Dictionary mapping output type to file path
        """
        import pandas as pd
        import csv
        from pathlib import Path
        
        runid_dir = Path(output_dir) / runid
        
        if not runid_dir.exists():
            raise FileNotFoundError(f"RunId directory not found: {runid_dir}")
        
        # Find results and samples files
        results_file = None
        samples_file = None
        
        for file_path in runid_dir.glob("strategy_unified_results_*.csv"):
            results_file = file_path
            break
            
        for file_path in runid_dir.glob("test_samples_*.csv"):
            samples_file = file_path
            break
        
        if not results_file or not results_file.exists():
            raise FileNotFoundError(f"Results file not found in {runid_dir}")
        
        if not samples_file or not samples_file.exists():
            raise FileNotFoundError(f"Samples file not found in {runid_dir}")
        
        try:
            # Load results and samples from CSV
            results_df = pd.read_csv(results_file)
            samples_df = pd.read_csv(samples_file)
            
            # Convert to required formats
            samples = samples_df.to_dict('records')
            
            # Group results by strategy
            all_results = {}
            for strategy in results_df['strategy'].unique():
                strategy_results = results_df[results_df['strategy'] == strategy]
                # Create ValidationResult-like objects with required attributes
                validation_results = []
                for _, row in strategy_results.iterrows():
                    result_obj = type('ValidationResult', (), {
                        'predicted_label': row['predicted_label'],
                        'true_label': row['true_label'],
                        'strategy_name': row['strategy']
                    })()
                    validation_results.append(result_obj)
                all_results[strategy] = validation_results
            
            # Calculate performance metrics for all strategies
            performance_metrics = self.calculate_metrics_for_all_strategies(all_results, samples)
            
            # Convert to dictionary format for CSV export
            performance_data = self.performance_metrics_to_dict_list(performance_metrics)
            
            # Generate human-readable report
            report_lines = self.generate_performance_report_lines(performance_metrics, samples,
                                                                model_name, prompt_template_file, 
                                                                data_source, command_line)
            
            # Generate timestamp for output files
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save performance metrics and report to runId directory
            performance_file = runid_dir / f"performance_metrics_{timestamp}.csv"
            with open(performance_file, 'w', newline='', encoding='utf-8') as f:
                if performance_data:
                    fieldnames = performance_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(performance_data)
            
            self.logger.info(f"Performance metrics saved to: {performance_file}")
            
            # Save evaluation report to runId directory
            report_file = runid_dir / f"evaluation_report_{timestamp}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Evaluation report saved to: {report_file}")
            
            return {
                'performance_metrics': performance_file,
                'evaluation_report': report_file,
                'total_samples': len(samples),
                'strategies_tested': list(results_df['strategy'].unique())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics from runId {runid}: {e}")
            raise


def create_evaluation_metrics() -> EvaluationMetrics:
    """
    Factory function to create an evaluation metrics calculator.
    
    Returns:
        EvaluationMetrics: Configured evaluation metrics instance
    """
    return EvaluationMetrics()


def calculate_metrics_from_runid(runid: str, output_dir: str = "outputs", 
                                model_name: str = "Unknown", 
                                prompt_template_file: str = "Unknown",
                                data_source: str = "Unknown",
                                command_line: str = "Unknown") -> Dict[str, any]:
    """
    Standalone function to calculate metrics from a saved runid folder.
    
    Args:
        runid: The run identifier (timestamp)
        output_dir: The base output directory
        model_name: Name of the model used
        prompt_template_file: Name of the prompt template file used
        data_source: Name of the data source used
        command_line: Command line that was used
        
    Returns:
        Dict containing metrics results
    """
    evaluator = create_evaluation_metrics()
    return evaluator.calculate_metrics_from_runid(runid, output_dir, model_name, 
                                                 prompt_template_file, data_source, command_line)