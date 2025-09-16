"""
Evaluation metrics for prompt strategy validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """Validation result structure"""
    strategy_name: str
    input_text: str
    predicted_label: Optional[str]
    true_label: Optional[str]
    response_text: str
    response_time: float
    metrics: Dict
    rationale: Optional[str] = None  # Added rationale field


class EvaluationMetrics:
    """
    Basic evaluation metrics calculator.
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.supported_metrics = ["accuracy", "precision", "recall", "f1_score"]
    
    def parse_prediction(self, response_text: str) -> str:
        """
        Parse model response to extract prediction.
        
        Args:
            response_text: Raw model response
            
        Returns:
            str: Parsed prediction
        """
        if not response_text or response_text.strip() == "":
            return "unknown"
        
        # Simple keyword-based parsing
        response_lower = response_text.lower()
        if "hate" in response_lower or "toxic" in response_lower:
            return "hate"
        elif "not hate" in response_lower or "clean" in response_lower:
            return "not_hate"
        else:
            return "uncertain"
    
    def calculate_metrics(self, predictions: List[str], true_labels: List[str]) -> Dict:
        """
        Calculate basic evaluation metrics.
        
        Args:
            predictions: List of predicted labels
            true_labels: List of ground truth labels
            
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