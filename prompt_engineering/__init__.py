"""
Hate Speech Detection Prompt Runner Package

A focused package for running prompt validation across different strategies:
- Policy-based prompts
- Persona-aware prompts  
- Combined prompts
- Baseline prompts

This package provides clean validation with direct dependencies and modular design,
focused on prompt performance assessment for hate speech detection.
"""

from .strategy_templates_loader import StrategyTemplatesLoader, PromptStrategy
from .evaluation_metrics_calc import EvaluationMetrics, ValidationResult
from .unified_dataset_loader import UnifiedDatasetLoader, load_dataset, get_dataset_info
from .azureai_mi_connector_wrapper import AzureAIConnector
from .persistence_helper import PersistenceHelper

__version__ = "1.0.0"
__all__ = [
    "StrategyTemplatesLoader",
    "PromptStrategy", 
    "EvaluationMetrics",
    "ValidationResult",
    "UnifiedDatasetLoader",
    "load_dataset",
    "get_dataset_info",
    "AzureAIConnector",
    "PersistenceHelper"
]