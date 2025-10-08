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

from .loaders import StrategyTemplatesLoader, PromptStrategy, UnifiedDatasetLoader, load_dataset, get_dataset_info
from .metrics import EvaluationMetrics, ValidationResult, PersistenceHelper
from .connector import AzureAIConnector

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