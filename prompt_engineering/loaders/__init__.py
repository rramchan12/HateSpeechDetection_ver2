"""
Loaders package for prompt engineering framework.

This package contains data and template loading utilities for the hate speech
detection framework.

Modules:
    strategy_templates_loader: Loading and managing prompt strategy templates
    unified_dataset_loader: Unified interface for loading test datasets
"""

from .strategy_templates_loader import (
    PromptTemplate,
    PromptStrategy, 
    StrategyTemplatesLoader,
    format_prompt_with_context
)

from .unified_dataset_loader import (
    DatasetType,
    UnifiedDatasetLoader,
    load_dataset,
    load_dataset_by_filename,
    get_dataset_info
)

__all__ = [
    # Strategy templates
    "PromptTemplate",
    "PromptStrategy",
    "StrategyTemplatesLoader", 
    "format_prompt_with_context",
    # Dataset loading
    "DatasetType",
    "UnifiedDatasetLoader",
    "load_dataset",
    "load_dataset_by_filename",
    "get_dataset_info"
]