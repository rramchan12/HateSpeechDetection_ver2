"""
Fine-Tuning Prompt Generator Package

Converts unified dataset into fine-tuning instruction format (JSONL).
"""

from .generator import FineTuningDataGenerator

__all__ = ['FineTuningDataGenerator']
