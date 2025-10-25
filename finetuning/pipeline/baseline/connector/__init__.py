"""
Accelerate-based connectors for unified inference and fine-tuning.

This package provides a unified interface for both inference and fine-tuning
using HuggingFace Accelerate for automatic multi-GPU support.
"""

from .accelerate_connector import AccelerateConnector

__all__ = ['AccelerateConnector']
