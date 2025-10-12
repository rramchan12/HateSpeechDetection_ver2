"""
Baseline implementations for hyperparameter optimization pipeline.

This module contains different baseline approaches:
- serial: Serial baseline implementation with hybrid F1-first selection
"""

# Re-export from serial implementation as default
from .serial import (
    HyperparameterOptimizer,
    Phase1ConsistencyVerifier,
    Phase2PerformanceOptimizer,
    Phase3RecallMaximizer,
    FinalOptimizationSelector,
    OptimizationVariant,
    OptimizationResults,
    PhaseResult,
    ConsistencyAnalysis,
    PerformanceAnalysis,
    RecallAnalysis,
    PatternDiscovery,
    FinalConfiguration,
    BiasAnalysis,
    create_hyperparameter_optimizer,
    create_phase1_verifier,
    create_phase2_optimizer,
    create_phase3_maximizer,
    create_final_selector,
    run_full_optimization,
    run_single_phase,
)

__all__ = [
    'HyperparameterOptimizer',
    'Phase1ConsistencyVerifier',
    'Phase2PerformanceOptimizer',
    'Phase3RecallMaximizer', 
    'FinalOptimizationSelector',
    'OptimizationVariant',
    'OptimizationResults',
    'PhaseResult',
    'ConsistencyAnalysis',
    'PerformanceAnalysis',
    'RecallAnalysis',
    'PatternDiscovery',
    'FinalConfiguration',
    'BiasAnalysis',
    'create_hyperparameter_optimizer',
    'create_phase1_verifier',
    'create_phase2_optimizer',
    'create_phase3_maximizer',
    'create_final_selector', 
    'run_full_optimization',
    'run_single_phase',
]