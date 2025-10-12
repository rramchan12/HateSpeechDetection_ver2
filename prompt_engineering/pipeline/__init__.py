"""
Hyperparameter Optimization Pipeline Package

This package provides different implementation approaches for hyperparameter optimization:
- baseline.serial: Serial baseline implementation with hybrid F1-first selection

Usage:
    from pipeline.baseline.serial import run_full_optimization, run_single_phase
    
    # Run full optimization
    results = run_full_optimization(
        model_id="gpt-oss-120b",
        data_source="canned_100_stratified"
    )
    
    # Run single phase
    phase1_results = run_single_phase(phase=1)
"""

# Re-export from baseline.serial for convenience
from .baseline.serial import (
    # Main classes
    HyperparameterOptimizer,
    Phase1ConsistencyVerifier,
    Phase2PerformanceOptimizer,
    Phase3RecallMaximizer,
    FinalOptimizationSelector,
    
    # Data classes
    OptimizationVariant,
    OptimizationResults,
    PhaseResult,
    ConsistencyAnalysis,
    PerformanceAnalysis,
    RecallAnalysis,
    PatternDiscovery,
    FinalConfiguration,
    BiasAnalysis,
    
    # Factory functions
    create_hyperparameter_optimizer,
    create_phase1_verifier,
    create_phase2_optimizer,
    create_phase3_maximizer,
    create_final_selector,
    
    # Convenience functions
    run_full_optimization,
    run_single_phase,
)

__version__ = "1.0.0"
__author__ = "Hate Speech Detection Optimization Team"

__all__ = [
    # Main classes
    'HyperparameterOptimizer',
    'Phase1ConsistencyVerifier',
    'Phase2PerformanceOptimizer', 
    'Phase3RecallMaximizer',
    'FinalOptimizationSelector',
    
    # Data classes
    'OptimizationVariant',
    'OptimizationResults',
    'PhaseResult',
    'ConsistencyAnalysis',
    'PerformanceAnalysis',
    'RecallAnalysis', 
    'PatternDiscovery',
    'FinalConfiguration',
    'BiasAnalysis',
    
    # Factory functions
    'create_hyperparameter_optimizer',
    'create_phase1_verifier',
    'create_phase2_optimizer',
    'create_phase3_maximizer', 
    'create_final_selector',
    
    # Convenience functions
    'run_full_optimization',
    'run_single_phase',
]