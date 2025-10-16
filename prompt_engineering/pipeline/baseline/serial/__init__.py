"""
Hyperparameter Optimization Pipeline Package

This package implements a comprehensive 3-phase hyperparameter optimization 
approach for hate speech detection models, integrating with the existing
evaluation and bias analysis framework.

Main Components:
- HyperparameterOptimizer: Core optimization engine
- Phase1ConsistencyVerifier: Consistency verification with <5% variance
- Phase2PerformanceOptimizer: Performance optimization for F1/precision
- Phase3RecallMaximizer: Recall maximization and pattern discovery  
- FinalOptimizationSelector: Final configuration selection

Success Criteria:
- Phase 1: <5% variance, maintain baseline performance
- Phase 2: >2% F1 improvement OR >5% precision improvement  
- Phase 3: >3% recall improvement AND ≥5 new patterns discovered
- Final: >85% F1-score with consistent bias performance across target groups

Usage:
    # Full pipeline
    python optimization_runner.py --full-pipeline
    
    # Specific phases
    python optimization_runner.py --phases 1,2 --data-source canned_100_stratified
    
    # Programmatic usage
    from pipeline import create_hyperparameter_optimizer
    optimizer = create_hyperparameter_optimizer()
    results = optimizer.run_optimization()
"""

from .hyperparameter_optimizer import (
    HyperparameterOptimizer, 
    OptimizationVariant,
    PhaseResult,
    OptimizationResults,
    create_hyperparameter_optimizer
)

from .phase1_consistency import (
    Phase1ConsistencyVerifier,
    ConsistencyAnalysis,
    create_phase1_verifier
)

from .phase2_performance import (
    Phase2PerformanceOptimizer,
    PerformanceAnalysis,
    create_phase2_optimizer
)

from .phase3_recall import (
    Phase3RecallMaximizer,
    RecallAnalysis,
    PatternDiscovery,
    create_phase3_maximizer
)

from .final_selector import (
    FinalOptimizationSelector,
    FinalConfiguration,
    BiasAnalysis,
    create_final_selector
)

__version__ = "1.0.0"
__author__ = "Hate Speech Detection Optimization Team"

# Package-level convenience functions

def run_full_optimization(model_id: str = "gpt-oss-120b", 
                         data_source: str = "canned_100_stratified",
                         output_dir: str = "outputs/optimization",
                         config_path: str = None) -> OptimizationResults:
    """
    Run the complete 3-phase optimization pipeline.
    
    Args:
        model_id: Model identifier for Azure AI
        data_source: Dataset for optimization
        output_dir: Directory for results
        config_path: Path to model configuration YAML
        
    Returns:
        OptimizationResults: Complete optimization results
    """
    from pathlib import Path
    
    optimizer = create_hyperparameter_optimizer(model_id, config_path)
    optimizer.output_dir = Path(output_dir)
    
    return optimizer.run_optimization(data_source)


def run_single_phase(phase: int, 
                    model_id: str = "gpt-oss-120b",
                    data_source: str = "canned_100_stratified", 
                    base_template: str = "all_combined.json",
                    output_dir: str = "outputs/optimization",
                    config_path: str = None):
    """
    Run a single optimization phase.
    
    Args:
        phase: Phase number (1, 2, or 3)
        model_id: Model identifier
        data_source: Dataset for optimization
        base_template: Base template file
        output_dir: Output directory
        config_path: Model configuration path
        
    Returns:
        Phase-specific results
    """
    from pathlib import Path
    
    optimizer = create_hyperparameter_optimizer(model_id, config_path)
    optimizer.output_dir = Path(output_dir)
    
    # Initialize baseline
    optimizer.initialize_baseline(data_source, "baseline_v1.json")
    
    if phase == 1:
        verifier = create_phase1_verifier(optimizer)
        return verifier.execute_phase1(base_template, data_source)
    elif phase == 2:
        optimizer = create_phase2_optimizer(optimizer)
        return optimizer.execute_phase2(base_template, data_source)
    elif phase == 3:
        maximizer = create_phase3_maximizer(optimizer)
        return maximizer.execute_phase3(base_template, data_source)
    else:
        raise ValueError(f"Invalid phase number: {phase}. Must be 1, 2, or 3.")


# Integration with existing evaluation metrics system
def integrate_with_evaluation_metrics():
    """
    Ensure integration with the existing evaluation metrics and bias analysis system.
    
    This function ensures that the optimization pipeline properly uses:
    - EvaluationMetrics class for comprehensive performance calculation
    - BiasMetrics analysis for LGBTQ+, Middle Eastern, and Mexican target groups
    - PerformanceMetrics for F1, precision, recall tracking
    - Enhanced CSV generation with bias columns
    """
    
    # Import existing evaluation system to verify compatibility
    try:
        import sys
        import os
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'metrics')
        sys.path.append(metrics_path)
        
        from metrics.evaluation_metrics_calc import (
            EvaluationMetrics, 
            PerformanceMetrics, 
            BiasMetrics,
            create_evaluation_metrics
        )
        
        return True
        
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to integrate with evaluation metrics system: {e}")
        return False


# Verify integration on import
_integration_success = integrate_with_evaluation_metrics()

if not _integration_success:
    import warnings
    warnings.warn("Optimization pipeline may not integrate properly with evaluation metrics system")


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