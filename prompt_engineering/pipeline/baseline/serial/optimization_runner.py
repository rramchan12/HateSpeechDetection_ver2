#!/usr/bin/env python3
"""
Hyperparameter Optimization CLI Runner

This module provides a command-line interface to run the complete 3-phase
hyperparameter optimization pipeline for hate speech detection.

Usage:
    python optimization_runner.py --data-source canned_100_stratified
    python optimization_runner.py --phases 1,2 --model gpt-oss-120b
    python optimization_runner.py --full-pipeline --output-dir results/optimization
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add prompt_engineering directory to path for imports (3 levels up)
prompt_engineering_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(prompt_engineering_path)

# Local imports from same directory
from hyperparameter_optimizer import HyperparameterOptimizer, create_hyperparameter_optimizer
from phase1_consistency import Phase1ConsistencyVerifier, create_phase1_verifier
from phase2_performance import Phase2PerformanceOptimizer, create_phase2_optimizer  
from phase3_recall import Phase3RecallMaximizer, create_phase3_maximizer
from final_selector import FinalOptimizationSelector, create_final_selector


def setup_logging(debug: bool = False, log_file: str = None):
    """
    Set up logging configuration.
    
    Args:
        debug: Enable debug logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization Pipeline for Hate Speech Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization pipeline
  python optimization_runner.py --full-pipeline
  
  # Run specific phases only
  python optimization_runner.py --phases 1,2 --data-source canned_100_stratified
  
  # Use specific model and output directory
  python optimization_runner.py --model gpt-5 --output-dir results/opt_gpt5
  
  # Debug mode with detailed logging
  python optimization_runner.py --debug --phases 1
        """
    )
    
    # Core optimization parameters
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run complete 3-phase optimization pipeline")
    
    parser.add_argument("--phases", type=str, default="1,2,3",
                       help="Phases to run (e.g., '1,2,3' or '1,3'). Default: all phases")
    
    parser.add_argument("--data-source", type=str, default="canned_100_stratified",
                       help="Dataset for optimization (default: canned_100_stratified)")
    
    parser.add_argument("--base-template", type=str, default="baseline_v1.json",
                       help="Base template file for optimization (default: baseline_v1.json)")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-oss-120b",
                       help="Model ID for optimization (default: gpt-oss-120b)")
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model configuration YAML file")
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, default="outputs/optimization",
                       help="Output directory for optimization results")
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    parser.add_argument("--log-file", type=str, default=None,
                       help="Optional log file path")
    
    # Phase-specific options
    parser.add_argument("--consistency-runs", type=int, default=5,
                       help="Number of runs for Phase 1 consistency testing (default: 5)")
    
    parser.add_argument("--performance-runs", type=int, default=3,
                       help="Number of runs for Phase 2 performance testing (default: 3)")
    
    parser.add_argument("--recall-runs", type=int, default=3,
                       help="Number of runs for Phase 3 recall testing (default: 3)")
    
    # Success criteria overrides
    parser.add_argument("--f1-threshold", type=float, default=0.85,
                       help="F1-score threshold for final success (default: 0.85)")
    
    parser.add_argument("--variance-threshold", type=float, default=0.05,
                       help="Variance threshold for Phase 1 (<5% default)")
    
    # Execution options
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline initialization (use existing baseline)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    return parser


def run_phase1_consistency(optimizer: HyperparameterOptimizer, args) -> dict:
    """
    Run Phase 1: Consistency Verification.
    
    Args:
        optimizer: Main optimizer instance
        args: Command line arguments
        
    Returns:
        dict: Phase 1 results
    """
    logger = logging.getLogger(__name__)
    logger.info("[START] Starting Phase 1: Consistency Verification")
    
    # Create Phase 1 verifier
    phase1 = create_phase1_verifier(optimizer)
    
    # Execute Phase 1
    results = phase1.execute_phase1(
        base_template=args.base_template,
        data_source=args.data_source
    )
    
    # Save results
    results_file = phase1.save_phase1_results(results)
    logger.info(f"[SAVE] Phase 1 results saved to: {results_file}")
    
    return results


def run_phase2_performance(optimizer: HyperparameterOptimizer, args) -> dict:
    """
    Run Phase 2: Performance Optimization.
    
    Args:
        optimizer: Main optimizer instance
        args: Command line arguments
        
    Returns:
        dict: Phase 2 results
    """
    logger = logging.getLogger(__name__)
    logger.info("[START] Starting Phase 2: Performance Optimization")
    
    # Create Phase 2 optimizer
    phase2 = create_phase2_optimizer(optimizer)
    
    # Execute Phase 2
    results = phase2.execute_phase2(
        base_template=args.base_template,
        data_source=args.data_source
    )
    
    # Save results
    results_file = phase2.save_phase2_results(results)
    logger.info(f"[SAVE] Phase 2 results saved to: {results_file}")
    
    return results


def run_phase3_recall(optimizer: HyperparameterOptimizer, args) -> dict:
    """
    Run Phase 3: Recall Maximization.
    
    Args:
        optimizer: Main optimizer instance
        args: Command line arguments
        
    Returns:
        dict: Phase 3 results
    """
    logger = logging.getLogger(__name__)
    logger.info("[START] Starting Phase 3: Recall Maximization")
    
    # Create Phase 3 maximizer
    phase3 = create_phase3_maximizer(optimizer)
    
    # Execute Phase 3
    results = phase3.execute_phase3(
        base_template=args.base_template,
        data_source=args.data_source
    )
    
    # Save results
    results_file = phase3.save_phase3_results(results)
    logger.info(f"[SAVE] Phase 3 results saved to: {results_file}")
    
    return results


def run_final_selection(optimizer: HyperparameterOptimizer, 
                       phase1_results: dict, phase2_results: dict, phase3_results: dict,
                       args) -> dict:
    """
    Run final configuration selection and validation.
    
    Args:
        optimizer: Main optimizer instance
        phase1_results: Results from Phase 1
        phase2_results: Results from Phase 2
        phase3_results: Results from Phase 3
        args: Command line arguments
        
    Returns:
        dict: Final selection results
    """
    logger = logging.getLogger(__name__)
    logger.info("[START] Starting Final Configuration Selection")
    
    # Create final selector
    selector = create_final_selector(optimizer)
    
    # Override F1 threshold if specified
    if args.f1_threshold != 0.85:
        selector.F1_THRESHOLD = args.f1_threshold
        logger.info(f"[THRESHOLD] Using custom F1 threshold: {args.f1_threshold}")
    
    # Evaluate all configurations
    configurations = selector.evaluate_all_configurations(
        phase1_results or {},
        phase2_results or {}, 
        phase3_results or {}
    )
    
    # Select optimal configuration
    optimal_config = selector.select_optimal_configuration(configurations)
    
    if not optimal_config:
        logger.error("[ERROR] No optimal configuration found")
        return {}
    
    # Run final validation
    final_results = selector.run_final_validation(optimal_config, args.data_source)
    
    # Generate optimization report
    report_lines = selector.generate_optimization_report(final_results)
    
    # Save final report
    report_file = optimizer.output_dir / f"optimization_report_{optimizer.run_id}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print report to console
    for line in report_lines:
        logger.info(line)
    
    logger.info(f"[REPORT] Final report saved to: {report_file}")
    
    return final_results


def main():
    """
    Main entry point for the optimization CLI.
    """
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_file = None
    if args.log_file:
        log_file = args.log_file
    elif not args.dry_run:
        # Create log file in output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"optimization_log_{timestamp}.log"
    
    logger = setup_logging(args.debug, log_file)
    
    if args.dry_run:
        logger.info("[DRY_RUN] DRY RUN MODE - Showing what would be executed")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Data Source: {args.data_source}")
        logger.info(f"  Base Template: {args.base_template}")
        logger.info(f"  Output Directory: {args.output_dir}")
        phases = [int(p) for p in args.phases.split(',')]
        logger.info(f"  Phases: {phases}")
        logger.info("  Execute with --dry-run removed to run optimization")
        return
    
    try:
        # Create hyperparameter optimizer
        logger.info("[INIT] Initializing Hyperparameter Optimizer")
        optimizer = create_hyperparameter_optimizer(
            model_id=args.model,
            config_path=args.config
        )
        optimizer.output_dir = Path(args.output_dir)
        optimizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[RUN_ID] Optimization Run ID: {optimizer.run_id}")
        
        # Initialize baseline if not skipping
        if not args.skip_baseline:
            logger.info("[BASELINE] Initializing baseline performance...")
            optimizer.initialize_baseline(args.data_source, "baseline_v1.json")
        else:
            logger.info("[SKIP] Skipping baseline initialization")
        
        # Parse phases to run
        phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
        if args.full_pipeline:
            phases_to_run = [1, 2, 3]
        
        logger.info(f"[PHASES] Running optimization phases: {phases_to_run}")
        
        # Execute phases
        phase1_results = None
        phase2_results = None  
        phase3_results = None
        
        if 1 in phases_to_run:
            phase1_results = run_phase1_consistency(optimizer, args)
        
        if 2 in phases_to_run:
            phase2_results = run_phase2_performance(optimizer, args)
        
        if 3 in phases_to_run:
            phase3_results = run_phase3_recall(optimizer, args)
        
        # Run final selection if any phases were executed
        if phase1_results or phase2_results or phase3_results:
            final_results = run_final_selection(
                optimizer, phase1_results, phase2_results, phase3_results, args
            )
            
            if final_results.get('success_criteria_met', {}).get('overall_success'):
                logger.info("[SUCCESS] OPTIMIZATION COMPLETED SUCCESSFULLY!")
                sys.exit(0)
            else:
                logger.warning("[WARNING] Optimization completed but did not meet all success criteria")
                sys.exit(2)
        else:
            logger.error("[ERROR] No phases were executed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Optimization interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"[ERROR] Optimization failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()