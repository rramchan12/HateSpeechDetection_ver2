"""
Phase 1: Consistency Verification Implementation

This module implements Conservative and Standard variants for Phase 1 of the
hyperparameter optimization pipeline, focusing on achieving <5% variance
across runs and maintaining baseline performance.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from hyperparameter_optimizer import OptimizationVariant, HyperparameterOptimizer


@dataclass
class ConsistencyAnalysis:
    """
    Detailed consistency analysis results for Phase 1.
    """
    variant_name: str
    run_count: int
    f1_scores: List[float]
    f1_mean: float
    f1_std: float
    variance_coefficient: float
    baseline_ratio: float
    consistency_achieved: bool
    baseline_maintained: bool
    recommendations: List[str]


class Phase1ConsistencyVerifier:
    """
    Specialized implementation for Phase 1: Consistency Verification.
    
    This phase tests Conservative and Standard variants to ensure:
    - <5% variance across multiple runs (coefficient of variation < 0.05)
    - Baseline performance maintenance (≥95% of baseline for conservative, ≥100% for standard)
    """
    
    def __init__(self, optimizer: HyperparameterOptimizer):
        """
        Initialize Phase 1 verifier.
        
        Args:
            optimizer: Main hyperparameter optimizer instance
        """
        self.optimizer = optimizer
        self.logger = logging.getLogger(f"{__name__}.Phase1")
        
    def get_conservative_strategy(self) -> str:
        """
        Get the strategy name for conservative variant from existing baseline template.
        
        Returns:
            str: Strategy name for conservative testing
        """
        return "baseline_conservative"
    
    def get_standard_strategy(self) -> str:
        """
        Get the strategy name for standard variant from existing baseline template.
        
        Returns:
            str: Strategy name for standard testing
        """
        return "baseline_standard"
    
    def run_consistency_test(self, template_path: str, variant_name: str,
                           data_source: str = "canned_100_stratified",
                           test_runs: int = 1) -> ConsistencyAnalysis:
        """
        Run consistency testing for a specific variant.
        
        Args:
            template_path: Path to variant template
            variant_name: Name of the variant being tested
            data_source: Dataset for testing
            test_runs: Number of runs for consistency measurement
            
        Returns:
            ConsistencyAnalysis: Detailed consistency analysis
        """
        self.logger.info(f"[CONSISTENCY] Running consistency test for {variant_name} ({test_runs} runs)")
        
        import sys
        import os
        # Add prompt_engineering path for imports
        prompt_engineering_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        sys.path.append(prompt_engineering_path)
        from prompt_runner import PromptRunner
        
        run_results = []
        f1_scores = []
        
        for run_idx in range(test_runs):
            self.logger.info(f"  [RUN] Run {run_idx + 1}/{test_runs}")
            
            try:
                runner = PromptRunner(
                    model_id=self.optimizer.model_id,
                    config_path=self.optimizer.config_path,
                    prompt_template_file=template_path
                )
                
                # Select the appropriate strategy based on variant
                if variant_name == "conservative":
                    strategy_to_test = [self.get_conservative_strategy()]
                else:  # standard
                    strategy_to_test = [self.get_standard_strategy()]
                
                # Run validation with the specific strategy for consistency
                results = runner.run_validation(
                    strategies=strategy_to_test,
                    data_source=data_source,
                    output_dir=str(self.optimizer.output_dir / f"phase1_{variant_name}_run_{run_idx}"),
                    sample_size=None,
                    use_concurrent=False,  # Sequential for consistency
                    max_workers=1,
                    batch_size=3  # Smaller batches for consistency
                )
                
                run_results.append(results)
                
                # Extract F1 score from performance metrics CSV file
                f1_score = self._extract_f1_from_run(results)
                f1_scores.append(f1_score)
                
                self.logger.info(f"    F1 Score: {f1_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Run {run_idx} failed: {e}")
                continue
        
        # Calculate consistency metrics
        analysis = self._analyze_consistency(f1_scores, variant_name)
        
        return analysis
    
    def _extract_f1_from_run(self, results: Dict[str, Any]) -> float:
        """
        Extract F1 score from run results by reading the performance metrics CSV file.
        
        Args:
            results: Results dictionary from prompt_runner
            
        Returns:
            float: F1 score from the performance metrics file, or 0.0 if not found
        """
        import pandas as pd
        import os
        from pathlib import Path
        
        try:
            # Get the output directory from results
            output_dir = results.get('output_directory', '')
            if not output_dir:
                self.logger.warning("No output directory in results")
                return 0.0
            
            # Look for performance_metrics CSV file
            run_dir = Path(output_dir)
            metrics_files = list(run_dir.glob('performance_metrics_*.csv'))
            
            if not metrics_files:
                self.logger.warning(f"No performance metrics file found in {run_dir}")
                return 0.0
            
            # Read the performance metrics CSV
            metrics_file = metrics_files[0]  # Take the first one if multiple exist
            df = pd.read_csv(metrics_file)
            
            if df.empty or 'f1_score' not in df.columns:
                self.logger.warning(f"No F1 score data in {metrics_file}")
                return 0.0
            
            # Return the F1 score from the first row (should be only one row for single strategy)
            f1_score = float(df['f1_score'].iloc[0])
            return f1_score
            
        except Exception as e:
            self.logger.warning(f"Failed to extract F1 score from CSV: {e}")
            return 0.0
    
    def _analyze_consistency(self, f1_scores: List[float], variant_name: str) -> ConsistencyAnalysis:
        """
        Analyze consistency from F1 scores across runs.
        
        Args:
            f1_scores: F1 scores from multiple runs
            variant_name: Name of the variant
            
        Returns:
            ConsistencyAnalysis: Comprehensive consistency analysis
        """
        import statistics
        
        if not f1_scores:
            return ConsistencyAnalysis(
                variant_name=variant_name,
                run_count=0,
                f1_scores=[],
                f1_mean=0.0,
                f1_std=0.0,
                variance_coefficient=float('inf'),
                baseline_ratio=0.0,
                consistency_achieved=False,
                baseline_maintained=False,
                recommendations=["[ERROR] No successful runs - check configuration"]
            )
        
        # Calculate statistics
        f1_mean = statistics.mean(f1_scores)
        f1_std = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
        variance_coefficient = (f1_std / f1_mean) if f1_mean > 0 and len(f1_scores) > 1 else float('nan')
        
        # Compare to baseline using correctly extracted metrics
        baseline_metrics = getattr(self.optimizer, 'baseline_metrics', {'f1_score': 0.0})
        baseline_f1 = baseline_metrics.get('f1_score', 0.0)
        baseline_ratio = (f1_mean / baseline_f1) if baseline_f1 > 0 else 0.0
        
        # Evaluate success criteria
        if len(f1_scores) == 1:
            # Single run - consistency cannot be meaningfully measured
            consistency_achieved = None  # Undefined for single run
        else:
            consistency_achieved = variance_coefficient < 0.05  # <5% variance
        
        # Baseline maintenance criteria depend on variant
        if variant_name == "conservative":
            baseline_maintained = baseline_ratio >= 0.95  # ≥95% of baseline
        else:  # standard
            baseline_maintained = baseline_ratio >= 1.0   # ≥100% of baseline
        
        # Generate recommendations
        recommendations = []
        
        # Handle consistency recommendations based on run count
        if len(f1_scores) == 1:
            recommendations.append(f"[INFO] Single run - consistency analysis skipped")
            recommendations.append(f"[SUGGESTION] Use --runs 3+ for meaningful consistency analysis")
        elif consistency_achieved:
            recommendations.append(f"[SUCCESS] Consistency achieved: {variance_coefficient:.3f} < 0.05")
        else:
            recommendations.append(f"[ERROR] Consistency not achieved: {variance_coefficient:.3f} ≥ 0.05")
            recommendations.append("[SUGGESTION] Consider: Lower temperature, fixed seed, smaller batch size")
        
        # Handle baseline maintenance recommendations
        if baseline_maintained:
            recommendations.append(f"[SUCCESS] Baseline maintained: {baseline_ratio:.3f}")
        else:
            recommendations.append(f"[ERROR] Baseline not maintained: {baseline_ratio:.3f}")
            recommendations.append("[SUGGESTION] Consider: Adjusting prompt strategy or hyperparameters")
        
        # Summary recommendation (removed [READY] message)
        if len(f1_scores) == 1:
            if baseline_maintained:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant shows baseline performance")
            else:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant needs performance improvement")
        else:
            if consistency_achieved and baseline_maintained:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant meets both consistency and performance criteria")
            elif consistency_achieved:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant is consistent but needs performance improvement")
            elif baseline_maintained:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant performs well but lacks consistency")
            else:
                recommendations.append(f"[RESULT] {variant_name.upper()} variant needs both consistency and performance improvement")
        
        analysis = ConsistencyAnalysis(
            variant_name=variant_name,
            run_count=len(f1_scores),
            f1_scores=f1_scores,
            f1_mean=f1_mean,
            f1_std=f1_std,
            variance_coefficient=variance_coefficient,
            baseline_ratio=baseline_ratio,
            consistency_achieved=consistency_achieved if consistency_achieved is not None else False,
            baseline_maintained=baseline_maintained,
            recommendations=recommendations
        )
        
        return analysis
    
    def execute_phase1(self, base_template: str = "all_combined.json",
                      data_source: str = "canned_100_stratified") -> Dict[str, ConsistencyAnalysis]:
        """
        Execute complete Phase 1: Consistency Verification.
        
        Args:
            base_template: Base template file to modify
            data_source: Dataset for consistency testing
            
        Returns:
            Dict[str, ConsistencyAnalysis]: Results for both conservative and standard variants
        """
        self.logger.info("[START] Starting Phase 1: Consistency Verification")
        
        # Ensure baseline exists
        if not self.optimizer.baseline_results:
            self.optimizer.initialize_baseline(data_source, "baseline_v1.json")
        
        results = {}
        
        # Test Conservative Variant
        self.logger.info("[CONSERVATIVE] Testing Conservative Variant...")
        try:
            conservative_analysis = self.run_consistency_test(
                "baseline_v1.json", "conservative", data_source
            )
            results["conservative"] = conservative_analysis
            
            self.logger.info(f"[CONSERVATIVE] Conservative Results:")
            for rec in conservative_analysis.recommendations:
                self.logger.info(f"  {rec}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Conservative variant failed: {e}")
        
        # Test Standard Variant
        self.logger.info("[STANDARD] Testing Standard Variant...")
        try:
            standard_analysis = self.run_consistency_test(
                "baseline_v1.json", "standard", data_source
            )
            results["standard"] = standard_analysis
            
            self.logger.info(f"[STANDARD] Standard Results:")
            for rec in standard_analysis.recommendations:
                self.logger.info(f"  {rec}")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Standard variant failed: {e}")
        
        # Overall Phase 1 assessment
        successful_variants = [name for name, analysis in results.items() 
                             if analysis.consistency_achieved and analysis.baseline_maintained]
        
        if successful_variants:
            self.logger.info(f"[SUCCESS] Phase 1 SUCCESS: {len(successful_variants)} variant(s) passed")
            self.logger.info(f"   Successful variants: {', '.join(successful_variants)}")
        else:
            self.logger.warning("[WARNING] Phase 1: No variants met all success criteria")
            self.logger.info("[SUGGESTION] Consider adjusting hyperparameters or success thresholds")
        
        return results
    
    def save_phase1_results(self, results: Dict[str, ConsistencyAnalysis]) -> str:
        """
        Save Phase 1 results to file.
        
        Args:
            results: Phase 1 analysis results
            
        Returns:
            str: Path to saved results file
        """
        results_file = self.optimizer.output_dir / f"phase1_results_{self.optimizer.run_id}.json"
        
        # Convert to serializable format
        serializable_results = {}
        for variant_name, analysis in results.items():
            serializable_results[variant_name] = {
                'variant_name': analysis.variant_name,
                'run_count': analysis.run_count,
                'f1_scores': analysis.f1_scores,
                'f1_mean': analysis.f1_mean,
                'f1_std': analysis.f1_std,
                'variance_coefficient': analysis.variance_coefficient,
                'baseline_ratio': analysis.baseline_ratio,
                'consistency_achieved': analysis.consistency_achieved,
                'baseline_maintained': analysis.baseline_maintained,
                'recommendations': analysis.recommendations
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"[SAVE] Phase 1 results saved: {results_file}")
        return str(results_file)


def create_phase1_verifier(optimizer: HyperparameterOptimizer) -> Phase1ConsistencyVerifier:
    """
    Factory function to create a Phase 1 consistency verifier.
    
    Args:
        optimizer: Main hyperparameter optimizer instance
        
    Returns:
        Phase1ConsistencyVerifier: Configured Phase 1 verifier
    """
    return Phase1ConsistencyVerifier(optimizer)