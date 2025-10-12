"""
Hyperparameter Optimizer for Hate Speech Detection

This module provides comprehensive hyperparameter optimization for hate speech detection models
using hybrid scoring logic that combines F1 scores with bias fairness metrics.

Key Features:
- Multi-run analysis and comparison
- Hybrid scoring (F1 performance + bias fairness)
- Comprehensive reporting with CSV outputs
- Template-based hyperparameter extraction
- Bias-aware model selection
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import os


@dataclass
class RunResult:
    """Container for a single run's results and metadata."""
    run_id: str
    run_path: Path
    template_file: str
    model_name: str
    strategies: Dict[str, Any]
    performance_metrics: pd.DataFrame
    bias_metrics: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class StrategyScore:
    """Comprehensive scoring for a strategy."""
    strategy_name: str
    run_id: str
    f1_score: float
    f1_normalized: float
    bias_score: float
    bias_normalized: float
    hybrid_score: float
    rank: int
    hyperparameters: Dict[str, Any]
    template_file: str
    model_name: str


class HyperparameterOptimiser:
    """
    Advanced hyperparameter optimizer for hate speech detection models.
    
    Uses hybrid scoring logic combining F1 performance with bias fairness metrics
    to select optimal configurations across multiple runs and strategies.
    """
    
    def __init__(self, 
                 prompt_templates_dir: str = "prompt_templates",
                 output_dir: str = None,
                 bias_weight: float = 0.3,
                 performance_weight: float = 0.7):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            prompt_templates_dir: Directory containing prompt template files
            output_dir: Directory for optimization results output (defaults to ./outputs)
            bias_weight: Weight for bias fairness in hybrid scoring (0.0-1.0)
            performance_weight: Weight for F1 performance in hybrid scoring (0.0-1.0)
        """
        self.prompt_templates_dir = Path(prompt_templates_dir)
        
        # Default to outputs folder within the hyperparam directory
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "outputs"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scoring weights (should sum to 1.0)
        self.bias_weight = bias_weight
        self.performance_weight = performance_weight
        
        if abs(bias_weight + performance_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {bias_weight + performance_weight}")
        
        # Protected demographic groups for bias analysis
        self.protected_groups = ['lgbtq', 'mexican', 'middle_east']
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Results storage
        self.run_results: List[RunResult] = []
        self.strategy_scores: List[StrategyScore] = []
        
        # Command line arguments (to be set externally)
        self.command_line_args = None

    def set_command_line_args(self, args_dict: dict):
        """Set command line arguments for inclusion in output reports."""
        self.command_line_args = args_dict

    def load_run_result(self, run_path: str, template_file: str = None, model_name: str = None) -> RunResult:
        """
        Load and parse results from a run directory.
        
        Args:
            run_path: Path to run directory (e.g., 'outputs/baseline_v1/run_20251012_133005')
            template_file: Name of template file used (if known)
            model_name: Model name used (if known)
            
        Returns:
            RunResult: Parsed run results with metadata
        """
        run_path = Path(run_path)
        run_id = run_path.name
        
        self.logger.info(f"Loading run results from: {run_path}")
        
        # Extract timestamp from run_id (e.g., "run_20251012_133005" -> "20251012_133005")
        timestamp = run_id.replace("run_", "") if run_id.startswith("run_") else run_id
        
        # Load performance metrics
        perf_file = run_path / f"performance_metrics_{timestamp}.csv"
        if not perf_file.exists():
            raise FileNotFoundError(f"Performance metrics file not found: {perf_file}")
        
        performance_df = pd.read_csv(perf_file)
        
        # Load bias metrics
        bias_file = run_path / f"bias_metrics_{timestamp}.csv"
        if not bias_file.exists():
            raise FileNotFoundError(f"Bias metrics file not found: {bias_file}")
        
        bias_df = pd.read_csv(bias_file)
        
        # Try to determine template file from run metadata if not provided
        if template_file is None:
            template_file = self._infer_template_from_run(run_path)
        
        # Load hyperparameters from template
        strategies = {}
        if template_file:
            template_path = self.prompt_templates_dir / template_file
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    strategies = template_data.get('strategies', {})        
        # Create metadata
        metadata = {
            'run_timestamp': self._extract_timestamp_from_run_id(run_id),
            'template_file': template_file,
            'model_name': model_name or 'unknown',
            'run_path': str(run_path),
            'strategies_count': len(performance_df) if not performance_df.empty else 0
        }
        
        return RunResult(
            run_id=run_id,
            run_path=run_path,
            template_file=template_file or 'unknown',
            model_name=model_name or 'unknown',
            strategies=strategies,
            performance_metrics=performance_df,
            bias_metrics=bias_df,
            metadata=metadata
        )
    
    def _infer_template_from_run(self, run_path: Path) -> Optional[str]:
        """Attempt to infer template file from run directory or logs."""
        # Check for log files that might contain template info
        log_files = list(run_path.glob("*.log"))
        if log_files:
            try:
                with open(log_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'baseline_v1' in content:
                        return 'baseline_v1.json'
                    elif 'all_combined' in content:
                        return 'all_combined.json'
            except Exception:
                pass
        
        # Default fallback
        return 'baseline_v1.json'
    
    def _extract_timestamp_from_run_id(self, run_id: str) -> str:
        """Extract timestamp from run_id format like 'run_20251012_133005'."""
        if '_' in run_id:
            parts = run_id.split('_')
            if len(parts) >= 3:
                return f"{parts[1]}_{parts[2]}"
        return "unknown"
    
    def calculate_f1_score(self, strategy_row: pd.Series) -> float:
        """Extract F1 score from performance metrics row."""
        return float(strategy_row.get('f1_score', 0.0))
    
    def calculate_bias_score(self, strategy_name: str, bias_df: pd.DataFrame) -> float:
        """
        Calculate bias fairness score for a strategy across protected groups.
        
        Higher scores indicate better fairness (lower bias).
        Score ranges from 0.0 (worst) to 1.0 (best).
        
        Args:
            strategy_name: Name of the strategy to evaluate
            bias_df: Bias metrics DataFrame
            
        Returns:
            float: Normalized bias fairness score
        """
        strategy_bias = bias_df[bias_df['strategy'] == strategy_name]
        
        if strategy_bias.empty:
            self.logger.warning(f"No bias data found for strategy: {strategy_name}")
            return 0.0
        
        # Calculate bias metrics for each protected group
        bias_scores = []
        
        for group in self.protected_groups:
            group_data = strategy_bias[strategy_bias['persona_tag'] == group]
            
            if group_data.empty:
                continue
                
            group_row = group_data.iloc[0]
            
            # Get FPR and FNR (lower is better)
            fpr = float(group_row.get('false_positive_rate', 0.0))
            fnr = float(group_row.get('false_negative_rate', 0.0))
            
            # Convert to fairness score (higher is better)
            # Perfect fairness would be FPR=0, FNR=0, giving score=1.0
            fairness_score = 1.0 - (fpr + fnr) / 2.0
            bias_scores.append(max(0.0, fairness_score))  # Ensure non-negative
        
        # Return average fairness across all groups
        if bias_scores:
            return np.mean(bias_scores)
        else:
            self.logger.warning(f"No valid bias scores calculated for: {strategy_name}")
            return 0.0
    
    def calculate_hybrid_score(self, f1_normalized: float, bias_normalized: float) -> float:
        """
        Calculate hybrid score combining F1 performance and bias fairness.
        
        Args:
            f1_normalized: Normalized F1 score (0.0-1.0)
            bias_normalized: Normalized bias score (0.0-1.0)
            
        Returns:
            float: Weighted hybrid score
        """
        return (self.performance_weight * f1_normalized) + (self.bias_weight * bias_normalized)
    
    def analyze_runs(self, run_configs: List[Dict[str, str]]) -> None:
        """
        Analyze multiple runs and calculate comprehensive scores.
        
        Args:
            run_configs: List of run configurations, each containing:
                        {'run_path': str, 'template_file': str, 'model_name': str}
        """
        self.logger.info(f"Analyzing {len(run_configs)} runs...")
        
        # Load all run results
        self.run_results = []
        for config in run_configs:
            try:
                run_result = self.load_run_result(
                    run_path=config['run_path'],
                    template_file=config.get('template_file'),
                    model_name=config.get('model_name', 'unknown')
                )
                self.run_results.append(run_result)
            except Exception as e:
                self.logger.error(f"Failed to load run {config['run_path']}: {e}")
                continue
        
        if not self.run_results:
            raise ValueError("No valid runs loaded for analysis")
        
        # Calculate scores for all strategies across all runs
        all_strategy_scores = []
        
        for run_result in self.run_results:
            for _, perf_row in run_result.performance_metrics.iterrows():
                strategy_name = perf_row['strategy']
                
                # Calculate raw scores
                f1_score = self.calculate_f1_score(perf_row)
                bias_score = self.calculate_bias_score(strategy_name, run_result.bias_metrics)
                
                # Get hyperparameters
                hyperparams = {}
                if strategy_name in run_result.strategies:
                    hyperparams = run_result.strategies[strategy_name].get('parameters', {})                
                all_strategy_scores.append({
                    'strategy_name': strategy_name,
                    'run_id': run_result.run_id,
                    'f1_score': f1_score,
                    'bias_score': bias_score,
                    'hyperparameters': hyperparams,
                    'template_file': run_result.template_file,
                    'model_name': run_result.model_name
                })
        
        # Normalize scores across all strategies
        if all_strategy_scores:
            f1_scores = [s['f1_score'] for s in all_strategy_scores]
            bias_scores = [s['bias_score'] for s in all_strategy_scores]
            
            f1_min, f1_max = min(f1_scores), max(f1_scores)
            bias_min, bias_max = min(bias_scores), max(bias_scores)
            
            # Avoid division by zero
            f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
            bias_range = bias_max - bias_min if bias_max > bias_min else 1.0
            
            # Create final scored strategies
            self.strategy_scores = []
            for i, score_data in enumerate(all_strategy_scores):
                # Normalize scores to 0-1 range
                f1_normalized = (score_data['f1_score'] - f1_min) / f1_range
                bias_normalized = (score_data['bias_score'] - bias_min) / bias_range
                
                # Calculate hybrid score
                hybrid_score = self.calculate_hybrid_score(f1_normalized, bias_normalized)
                
                strategy_score = StrategyScore(
                    strategy_name=score_data['strategy_name'],
                    run_id=score_data['run_id'],
                    f1_score=score_data['f1_score'],
                    f1_normalized=f1_normalized,
                    bias_score=score_data['bias_score'],
                    bias_normalized=bias_normalized,
                    hybrid_score=hybrid_score,
                    rank=0,  # Will be set after sorting
                    hyperparameters=score_data['hyperparameters'],
                    template_file=score_data['template_file'],
                    model_name=score_data['model_name']
                )
                
                self.strategy_scores.append(strategy_score)
            
            # Sort by hybrid score (descending) and assign ranks
            self.strategy_scores.sort(key=lambda x: x.hybrid_score, reverse=True)
            for i, strategy_score in enumerate(self.strategy_scores):
                strategy_score.rank = i + 1
            
        self.logger.info(f"Analysis complete. Evaluated {len(all_strategy_scores)} strategy configurations.")
    
    def select_optimal_configuration(self) -> StrategyScore:
        """
        Select the optimal hyperparameter configuration based on hybrid scoring.
        
        Returns:
            StrategyScore: The top-ranked strategy configuration
        """
        if not self.strategy_scores:
            raise ValueError("No strategy scores available. Run analyze_runs() first.")
        
        optimal = self.strategy_scores[0]  # Already sorted by hybrid score
        
        self.logger.info(f"Optimal configuration selected:")
        self.logger.info(f"  Strategy: {optimal.strategy_name}")
        self.logger.info(f"  Run ID: {optimal.run_id}")
        self.logger.info(f"  F1 Score: {optimal.f1_score:.4f}")
        self.logger.info(f"  Bias Score: {optimal.bias_score:.4f}")
        self.logger.info(f"  Hybrid Score: {optimal.hybrid_score:.4f}")
        
        return optimal
    
    def generate_results_table(self) -> pd.DataFrame:
        """
        Generate comprehensive results table with all strategies and scores.
        
        Returns:
            pd.DataFrame: Complete results table with rankings and scores
        """
        if not self.strategy_scores:
            raise ValueError("No strategy scores available. Run analyze_runs() first.")
        
        # Prepare data for DataFrame
        table_data = []
        
        for score in self.strategy_scores:
            # Flatten hyperparameters into columns
            hyperparams = score.hyperparameters.copy()
            
            row = {
                'rank': score.rank,
                'strategy_name': score.strategy_name,
                'run_id': score.run_id,
                'template_file': score.template_file,
                'model_name': score.model_name,
                'f1_score': round(score.f1_score, 4),
                'f1_normalized': round(score.f1_normalized, 4),
                'bias_score': round(score.bias_score, 4),
                'bias_normalized': round(score.bias_normalized, 4),
                'hybrid_score': round(score.hybrid_score, 4),
            }
            
            # Add hyperparameter columns
            for param_name, param_value in hyperparams.items():
                row[f'hyperparam_{param_name}'] = param_value
            
            table_data.append(row)
        
        results_df = pd.DataFrame(table_data)
        
        # Sort by rank for clarity
        results_df = results_df.sort_values('rank')
        
        return results_df
    
    def save_results(self, filename_prefix: str = None) -> Dict[str, str]:
        """
        Save optimization results to CSV files and JSON reports.
        
        Args:
            filename_prefix: Optional prefix for output filenames
            
        Returns:
            Dict[str, str]: Dictionary mapping result type to output file path
        """
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"hyperparam_optimization_{timestamp}"
        
        output_files = {}
        
        # Generate and save results table
        results_df = self.generate_results_table()
        csv_path = self.output_dir / f"{filename_prefix}_results.csv"
        results_df.to_csv(csv_path, index=False)
        output_files['results_table'] = str(csv_path)
        
        # Save optimal configuration details
        if self.strategy_scores:
            optimal = self.select_optimal_configuration()
            
            optimal_report = {
                'optimization_summary': {
                    'total_strategies_evaluated': len(self.strategy_scores),
                    'total_runs_analyzed': len(self.run_results),
                    'scoring_weights': {
                        'performance_weight': self.performance_weight,
                        'bias_weight': self.bias_weight
                    },
                    'protected_groups': self.protected_groups,
                    'command_line_args': self.command_line_args
                },
                'optimal_configuration': {
                    'rank': optimal.rank,
                    'strategy_name': optimal.strategy_name,
                    'run_id': optimal.run_id,
                    'template_file': optimal.template_file,
                    'model_name': optimal.model_name,
                    'scores': {
                        'f1_score': optimal.f1_score,
                        'f1_normalized': optimal.f1_normalized,
                        'bias_score': optimal.bias_score,
                        'bias_normalized': optimal.bias_normalized,
                        'hybrid_score': optimal.hybrid_score
                    },
                    'hyperparameters': optimal.hyperparameters
                },
                'top_5_alternatives': [
                    {
                        'rank': score.rank,
                        'strategy_name': score.strategy_name,
                        'run_id': score.run_id,
                        'hybrid_score': score.hybrid_score,
                        'f1_score': score.f1_score,
                        'bias_score': score.bias_score
                    }
                    for score in self.strategy_scores[1:6]  # Next 5 best
                ]
            }
            
            json_path = self.output_dir / f"{filename_prefix}_optimal_config.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(optimal_report, f, indent=2, ensure_ascii=False)
            output_files['optimal_config'] = str(json_path)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        for result_type, file_path in output_files.items():
            self.logger.info(f"  {result_type}: {file_path}")
        
        return output_files
    
    def print_summary(self, top_n: int = 5) -> None:
        """
        Print a formatted summary of optimization results.
        
        Args:
            top_n: Number of top strategies to display
        """
        if not self.strategy_scores:
            print("No results available. Run analyze_runs() first.")
            return
        
        print("=" * 80)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Total Strategies Evaluated: {len(self.strategy_scores)}")
        print(f"Total Runs Analyzed: {len(self.run_results)}")
        print(f"Scoring Weights: Performance={self.performance_weight:.1%}, Bias={self.bias_weight:.1%}")
        print(f"Protected Groups: {', '.join(self.protected_groups)}")
        print()
        
        print(f"TOP {top_n} CONFIGURATIONS:")
        print("-" * 80)
        
        for i, score in enumerate(self.strategy_scores[:top_n]):
            print(f"#{score.rank} - {score.strategy_name} (Run: {score.run_id})")
            print(f"    Hybrid Score: {score.hybrid_score:.4f}")
            print(f"    F1 Score: {score.f1_score:.4f} (normalized: {score.f1_normalized:.4f})")
            print(f"    Bias Score: {score.bias_score:.4f} (normalized: {score.bias_normalized:.4f})")
            print(f"    Template: {score.template_file}")
            
            # Show key hyperparameters
            key_params = ['temperature', 'max_tokens', 'top_p']
            hyperparam_strs = []
            for param in key_params:
                if param in score.hyperparameters:
                    hyperparam_strs.append(f"{param}={score.hyperparameters[param]}")
            
            if hyperparam_strs:
                print(f"    Key Params: {', '.join(hyperparam_strs)}")
            print()


# Example usage and testing functions
def create_example_analysis():
    """Example function showing how to use the HyperparameterOptimiser."""
    
    # Initialize optimizer
    optimizer = HyperparameterOptimiser(
        prompt_templates_dir="prompt_templates",
        # output_dir defaults to ./outputs
        bias_weight=0.3,
        performance_weight=0.7
    )
    
    # Define runs to analyze
    run_configs = [
        {
            'run_path': 'outputs/baseline_v1/run_20251012_133005',
            'template_file': 'baseline_v1.json',
            'model_name': 'gpt-oss-120b'
        },
        # Add more runs as needed
    ]
    
    # Analyze runs
    optimizer.analyze_runs(run_configs)
    
    # Get optimal configuration
    optimal_config = optimizer.select_optimal_configuration()
    
    # Print summary
    optimizer.print_summary()
    
    # Save results
    output_files = optimizer.save_results()
    
    return optimizer, optimal_config, output_files


if __name__ == "__main__":
    # Example usage
    optimizer, optimal_config, files = create_example_analysis()
    print(f"Optimization complete. Results saved to: {files}")