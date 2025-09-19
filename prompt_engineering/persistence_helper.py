"""
Persistence helper for saving validation results and performance metrics.
Handles all file I/O operations for the prompt strategy validator.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from evaluation_metrics_calc import EvaluationMetrics, ValidationResult


class PersistenceHelper:
    """
    Helper class for persisting validation results to various output formats.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the persistence helper.
        
        Args:
            output_dir: Directory to save output files (defaults to outputs)
        """
        self.logger = logging.getLogger(__name__)
        
        # Set default output directory if not provided
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "outputs"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Track current incremental files
        self._current_results_file = None
        self._current_samples_file = None
        self._results_writer = None
        self._samples_writer = None
        self._results_csv_file = None
        self._samples_csv_file = None
        
        self.logger.info(f"PersistenceHelper initialized with output directory: {self.output_dir}")
    
    def generate_timestamp(self) -> str:
        """
        Generate a timestamp string for file naming.
        
        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_runid_directory(self, timestamp: str) -> Path:
        """
        Create a runId-specific directory for organizing outputs.
        
        Args:
            timestamp: Timestamp for runId naming
            
        Returns:
            Path: Path to the created runId directory
        """
        runid_dir = self.output_dir / f"run_{timestamp}"
        runid_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created runId directory: {runid_dir}")
        return runid_dir
    
    def initialize_incremental_storage(self, timestamp: str, runid_dir: Path) -> None:
        """
        Initialize incremental storage for results and samples.
        
        Args:
            timestamp: Timestamp for file naming
            runid_dir: Directory for the specific run
        """
        # Initialize results file
        self._current_results_file = runid_dir / f"strategy_unified_results_{timestamp}.csv"
        self._results_csv_file = open(self._current_results_file, 'w', newline='', encoding='utf-8')
        
        # Initialize samples file  
        self._current_samples_file = runid_dir / f"test_samples_{timestamp}.csv"
        self._samples_csv_file = open(self._current_samples_file, 'w', newline='', encoding='utf-8')
        
        self.logger.info(f"Initialized incremental storage: results={self._current_results_file}, samples={self._current_samples_file}")
    
    def save_result_incrementally(self, result_dict: Dict) -> None:
        """
        Save a single validation result incrementally to CSV.
        
        Args:
            result_dict: Dictionary containing validation result data
        """
        if self._results_writer is None and self._results_csv_file is not None:
            # Initialize CSV writer with headers on first write
            fieldnames = result_dict.keys()
            self._results_writer = csv.DictWriter(self._results_csv_file, fieldnames=fieldnames)
            self._results_writer.writeheader()
        
        if self._results_writer is not None:
            self._results_writer.writerow(result_dict)
            self._results_csv_file.flush()  # Ensure data is written immediately
    
    def save_sample_incrementally(self, sample_dict: Dict) -> None:
        """
        Save a single test sample incrementally to CSV.
        
        Args:
            sample_dict: Dictionary containing test sample data
        """
        if self._samples_writer is None and self._samples_csv_file is not None:
            # Initialize CSV writer with headers on first write
            fieldnames = sample_dict.keys()
            self._samples_writer = csv.DictWriter(self._samples_csv_file, fieldnames=fieldnames)
            self._samples_writer.writeheader()
        
        if self._samples_writer is not None:
            self._samples_writer.writerow(sample_dict)
            self._samples_csv_file.flush()  # Ensure data is written immediately
    
    def finalize_incremental_storage(self) -> Dict[str, Path]:
        """
        Finalize incremental storage by closing files and returning paths.
        
        Returns:
            Dict[str, Path]: Dictionary with paths to saved files
        """
        output_paths = {}
        
        if self._results_csv_file is not None:
            self._results_csv_file.close()
            output_paths['detailed_results'] = self._current_results_file
            self.logger.info(f"Finalized results file: {self._current_results_file}")
        
        if self._samples_csv_file is not None:
            self._samples_csv_file.close()
            output_paths['test_samples'] = self._current_samples_file
            self.logger.info(f"Finalized samples file: {self._current_samples_file}")
        
        # Reset state
        self._current_results_file = None
        self._current_samples_file = None
        self._results_writer = None
        self._samples_writer = None
        self._results_csv_file = None
        self._samples_csv_file = None
        
        return output_paths
    
    def save_detailed_results(self, detailed_results: List[Dict], timestamp: str) -> Path:
        """
        Save detailed validation results to CSV.
        
        Args:
            detailed_results: List of detailed result dictionaries
            timestamp: Timestamp for file naming
            
        Returns:
            Path: Path to the saved CSV file
        """
        csv_path = self.output_dir / f"strategy_unified_results_{timestamp}.csv"
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if detailed_results:
                    fieldnames = detailed_results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(detailed_results)
            
            self.logger.info(f"Detailed results saved to: {csv_path}")
            return csv_path
            
        except Exception as e:
            self.logger.error(f"Error saving detailed results: {e}")
            raise
    
    def save_test_samples(self, samples: List[Dict], timestamp: str) -> Path:
        """
        Save test samples to CSV.
        
        Args:
            samples: List of test sample dictionaries
            timestamp: Timestamp for file naming
            
        Returns:
            Path: Path to the saved CSV file
        """
        csv_path = self.output_dir / f"test_samples_{timestamp}.csv"
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if samples:
                    fieldnames = samples[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(samples)
            
            self.logger.info(f"Test samples saved to: {csv_path}")
            return csv_path
            
        except Exception as e:
            self.logger.error(f"Error saving test samples: {e}")
            raise
    
    def save_performance_metrics(self, performance_data: List[Dict], timestamp: str) -> Path:
        """
        Save performance metrics to CSV.
        
        Args:
            performance_data: List of performance metric dictionaries
            timestamp: Timestamp for file naming
            
        Returns:
            Path: Path to the saved CSV file
        """
        csv_path = self.output_dir / f"performance_metrics_{timestamp}.csv"
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if performance_data:
                    fieldnames = performance_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(performance_data)
            
            self.logger.info(f"Performance metrics saved to: {csv_path}")
            return csv_path
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            raise
    
    def save_evaluation_report(self, report_lines: List[str], timestamp: str) -> Path:
        """
        Save human-readable evaluation report to text file.
        
        Args:
            report_lines: List of report lines
            timestamp: Timestamp for file naming
            
        Returns:
            Path: Path to the saved text file
        """
        report_path = self.output_dir / f"evaluation_report_{timestamp}.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Evaluation report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {e}")
            raise
    
    def calculate_and_save_comprehensive_results(self, all_results: Dict[str, List], 
                                               detailed_results: List[Dict], 
                                               samples: List[Dict], 
                                               timestamp: str) -> Dict[str, Path]:
        """
        Calculate metrics and save comprehensive evaluation results to all output formats.
        
        Args:
            all_results: Results grouped by strategy
            detailed_results: Detailed prediction results
            samples: Original test samples
            timestamp: Timestamp for file naming
            
        Returns:
            Dict[str, Path]: Dictionary mapping output type to file path
        """
        output_paths = {}
        
        try:
            # Initialize evaluation metrics calculator
            evaluator = EvaluationMetrics()
            
            # Save detailed results and test samples
            output_paths['detailed_results'] = self.save_detailed_results(detailed_results, timestamp)
            output_paths['test_samples'] = self.save_test_samples(samples, timestamp)
            
            # Calculate performance metrics for all strategies
            performance_metrics = evaluator.calculate_metrics_for_all_strategies(all_results, samples)
            
            # Convert to dictionary format for CSV export
            performance_data = evaluator.performance_metrics_to_dict_list(performance_metrics)
            
            # Generate human-readable report
            report_lines = evaluator.generate_performance_report_lines(performance_metrics, samples)
            
            # Save performance metrics and report
            output_paths['performance_metrics'] = self.save_performance_metrics(performance_data, timestamp)
            output_paths['evaluation_report'] = self.save_evaluation_report(report_lines, timestamp)
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive results saving: {e}")
            raise
    
    def load_canned_samples(self, samples_file: Optional[Path] = None) -> List[Dict]:
        """
        Load canned test samples from JSON file.
        
        Args:
            samples_file: Path to samples file (defaults to canned_basic_all.json)
            
        Returns:
            List[Dict]: List of sample dictionaries
        """
        if samples_file is None:
            samples_file = Path(__file__).parent / "data_samples" / "canned_basic_all.json"
        
        try:
            with open(samples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('samples', [])
                
        except Exception as e:
            self.logger.error(f"Error loading canned samples from {samples_file}: {e}")
            raise
    
    def get_output_directory(self) -> Path:
        """
        Get the current output directory.
        
        Returns:
            Path: Output directory path
        """
        return self.output_dir
    
    def set_output_directory(self, output_dir: Path) -> None:
        """
        Set a new output directory.
        
        Args:
            output_dir: New output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Output directory changed to: {self.output_dir}")


def create_persistence_helper(output_dir: Optional[Path] = None) -> PersistenceHelper:
    """
    Factory function to create a persistence helper.
    
    Args:
        output_dir: Directory to save output files
        
    Returns:
        PersistenceHelper: Configured persistence helper instance
    """
    return PersistenceHelper(output_dir=output_dir)