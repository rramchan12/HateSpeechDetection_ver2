"""
Exploratory Data Analysis for Unified Dataset
==============================================

This module provides comprehensive EDA capabilities for the unified hate speech detection dataset
that combines HateXplain and ToxiGen data filtered for LGBTQ, Mexican, and Middle East target groups.

Features:
- Dataset loading and basic statistics
- Distribution analysis (splits, labels, target groups, sources)
- Data quality assessment
- Visualization capabilities
- Export functionality for analysis results

Author: HateSpeechDetection_ver2 Project
Date: September 2025
"""

import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class UnifiedDatasetStats:
    """Statistics container for unified dataset analysis."""
    total_entries: int
    split_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    label_binary_distribution: Dict[str, int]
    label_multiclass_distribution: Dict[str, int]
    target_group_distribution: Dict[str, int]
    persona_tag_distribution: Dict[str, int]
    avg_text_length: float
    synthetic_ratio: float
    rationale_coverage: float


class UnifiedDatasetEDA:
    """
    Comprehensive EDA class for the unified hate speech detection dataset.
    
    Provides methods for loading, analyzing, and visualizing the combined
    HateXplain + ToxiGen dataset filtered for specific target groups.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the EDA analyzer.
        
        Args:
            data_dir: Path to unified dataset directory. If None, uses default project structure.
            output_dir: Path for saving analysis outputs. If None, uses eda/ directory.
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[1]
            self.data_dir = self.project_root / "data" / "processed" / "unified"
        else:
            self.data_dir = Path(data_dir)
            
        if output_dir is None:
            self.output_dir = self.project_root / "eda" / "outputs"
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.train_data = None
        self.val_data = None 
        self.test_data = None
        self.combined_data = None
        self.stats = None
        
        logger.info(f"Initialized UnifiedDatasetEDA")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def load_dataset(self) -> Dict[str, Any]:
        """
        Load the unified dataset from JSON files.
        
        Returns:
            Dictionary containing loaded data and metadata
        """
        logger.info("Loading unified dataset...")
        
        # Define file paths
        train_file = self.data_dir / "unified_train.json"
        val_file = self.data_dir / "unified_val.json"
        test_file = self.data_dir / "unified_test.json"
        stats_file = self.data_dir / "unified_dataset_stats.json"
        
        # Check file existence
        required_files = [train_file, val_file, test_file, stats_file]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Load data splits
        with open(train_file, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
            
        with open(val_file, 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)
            
        with open(test_file, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            
        # Load statistics
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_raw = json.load(f)
            
        # Combine all data for comprehensive analysis
        self.combined_data = self.train_data + self.val_data + self.test_data
        
        # Convert to stats object
        self.stats = UnifiedDatasetStats(
            total_entries=stats_raw['total_entries'],
            split_distribution={
                'train': len(self.train_data),
                'val': len(self.val_data),
                'test': len(self.test_data)
            },
            source_distribution=stats_raw['source_distribution'],
            label_binary_distribution=stats_raw['label_binary_distribution'],
            label_multiclass_distribution=stats_raw['label_multiclass_distribution'],
            target_group_distribution=stats_raw['target_group_distribution'],
            persona_tag_distribution=stats_raw['persona_tag_distribution'],
            avg_text_length=stats_raw['avg_text_length'],
            synthetic_ratio=stats_raw['synthetic_ratio'],
            rationale_coverage=stats_raw['rationale_coverage']
        )
        
        logger.info(f"Loaded dataset successfully:")
        logger.info(f"  Train: {len(self.train_data):,} entries")
        logger.info(f"  Val: {len(self.val_data):,} entries")
        logger.info(f"  Test: {len(self.test_data):,} entries")
        logger.info(f"  Total: {len(self.combined_data):,} entries")
        
        return {
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data,
            'combined_data': self.combined_data,
            'stats': self.stats
        }
    
    def analyze_basic_statistics(self) -> Dict[str, Any]:
        """
        Generate basic dataset statistics.
        
        Returns:
            Dictionary containing basic analysis results
        """
        if self.combined_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        logger.info("Analyzing basic dataset statistics...")
        
        analysis = {
            'total_entries': len(self.combined_data),
            'split_counts': self.stats.split_distribution,
            'source_counts': self.stats.source_distribution,
            'avg_text_length': self.stats.avg_text_length,
            'synthetic_ratio': self.stats.synthetic_ratio,
            'rationale_coverage': self.stats.rationale_coverage
        }
        
        # Calculate split percentages
        total = analysis['total_entries']
        analysis['split_percentages'] = {
            split: (count / total) * 100 
            for split, count in analysis['split_counts'].items()
        }
        
        # Calculate source percentages
        analysis['source_percentages'] = {
            source: (count / total) * 100 
            for source, count in analysis['source_counts'].items()
        }
        
        logger.info("Basic statistics analysis completed")
        return analysis
    
    def plot_split_distribution(self, save_plot: bool = True) -> plt.Figure:
        """
        Create bar plot for train/val/test split distribution.
        
        Args:
            save_plot: Whether to save the plot to output directory
            
        Returns:
            Matplotlib figure object
        """
        if self.stats is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        logger.info("Creating split distribution plot...")
        
        # Prepare data
        splits = list(self.stats.split_distribution.keys())
        counts = list(self.stats.split_distribution.values())
        total = sum(counts)
        percentages = [(count / total) * 100 for count in counts]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Absolute counts
        bars1 = ax1.bar(splits, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Split Distribution - Absolute Counts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Entries', fontsize=12)
        ax1.set_xlabel('Data Split', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Percentages
        bars2 = ax2.bar(splits, percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Split Distribution - Percentages', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xlabel('Data Split', fontsize=12)
        ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Overall styling
        plt.suptitle('Unified Dataset - Train/Val/Test Split Distribution', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Add dataset info as text
        info_text = f"Total Entries: {total:,}\nFiltered Target Groups: LGBTQ, Mexican, Middle East"
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if save_plot:
            plot_path = self.output_dir / "split_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Split distribution plot saved to: {plot_path}")
        
        return fig
    
    def plot_source_distribution(self, save_plot: bool = True) -> plt.Figure:
        """
        Create bar plot for HateXplain vs ToxiGen source distribution.
        
        Args:
            save_plot: Whether to save the plot to output directory
            
        Returns:
            Matplotlib figure object
        """
        if self.stats is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        logger.info("Creating source distribution plot...")
        
        # Prepare data
        sources = list(self.stats.source_distribution.keys())
        counts = list(self.stats.source_distribution.values())
        total = sum(counts)
        percentages = [(count / total) * 100 for count in counts]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color scheme
        colors = ['#e74c3c', '#3498db']  # Red for HateXplain, Blue for ToxiGen
        
        # Plot 1: Absolute counts
        bars1 = ax1.bar(sources, counts, color=colors)
        ax1.set_title('Source Dataset Distribution - Absolute Counts', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Entries', fontsize=12)
        ax1.set_xlabel('Source Dataset', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Percentages
        bars2 = ax2.bar(sources, percentages, color=colors)
        ax2.set_title('Source Dataset Distribution - Percentages', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xlabel('Source Dataset', fontsize=12)
        ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Overall styling
        plt.suptitle('Unified Dataset - Source Distribution (HateXplain vs ToxiGen)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Add dataset info as text
        info_text = f"Total Entries: {total:,}\nSynthetic Ratio: {self.stats.synthetic_ratio:.1%}"
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if save_plot:
            plot_path = self.output_dir / "source_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Source distribution plot saved to: {plot_path}")
        
        return fig
    
    def print_summary_report(self):
        """Print a comprehensive summary report of the dataset."""
        if self.stats is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        print("\n" + "="*60)
        print("UNIFIED DATASET EDA SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total Entries: {self.stats.total_entries:,}")
        print(f"  Filtered Target Groups: LGBTQ, Mexican, Middle East")
        print(f"  Average Text Length: {self.stats.avg_text_length:.1f} characters")
        print(f"  Synthetic Ratio: {self.stats.synthetic_ratio:.1%}")
        print(f"  Rationale Coverage: {self.stats.rationale_coverage:.1%}")
        
        print(f"\nSplit Distribution:")
        total = self.stats.total_entries
        for split, count in self.stats.split_distribution.items():
            percentage = (count / total) * 100
            print(f"  {split.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nSource Distribution:")
        for source, count in self.stats.source_distribution.items():
            percentage = (count / total) * 100
            print(f"  {source.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nLabel Distribution (Binary):")
        for label, count in self.stats.label_binary_distribution.items():
            percentage = (count / total) * 100
            print(f"  {label.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nTarget Group Distribution:")
        for group, count in self.stats.target_group_distribution.items():
            percentage = (count / total) * 100
            print(f"  {group.upper()}: {count:,} ({percentage:.1f}%)")
        
        print("="*60)


def main():
    """Main function to demonstrate EDA capabilities."""
    print("Unified Dataset EDA - Demo")
    print("="*40)
    
    # Initialize EDA analyzer
    eda = UnifiedDatasetEDA()
    
    # Load dataset
    data = eda.load_dataset()
    
    # Generate basic analysis
    basic_stats = eda.analyze_basic_statistics()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    split_fig = eda.plot_split_distribution(save_plot=True)
    source_fig = eda.plot_source_distribution(save_plot=True)
    
    # Print summary report
    eda.print_summary_report()
    
    # Show plots
    plt.show()
    
    print(f"\n>> EDA completed! Outputs saved to: {eda.output_dir}")


if __name__ == "__main__":
    main()