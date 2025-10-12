# Hyperparameter Optimizer for Hate Speech Detection

## Overview

The `HyperparameterOptimiser` class provides hyperparameter optimization for hate speech detection models using a hybrid scoring system that combines F1 performance metrics with bias fairness analysis.

## Features

- **Hybrid Scoring**: Combines F1 performance (default 70%) with bias fairness (default 30%)
- **Multi-Run Analysis**: Analyzes results from multiple runs and strategies
- **Bias-Aware Selection**: Evaluates fairness across protected demographic groups
- **Reporting**: Generates CSV tables and JSON reports
- **Template Integration**: Extracts hyperparameters from prompt template files

## Quick Start

### Using the Optimization Runner

The fastest way to see the optimizer in action is to use the comprehensive runner:

```bash
# Run with default settings (all analyses, output to ./outputs)
python optimization_runner.py --run-id run_20251012_133005

# Run specific analysis mode
python optimization_runner.py --run-id run_20251012_133005 --mode balanced

# Run with custom output directory
python optimization_runner.py --run-id run_20251012_133005 --output-dir ./custom_results

# Run performance-focused analysis only
python optimization_runner.py --run-id run_20251012_133005 --mode performance

# Run all analyses with custom output location
python optimization_runner.py --run-id run_20251012_133005 --mode all --output-dir ./my_analysis
```

**Available Analysis Modes:**

- `all` - Run all analysis types (performance, fairness, balanced, multi)
- `performance` - Performance-focused analysis (90% performance, 10% bias)
- `fairness` - Fairness-focused analysis (50% performance, 50% bias)  
- `balanced` - Balanced analysis (70% performance, 30% bias)
- `multi` - Multi-configuration analysis with comprehensive reporting

**Output:** All results are saved to the specified output directory (default: `hyperparam/outputs/{run_id}`).

### 1. Basic Usage

```python
from hyperparameter_optimiser import HyperparameterOptimiser

# Initialize optimizer
optimizer = HyperparameterOptimiser(
    prompt_templates_dir='prompt_templates',
    output_dir='optimization_results',
    bias_weight=0.3,      # 30% weight for bias fairness
    performance_weight=0.7  # 70% weight for F1 performance
)

# Define runs to analyze
run_configs = [
    {
        'run_path': 'outputs/baseline_v1/run_20251012_133005',
        'template_file': 'baseline_v1.json',
        'model_name': 'gpt-oss-120b'
    },
    # Add more runs...
]

# Analyze and optimize
optimizer.analyze_runs(run_configs)
optimal_config = optimizer.select_optimal_configuration()
optimizer.print_summary()
files = optimizer.save_results()
```

### 2. Input Data Structure

The optimizer expects run directories with the following structure:

```
outputs/baseline_v1/run_20251012_133005/
├── performance_metrics_20251012_133005.csv  # F1, accuracy, precision, recall
├── bias_metrics_20251012_133005.csv        # FPR/FNR by demographic group
├── evaluation_report_20251012_133005.txt   # Optional
└── validation_log_20251012_133005.log      # Optional
```

### 3. Template Files

Hyperparameters are extracted from JSON template files:

```json
{
  "strategies": {
    "baseline_standard": {
      "name": "baseline_standard",
      "parameters": {
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
      }
    }
  }
}
```

## Scoring System

### Hybrid Score Formula

```
Hybrid Score = (Performance Weight × F1 Normalized) + (Bias Weight × Bias Normalized)
```

### F1 Score Normalization

F1 scores are normalized to 0-1 range across all strategies:
```
F1 Normalized = (F1 Score - Min F1) / (Max F1 - Min F1)
```

### Bias Score Calculation

For each protected group (lgbtq, mexican, middle_east):
```
Group Fairness = 1.0 - (FPR + FNR) / 2.0
Bias Score = Average(Group Fairness across all groups)
```

## Output Files

### 1. Results Table (CSV)

| rank | strategy_name | run_id | f1_score | bias_score | hybrid_score | hyperparam_temperature | ... |
|------|---------------|--------|----------|------------|-------------|----------------------|-----|
| 1 | baseline_standard | run_123 | 0.6263 | 0.647 | 1.0000 | 0.1 | ... |
| 2 | baseline_focused | run_123 | 0.6000 | 0.616 | 0.7681 | 0.05 | ... |

### 2. Optimal Configuration (JSON)

```json
{
  "optimization_summary": {
    "total_strategies_evaluated": 6,
    "total_runs_analyzed": 1,
    "scoring_weights": {
      "performance_weight": 0.7,
      "bias_weight": 0.3
    }
  },
  "optimal_configuration": {
    "strategy_name": "baseline_standard",
    "scores": {
      "f1_score": 0.6263,
      "bias_score": 0.647,
      "hybrid_score": 1.0
    },
    "hyperparameters": {
      "temperature": 0.1,
      "max_tokens": 512,
      "top_p": 1.0
    }
  }
}
```



## Advanced Configuration

### Custom Scoring Weights

```python
# Prioritize performance over fairness
optimizer = HyperparameterOptimiser(
    bias_weight=0.1,      # 10% bias weight
    performance_weight=0.9  # 90% performance weight
)

# Prioritize fairness over performance  
optimizer = HyperparameterOptimiser(
    bias_weight=0.6,      # 60% bias weight
    performance_weight=0.4  # 40% performance weight
)
```

### Multiple Run Analysis

```python
run_configs = [
    {
        'run_path': 'outputs/experiment_1/run_20251012_100000',
        'template_file': 'baseline_v1.json',
        'model_name': 'gpt-4o'
    },
    {
        'run_path': 'outputs/experiment_2/run_20251012_110000', 
        'template_file': 'baseline_v2.json',
        'model_name': 'gpt-oss-120b'
    }
]
```

## Command Line Usage

### Arguments

- **`--run-id`** (required): Run ID containing all configuration info (e.g., `run_20251012_133005`)
- **`--mode`** (optional): Analysis mode - `all`, `performance`, `fairness`, `balanced`, or `multi` (default: `all`)
- **`--output-dir`** (optional): Output directory for results (default: `hyperparam/outputs/{run_id}`)

### Usage Examples

```bash
# Basic usage - run all analyses with default output to hyperparam/outputs/{run_id}
python optimization_runner.py --run-id run_20251012_133005

# Run specific analysis type
python optimization_runner.py --run-id run_20251012_133005 --mode performance
python optimization_runner.py --run-id run_20251012_133005 --mode fairness
python optimization_runner.py --run-id run_20251012_133005 --mode balanced
python optimization_runner.py --run-id run_20251012_133005 --mode multi

# Custom output directory
python optimization_runner.py --run-id run_20251012_133005 --output-dir ./experiment_1_results

# Combined options
python optimization_runner.py --run-id run_20251012_133005 --mode balanced --output-dir ./balanced_analysis
```

### Analysis Mode Details

| Mode | Performance Weight | Bias Weight | Description |
|------|-------------------|-------------|-------------|
| `performance` | 90% | 10% | Optimizes primarily for F1 performance |
| `fairness` | 50% | 50% | Balances performance and bias equally |
| `balanced` | 70% | 30% | Standard balanced approach |
| `multi` | 70% | 30% | Comprehensive analysis with detailed reporting |
| `all` | - | - | Runs all above modes for comparison |

### Generated Output Files

All results are saved to the specified output directory:

- `optimization_results.csv` - Detailed results table with rankings
- `optimization_summary.json` - Structured optimization summary
- `comprehensive_analysis.csv` - Multi-run analysis results (when applicable)
- `comprehensive_analysis.json` - Complete analysis report (when applicable)

## API Reference

### HyperparameterOptimiser Class

#### Constructor

```python
HyperparameterOptimiser(
    prompt_templates_dir: str = "prompt_templates",
    output_dir: str = "hyperparam_optimization_results", 
    bias_weight: float = 0.3,
    performance_weight: float = 0.7
)
```

#### Key Methods

- `analyze_runs(run_configs: List[Dict])` - Analyze multiple runs
- `select_optimal_configuration() -> StrategyScore` - Get best config
- `print_summary(top_n: int = 5)` - Display results
- `save_results(filename_prefix: str = None) -> Dict[str, str]` - Export files
- `generate_results_table() -> pd.DataFrame` - Get results as DataFrame

### StrategyScore Dataclass

```python
@dataclass
class StrategyScore:
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
```

## Protected Groups

The optimizer evaluates bias fairness across these demographic groups:

- `lgbtq` - LGBTQ+ community
- `mexican` - Mexican/Latino community  
- `middle_east` - Middle Eastern community

Bias metrics are calculated using False Positive Rate (FPR) and False Negative Rate (FNR) for each group.

## Best Practices

1. **Multiple Runs**: Analyze multiple runs for robust optimization
2. **Balanced Weights**: Start with 70/30 performance/bias split
3. **Template Management**: Keep hyperparameter templates organized
4. **Result Review**: Always review the top 5 configurations
5. **Bias Monitoring**: Pay attention to bias scores, not just F1
6. **Documentation**: Save optimization reports for reproducibility

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure run directories contain required CSV files
2. **Template Missing**: Verify prompt template files exist and are valid JSON
3. **Empty Results**: Check that performance/bias CSV files have data
4. **Import Errors**: Run from correct directory with proper Python path

### File Naming Convention

Expected files in run directories:

- `performance_metrics_{timestamp}.csv`
- `bias_metrics_{timestamp}.csv`

Where `{timestamp}` matches the run directory name (e.g., `20251012_133005` for `run_20251012_133005`).
