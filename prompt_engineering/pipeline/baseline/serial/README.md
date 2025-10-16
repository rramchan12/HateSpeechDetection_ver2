# Hyperparameter Optimization Pipeline for Hate Speech Detection

This pipeline implements a comprehensive 3-phase hyperparameter optimization approach to find the optimal configuration for hate speech detection models, with a focus on achieving high performance while maintaining fairness across target groups (LGBTQ+, Middle Eastern, and Mexican communities).

## 🎯 Success Criteria

### Phase 1: Consistency Verification
- **Conservative variant**: Tests baseline_conservative strategy from baseline_v1.json
- **Standard variant**: Tests baseline_standard strategy from baseline_v1.json

### Phase 2: Performance Optimization  
- **Balanced variant**: Tests baseline_balanced strategy from baseline_v1.json
- **Focused variant**: Tests baseline_focused strategy from baseline_v1.json

### Phase 3: Recall Maximization
- **Creative variant**: Tests baseline_creative strategy from baseline_v1.json
- **Exploratory variant**: Tests baseline_exploratory strategy from baseline_v1.json

### Final Selection: Hybrid F1-First Approach
- **Primary Criterion**: F1 Score (balances precision and recall automatically)
- **F1 Threshold**: >70% F1-score (realistic threshold for hate speech detection)
- **Bias Threshold**: >80% bias balance score
- **Ranking System**: 0-100 scale based on F1 score + bias balance + success bonuses

## 🚀 Quick Start

### Full Pipeline
```bash
# Run complete 3-phase optimization
cd prompt_engineering/pipeline
python optimization_runner.py --full-pipeline

# With specific data source and model
python optimization_runner.py --full-pipeline \
    --data-source canned_100_stratified \
    --model gpt-oss-120b \
    --output-dir results/optimization
```

### Individual Phases
```bash
# Run only Phase 1 (Consistency)
python optimization_runner.py --phases 1

# Run Phases 2 and 3 (Performance + Recall)  
python optimization_runner.py --phases 2,3

# Debug mode with detailed logging
python optimization_runner.py --phases 1 --debug
```

### Programmatic Usage
```python
from pipeline import run_full_optimization, run_single_phase

# Full optimization
results = run_full_optimization(
    model_id="gpt-oss-120b",
    data_source="canned_100_stratified", 
    output_dir="results/opt"
)

# Single phase
phase1_results = run_single_phase(
    phase=1,
    model_id="gpt-oss-120b"
)
```

## 📋 Pipeline Components

### Core Engine (`hyperparameter_optimizer.py`)
- **HyperparameterOptimizer**: Main optimization coordinator
- **OptimizationVariant**: Configuration variant definitions
- **OptimizationResults**: Complete pipeline results

### Phase 1: Consistency Verification (`phase1_consistency.py`)
- **Conservative Strategy**: baseline_conservative (temp 0.0, max_tokens 256, deterministic)
- **Standard Strategy**: baseline_standard (temp 0.1, max_tokens 512, reliable)
- **Consistency Analysis**: Tracks performance variance and baseline maintenance

### Phase 2: Performance Optimization (`phase2_performance.py`)
- **Balanced Strategy**: baseline_balanced (temp 0.2, max_tokens 400, OpenAI best practices)
- **Focused Strategy**: baseline_focused (temp 0.05, max_tokens 200, high-confidence)
- **Performance Analysis**: F1/precision improvement tracking with bias monitoring

### Phase 3: Recall Maximization (`phase3_recall.py`)
- **Creative Strategy**: baseline_creative (temp 0.3, max_tokens 768, nuanced reasoning)
- **Exploratory Strategy**: baseline_exploratory (temp 0.5, max_tokens 1024, pattern discovery)
- **Recall Analysis**: Tracks recall improvements and pattern discovery

### Final Selection (`final_selector.py`)
- **Hybrid F1-First Selection**: Primary ranking by F1 score with strategic adjustments
- **Configuration Ranking**: 0-100 scale ranking system for all 6 configurations
- **Bias Analysis**: FPR/FNR consistency across target groups
- **Final Validation**: Comprehensive testing of optimal configuration

## 🎛️ Configuration Options

### Model Configuration
```bash
--model gpt-oss-120b          # Model identifier  
--config path/to/config.yaml  # Custom model configuration
```

### Data Sources
```bash
--data-source canned_100_stratified    # Stratified 100-sample dataset
--data-source canned_100_size_varied   # Size-varied 100-sample dataset  
--data-source canned_50_quick          # Quick 50-sample dataset
--data-source unified                  # Full unified dataset
```

### Execution Control
```bash
--phases 1,2,3           # Phases to execute
--consistency-runs 5     # Phase 1 consistency test runs
--performance-runs 3     # Phase 2 performance test runs  
--recall-runs 3          # Phase 3 recall test runs
--skip-baseline          # Skip baseline initialization
--dry-run               # Show execution plan without running
```

### Success Thresholds
```bash
--f1-threshold 0.70         # Final F1-score requirement (default: 70%)
--bias-threshold 0.80       # Bias balance requirement (default: 80%)
--variance-threshold 0.05   # Phase 1 variance limit (default: 5%)
```

## 📊 Output Structure

```
outputs/optimization/
├── opt_20241011_143022/              # Run directory
│   ├── baseline/                     # Baseline results
│   ├── phase1_conservative_run_0/    # Phase 1 conservative tests
│   ├── phase1_standard_run_0/        # Phase 1 standard tests
│   ├── phase2_balanced_run_0/        # Phase 2 balanced tests  
│   ├── phase2_focused_run_0/         # Phase 2 focused tests
│   ├── phase3_creative_run_0/        # Phase 3 creative tests
│   ├── phase3_exploratory_run_0/     # Phase 3 exploratory tests
│   ├── final_validation/             # Final validation results
│   ├── phase1_results_*.json         # Phase 1 analysis
│   ├── phase2_results_*.json         # Phase 2 analysis  
│   ├── phase3_results_*.json         # Phase 3 analysis
│   ├── optimization_results_*.json   # Complete results
│   ├── optimization_report_*.txt     # Final report
│   └── optimization_log_*.log        # Execution log
```

## 🔧 Template Generation

The pipeline automatically generates optimized templates for each variant:

### Phase 1 Templates
- **Conservative**: `conservative_all_combined.json` - Stability-focused
- **Standard**: `standard_all_combined.json` - Balanced performance

### Phase 2 Templates  
- **Balanced**: `balanced_all_combined.json` - F1-score optimized
- **Focused**: `focused_all_combined.json` - Precision optimized

### Phase 3 Templates
- **Creative**: `creative_all_combined.json` - Recall maximized
- **Exploratory**: `exploratory_all_combined.json` - Pattern discovery

## 📈 Integration with Evaluation System

The pipeline integrates with the existing evaluation framework:

- **Strategy Testing**: Tests all 6 baseline_v1.json strategies across 3 phases
- **F1-First Selection**: Uses F1 score as primary ranking criterion (balances precision + recall)
- **BiasMetrics**: FPR/FNR analysis by target groups (LGBTQ+, Middle Eastern, Mexican)
- **Hybrid Ranking**: 0-100 scale ranking combining F1 score, bias balance, and success bonuses
- **Configuration Export**: Exports optimal configuration for production deployment

## 🎯 Success Validation

### Phase Execution Model
- **Phase 1**: Tests 2 variants (conservative + standard), returns both results
- **Phase 2**: Tests 2 variants (balanced + focused), returns both results  
- **Phase 3**: Tests 2 variants (creative + exploratory), returns both results
- **No phase-level filtering**: All 6 configurations passed to final selector

### Hybrid F1-First Selection Process
1. **Primary Ranking**: F1 score (0-70 points) - balances precision and recall automatically
2. **Secondary Factor**: Bias balance score (0-25 points) - fairness consideration
3. **Balance Bonus**: +5 points if both F1 > 60% and bias > 60%
4. **Total Ranking**: 0-100 scale for clear comparison

### Final Selection Thresholds
- **F1 Threshold**: >70% F1-score (realistic for hate speech detection)
- **Bias Threshold**: >80% bias balance score
- **Fallback Strategy**: Select highest F1 if no configuration meets all thresholds

## 🚨 Troubleshooting

### Common Issues
```bash
# Template file not found
Error: Template file 'xyz.json' not found
Solution: Ensure base template exists in prompt_templates/ directory

# Model configuration issues  
Error: Model 'gpt-oss-20b' not found
Solution: Use available models (gpt-oss-120b, gpt-5) or update config

# Insufficient baseline performance
Warning: Phase 1 baseline maintenance failed
Solution: Review baseline configuration or adjust success thresholds
```

### Debug Mode
```bash
# Enable detailed logging
python optimization_runner.py --debug --log-file debug.log

# Check intermediate results
ls outputs/optimization/opt_*/phase*_results_*.json
```

## 📚 Examples

### Complete Optimization Run
```bash
# Full pipeline with custom settings
python optimization_runner.py \
    --full-pipeline \
    --model gpt-oss-120b \
    --data-source canned_100_stratified \
    --output-dir results/production_opt \
    --f1-threshold 0.87 \
    --debug
```

### Research and Development
```bash
# Test individual phases for research
python optimization_runner.py --phases 1 --consistency-runs 10
python optimization_runner.py --phases 2 --performance-runs 5  
python optimization_runner.py --phases 3 --recall-runs 5

# Compare different datasets
for dataset in canned_100_stratified canned_100_size_varied; do
    python optimization_runner.py --phases 1,2 --data-source $dataset
done
```

## 🤝 Contributing

### Adding New Variants
1. Modify the `get_phase_variants()` method in `hyperparameter_optimizer.py`
2. Add template generation logic in the respective phase module
3. Update success criteria and analysis methods
4. Add tests for the new variant

### Extending Evaluation Metrics
1. Enhance bias analysis in `final_selector.py`
2. Add new pattern discovery logic in `phase3_recall.py`
3. Update reporting in `optimization_runner.py`

## 📄 License

This optimization pipeline is part of the Hate Speech Detection project and follows the same licensing terms.

---

For technical support or questions about the optimization pipeline, please refer to the main project documentation or create an issue in the project repository.