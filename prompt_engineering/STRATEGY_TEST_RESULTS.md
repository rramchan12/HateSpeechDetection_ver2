# Strategy Test Results

## Overview

This document contains test results and analysis for all four prompt strategies evaluated against Azure AI models using our comprehensive dataset options including canned samples and the unified hate speech dataset with incremental storage and runId-based output organization.

## Test Configuration

- **Model Options**: GPT-OSS-20B (default), GPT-5, and other YAML-configured models
- **Dataset Options**:
  - Canned basic samples (5 curated examples): `canned_basic_all`
  - Canned diverse samples (100 stratified examples): `canned_100_all`  
  - Unified dataset (12,589+ filtered samples): `unified`
- **Strategies Tested**: Baseline, Policy, Persona, Combined
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Response Time
- **Output Format**: JSON with rationale explanations
- **Storage**: Incremental CSV writing with runId-based folder organization
- **Sample Size Control**: Configurable sampling for all data sources

## Available Data Sources

### Canned Datasets

#### canned_basic_all (5 samples)
1. **Sample 1** (Label: hate, Target: LGBTQ)
   - Text: "LGBTQ people are destroying traditional values and shouldn't be allowed near children"
   - Persona Tag: lgbtq

2. **Sample 2** (Label: normal, Target: General)
   - Text: "I support equal rights for all people regardless of their background"
   - Persona Tag: general

3. **Sample 3** (Label: hate, Target: Mexican)
   - Text: "Mexicans are taking all our jobs and ruining this country"
   - Persona Tag: hispanic

4. **Sample 4** (Label: normal, Target: General)
   - Text: "Learning about different cultures helps us become more understanding"
   - Persona Tag: general

5. **Sample 5** (Label: hate, Target: Middle East)
   - Text: "Middle Eastern people are all terrorists and can't be trusted"
   - Persona Tag: middle_eastern

#### canned_100_all (100 samples)
- Stratified sampling across all target groups (LGBTQ, Mexican, Middle East)
- Balanced hate/normal distribution
- Diverse persona tags and sources
- Extracted from unified dataset for representative testing

### Unified Dataset
- Full unified test dataset with 12,589+ samples
- Filtered to LGBTQ, Mexican, and Middle East target groups
- Comprehensive coverage with configurable sampling
- Memory-efficient processing with incremental storage

## Current Framework Capabilities

### Multi-Model Support
```bash
# Use default GPT-OSS-20B
python prompt_runner.py --data-source canned_basic_all --strategies all

# Switch to GPT-5
python prompt_runner.py --model gpt-5 --data-source unified --strategies baseline --sample-size 25
```

### Flexible Data Source Selection
```bash
# Basic canned samples (5 total)
python prompt_runner.py --data-source canned_basic_all --strategies all

# Diverse canned samples with sampling
python prompt_runner.py --data-source canned_100_all --strategies all --sample-size 10

# Unified dataset with custom sampling
python prompt_runner.py --data-source unified --sample-size 50 --strategies policy persona
```

### Metrics Recalculation
```bash
# Recalculate metrics from previous run results
python prompt_runner.py --metrics-only --run-id run_20250920_015821
```

## Strategy Performance Framework

### Strategy Descriptions

#### 1. Baseline Strategy
- **Approach**: Direct hate speech classification without specialized prompting
- **System Prompt**: Basic content moderation assistant role
- **Parameters**: Standard temperature (0.1), max_tokens (512)
- **Expected Performance**: Good baseline accuracy, less nuanced understanding

#### 2. Policy Strategy  
- **Approach**: Platform guidelines-based content moderation
- **System Prompt**: Incorporates community standards and hate speech definitions
- **Parameters**: Optimized for policy compliance (temperature 0.1)
- **Expected Performance**: Strong policy adherence, clear rule-based decisions

#### 3. Persona Strategy
- **Approach**: Multi-perspective evaluation with target group bias awareness
- **System Prompt**: Incorporates identity and bias considerations
- **Parameters**: Balanced for nuanced understanding (temperature 0.1)
- **Expected Performance**: Enhanced bias detection, contextual awareness

#### 4. Combined Strategy
- **Approach**: Fusion of policy and persona strategies
- **System Prompt**: Comprehensive bias-aware policy-based classification
- **Parameters**: Optimized for comprehensive analysis (temperature 0.1)
- **Expected Performance**: Best overall performance, balanced approach

## Output File Organization

All test runs generate files in timestamped runId directories (`outputs/run_YYYYMMDD_HHMMSS/`):

### 1. Validation Results (`validation_results.csv`)
Individual sample results for each strategy:
- Columns: `strategy`, `sample_id`, `text`, `true_label`, `predicted_label`, `response_time`, `rationale`
- Incremental writing during validation for memory efficiency
- Complete audit trail of all predictions

### 2. Performance Metrics (`performance_metrics.csv`)
Strategy performance comparison with detailed metrics:
- Columns: `strategy`, `accuracy`, `precision`, `recall`, `f1_score`, `true_positive`, `true_negative`, `false_positive`, `false_negative`
- Calculated from stored results, not in-memory data
- Ready for analysis and visualization

### 3. Test Samples (`test_samples.csv`)
Test samples used in unified format:
- Columns: `text`, `label_binary`, `target_group_norm`, `persona_tag`, `source_dataset`
- Preserves original dataset structure for traceability
- Enables sample-specific analysis

### 4. Evaluation Report (`evaluation_report.txt`)
Human-readable summary with:
- Test configuration and parameters
- Individual prediction results with indicators
- Strategy performance summary and analysis
- Response time statistics and patterns

## Running Strategy Tests

### Quick Testing Commands

```bash
# Test all strategies with basic canned samples
python prompt_runner.py --data-source canned_basic_all --strategies all

# Test specific strategies with diverse samples and sampling
python prompt_runner.py --data-source canned_100_all --strategies baseline policy --sample-size 5

# Comprehensive evaluation with unified dataset
python prompt_runner.py --data-source unified --strategies all --sample-size 100

# Test single strategy quickly
python prompt_runner.py --data-source canned_basic_all --strategies baseline
```

### Connection and Metrics Testing

```bash
# Test Azure AI connection
python prompt_runner.py --test-connection

# Recalculate metrics from existing results
python prompt_runner.py --metrics-only --run-id run_20250920_015821

# Debug mode for detailed logging
python prompt_runner.py --debug --data-source canned_basic_all --strategies all
```

### Advanced Usage Examples

```bash
# Reproducible sampling with custom seed
python prompt_runner.py --data-source unified --sample-size 25 --random-seed 42 --strategies all

# Multi-model comparison
python prompt_runner.py --model gpt-oss-20b --data-source canned_100_all --strategies all --sample-size 10
python prompt_runner.py --model gpt-5 --data-source canned_100_all --strategies all --sample-size 10
```

## Performance Analysis Framework

### Key Metrics to Track

1. **Accuracy**: Overall correctness across all samples
2. **Precision**: True positive rate (how many predicted hate are actually hate)
3. **Recall**: Sensitivity (how many actual hate cases are caught)
4. **F1-Score**: Harmonic mean of precision and recall
5. **Response Time**: Average time per prediction
6. **Rationale Quality**: Consistency and clarity of explanations

### Analysis Dimensions

1. **Strategy Comparison**: Relative performance across approaches
2. **Target Group Analysis**: Performance variation by identity group
3. **Sample Complexity**: Easy vs. challenging case performance
4. **Response Time Patterns**: Efficiency across strategies
5. **Rationale Consistency**: Quality of model explanations

## Expected Usage Patterns

### Development and Testing
```bash
# Quick validation during development
python prompt_runner.py --data-source canned_basic_all --strategies baseline

# Strategy comparison testing
python prompt_runner.py --data-source canned_100_all --strategies all --sample-size 10
```

### Comprehensive Evaluation
```bash
# Full strategy evaluation
python prompt_runner.py --data-source unified --strategies all --sample-size 200

# Model comparison testing
python prompt_runner.py --model gpt-5 --data-source unified --strategies all --sample-size 100
```

### Performance Monitoring
```bash
# Metrics recalculation for analysis
python prompt_runner.py --metrics-only --run-id [specific_run_id]

# Baseline performance tracking
python prompt_runner.py --data-source canned_100_all --strategies baseline --sample-size 25
```

## Architecture Benefits for Testing

### Incremental Storage
- Memory-efficient processing of large datasets
- Real-time result availability during long runs
- Robust handling of interrupted processes

### runId Organization
- Clear separation of test runs
- Easy comparison across different configurations
- Audit trail for all experiments

### Flexible Sampling
- Quick testing with small samples
- Comprehensive evaluation with large samples
- Reproducible results with seed control

### Multi-Model Support
- Easy comparison across different models
- YAML-based configuration management
- Environment variable support for deployment

## Next Steps for Testing

1. **Baseline Establishment**: Run comprehensive tests with all strategies on `canned_100_all`
2. **Large-Scale Validation**: Execute unified dataset testing with 500+ samples
3. **Performance Analysis**: Compare strategies across different target groups
4. **Model Comparison**: Test GPT-5 vs GPT-OSS-20B performance
5. **Rationale Analysis**: Evaluate explanation quality across strategies
6. **Optimization**: Fine-tune prompt templates based on results

## Updating This Document

To populate this document with actual results:

1. Run strategy tests with desired configuration
2. Analyze output files in the generated runId folder
3. Extract key metrics and findings from `performance_metrics.csv`
4. Review individual predictions in `validation_results.csv`
5. Summarize insights from `evaluation_report.txt`
6. Update this document with concrete results and analysis

---

*This document serves as a template and will be updated with actual test results as comprehensive evaluations are completed.*