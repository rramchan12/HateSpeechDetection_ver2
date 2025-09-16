# Strategy Test Results

## Overview

This document contains the latest test results for all four prompt strategies evaluated against the GPT-OSS-20B model using our unified hate speech dataset.

## Test Configuration

- **Model**: GPT-OSS-20B via Azure AI Inference
- **Test Samples**: 5 diverse examples from unified dataset
- **Strategies Tested**: Baseline, Persona, Policy, Combined
- **Evaluation Metrics**: Accuracy, Response Time, Individual Prediction Analysis

## Latest Test Run Results

### Test Samples Used

1. **Sample 1** (Label: hate)
   - Text: "LGBTQ people are destroying traditional values and shouldn't be allowed near children"
   - Target Group: LGBTQ

2. **Sample 2** (Label: normal)
   - Text: "I support equal rights for all people regardless of their background"
   - Target Group: General

3. **Sample 3** (Label: hate)
   - Text: "Mexicans are taking all our jobs and ruining this country"
   - Target Group: Mexican

4. **Sample 4** (Label: normal)
   - Text: "Learning about different cultures helps us become more understanding"
   - Target Group: General

5. **Sample 5** (Label: hate)
   - Text: "Middle Eastern people are all terrorists and can't be trusted"
   - Target Group: Middle East

### Strategy Performance Summary

| Strategy | Accuracy | Precision | Recall | F1-Score | Avg Response Time |
|----------|----------|-----------|--------|----------|------------------|
| Baseline | TBD% | TBD | TBD | TBD | TBD ms |
| Persona | TBD% | TBD | TBD | TBD | TBD ms |
| Policy | TBD% | TBD | TBD | TBD | TBD ms |
| Combined | TBD% | TBD | TBD | TBD | TBD ms |

*Results will be updated after next test run with `python runner.py --test-strategy all`*

### Individual Prediction Results

#### Baseline Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: normal)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: normal)
- Sample 5: TBD (Expected: hate)

#### Persona Strategy  
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: normal)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: normal)
- Sample 5: TBD (Expected: hate)

#### Policy Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: normal)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: normal)
- Sample 5: TBD (Expected: hate)

#### Combined Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: normal)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: normal)
- Sample 5: TBD (Expected: hate)

## Analysis

### Key Findings
- Analysis will be updated after next test run
- Best performing strategy: TBD
- Most challenging samples: TBD
- Response time insights: TBD
- Rationale quality: TBD

### Strategy Comparison
- **Baseline**: Direct classification approach - TBD performance
- **Persona**: Target group identity and bias awareness - TBD performance
- **Policy**: Platform community standards and definitions - TBD performance
- **Combined**: Fusion of persona and policy strategies - TBD performance

## Output Files Generated

All test runs generate timestamped files in `validation_outputs/`:

1. **Detailed Results**: `strategy_unified_results_TIMESTAMP.csv`
   - Individual sample results for each strategy
   - Columns: strategy, sample_id, text, true_label, predicted_label, response_time, rationale

2. **Performance Metrics**: `performance_metrics_TIMESTAMP.csv`
   - Strategy performance comparison with confusion matrix data
   - Columns: strategy, accuracy, precision, recall, f1_score, true_positive, true_negative, false_positive, false_negative

3. **Test Samples**: `test_samples_TIMESTAMP.csv`
   - Test samples used with unified format
   - Columns: text, label_binary, target_group_norm, persona_tag, source_dataset

4. **Human-Readable Report**: `evaluation_report_TIMESTAMP.txt`
   - Test samples with true labels
   - Individual prediction results with clear indicators
   - Strategy performance summary and analysis

## Running Strategy Tests

To generate updated results and populate this document:

```bash
# Run comprehensive strategy testing with default settings
python runner.py

# Run with specific strategies and sample size
python runner.py --test-strategy all --sample-size 10

# Test single strategy quickly
python runner.py --test-prompt baseline

# Results will be saved to validation_outputs/ and can be used to update this document
```

## CLI Commands Summary

- `python runner.py --test-connection`: Test Azure AI endpoint connectivity
- `python runner.py --test-prompt <strategy>`: Quick single-sample test
- `python runner.py --test-strategy all`: Comprehensive evaluation with metrics
- `python runner.py`: Default comprehensive evaluation (20 samples)

## Next Steps

1. Run `python runner.py --test-strategy all` to generate updated results
2. Analyze performance patterns across different target groups
3. Investigate any prediction inconsistencies and rationale quality
4. Consider fine-tuning prompt templates based on results
5. Expand test sample size for more robust evaluation
6. Compare rationale quality across strategies

---

*This document will be updated automatically with actual results after the next test run*