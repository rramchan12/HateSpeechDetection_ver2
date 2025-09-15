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

2. **Sample 2** (Label: not_hate)
   - Text: "I support equal rights for all people regardless of their background"
   - Target Group: General

3. **Sample 3** (Label: hate)
   - Text: "Mexicans are taking all our jobs and ruining this country"
   - Target Group: Mexican

4. **Sample 4** (Label: not_hate)
   - Text: "Learning about different cultures helps us become more understanding"
   - Target Group: General

5. **Sample 5** (Label: hate)
   - Text: "Middle Eastern people are all terrorists and can't be trusted"
   - Target Group: Middle East

### Strategy Performance Summary

| Strategy | Accuracy | Correct Predictions | Total Samples | Avg Response Time |
|----------|----------|-------------------|---------------|------------------|
| Baseline | TBD% | TBD/5 | 5 | TBD ms |
| Persona | TBD% | TBD/5 | 5 | TBD ms |
| Policy | TBD% | TBD/5 | 5 | TBD ms |
| Combined | TBD% | TBD/5 | 5 | TBD ms |

*Results will be updated after next test run with `python runner.py --test-strategies`*

### Individual Prediction Results

#### Baseline Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: not_hate)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: not_hate)
- Sample 5: TBD (Expected: hate)

#### Persona Strategy  
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: not_hate)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: not_hate)
- Sample 5: TBD (Expected: hate)

#### Policy Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: not_hate)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: not_hate)
- Sample 5: TBD (Expected: hate)

#### Combined Strategy
- Sample 1: TBD (Expected: hate)
- Sample 2: TBD (Expected: not_hate)
- Sample 3: TBD (Expected: hate)
- Sample 4: TBD (Expected: not_hate)
- Sample 5: TBD (Expected: hate)

## Analysis

### Key Findings
- Analysis will be updated after next test run
- Best performing strategy: TBD
- Most challenging samples: TBD
- Response time insights: TBD

### Strategy Comparison
- **Baseline**: Direct classification approach - TBD performance
- **Persona**: Target group identity and bias awareness - TBD performance
- **Policy**: Platform community standards and definitions - TBD performance
- **Combined**: Fusion of persona and policy strategies - TBD performance

## Output Files Generated

All test runs generate timestamped files in `validation_outputs/`:

1. **Detailed Results**: `strategy_test_results_TIMESTAMP.csv`
   - Individual sample results for each strategy
   - Columns: timestamp, strategy, sample_id, input_text, predicted_label, true_label, response_time, response_text, correct

2. **Summary Metrics**: `strategy_summary_TIMESTAMP.csv`
   - Strategy performance comparison
   - Columns: strategy, accuracy, correct_predictions, total_samples, avg_response_time

3. **Human-Readable Report**: `strategy_report_TIMESTAMP.txt`
   - Test samples with true labels
   - Individual prediction results with ✓/✗ indicators
   - Strategy performance summary and best performer identification

## Running Strategy Tests

To generate updated results and populate this document:

```bash
# Run comprehensive strategy testing
python runner.py --test-strategies

# Results will be saved to validation_outputs/ and can be used to update this document
```

## Next Steps

1. Run `python runner.py --test-strategies` to generate updated results
2. Analyze performance patterns across different target groups
3. Investigate any prediction inconsistencies
4. Consider fine-tuning prompt templates based on results
5. Expand test sample size for more robust evaluation

---

*This document will be updated automatically with actual results after the next test run*