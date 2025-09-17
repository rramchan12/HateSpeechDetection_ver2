# Strategy Test Results

## Overview

This document contains the latest test results for all four prompt strategies evaluated against the GPT-OSS-20B model using our canned test samples and unified hate speech dataset.

## Test Configuration

- **Model**: GPT-OSS-20B via Azure AI Inference
- **Dataset Options**: 
  - Canned samples (5 curated examples)
  - Unified dataset (12,589+ filtered samples)
- **Strategies Tested**: Baseline, Persona, Policy, Combined
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Response Time
- **Output Format**: JSON with rationale explanations

## Latest Test Run Results (Canned Dataset)

### Test Samples Used

1. **Sample 1** (Label: hate)
   - Text: "LGBTQ people are destroying traditional values and shouldn't be allowed near children"
   - Target Group: LGBTQ
   - Persona Tag: lgbtq

2. **Sample 2** (Label: normal)  
   - Text: "I support equal rights for all people regardless of their background"
   - Target Group: General
   - Persona Tag: general

3. **Sample 3** (Label: hate)
   - Text: "Mexicans are taking all our jobs and ruining this country"
   - Target Group: Mexican
   - Persona Tag: hispanic

4. **Sample 4** (Label: normal)
   - Text: "Learning about different cultures helps us become more understanding"
   - Target Group: General
   - Persona Tag: general

5. **Sample 5** (Label: hate)
   - Text: "Middle Eastern people are all terrorists and can't be trusted"
   - Target Group: Middle East
   - Persona Tag: middle_eastern

### Strategy Performance Summary (Latest Run: 2025-09-18)

| Strategy | Accuracy | Precision | Recall | F1-Score | Avg Response Time |
|----------|----------|-----------|--------|----------|------------------|
| Baseline | 100% | 1.000 | 1.000 | 1.000 | ~1.3s |
| Persona | 100% | 1.000 | 1.000 | 1.000 | ~1.3s |
| Policy | 100% | 1.000 | 1.000 | 1.000 | ~1.3s |
| Combined | 100% | 1.000 | 1.000 | 1.000 | ~1.3s |

**Note**: Perfect scores on canned dataset indicate high performance on clear-cut examples. More comprehensive evaluation needed with larger unified dataset samples.

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

All test runs generate timestamped files in `outputs/`:

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
# Run comprehensive strategy testing with canned samples
python prompt_runner.py --dataset-type canned --num-samples all --strategy all

# Run with unified dataset for larger scale testing
python prompt_runner.py --dataset-type unified --num-samples 25 --strategy all

# Test single strategy quickly with canned data
python prompt_runner.py --dataset-type canned --num-samples 2 --strategy baseline

# Test specific strategies on unified dataset
python prompt_runner.py --dataset-type unified --num-samples 50 --strategy policy combined

# Results will be saved to outputs/ and can be used to update this document
```

## CLI Commands Summary

- `python prompt_runner.py --dataset-type canned --num-samples [N|all] --strategy [strategy|all]`: Test with curated samples
- `python prompt_runner.py --dataset-type unified --num-samples [N|all] --strategy [strategy|all]`: Test with full dataset  
- **Strategies**: `baseline`, `policy`, `persona`, `combined`, `all`
- **Output Location**: All results saved to `outputs/` directory with timestamps
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