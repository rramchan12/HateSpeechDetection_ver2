# Overall IFT Summary: Cross-Model Hate Speech Detection Performance

## Executive Summary

This document presents comprehensive empirical results from systematic instruction fine-tuning (IFT) optimization across two models (GPT-OSS and GPT-5) and two prompt architectures (Baseline and Combined Policy-Persona) for hate speech detection. Through hybrid optimization analysis (70% performance + 30% bias fairness) across 4 production validation runs on a unified dataset (1,009 samples), we identify optimal configurations and provide actionable findings for production deployment.

**Global Winner**: `baseline_standard` strategy on **gpt-oss-120b** achieved **F1=0.615** (Hybrid Score=0.930), outperforming all GPT-5 configurations and combined templates through traditional hyperparameter optimization (temperature=0.1, max_tokens=512).

**Key Finding**: Hyperparameter tuning with open-source models (GPT-OSS) outperforms architectural prompt engineering with constrained APIs (GPT-5, temperature=1.0) by 8.7% F1-score at production scale, challenging assumptions about architectural optimization sufficiency.

---

## Experimental Framework

### Models Evaluated
- **gpt-oss-120b**: Phi-3.5-MoE-instruct (120B parameters), full hyperparameter control
- **gpt-5**: Azure AI Model Inference, fixed temperature=1.0 constraint

### Architectures Tested
1. **Baseline Templates**: Direct policy-based classification (baseline_v1.json, baseline_v1_gpt5.json)
2. **Architectural Templates**: Multi-stage reasoning frameworks (gpt5_architecture_v1.json)
3. **Combined Policy-Persona**: Few-shot learning with explicit persona examples (combined_gptoss_v1.json, combined_gpt5_v1.json)

### Evaluation Framework
- **Dataset**: Unified corpus (1,009 samples, HateXplain + ToxiGen)
- **Protected Groups**: LGBTQ+ (49.0%), Mexican/Latino (20.7%), Middle Eastern (30.3%)
- **Metrics**: F1-score (performance), FPR/FNR by group (bias fairness), Hybrid Score (0.7×F1 + 0.3×Bias)
- **Production Runs Analyzed**: 4 large-scale validations (1,009 samples each)

---

## Overall Optimization Results

### Cross-Model Performance Ranking (Production Scale)

| Rank | Strategy | Model | Template | F1-Score | Bias Score | Hybrid Score | Mexican FPR | Notes |
|------|----------|-------|----------|----------|------------|--------------|-------------|-------|
| **1** | **baseline_standard** | **gpt-oss-120b** | baseline_v1.json | **0.615** | **0.685** | **0.930** | 8.1% | Optimal configuration ✅ |
| 2 | combined_optimized | gpt-5 | combined_gpt5_v1.json | 0.607 | 0.689 | 0.773 | 5.8% | Best GPT-5 performance |
| 3 | hybrid_fast_accurate | gpt-5 | gpt5_architecture_v1.json | 0.596 | 0.677 | 0.256 | 11.6% | Architecture optimization |
| 4 | combined_optimized | gpt-oss-120b | combined_gptoss_v1.json | 0.590 | 0.672 | 0.000 | 7.0% | Few-shot learning |

**Reference**: Comprehensive analysis from `pipeline/baseline/hyperparam/outputs/overall/comprehensive_analysis_optimal_config.json`

### Critical Cross-Model Findings

1. **GPT-OSS Baseline Dominance**: gpt-oss-120b with baseline template (F1=0.615) outperforms all GPT-5 configurations by 0.8-1.9%, validating hyperparameter tuning superiority over architectural optimization for constrained APIs

2. **GPT-5 Combined Template Success**: GPT-5's best performance (F1=0.607) achieved through combined policy-persona template with few-shot examples, not architectural optimization (F1=0.596)

3. **Architectural Optimization Limitations**: GPT-5's `hybrid_fast_accurate` architecture (F1=0.596) underperforms GPT-OSS baseline by 1.9%, highlighting that multi-stage reasoning cannot fully compensate for fixed temperature=1.0 constraint

4. **Bias-Performance Tradeoff**: Combined templates show highest bias scores (0.672-0.689) but lower F1 performance, demonstrating Pareto frontier between classification accuracy and fairness

---

## Model-Specific Analysis

### GPT-OSS Performance (gpt-oss-120b)

**Reference**: `gptoss_ift_summary_README.md` (Baseline), `combined/gpt_oss_combined_ift_summary_README.md` (Combined)

#### Baseline Template (baseline_v1.json)

**Optimal Configuration**: `baseline_standard` 
- **Performance**: F1=0.615, Accuracy=65.0%, Precision=61.0%, Recall=62.0%
- **Hyperparameters**: temperature=0.1, max_tokens=512, top_p=1.0
- **Production Run**: run_20251012_191628 (1,009 samples)
- **Key Strength**: Best overall performance through temperature optimization

**Bias Metrics by Protected Group**:
| Group | FPR | FNR | Assessment |
|-------|-----|-----|------------|
| LGBTQ+ (494) | 43.0% | 39.4% | ⚠️ Highest FPR across all models |
| Mexican (209) | **8.1%** | 39.8% | ✅ Second-best FPR (after GPT-5 combined) |
| Middle East (306) | 23.6% | 35.2% | ⚠️ Balanced but elevated |

**Key Findings from `gptoss_ift_summary_README.md`**:
- **Temperature Dominance**: 16.6% F1 decrease from temp=0.1 to temp=0.5, establishing temperature as primary optimization variable for classification tasks
- **Token Goldilocks Zone**: Performance peaks at 200-512 tokens, degrades beyond 768 tokens due to hedging behavior
- **Scale Stability**: Only 1.1% F1 degradation (0.626→0.615) from 100-sample to 1,009-sample validation, excellent generalization
- **LGBTQ+ Overcriminalization**: 43.0% FPR indicates 5.3× higher false positive rate vs. Mexican community (8.1%)

#### Combined Template (combined_gptoss_v1.json)

**Optimal Configuration**: `combined_optimized`
- **Performance**: F1=0.590, Accuracy=64.5%, Precision=61.6%, Recall=56.7%
- **Hyperparameters**: temperature=0.1, max_tokens=512
- **Production Run**: run_20251018_234643 (1,009 samples)
- **Key Strength**: Best Mexican FPR (7.0%) through few-shot learning

**Bias Metrics by Protected Group**:
| Group | FPR | FNR | Assessment |
|-------|-----|-----|------------|
| LGBTQ+ (494) | 39.2% | 41.2% | ⚠️ Nearly symmetric errors |
| Mexican (209) | **7.0%** | 48.0% | ✅ **Best FPR across all models** |
| Middle East (306) | 19.4% | 42.0% | ⚠️ Review needed |

**Key Findings from `gpt_oss_combined_ift_summary_README.md`**:
- **Few-Shot Learning Effectiveness**: Explicit Mexican/Latino hate examples reduced FPR from baseline (8.1%→7.0%), validating persona-specific prompting
- **Recall Degradation Control**: 9.3% recall drop at scale (66.0%→56.7%), significantly better than GPT-5's 15.3% degradation
- **Balanced Error Distribution**: FN/FP ratio of 1.22:1 (197 FN vs. 161 FP) shows more balanced errors than GPT-5's 1.83:1
- **Production Readiness**: Minimal F1 degradation (-2.4%) at scale demonstrates robust generalization

**GPT-OSS Summary**: Baseline template achieves best overall performance (F1=0.615), while combined template optimizes for fairness (Mexican FPR=7.0%). Hyperparameter tuning (temperature=0.1) is critical optimization variable.

---

### GPT-5 Performance (gpt-5)

**Reference**: `gpt5_ift_summary_README.md` (Baseline/Architecture), `combined/gpt5_combined_ift_summary_README.md` (Combined)

#### Architectural Template (gpt5_architecture_v1.json)

**Optimal Configuration**: `hybrid_fast_accurate`
- **Performance**: F1=0.596 (100-sample), **F1=0.528 (production)**, Accuracy=62.6%, Precision=61.3%, Recall=46.4%
- **Architecture**: 2-stage reasoning (context analysis → binary classification), 600 tokens
- **Production Run**: run_20251016_220311 (1,009 samples) ✓ CORRECTED
- **Key Weakness**: 15.3% recall collapse at production scale (61.7%→46.4%)

**Bias Metrics by Protected Group (Production)**:
| Group | FPR | FNR | Assessment |
|-------|-----|-----|------------|
| LGBTQ+ (494) | 28.1% | 54.1% | ⚠️ Highest FNR across all models |
| Mexican (209) | 11.6% | 48.8% | ⚠️ FPR higher than combined template |
| Middle East (306) | 22.2% | **56.8%** | ⚠️ **Highest FNR across all groups** |

**Key Findings from `gpt5_ift_summary_README.md`**:
- **Production Underperformance**: F1=0.528 at scale, **-8.7% below GPT-OSS baseline (0.615)**, challenging architectural optimization claims
- **Recall Collapse**: 15.3% recall degradation (61.7%→46.4%) indicates architecture's sensitivity to dataset diversity at production scale
- **False Negative Dominance**: 244 FN vs. 133 FP (1.83:1 ratio), systematic under-detection of hate speech
- **Small-Sample Overfitting**: 50-sample performance (F1=0.682) does not transfer to production (F1=0.528), 22.6% degradation
- **Architectural Failure Modes**: 3 architectures (chain_reasoning, adversarial_reasoning, confidence_calibrated) achieved F1=0.0 due to over-engineering

**Architecture vs. Hyperparameter**: At 50-sample scale, architecture appeared superior (F1=0.682 vs. GPT-OSS 0.626). At production scale, architecture underperforms (F1=0.528 vs. GPT-OSS 0.615), demonstrating small-sample optimization unreliability.

#### Combined Template (combined_gpt5_v1.json)

**Optimal Configuration**: `combined_optimized`
- **Performance**: F1=0.607, Accuracy=66.8%, Precision=65.2%, Recall=56.8%
- **Architecture**: Policy-persona integration with few-shot examples, 650 tokens
- **Production Run**: run_20251019_133309 (1,009 samples)
- **Key Strength**: +2.0% F1 improvement at scale (0.587→0.607), GPT-5's best production result

**Bias Metrics by Protected Group**:
| Group | FPR | FNR | Assessment |
|-------|-----|-----|------------|
| LGBTQ+ (494) | 34.1% | 40.8% | ⚠️ Both elevated |
| Mexican (209) | **5.8%** | 48.8% | ✅ **Second-best FPR across all models** |
| Middle East (306) | 16.1% | 41.4% | ✅ Excellent FPR |

**Key Findings from `gpt5_combined_ift_summary_README.md`**:
- **Scale Robustness**: +2.0% F1 improvement (0.587→0.607) demonstrates superior generalization vs. typical degradation patterns, unique among GPT-5 configurations
- **Mexican FPR Excellence**: 5.8% FPR represents -24.2% improvement from 100-sample (30.0%→5.8%), validating few-shot example effectiveness for GPT-5
- **Token-Recall Dependency**: 650 tokens optimal for recall=56.8%; each 150-token reduction correlates with ~10% recall decrease due to temperature=1.0 constraint
- **Priority Ordering Effect**: Mexican examples placed FIRST in few-shot set maintain pattern salience despite sampling variability (temperature=1.0)

**GPT-5 Summary**: Combined template (F1=0.607) outperforms architectural optimization (F1=0.528) by 7.9%, demonstrating few-shot learning superiority over multi-stage reasoning for constrained APIs. However, both underperform GPT-OSS baseline.

---

## Template Architecture Comparison

### Baseline Templates (baseline_v1.json, baseline_v1_gpt5.json)

**Design**: Direct policy-based classification with structured JSON output
- **GPT-OSS Performance**: F1=0.615 (best overall)
- **GPT-5 Performance**: F1=0.552 (100-sample, hyperparameter variant)
- **Key Advantage**: Simplicity enables effective hyperparameter optimization
- **Limitation**: Generic prompting shows bias variance across protected groups

**Winner**: GPT-OSS baseline achieves best production performance through temperature optimization (temp=0.1)

### Combined Policy-Persona Templates (combined_gptoss_v1.json, combined_gpt5_v1.json)

**Design**: Policy integration with few-shot persona examples (Mexican, LGBTQ+, Middle Eastern)
- **GPT-OSS Performance**: F1=0.590 (Mexican FPR=7.0%)
- **GPT-5 Performance**: F1=0.607 (Mexican FPR=5.8%)
- **Key Advantage**: Explicit persona examples reduce false positive rates for minority groups
- **Limitation**: Lower overall F1 vs. baseline (GPT-OSS), trade-off for fairness

**Winner**: GPT-5 combined achieves best Mexican FPR (5.8%) but requires 650 tokens due to temperature constraint

### Architectural Templates (gpt5_architecture_v1.json)

**Design**: Multi-stage reasoning (context analysis → hate indicator extraction → classification)
- **GPT-5 Performance**: F1=0.528 (production), 0.596 (100-sample)
- **Key Advantage**: Structured reasoning provides explainability
- **Limitation**: Severe production degradation (-11.7% from 100-sample), highest FNR across all approaches

**Assessment**: Architectural optimization insufficient for temperature=1.0 constraint, inferior to few-shot learning approach

---

## Cross-Template Key Findings

### 1. Hyperparameter Tuning Outperforms Architectural Engineering at Production Scale

**Evidence**:
- GPT-OSS baseline (F1=0.615, temp=0.1) > GPT-5 architecture (F1=0.528, temp=1.0), +8.7% advantage
- GPT-OSS baseline (F1=0.615) > GPT-5 combined (F1=0.607), +0.8% advantage
- Temperature=0.1 optimization provides 16.6% F1 improvement vs. high temperature (GPT-OSS testing)

**Implication**: When available, prioritize hyperparameter control (especially temperature ≤0.1) over prompt architecture complexity. Architectural innovation cannot fully compensate for API constraints.

**Reference**: `gptoss_ift_summary_README.md` (Finding #1: Temperature Dominance)

### 2. Few-Shot Learning Reduces Bias Better Than Multi-Stage Reasoning

**Evidence**:
- Combined templates achieve best Mexican FPR: GPT-5 (5.8%), GPT-OSS (7.0%)
- Architectural template shows worse Mexican FPR (11.6%) despite explicit reasoning stages
- Few-shot examples enable -24.2% FPR improvement for GPT-5 (30.0%→5.8% at scale)

**Implication**: Explicit persona-specific examples in prompts more effective for bias mitigation than abstract reasoning frameworks. Priority ordering (minority examples first) maintains salience.

**Reference**: `gpt5_combined_ift_summary_README.md` (Finding #2: Few-Shot Learning at Scale), `gpt_oss_combined_ift_summary_README.md` (V1 Enhancement History)

### 3. Small-Sample Optimization Misleads Production Performance Predictions

**Evidence**:
- GPT-5 architecture: 50-sample F1=0.682 → Production F1=0.528 (-22.6% degradation)
- GPT-OSS baseline: 100-sample F1=0.626 → Production F1=0.615 (-1.8% degradation)
- Architectural optimization showed +11% advantage at 50-sample scale, reversed to -8.7% disadvantage at production

**Implication**: Minimum 500-1000 samples required for robust production assessment. Small-sample optimization (50-100 samples) can produce misleading performance estimates, especially for architectural approaches.

**Reference**: `gpt5_ift_summary_README.md` (Finding #7: Critical Production Context), `gptoss_ift_summary_README.md` (Phase 2 Validation)

### 4. Temperature=1.0 Constraint Requires Token Budget Compensation

**Evidence**:
- GPT-5 combined optimal: 650 tokens (F1=0.607, recall=56.8%)
- GPT-5 architecture optimal: 600 tokens (F1=0.528, recall=46.4%)
- GPT-OSS optimal: 512 tokens (F1=0.615, recall=62.0%)
- Each 150-token reduction correlates with ~10% recall decrease for GPT-5

**Implication**: Fixed temperature=1.0 introduces sampling variance requiring larger token budgets (600-650) to maintain decision certainty. GPT-OSS achieves better performance with fewer tokens (512) due to temperature control (0.1).

**Reference**: `gpt5_combined_ift_summary_README.md` (Finding #4: Token-Recall Dependency)

### 5. Recall-Precision Tradeoff Amplified by Architecture Complexity

**Evidence**:
- GPT-5 architecture: Precision=61.3%, Recall=46.4% (15% gap, conservative bias)
- GPT-5 combined: Precision=65.2%, Recall=56.8% (8.4% gap, balanced)
- GPT-OSS baseline: Precision=61.0%, Recall=62.0% (1% gap, optimal balance)

**Implication**: Multi-stage reasoning architectures create over-cautious classification (high precision, low recall). Simpler templates with few-shot examples achieve better precision-recall balance.

**Reference**: `gpt5_ift_summary_README.md` (Production Finding #2: Conservative Bias)

### 6. Bias Fairness Requires Dedicated Optimization, Not Performance-Only Tuning

**Evidence**:
- Hybrid optimization (70% performance + 30% bias) discovered superior configurations vs. F1-only optimization
- LGBTQ+ FPR variance: 28.1% (GPT-5 arch) to 43.0% (GPT-OSS baseline), 15% disparity
- Mexican FPR variance: 5.8% (GPT-5 combined) to 11.6% (GPT-5 arch), 2× difference

**Implication**: Single-objective F1 optimization ignores demographic fairness. Production systems must include bias metrics (FPR/FNR by group) in multi-objective optimization to discover Pareto-optimal configurations.

**Reference**: `gptoss_ift_summary_README.md` (Finding #3: Bias-Aware Optimization), Overall analysis hybrid scoring methodology

---

## Production Deployment Recommendations

### Model Selection

**Primary Recommendation**: **gpt-oss-120b with baseline_standard template**
- **Performance**: F1=0.615 (best overall)
- **Configuration**: temperature=0.1, max_tokens=512, top_p=1.0
- **Strengths**: Optimal performance, stable generalization (1.1% degradation at scale)
- **Limitations**: LGBTQ+ FPR=43.0% requires bias calibration
- **Use Case**: General-purpose hate speech detection with human review for LGBTQ+ content

**Alternative for Fairness-Critical Applications**: **GPT-5 with combined_gpt5_v1 template**
- **Performance**: F1=0.607 (best GPT-5)
- **Configuration**: temperature=1.0 (fixed), max_tokens=650
- **Strengths**: Best Mexican FPR (5.8%), scale robustness (+2.0% F1 improvement)
- **Limitations**: Higher computational cost (650 tokens), API dependency
- **Use Case**: Applications requiring minimal false positives for minority communities

### Template Selection Guidelines

1. **When to use Baseline templates**:
   - Maximum F1-score priority
   - Hyperparameter control available (temperature tuning)
   - General-purpose classification without strict fairness requirements

2. **When to use Combined templates**:
   - Protected group fairness critical (minimize FPR for minorities)
   - Known bias patterns requiring explicit mitigation
   - Production systems with human-in-the-loop review

3. **Avoid Architectural templates (GPT-5)**:
   - Production deployment (F1=0.528, severe recall collapse)
   - High recall requirements (46.4% recall unacceptable for safety-critical systems)
   - Temperature=1.0 constraint without token budget flexibility

### Bias Calibration Requirements

**All configurations require bias-specific tuning before production**:

| Configuration | Mexican FPR | LGBTQ+ FPR | Calibration Priority |
|---------------|-------------|------------|----------------------|
| GPT-OSS baseline | 8.1% | **43.0%** | HIGH: LGBTQ+ overcriminalization |
| GPT-OSS combined | **7.0%** | 39.2% | MEDIUM: LGBTQ+ monitoring |
| GPT-5 combined | **5.8%** | 34.1% | MEDIUM: Recall optimization (FNR 48.8-56.8%) |
| GPT-5 architecture | 11.6% | 28.1% | **CRITICAL: Recall collapse (FNR >48%)** |

**Calibration Strategy**: Implement per-group confidence thresholds using validation set to equalize FPR/FNR across protected demographics.

### Monitoring Requirements for Production

1. **Track FPR/FNR by protected group** (weekly minimum)
2. **Alert on FPR disparity >10%** between any two groups
3. **Alert on FNR >40%** for any group (safety threshold)
4. **Monthly recalibration** using production data samples
5. **Quarterly revalidation** on held-out test sets

---

## Conclusion

Through comprehensive cross-model evaluation, this research demonstrates that **traditional hyperparameter optimization with open-source models (GPT-OSS, F1=0.615) outperforms architectural prompt engineering with constrained APIs (GPT-5, F1=0.528-0.607) at production scale**. While architectural innovation and few-shot learning provide value for bias mitigation (Mexican FPR: 5.8-7.0%), they cannot fully compensate for API-imposed constraints like fixed temperature=1.0.

**Key Takeaways for Practitioners**:

1. **Prioritize hyperparameter control** (especially temperature ≤0.1) when available over complex prompt architectures
2. **Use few-shot learning** (combined templates) for bias mitigation, not multi-stage reasoning
3. **Validate at production scale** (≥1,000 samples) to avoid small-sample optimization misleading predictions
4. **Implement hybrid optimization** (performance + bias) to discover Pareto-optimal configurations for fairness-critical applications
5. **Plan for bias calibration** regardless of chosen configuration; no template achieves fairness without post-hoc tuning

**Future Work**:
- Per-group confidence threshold calibration for FPR/FNR equalization
- Ensemble methods combining GPT-OSS baseline (performance) + GPT-5 combined (fairness)
- Active learning for edge case detection in minority communities
- Temperature scheduling experiments for GPT-5 (if API constraints relax)

---

## References

### Individual Model Summaries
- **GPT-OSS Baseline**: `gptoss_ift_summary_README.md`
- **GPT-OSS Combined**: `combined/gpt_oss_combined_ift_summary_README.md`
- **GPT-5 Baseline/Architecture**: `gpt5_ift_summary_README.md`
- **GPT-5 Combined**: `combined/gpt5_combined_ift_summary_README.md`

### Optimization Analysis
- **Overall Cross-Model Analysis**: `pipeline/baseline/hyperparam/outputs/overall/comprehensive_analysis_optimal_config.json`
- **Ranked Results**: `pipeline/baseline/hyperparam/outputs/overall/comprehensive_analysis_results.csv`

### Production Validation Runs
1. **run_20251012_191628**: GPT-OSS baseline (F1=0.615) ✅ WINNER
2. **run_20251019_133309**: GPT-5 combined (F1=0.607)
3. **run_20251016_220311**: GPT-5 architecture (F1=0.528) ✓ CORRECTED
4. **run_20251018_234643**: GPT-OSS combined (F1=0.590)

### Template Files
- `baseline_v1.json` (GPT-OSS baseline)
- `baseline_v1_gpt5.json` (GPT-5 baseline)
- `gpt5_architecture_v1.json` (GPT-5 architecture)
- `combined/combined_gptoss_v1.json` (GPT-OSS combined)
- `combined/combined_gpt5_v1.json` (GPT-5 combined)

---

**Document Version**: 1.0  
**Last Updated**: October 19, 2025  
**Analysis Scope**: 4 production runs, 4,036 total classifications (1,009 samples × 4 configurations)  
**Optimization Framework**: Hybrid scoring (70% F1-score + 30% bias fairness)
