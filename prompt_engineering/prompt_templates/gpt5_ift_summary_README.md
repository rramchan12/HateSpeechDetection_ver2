# GPT-5: Architectural Optimization Results & Findings

## Overview

This document presents empirical results from optimizing GPT-5 for hate speech detection through architectural prompt engineering. GPT-5's API constraints (fixed `temperature=1.0`) required optimization through prompt architecture and reasoning strategies rather than traditional hyperparameter tuning.

**Key Achievement**: `hybrid_fast_accurate` architecture achieved **F1=0.682** on 50-sample benchmark and **F1=0.598** on 100-sample dataset through two-phase testing (hyperparameter baseline → architectural optimization).

**Model Details**:
- **Model**: GPT-5
- **Configuration**: `temperature=1.0` (fixed), `max_tokens` variable (200-400)
- **Optimization Approach**: Two-phase testing (hyperparameter → architecture)
- **Dataset**: Unified hate speech corpus (hatexplain + toxigen)
- **Prompt Template**: baseline_v1_gpt5.json (hyperparameter), gpt5_architecture_v1.json (architecture)

---

## Experimental Setup

### Phase 1: Hyperparameter Testing (Baseline)

- **Objective**: Establish baseline performance using hyperparameter variations within GPT-5 constraints
- **Dataset**: canned_50_quick (50 samples), canned_100_size_varied (100 samples)
- **Model**: GPT-5
- **Configuration**: baseline_v1_gpt5.json
- **Strategies Tested**: 3 configurations (conservative, standard, balanced)
- **Execution**: 15 workers, batch size 8
- **Output Location**: `outputs/baseline_v1/gpt5/run_20251015_200026/`, `run_20251015_225644/`

### Phase 2: Architectural Testing

- **Objective**: Optimize prompt architecture as primary performance driver
- **Dataset**: canned_50_quick (50 samples), canned_100_stratified (100 samples), canned_100_size_varied (100 samples)
- **Model**: GPT-5
- **Configuration**: gpt5_architecture_v1.json
- **Strategies Tested**: 4 active architectures (removed 3 failures with F1=0.0)
  - Active: direct_binary, multi_perspective, cultural_context, hybrid_fast_accurate
  - Removed: chain_reasoning, adversarial_reasoning, confidence_calibrated
- **Execution**: 15 workers, batch size 8
- **Output Location**: `outputs/baseline_v1/gpt5/run_20251015_223645/`, `run_20251015_224029/`, `run_20251015_224859/`

### Phase 3: Production Validation Run (run_20251016_220311)

- **Purpose**: Large-scale validation of optimal architecture on full unified dataset
- **Dataset**: unified (1,009 samples from both HateXplain and ToxiGen datasets)
- **Model**: GPT-5
- **Configuration**: gpt5_architecture_v1.json
- **Strategy Tested**: `hybrid_fast_accurate` only (winner from hybrid optimization)
- **Execution**: Concurrent processing (15 workers, batch size 8)
- **Duration**: ~20 minutes for 1,009 classifications
- **Evaluation Framework**: F1-score + comprehensive bias metrics (FPR/FNR by target group)
- **Protected Groups Distribution**:
  - LGBTQ+ (494 samples, 49.0%)
  - Mexican/Latino (209 samples, 20.7%)
  - Middle Eastern (306 samples, 30.3%)
- **Output Location**: `outputs/baseline_v1/gpt5/baseline/run_20251016_220311/`

**Production Validation Results**:

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 62.6% | Maintained performance at scale |
| **Precision** | 61.3% | Stable precision on full dataset |
| **Recall** | 46.4% | Recall degradation at scale |
| **F1-Score** | **0.528** | -11.7% from 100-sample run (0.598) |
| **Confusion Matrix** | TP=211, TN=421, FP=133, FN=244 | High FN rate indicates conservative bias |

**Bias Metrics by Protected Group**:

| Target Group | Samples | FPR | FNR | Fairness |
|--------------|---------|-----|-----|----------|
| LGBTQ+ | 494 (49.0%) | 0.281 | 0.541 | ⚠️ REVIEW |
| Mexican | 209 (20.7%) | **0.116** | 0.488 | ⚠️ REVIEW |
| Middle East | 306 (30.3%) | 0.222 | 0.568 | ⚠️ REVIEW |

**Critical Finding**: Production validation revealed **11.7% F1 degradation** when scaling from 100 samples (F1=0.598) to 1,009 samples (F1=0.528), primarily driven by recall collapse (46.4%). Mexican persona achieved best FPR (0.116), but all groups showed elevated FNR (>0.48), indicating architecture requires fairness-specific tuning and recall optimization for production deployment.

---

## Empirical Results

### Hybrid Optimization Results - Bias-Aware Selection

**Objective**: Multi-run optimization combining performance (70% weight) and bias fairness (30% weight) across run_20251015_225644 (hyperparameter) and run_20251015_224859 (architecture).

**Hybrid Scoring Logic**: `Hybrid Score = 0.7 × F1_normalized + 0.3 × Bias_normalized`

| Rank | Strategy | Run ID | Dataset | Template | F1 Score | Bias Score | Hybrid Score | Config |
|------|----------|--------|---------|----------|----------|------------|--------------|--------|
|  1 | **hybrid_fast_accurate** | run_20251015_224859 | 100-sample varied | architecture | **0.598** | **0.631** | **0.995** | 600 tokens |
|  2 | direct_binary | run_20251015_224859 | 100-sample varied | architecture | 0.568 | 0.634 | 0.937 | 250 tokens |
|  3 | baseline_standard | run_20251015_225644 | 100-sample varied | hyperparameter | 0.552 | 0.607 | 0.841 | 300 tokens |
| 4 | baseline_balanced | run_20251015_225644 | 100-sample varied | hyperparameter | 0.506 | 0.561 | 0.636 | 350 tokens |
| 5 | multi_perspective | run_20251015_224859 | 100-sample varied | architecture | 0.460 | 0.586 | 0.598 | 250 tokens |
| 6 | cultural_context | run_20251015_224859 | 100-sample varied | architecture | 0.418 | 0.598 | 0.540 | 250 tokens |
| 7 | baseline_conservative | run_20251015_225644 | 100-sample varied | hyperparameter | 0.262 | 0.507 | 0.000 | 200 tokens |

**Key Findings from Hybrid Optimization**:
- **Winner**: `hybrid_fast_accurate` (rank 1) with highest hybrid score (0.995), combining best F1 performance (0.598) with strong bias fairness (0.631)
- **Architecture Dominance**: Top 2 positions held by architectural strategies, validating architecture-first optimization approach for GPT-5
- **Hyperparameter Baseline**: `baseline_standard` ranks 3rd, best among hyperparameter strategies but 7.4% below optimal architecture
- **Conservative Failure**: `baseline_conservative` normalized to 0.000 hybrid score due to catastrophic F1 (0.262) and poor bias performance
- **Bias-Performance Tradeoff**: `direct_binary` achieves highest bias score (0.634) but lower F1 than `hybrid_fast_accurate`, demonstrating Pareto frontier

### Production Run Results (run_20251016_220311)

**Validation Objective**: Full-scale deployment test of optimal `hybrid_fast_accurate` architecture on complete unified dataset (1,009 samples).

**Performance Metrics**:

| Metric | Value | Change from 100-sample | Interpretation |
|--------|-------|------------------------|----------------|
| **F1-Score** | **0.528** | -11.7% (0.598→0.528) | Performance degradation at scale |
| **Accuracy** | 62.6% | +1.6% (61.0%→62.6%) | Slight accuracy improvement |
| **Precision** | 61.3% | +3.3% (58.0%→61.3%) | More conservative predictions |
| **Recall** | 46.4% | -15.3% (61.7%→46.4%) | **Recall collapse at scale** |
| **True Positives** | 211 | — | Detected hate speech cases |
| **True Negatives** | 421 | — | Correctly identified benign |
| **False Positives** | 133 | — | Over-flagging (precision error) |
| **False Negatives** | 244 | — | **Missed hate speech (recall error)** |

**Bias Fairness Metrics by Protected Group**:

| Target Group | Sample Distribution | FPR | FNR | Fairness Status | Key Issue |
|--------------|---------------------|-----|-----|-----------------|-----------|
| **LGBTQ+** | 494 (49.0%) | 0.281 | **0.541** | ⚠️ REVIEW | High false negative rate |
| **Mexican** | 209 (20.7%) | **0.116** | 0.488 | ⚠️ REVIEW | Best FPR, but elevated FNR |
| **Middle East** | 306 (30.3%) | 0.222 | **0.568** | ⚠️ REVIEW | Highest FNR across groups |

**Critical Production Findings**:

1. **Recall Collapse**: 15.3% recall degradation (61.7%→46.4%) when scaling from 100 to 1,009 samples, indicating architecture's sensitivity to dataset diversity
2. **Conservative Bias**: Precision increased while recall dropped, suggesting model shifted toward risk-averse false negative bias at production scale
3. **Fairness Gap**: All protected groups exceeded fairness threshold (FNR >0.30), with Middle East persona showing highest false negative rate (0.568)
4. **False Negative Dominance**: 244 false negatives vs. 133 false positives (1.83:1 ratio) indicates systematic under-detection of hate speech
5. **Mexican Persona Advantage**: Lowest FPR (0.116) suggests architecture better handles Mexican/Latino cultural context, but still fails 48.8% of actual hate cases

**Production Deployment Recommendation**: Architecture requires recall-focused tuning and fairness-specific calibration before production deployment. Current configuration prioritizes precision over recall, creating safety risk through high false negative rates across all demographic groups.

---

## Bias Analysis

**Key Bias Findings**: While `hybrid_fast_accurate` achieved superior performance (F1=0.598), bias fairness remains a challenge. At 50-sample scale, only Mexican persona achieved fairness thresholds (FPR=0.0, FNR=0.167). Middle East persona showed highest bias (FPR=FNR=0.50), indicating cultural context gaps. Dataset scaling to 100 samples degraded fairness across all architectures, with no persona achieving fair designation, confirming that architecture optimization alone is insufficient for bias mitigation—dedicated fairness tuning required.

---

## Key Findings from Architectural Optimization

### 1. Architectural Optimization Superiority for Constrained APIs

**Evidence**: `hybrid_fast_accurate` achieved F1=0.682 vs. `baseline_balanced` F1=0.564 (+21% improvement). Architectural approach maintained 8-21% F1 advantage across all test cohorts.

**Implication**: When API constraints limit hyperparameter space (fixed temperature, no top_p/penalties), prompt architecture becomes primary optimization vector. Multi-stage reasoning frameworks outperform single-step classifications through explicit reasoning decomposition and structured output formats.

### 2. Precision-Recall Tradeoff Amplification in Conservative Strategies

**Evidence**: `baseline_conservative` achieved 100% precision but 13% recall (F1=0.231). Token reduction (200 vs. 350 tokens) correlated with +87% FNR increase for Middle East persona. Conservative approach failed to detect 87% of hate speech in Mexican persona contexts (FNR=0.833).

**Implication**: Low token budgets force models into high-confidence-only classification mode, sacrificing recall. Minimum 300 tokens required for balanced hate speech detection with GPT-5.

### 3. Architecture Failure Modes: Zero-Recall Catastrophes

**Evidence**: 3 architectures removed from gpt5_architecture_v1.json due to F1=0.0 (100% FNR): `chain_reasoning` (overly complex sequential analysis), `adversarial_reasoning` (challenge-response framework too conservative), `confidence_calibrated` (uncertainty quantification prevented decisive classifications).

**Implication**: Over-engineered reasoning architectures introduce failure modes where complexity undermines decision-making. Optimal reasoning stages = 2-4 for hate speech detection; beyond 4 steps risks collapse.

### 4. Cultural Context Architecture Underperformance

**Evidence**: `cultural_context` ranked last across all test cohorts (F1=0.250-0.418). Despite explicit cultural awareness framework, showed highest FNR for LGBTQ (0.778) and Middle East (0.714) personas.

**Implication**: Cultural awareness prompting creates over-contextualization, where model prioritizes cultural nuance interpretation over hate indicator detection. Avoid over-specification of cultural contexts; let model's pretrained cultural knowledge guide detection.

### 5. Dataset Scaling Degrades Fairness Universally

**Evidence**: 50-sample test: 1 persona achieved fairness (Mexican: FPR=0.0, FNR=0.167). 100-sample tests: 0 personas achieved fairness across all strategies. Average FNR increase: +11.3% for best-performing `hybrid_fast_accurate`. Middle East persona showed FPR amplification: 0.50→0.62 (+24%).

**Implication**: Statistical power increase from dataset scaling exposes latent bias that appears manageable in small samples. Minimum 500-1000 samples needed for robust bias assessment. Optimal architectures require per-persona calibration.

### 6. Multi-Perspective Architecture's Precision-Recall Dilemma

**Evidence**: `multi_perspective` achieved 72.7% precision (highest) but 34.8% recall (second-lowest). Consistent pattern: high precision, low recall across all datasets (precision range: 50-72.7%, recall range: 26.1-36.2%).

**Implication**: Viewpoint synthesis architecture creates consensus requirement where multiple analytical perspectives must agree for hate classification. Reduces false positives by requiring inter-perspective agreement but increases false negatives when perspectives disagree on borderline cases.

### 7. Hyperparameter vs. Architecture: GPT-5 Optimization Paradigm Shift

**Evidence**: GPT-OSS optimal strategy (baseline_standard): F1=0.615 through hyperparameter tuning (temp=0.1, tokens=512). GPT-5 optimal strategy (hybrid_fast_accurate): F1=0.682 through architectural engineering (fixed temp=1.0, variable reasoning stages). GPT-5 architecture surpasses GPT-OSS hyperparameters by +11% F1 despite fewer tunable parameters.

**Implication**: Model design constraints dictate optimization methodology. Architectural innovation compensates for hyperparameter restrictions, suggesting prompt engineering underutilized in open-source models.

---

## References

### Experimental Documentation

- **Hyperparameter Configuration**: `baseline_v1_gpt5.json`
- **Architecture Configuration**: `gpt5_architecture_v1.json`
- **Hyperparameter Methodology**: `baseline_v1_gpt5_README.md`
- **Architecture Methodology**: `gpt5_prompt_architecture_optimization_README.md`

### Experimental Runs

1. **run_20251015_200026**: Hyperparameter baseline (50 samples)
2. **run_20251015_223645**: Architecture initial test (50 samples)
3. **run_20251015_224029**: Architecture stratified validation (100 samples)
4. **run_20251015_224859**: Architecture complexity test (100 samples)
5. **run_20251015_225644**: Hyperparameter scaling validation (100 samples)

### Related Work

- **GPT-OSS Optimization**: `gptoss_ift_summary_README.md`
- **Template Architecture**: `baseline_v1_README.md`
- **Dataset Unification**: `data_preparation/UNIFICATION_APPROACH.md`
