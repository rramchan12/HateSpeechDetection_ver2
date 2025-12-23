# Baseline Prompt Engineering Optimization Guide - baseline_v1.json

## Overview

This document describes the systematic hyperparameter optimization for hate speech detection using baseline prompt strategies. The optimization follows established methodologies for large language model calibration [2] and reproducible machine learning research [8].

**Key Achievement**: Through three-phase systematic testing—initial benchmarking (run_20251011_085450, 100 samples), automated optimization (run_20251012_133005, 100 samples), and large-scale validation (run_20251012_191628, 1,009 samples)—we identified and validated `baseline_standard` as the optimal configuration, achieving **F1-score of 0.615** on the full unified dataset with comprehensive bias analysis across protected demographic groups.

All strategies use identical core prompt structure but vary LLM sampling parameters (temperature, max_tokens, top_p, frequency_penalty, presence_penalty) to explore the performance-fairness trade-off space [2,3,5]. This ablation study approach enables systematic identification of optimal hyperparameter combinations for hate speech classification.

**For detailed empirical results and findings**: See companion document `gptoss_ift_summary_README.md`

## Core Baseline Prompt Structure

All variants use identical system and user prompts focused on simple, direct hate speech classification without external context (policy/persona information).

### System Prompt

- **Role**: Content moderation assistant
- **Task**: Binary classification (hate/normal)
- **Output Format**: Strict JSON with classification + rationale
- **Rules**: Exact label matching, brief explanations required
- **Context**: General hate speech understanding (no specialized domain knowledge)

### User Prompt Template

Simple, minimal context classification request:

```text
Classify the following text as either 'hate' or 'normal'.
Text: "{text}"
Respond in JSON format: {"classification": "[hate/normal]", "rationale": "[brief explanation]"}
```

## Hyperparameter Analysis & Strategy Matrix

| Strategy | max_tokens | temperature | top_p | freq_penalty | pres_penalty | Purpose | Expected Behavior |
|----------|------------|-------------|-------|--------------|--------------|---------|-------------------|
| **baseline_conservative** | 256 | 0.0 | 0.9 | 0.0 | 0.0 | Deterministic, consistent | Most reliable, lowest variance |
| **baseline_standard** | 512 | 0.1 | 1.0 | 0.0 | 0.0 | Balanced reliability | Original baseline reference |
| **baseline_creative** | 768 | 0.3 | 0.95 | 0.1 | 0.1 | Nuanced reasoning | Better edge case handling |
| **baseline_focused** | 200 | 0.05 | 0.8 | 0.2 | 0.0 | Concise, confident | Fast, decisive classifications |
| **baseline_exploratory** | 1024 | 0.5 | 0.85 | 0.3 | 0.2 | High diversity | Discovery of new patterns |
| **baseline_balanced** | 400 | 0.2 | 0.9 | 0.05 | 0.05 | OpenAI best practices | Optimal general performance |

## Parameter Impact Analysis

###  **Temperature** (Randomness/Creativity) [2,3]
- **0.0 (Conservative)**: Completely deterministic, most consistent responses [2]
- **0.05 (Focused)**: Near-deterministic with minimal variation
- **0.1 (Standard)**: Low randomness, reliable classifications [2]
- **0.2 (Balanced)**: Moderate creativity, optimal for classification tasks [2]
- **0.3 (Creative)**: Higher creativity for nuanced reasoning
- **0.5 (Exploratory)**: High creativity, may discover edge patterns

###  **top_p** (Nucleus Sampling) [3]
- **0.8 (Focused)**: Very focused token selection, conservative choices
- **0.9 (Conservative/Balanced)**: Balanced focus, excludes low-probability tokens [3]
- **0.95 (Creative)**: Allows more diverse token choices
- **1.0 (Standard)**: No nucleus sampling, considers all tokens
- **0.85 (Exploratory)**: Moderate diversity with some constraint

###  **max_tokens** (Response Length)
- **200 (Focused)**: Forces concise responses, may miss nuance
- **256 (Conservative)**: Short but adequate explanations
- **400 (Balanced)**: Optimal for classification + rationale
- **512 (Standard)**: Comfortable length for detailed reasoning
- **768 (Creative)**: Allows detailed analysis of complex cases
- **1024 (Exploratory)**: Maximum flexibility for comprehensive reasoning

###  **frequency_penalty** (Repetition Reduction)
- **0.0**: No penalty, natural language patterns
- **0.05**: Minimal repetition reduction
- **0.1**: Light repetition penalty, maintains coherence
- **0.2**: Moderate penalty, encourages varied vocabulary
- **0.3**: Higher penalty, forces diverse expression

###  **presence_penalty** (Topic Diversity)
- **0.0**: No penalty, focused responses
- **0.05**: Minimal topic expansion
- **0.1**: Light encouragement of topic breadth
- **0.2**: Moderate topic diversity encouragement

## Strategy Explanations & Usage Guide

This section explains each baseline strategy, its design rationale, and recommended use cases. The strategies are presented in a progressive order from most deterministic to most exploratory.

### baseline_conservative - Maximum Reproducibility

**Configuration**: `temp=0.0, max_tokens=256, top_p=0.9, freq_penalty=0.0, pres_penalty=0.0`

**Purpose**: Establish a deterministic baseline for reproducible research [8]. This strategy produces identical outputs for the same input across multiple runs, making it ideal for establishing ground truth and debugging.

**Design Rationale**:
- **Temperature 0.0**: Completely deterministic sampling—always selects the highest probability token [2]
- **256 tokens**: Short responses force concise, decisive classifications without over-reasoning
- **top_p 0.9**: Conservative nucleus sampling excludes low-probability tokens
- **No penalties**: Natural language patterns without artificial constraints

**When to Use**:
- Establishing baseline performance metrics for comparison
- Reproducibility testing and debugging
- Creating consistent test datasets
- Validating model behavior across runs

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_conservative \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: Highest consistency, lowest variance across runs. Best for reliability benchmarking.

---

### baseline_standard - Balanced Production Configuration

**Configuration**: `temp=0.1, max_tokens=512, top_p=1.0, freq_penalty=0.0, pres_penalty=0.0`

**Purpose**: Production-validated optimal configuration that balances performance with bias fairness [2]. This strategy achieved the best hybrid score (70% performance + 30% bias fairness) across three independent experimental runs.

**Design Rationale**:
- **Temperature 0.1**: Low randomness ensures reliable classifications while allowing minimal variation [2]
- **512 tokens**: Comfortable response length for detailed reasoning and bias-aware evaluation
- **top_p 1.0**: No nucleus sampling constraint—model considers full token distribution
- **No penalties**: Maintains natural language coherence without artificial repetition constraints

**When to Use**:
- Production deployment of hate speech detection systems
- Applications requiring bias-fairness considerations
- Scenarios where explanation quality matters for human review
- Default choice for most hate speech classification tasks

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_standard \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: Reference performance level with stable accuracy. Optimal for deployment scenarios requiring both performance and fairness.

---

### baseline_focused - Fast Decisive Classification

**Configuration**: `temp=0.05, max_tokens=200, top_p=0.8, freq_penalty=0.2, pres_penalty=0.0`

**Purpose**: Maximize pure F1-score performance through concise, confident classifications. This strategy achieved the highest F1-score in initial benchmarking tests.

**Design Rationale**:
- **Temperature 0.05**: Near-deterministic with minimal variation—forces confident decisions
- **200 tokens**: Shortest response length forces decisive classifications without hedging
- **top_p 0.8**: Very focused token selection—only considers top 80% probability mass [3]
- **Frequency penalty 0.2**: Encourages varied vocabulary in brief explanations

**When to Use**:
- High-throughput classification scenarios where speed matters
- Applications prioritizing pure performance over explanation quality
- Batch processing of large datasets
- Situations where concise decisions are preferred over detailed reasoning

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_focused \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: Highest F1-score but may show moderate bias concerns. Best for performance-focused deployments with human oversight.

---

### baseline_balanced - OpenAI Best Practices

**Configuration**: `temp=0.2, max_tokens=400, top_p=0.9, freq_penalty=0.05, pres_penalty=0.05`

**Purpose**: Follow OpenAI's recommended parameter ranges for classification tasks [2]. Provides a middle ground between deterministic and creative sampling.

**Design Rationale**:
- **Temperature 0.2**: Moderate creativity—optimal range for many classification tasks [2]
- **400 tokens**: Balanced length for classification + rationale without verbosity
- **top_p 0.9**: Standard nucleus sampling—excludes low-probability outliers
- **Minimal penalties (0.05)**: Light touch to encourage natural language variation

**When to Use**:
- General-purpose classification when optimal hyperparameters are unknown
- Exploring performance across different prompt designs
- Baseline for comparing against other configurations
- Following established best practices from LLM literature

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_balanced \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: Moderate performance across metrics. Good starting point for hyperparameter exploration.

---

### baseline_creative - Nuanced Edge Case Handling

**Configuration**: `temp=0.3, max_tokens=768, top_p=0.95, freq_penalty=0.1, pres_penalty=0.1`

**Purpose**: Handle nuanced cases requiring more detailed reasoning. Higher creativity allows the model to explore less obvious classification patterns.

**Design Rationale**:
- **Temperature 0.3**: Higher creativity for exploring edge cases and ambiguous content
- **768 tokens**: Extended response length for detailed analysis of complex cases
- **top_p 0.95**: Broader token selection allows more diverse reasoning paths
- **Light penalties (0.1)**: Encourages varied expression in longer explanations

**When to Use**:
- Analyzing difficult or ambiguous hate speech cases
- Research on model reasoning capabilities
- Identifying edge cases in datasets
- Quality assurance for challenging content

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_creative \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: More detailed explanations but may show performance degradation. Best for analysis rather than production deployment.

---

### baseline_exploratory - Pattern Discovery & Dataset Analysis

**Configuration**: `temp=0.5, max_tokens=1024, top_p=0.85, freq_penalty=0.3, pres_penalty=0.2`

**Purpose**: Discover new patterns and edge cases through high-diversity sampling. Not recommended for production but valuable for dataset analysis and robustness testing.

**Design Rationale**:
- **Temperature 0.5**: High creativity explores diverse classification reasoning [3]
- **1024 tokens**: Maximum flexibility for comprehensive reasoning
- **top_p 0.85**: Constrains extremely low-probability tokens while allowing exploration
- **Higher penalties (0.3, 0.2)**: Forces diverse expression and topic coverage

**When to Use**:
- Dataset analysis and pattern discovery
- Model robustness testing across diverse sampling strategies
- Research on model behavior under high-temperature sampling
- Identifying unusual hate speech patterns or model failure modes

**Testing Command**:
```bash
python prompt_runner.py --strategies baseline_exploratory \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Expected Behavior**: Highest variance, lowest consistency. May discover novel patterns but shows worst F1-score performance. Use for exploration, not production.

---

### Multi-Strategy Comparison

**Compare Top Performers**:
```bash
python prompt_runner.py \
  --strategies baseline_conservative,baseline_standard,baseline_focused \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json
```

**Test All Strategies**:
```bash
python prompt_runner.py --strategies all \
  --sample-size 100 \
  --prompt-template-file baseline_v1.json \
  --max-workers 6
```

**Large-Scale Validation**:
```bash
python prompt_runner.py --strategies all \
  --sample-size 500 \
  --output-dir outputs/baseline_optimization \
  --prompt-template-file baseline_v1.json
```

## Performance Metrics to Track

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True hate / (True hate + False hate)
- **Recall**: True hate / (True hate + False normal)  
- **F1-Score**: Harmonic mean of precision and recall

### Secondary Metrics - Bias and Fairness [5,6,7,8]
In addition to standard classification metrics, Bias and Fairness will be evaluated using False Positive Rate (FPR) and False Negative Rate (FNR), following established algorithmic fairness frameworks [5,6].

**False Positive Rate (FPR) by Community**: Measures how often normal/benign content gets wrongly labeled as hate speech for each target group (LGBTQ+, Middle Eastern, Mexican). Lower FPR means fewer false alarms. This metric addresses representational harm in automated content moderation [8].

**False Negative Rate (FNR) by Community**: Measures how often actual hate speech gets missed and labeled as normal for each target group. Lower FNR means better detection of real hate speech. This is critical for allocational fairness in hate speech detection [6].

**Goal**: Ensure FPR and FNR rates are similar across all three communities to avoid bias against any specific group, following equalized odds fairness criteria [6].

### Target Group Analysis
Track performance and bias metrics by community:
- **LGBTQ+ (48.8% of dataset)**: Anti-LGBTQ+ slurs and discrimination
- **Middle Eastern (28.5%)**: Anti-Arab, Islamophobic content
- **Mexican/Latino (22.6%)**: Anti-Mexican slurs and stereotypes

**Bias Evaluation**: Ensure consistent FPR and FNR across all three target groups to avoid algorithmic bias against specific communities.

## Testing Commands Reference

```bash
# Individual strategy testing
python prompt_runner.py --strategies baseline_conservative --sample-size 100 --prompt-template-file baseline_v1.json

# Multiple strategy comparison
python prompt_runner.py --strategies baseline_conservative,baseline_standard,baseline_balanced --sample-size 100 --prompt-template-file baseline_v1.json

# Full baseline optimization suite
python prompt_runner.py --strategies all --sample-size 100 --prompt-template-file baseline_v1.json --max-workers 6

# Performance analysis with detailed metrics
python prompt_runner.py --strategies all --sample-size 500 --output-dir outputs/baseline_optimization --prompt-template-file baseline_v1.json
```

## Notes

- All strategies use `"response_format": "json_object"` for consistent parsing
- Prompt structure remains identical across all variants for fair comparison
- Focus on parameter optimization rather than prompt engineering in this phase
- Results should inform next-phase prompt content optimization

## References

### Language Models & Sampling Methods

**[1] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020).** "Language models are few-shot learners." *Advances in Neural Information Processing Systems*, 33, 1877-1901.  
**URL**: https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html  
**Used in**: Core LLM methodology, few-shot classification framework (Overview, System Prompt design)

**[2] Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021).** "Calibrate before use: Improving few-shot performance of language models." *Proceedings of the 38th International Conference on Machine Learning*, 139, 12697-12706.  
**URL**: https://proceedings.mlr.press/v139/zhao21c.html  
**Used in**: Temperature optimization (baseline_conservative, baseline_standard), hyperparameter calibration methodology (Sections: Parameter Impact Analysis - Temperature, Key Findings - Temperature Impact)

**[3] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019).** "The curious case of neural text degeneration." *arXiv preprint arXiv:1904.09751*.  
**URL**: https://arxiv.org/abs/1904.09751  
**Used in**: Nucleus sampling (top_p) configuration (Parameter Impact Analysis - top_p, Key Findings - Nucleus Sampling Effects)

### Algorithmic Fairness & Bias Evaluation

**[4] Barocas, S., Hardt, M., & Narayanan, A. (2023).** *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.  
**URL**: https://fairmlbook.org/  
**Used in**: Comprehensive fairness framework, bias evaluation methodology (Secondary Metrics - Bias and Fairness, Optimization Methodology - hybrid scoring)

**[5] Hardt, M., Price, E., & Srebro, N. (2016).** "Equality of opportunity in supervised learning." *Advances in Neural Information Processing Systems*, 29, 3315-3323.  
**URL**: https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html  
**Used in**: Equalized odds fairness criteria, FPR/FNR evaluation across protected groups (Secondary Metrics, Bias-Performance Trade-off analysis)

**[6] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021).** "A survey on bias and fairness in machine learning." *ACM Computing Surveys*, 54(6), 1-35.  
**URL**: https://doi.org/10.1145/3457607  
**Used in**: Bias taxonomy, protected group analysis methodology (Target Group Analysis, Bias Evaluation framework)

**[7] Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020).** "Language (technology) is power: A critical survey of bias in NLP." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 5454-5476.  
**URL**: https://aclanthology.org/2020.acl-main.485/  
**Used in**: Representational harm analysis, allocational fairness in hate speech detection (Secondary Metrics - Bias and Fairness)

### Experimental Methodology & Reproducibility

**[8] Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., ... & Larochelle, H. (2021).** "Improving reproducibility in machine learning research." *Journal of Machine Learning Research*, 22(164), 1-20.  
**URL**: https://jmlr.org/papers/v22/20-303.html  
**Used in**: Reproducibility framework, deterministic baseline establishment (Progressive Testing Strategy, baseline_conservative design, Optimization Workflow - Step 3 validation)

**[9] Dodge, J., Gururangan, S., Card, D., Schwartz, R., & Smith, N. A. (2019).** "Show your work: Improved reporting of experimental results." *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 2185-2194.  
**URL**: https://aclanthology.org/D19-1224/  
**Used in**: Experimental reporting standards, hyperparameter documentation (Hyperparameter Analysis & Strategy Matrix, Testing Commands Reference)

### Cross-Referenced Documentation

**Empirical Results & Findings**: See `gptoss_ift_summary_README.md` for:

- Complete performance rankings from experimental runs
- Detailed empirical findings from hyperparameter optimization
- Bias analysis and validation results across protected groups
- Production deployment recommendations with bias mitigation strategies

**Hyperparameter Optimization Framework**: `prompt_engineering/pipeline/baseline/hyperparam/README.md` - Details the HyperparameterOptimiser implementation, hybrid scoring methodology (70% performance + 30% bias fairness), and automated configuration selection process.

### Strategy-Reference Mapping

| Strategy | Primary References | Application |
|----------|-------------------|-------------|
| baseline_conservative | [2], [8] | Deterministic sampling (temp=0.0), reproducibility |
| baseline_standard | [1], [2] | Balanced parameters (temp=0.1), original baseline |
| baseline_creative | [2], [3] | Moderate temperature (temp=0.3), nucleus sampling |
| baseline_focused | [2], [3] | Low temperature (temp=0.05), narrow top_p (0.8) |
| baseline_exploratory | [3] | High temperature (temp=0.5), diverse sampling |
| baseline_balanced | [2] | OpenAI best practices (temp=0.2) |
| **All strategies** | [4], [5], [6], [7] | Bias evaluation, fairness metrics |
| **Testing methodology** | [8], [9] | Reproducibility, experimental reporting |
