# Baseline Prompt Engineering Optimization Guide - baseline_v1.json

## Overview
This file contains multiple baseline prompt variations with different hyperparameter configurations to systematically optimize hate speech detection performance. This approach follows established methodologies in prompt engineering optimization [4] and hyperparameter tuning for large language models [1,2]. All variants use the same core prompt structure but vary key LLM parameters to explore the performance space, consistent with ablation study methodologies [2].

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
```
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

### 🎯 **Temperature** (Randomness/Creativity) [2,3]
- **0.0 (Conservative)**: Completely deterministic, most consistent responses [2]
- **0.05 (Focused)**: Near-deterministic with minimal variation
- **0.1 (Standard)**: Low randomness, reliable classifications [2]
- **0.2 (Balanced)**: Moderate creativity, optimal for classification tasks [2]
- **0.3 (Creative)**: Higher creativity for nuanced reasoning
- **0.5 (Exploratory)**: High creativity, may discover edge patterns

### 🎯 **top_p** (Nucleus Sampling) [3]
- **0.8 (Focused)**: Very focused token selection, conservative choices
- **0.9 (Conservative/Balanced)**: Balanced focus, excludes low-probability tokens [3]
- **0.95 (Creative)**: Allows more diverse token choices
- **1.0 (Standard)**: No nucleus sampling, considers all tokens
- **0.85 (Exploratory)**: Moderate diversity with some constraint

### 🎯 **max_tokens** (Response Length)
- **200 (Focused)**: Forces concise responses, may miss nuance
- **256 (Conservative)**: Short but adequate explanations
- **400 (Balanced)**: Optimal for classification + rationale
- **512 (Standard)**: Comfortable length for detailed reasoning
- **768 (Creative)**: Allows detailed analysis of complex cases
- **1024 (Exploratory)**: Maximum flexibility for comprehensive reasoning

### 🎯 **frequency_penalty** (Repetition Reduction)
- **0.0**: No penalty, natural language patterns
- **0.05**: Minimal repetition reduction
- **0.1**: Light repetition penalty, maintains coherence
- **0.2**: Moderate penalty, encourages varied vocabulary
- **0.3**: Higher penalty, forces diverse expression

### 🎯 **presence_penalty** (Topic Diversity)
- **0.0**: No penalty, focused responses
- **0.05**: Minimal topic expansion
- **0.1**: Light encouragement of topic breadth
- **0.2**: Moderate topic diversity encouragement

## Progressive Testing Strategy [9,10]

### Phase 1: Deterministic Baseline [10]
**Goal**: Establish most consistent performance baseline following reproducibility best practices [10]
```bash
python prompt_runner.py --strategies baseline_conservative --prompt-template-file baseline_v1.json
```
- Expected: Highest consistency, lowest variance
- Use for: Reliability benchmarking

### Phase 2: Standard Comparison
**Goal**: Compare against original baseline parameters
```bash
python prompt_runner.py --strategies baseline_standard --prompt-template-file baseline_v1.json
```
- Expected: Reference performance level
- Use for: Performance comparison anchor

### Phase 3: Optimization Exploration
**Goal**: Find optimal parameter combinations
```bash
# Test all variants
python prompt_runner.py --strategies baseline_focused,baseline_balanced,baseline_creative --prompt-template-file baseline_v1.json

# Full exploration
python prompt_runner.py --strategies all --prompt-template-file baseline_v1.json
```

### Phase 4: Edge Case Discovery
**Goal**: Identify challenging cases and patterns
```bash
python prompt_runner.py --strategies baseline_exploratory --prompt-template-file baseline_v1.json
```
- Expected: Discovery of new hate speech patterns
- Use for: Dataset analysis and model robustness testing

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

## Optimization Methodology

### Step 1: Baseline Establishment
Run `baseline_conservative` and `baseline_standard` to establish performance bounds:
- Conservative: Lower bound (high consistency)
- Standard: Reference point (original performance)

### Step 2: Performance Optimization
Test `baseline_focused` and `baseline_balanced` for improved metrics:
- Focused: Optimize for precision
- Balanced: Optimize for overall F1-score

### Step 3: Recall Enhancement
Use `baseline_creative` and `baseline_exploratory` to maximize hate detection:
- Creative: Better nuanced reasoning
- Exploratory: Discover missed patterns

### Step 4: Parameter Tuning
Based on results, create custom variants by mixing successful parameters:
```json
"baseline_custom": {
  "parameters": {
    "max_tokens": [best_from_testing],
    "temperature": [optimal_value],
    "top_p": [best_performance],
    "frequency_penalty": [tuned_value],
    "presence_penalty": [optimized_setting]
  }
}
```

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

## Success Criteria

### Phase 1 Success: Consistency Verification
- Conservative variant shows <5% variance across runs
- Standard variant matches or exceeds original baseline performance

### Phase 2 Success: Performance Optimization  
- Balanced variant shows >2% F1-score improvement over standard
- Focused variant shows >5% precision improvement

### Phase 3 Success: Recall Maximization
- Creative variant shows >3% recall improvement
- Exploratory variant discovers ≥5 new hate speech patterns

### Final Success: Optimal Configuration
- Identify single best-performing parameter combination
- Achieve >85% F1-score on balanced dataset
- Maintain consistent performance across LGBTQ+, Middle Eastern, and Mexican target groups

## Notes
- All strategies use `"response_format": "json_object"` for consistent parsing
- Prompt structure remains identical across all variants for fair comparison
- Focus on parameter optimization rather than prompt engineering in this phase
- Results should inform next-phase prompt content optimization

## References

### Hyperparameter Optimization & Few-Shot Learning

[1] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). "Language models are few-shot learners." *Advances in Neural Information Processing Systems*, 33, 1877-1901.

[2] Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). "Calibrate before use: Improving few-shot performance of language models." *Proceedings of the 38th International Conference on Machine Learning*, 139, 12697-12706.

[3] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). "The curious case of neural text degeneration." *arXiv preprint arXiv:1904.09751*.

### Prompt Engineering (General Surveys - Not Chain-of-Thought)

[4] Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2023). "Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing." *ACM Computing Surveys*, 55(9), 1-35.

### Algorithmic Fairness & Bias Evaluation

[5] Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.

[6] Hardt, M., Price, E., & Srebro, N. (2016). "Equality of opportunity in supervised learning." *Advances in Neural Information Processing Systems*, 29, 3315-3323.

[7] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). "A survey on bias and fairness in machine learning." *ACM Computing Surveys*, 54(6), 1-35.

[8] Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). "Language (technology) is power: A critical survey of bias in NLP." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 5454-5476.

### Experimental Methodology & Reproducibility

[9] Dodge, J., Gururangan, S., Card, D., Schwartz, R., & Smith, N. A. (2019). "Show your work: Improved reporting of experimental results." *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 2185-2194.

[10] Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., ... & Larochelle, H. (2021). "Improving reproducibility in machine learning research." *Journal of Machine Learning Research*, 22(164), 1-20.