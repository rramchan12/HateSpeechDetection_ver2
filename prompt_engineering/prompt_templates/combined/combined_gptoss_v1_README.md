# Combined Policy-Persona Prompts for GPT-OSS (v1 - Few-Shot Enhanced)

## Overview

This configuration file provides empirically optimized policy-persona hybrid prompts for hate speech detection using the gpt-oss-120b model (Phi-3.5-MoE-instruct). All strategies combine X platform's hateful conduct policy with community-informed perspectives from the three most represented demographic groups in the unified dataset.

**Optimization Basis**: Hyperparameters are derived from systematic testing documented in `gptoss_ift_summary_README.md`, which evaluated 6 baseline configurations across 1,109 total samples (100-sample optimization + 1,009-sample production validation).

**V1 Enhancement (October 2025)**: Added few-shot examples for Mexican/Latino detection and restored LGBTQ+ context based on iterative validation testing (v2, v3, Option B/C comparison). Empirical results showed **71% accuracy** and **0.667 F1-score** with conservative strategy, achieving **-50% improvement** in Mexican/Latino false negative rate (83% → 33%).

**Key Design Principle**: Integrate explicit policy guidelines with community perspectives AND concrete examples to reduce both false positives (overcriminalization) and false negatives (undercriminalization) across protected demographic groups.

---

## Design Rationale

### Policy-Persona Integration Approach

This prompt design integrates two complementary frameworks for hate speech detection:

#### 1. Policy Foundation: X Platform Hateful Conduct Policy

**X Platform Hateful Conduct Policy** serves as the authoritative foundation for classification decisions. The policy explicitly prohibits content that "directly attacks people based on race, ethnicity, national origin, sexual orientation, gender, gender identity, religious affiliation, age, disability, or serious disease" through:

- Slurs and derogatory language
- Dehumanization of groups
- Incitement of fear, harassment, or discrimination
- Hateful references to violence
- Reinforcement of negative stereotypes

Reference: X Corp. Hateful Conduct Policy (https://help.twitter.com/en/rules-and-policies/hateful-conduct-policy)

**How We Use Policy in System Prompt:**

```text
You are an expert content moderation assistant trained to classify social media posts 
as either 'hate' or 'normal' using X platform's hateful conduct policy and 
community-informed perspectives.

You MUST respond in strict JSON format:
{"classification": "hate", "rationale": "brief explanation"} or 
{"classification": "normal", "rationale": "brief explanation"}

Classification Rules:
- Use EXACTLY 'hate' if the text violates X's hateful conduct policy OR targets 
  protected communities
- Use EXACTLY 'normal' if the text does not violate policy AND does not harm 
  protected communities

Detection Focus:
- Direct attacks: Explicit slurs, derogatory terms, and hateful language
- Coded/subtle hate: Dog whistles, stereotypes, dehumanizing comparisons, 
  and veiled attacks
- Context matters: Consider whether text criticizes policies/ideas vs. attacks people 
  based on protected characteristics
- Distinguish community self-reference from external attacks (e.g., LGBTQ+ individuals 
  using in-group terms vs. outsiders using slurs)
```

**How We Use Policy in User Template:**

```text
X PLATFORM HATEFUL CONDUCT POLICY:
Content violates policy if it directly attacks people based on race, ethnicity, 
national origin, sexual orientation, gender, gender identity, religious affiliation, 
age, disability, or serious disease.

Prohibited Content:
• Slurs and derogatory language targeting protected categories (both explicit and coded)
• Dehumanization of groups based on protected characteristics (comparing to animals, 
  objects, diseases, or subhumans)
• Incitement of fear, harassment, or discrimination against protected groups
• Hateful references to violence or violent events targeting protected categories
• Content reinforcing negative stereotypes about protected groups

Note: Criticism of policies, governments, or ideologies is NOT hate speech. Attacks must 
target PEOPLE based on protected characteristics.
```

The policy provides clear, platform-specific guidelines that anchor classification decisions in established content moderation standards, reducing ambiguity and improving consistency. Updated guidance explicitly distinguishes between attacking people (prohibited) and criticizing policies/ideologies (permitted), addressing false positives from policy discussions. Emphasis on both explicit and coded hate speech patterns improves detection of subtle attacks that contribute to high false negative rates.

---

#### 2. Community Perspectives: LGBTQ+, Mexican/Latino, Middle Eastern

**Community Perspectives** address dataset composition and documented bias patterns from production validation (1,009 samples):

**LGBTQ+ Community**: Validation testing revealed 31-50% false positive rate (FPR) across strategies, indicating overcriminalization of benign LGBTQ+ discourse, while false negative rates (FNR) of 33-67% show missed hate speech detection. Community perspective explicitly evaluates for anti-LGBTQ+ slurs (faggots, trannies), attacks on sexual orientation/gender identity, denial of LGBTQ+ identities, and harmful stereotypes (predators, groomers). **Critical distinction with enhanced emphasis**: in-group reclamation (LGBTQ+ person saying 'we're queer') vs. out-group slurs. The focused variant includes **"CRITICAL:"** prefix and concrete examples to reduce false positives while improving detection of coded homophobia/transphobia. Token allocation increased from 200 to 256 tokens to accommodate restored context.

**Mexican/Latino Community**: Validation showed 0% FPR (no overcriminalization) but critically high FNR of 33-83%, indicating severe undercriminalization where most hate speech against Mexicans/Latinos is missed. Community perspective provides expanded detection guidance with **explicit few-shot examples** showing the critical distinction between hate and policy discussion. The examples demonstrate: (1) generalizing ALL Mexicans as "illegal/criminals/invaders" = HATE, (2) dehumanization (comparing to animals) = HATE, (3) coded attacks like "They're not sending their best" = HATE, while (4) policy discussions like "We need stronger border enforcement" = NORMAL. This few-shot approach achieved **-50% improvement** in false negative rate (83% → 33%) across all strategies by teaching the model concrete patterns of immigration-based hate vs. legitimate policy debate.

**Middle Eastern Community**: Validation revealed 17-67% FPR (overcriminalization, especially in focused variant) and 50-62% FNR (missing hate). Community perspective assesses Islamophobic content, anti-Arab slurs (towelhead, terrorist), terrorism associations, and cultural/religious attacks. Critical clarification: Discussing terrorism, geopolitical conflicts, or specific terrorist groups is NOT hate speech unless it attributes these characteristics to ALL Middle Eastern people. This distinction reduces false positives while maintaining detection of actual anti-Arab and anti-Muslim hate.

**How We Use Personas in User Template:**

```text
COMMUNITY PERSPECTIVES (Primary Dataset Groups):

LGBTQ+ Community:
- Explicit slurs: faggots, trannies, dykes, queers (when used as attacks)
- Attacks on sexual orientation or gender identity
- Denial of LGBTQ+ identities or existence
- Harmful stereotypes (predators, mentally ill, groomers)
- Context: Distinguish in-group reclamation (LGBTQ+ person saying 'we're queer') 
  from out-group slurs

Mexican/Latino Community:
- Explicit slurs: beaners, wetbacks, illegals (when dehumanizing), anchor babies
- Immigration-based attacks: Calling all Mexicans 'illegal', 'criminals', 'invaders'
- Ethnic stereotypes: lazy, drug dealers, rapists, job stealers
- Dehumanization: Comparing to animals, diseases, or infestations
- Coded hate: 'They're not sending their best', 'go back to your country' 
  when targeting ethnicity

FEW-SHOT EXAMPLES - Mexican/Latino Hate vs. Policy Discussion:
HATE: "These Mexicans are all illegals, send them back" (generalizes ALL Mexicans)
HATE: "Mexico isn't sending their best, they're rapists and criminals" (attacks ALL)
HATE: "Mexicans are like rats crossing the border" (dehumanization)
NORMAL: "We need stronger border enforcement policies" (policy, not attacking people)
NORMAL: "Immigration system should prioritize legal pathways" (policy critique)

Middle Eastern Community:
- Explicit slurs: sand n*****, towelhead, camel jockey, terrorist
- Religious attacks: Islamophobia, calling all Muslims terrorists or extremists
- Terrorism associations: Assuming Middle Eastern people are terrorists or 
  support terrorism
- Cultural/religious attacks: Mocking Islamic practices, hijab, or Middle Eastern culture
- Note: Discussing terrorism, geopolitical conflicts, or specific terrorist groups 
  is NOT hate speech unless it attributes these to ALL Middle Eastern people
```

Community perspectives provide nuanced, group-specific evaluation criteria that help the model understand context-dependent harm patterns often missed by general policy guidelines alone. **V1 Enhancement**: Added explicit few-shot examples for Mexican/Latino detection based on empirical testing showing 83% false negative rate. The few-shot approach provides concrete contrast between attacking ALL people (HATE) vs. discussing policies (NORMAL), achieving -50% FNR improvement (83%→33%). Enhanced LGBTQ+ guidance with "CRITICAL:" emphasis and increased tokens (200→256) to accommodate restored in-group reclamation context, reducing false positives while maintaining hate detection accuracy. Updated Middle Eastern clarification distinguishes discussing topics vs. attacking people to reduce overcriminalization.

---

#### 3. Systematic Analysis Framework

**Four-Step Evaluation Process** ensures comprehensive detection:

1. Detection of slurs and direct attacks on protected characteristics
2. Identification of dehumanization and stereotyping
3. Assessment of community harm (from affected groups' perspectives)
4. Policy compliance verification

**How We Use Framework in User Template:**

```text
Analyze this text systematically:
1. Does it contain slurs (explicit or coded) or direct attacks on protected characteristics?
2. Does it dehumanize, stereotype, or generalize negatively about any protected community?
3. Does it incite fear, harassment, or violence against protected groups?
4. Is it attacking PEOPLE based on characteristics, or criticizing policies/ideas?
5. Would members of affected communities reasonably find this harmful?
```

This framework addresses both explicit hate (easily detectable slurs) and implicit hate (coded language, stereotypes, context-dependent attacks) documented in hate speech research. Enhanced five-step analysis adds explicit checks for coded language (Step 1), dehumanization and negative generalizations (Step 2), incitement (Step 3), and the critical distinction between attacking people vs. criticizing policies (Step 4). These additions directly address the high false negative rates (33-83% across groups) by providing more comprehensive detection guidance.

---

## Strategy Details

### 1. combined_optimized (RECOMMENDED)

**Purpose**: Optimal balance between performance (F1-score) and bias fairness across protected groups, enhanced with few-shot examples.

**Empirical Validation (Baseline)**: Based on `baseline_standard` configuration which achieved:

- F1-Score: 0.626 (optimization run), 0.615 (production validation)
- Hybrid Score: 1.000 (70% performance + 30% bias fairness)
- Bias Score: 0.647 (best fairness across all configurations)

**V1 Enhancement Results (50-sample validation, October 2025)**:

- F1-Score: 0.571 (few-shot variant)
- Accuracy: 58%
- Mexican/Latino FNR: 33% (improved from v3's 83%, **-50% reduction**)
- LGBTQ+ FPR: 50%, FNR: 33%
- Middle Eastern FPR: 67%, FNR: 50%

**Hyperparameters**:
```json
{
  "max_tokens": 512,
  "temperature": 0.1,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Rationale**:

- **Temperature 0.1**: Empirical testing showed 16.6% F1-score decrease from temp=0.1 to temp=0.5, confirming that low temperature (≤0.1) is critical for classification tasks requiring deterministic pattern recognition rather than creative exploration.
- **Max Tokens 512**: Non-monotonic optimization curve revealed "goldilocks zone" at 512 tokens, enabling sufficient explanation for bias-aware evaluation while avoiding hedging behavior observed at 768+ tokens.
- **Top P 1.0**: At low temperatures, nucleus sampling has minimal impact (<2% F1 variance across top_p values 0.8-1.0), so maximum value (1.0) is used for consistency.
- **Zero Penalties**: Frequency and presence penalties showed <0.05 F1 variance across tested values (0.0-0.3), indicating negligible effect on short-form classification tasks.

**Prompt Design**:

- **Policy Foundation**: X platform's hateful conduct policy provides authoritative framework for prohibited content categories (slurs, dehumanization, incitement, stereotypes).
- **Community Perspectives**: Explicit LGBTQ+, Mexican/Latino, and Middle Eastern community lenses address dataset composition and documented bias patterns (LGBTQ+ overcriminalization: 43.0% FPR vs. 8.1% for Mexican/Latino).
- **Few-Shot Examples**: Added 5-example block for Mexican/Latino detection showing explicit contrast between hate (generalizing ALL people) and normal (policy discussion). Empirical testing showed this approach reduced Mexican FNR from 83% to 33% (-50% improvement).
- **Systematic Analysis**: Three-step evaluation process ensures consistent detection of both explicit hate (slurs) and implicit hate (coded language, stereotypes).

**Use Case**: Production hate speech detection requiring balanced performance and demographic fairness, particularly for immigration-related content.

---

### 2. combined_focused

**Purpose**: Performance-optimized variant with restored LGBTQ+ context and few-shot examples.

**Empirical Validation (Baseline)**: Based on `baseline_focused` configuration which achieved:

- F1-Score: 0.600 (second-best in optimization run)
- Hybrid Score: 0.768 (second rank)
- Bias Score: 0.616

**V1 Enhancement Results (50-sample validation, October 2025)**:

- F1-Score: 0.400 (few-shot variant)
- Accuracy: 52%
- Mexican/Latino FNR: 33% (improved from v3's 50%, **-17% reduction**)
- LGBTQ+ FPR: 44%, FNR: 89% (context restoration insufficient at 256 tokens)
- Middle Eastern FPR: 33%, FNR: 63%

**Hyperparameters**:
```json
{
  "max_tokens": 200,
  "temperature": 0.05,
  "top_p": 0.8,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Rationale**:

- **Temperature 0.05**: Extremely low temperature for maximum determinism, forcing decisive classifications without over-reasoning.
- **Max Tokens 256**: Increased from original 200 tokens to accommodate restored LGBTQ+ context with in-group reclamation explanation. Token budget still constrains detailed explanations, optimized for concise decision-making.
- **Top P 0.8**: Slightly restrictive nucleus sampling paired with ultra-low temperature for focused token selection.

**Prompt Design**:

- **Condensed Policy Summary**: Streamlined policy presentation to fit within 256-token constraint.
- **LGBTQ+ Context Restoration**: Added "CRITICAL:" prefix and concrete example ('we're queer') to emphasize in-group reclamation distinction. Despite increased tokens, validation showed insufficient capacity (89% FNR for LGBTQ+).
- **Few-Shot Examples**: Compact 2-example format for Mexican/Latino detection showing hate vs. policy discussion contrast.
- **Streamlined Community Perspectives**: Abbreviated guidance maintaining core evaluation framework.

**Use Case**: Cost-efficient deployments where inference speed is prioritized. **Note**: High LGBTQ+ FNR (89%) suggests this variant needs further optimization or should be avoided for LGBTQ+-sensitive content.

---

### 3. combined_conservative (BEST OVERALL - V1 VALIDATION)

**Purpose**: High-precision variant minimizing false positives in risk-averse scenarios, enhanced with few-shot examples.

**Empirical Validation (Baseline)**: Based on `baseline_conservative` configuration which achieved:

- F1-Score: 0.594 (third-best in optimization run)
- Precision: 0.569-0.617 (strongest precision among top-3)
- Hybrid Score: 0.742 (third rank)

**V1 Enhancement Results (50-sample validation, October 2025)** ⭐:

- **F1-Score: 0.667** (BEST across all variants)
- **Accuracy: 71%** (BEST across all variants, +11% from v3 baseline)
- Mexican/Latino FNR: 33% (improved from v3's 67%, **-34% reduction**)
- **LGBTQ+ FPR: 25%, FNR: 25%** (perfect balance, BEST across all variants)
- Middle Eastern FPR: 33%, FNR: 50%

**Key Achievement**: Conservative strategy achieved the highest overall performance when combined with few-shot examples, demonstrating that deterministic sampling (temp=0.0) benefits most from concrete example-based learning.

**Hyperparameters**:
```json
{
  "max_tokens": 256,
  "temperature": 0.0,
  "top_p": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Rationale**:

- **Temperature 0.0**: Fully deterministic greedy decoding for maximum reproducibility and consistency. Enables exact replication of classifications for auditing and debugging. **Empirically demonstrated**: Deterministic sampling achieves best performance when combined with few-shot examples (71% accuracy vs. 58-60% for other strategies).
- **Max Tokens 256**: Moderate token budget balancing explanation detail with decisiveness. Empirical data showed 256 tokens as optimal for conservative strategy when combined with concrete examples.
- **Top P 0.9**: Standard nucleus sampling value (minimal impact at temp=0.0 due to greedy decoding).

**Prompt Design**:

- **Structured Policy Presentation**: Clear prohibited content categories with explicit "attacking PEOPLE vs. policies" distinction.
- **Few-Shot Examples**: Compact 2-example format demonstrating hate vs. policy discussion contrast. Conservative strategy benefited most from these concrete examples due to deterministic nature.
- **Focused Community Analysis**: LGBTQ+ in-group reclamation context, Mexican/Latino immigration-based hate patterns, Middle Eastern terrorism generalization distinctions.

**Use Case**: **RECOMMENDED FOR PRODUCTION** - Conservative deployments requiring reproducible classifications and minimized false positive rates. Achieved best overall balance (71% accuracy, 0.667 F1, LGBTQ+ 25%/25%, Mexican 33% FNR) across all tested variants.

---

## V1 Enhancement History

### Iterative Development Process (October 2025)

**Problem Identified**: Initial v1 prompts (pattern-based simplification from v0) showed Mexican/Latino detection regression:
- V3 baseline: 83% FNR for Mexican/Latino community (missing 83% of hate speech)
- LGBTQ+ focused variant: FPR degraded 19%→25%, FNR degraded 33%→44%
- Pattern-based guidance alone insufficient for model to distinguish hate vs. policy discussion

**Solution Exploration**: Tested two approaches on 50-sample balanced dataset:

1. **Option B (Few-Shot Examples)**: Added explicit 5-example block showing hate vs. normal contrast
   - Conservative strategy: **71% accuracy, 0.667 F1** ✅✅✅ BEST
   - Mexican FNR: Consistent 33% across all strategies (-50% improvement)
   - LGBTQ+ balance: Conservative achieved 25%/25% FPR/FNR

2. **Option C (Hybrid Patterns + Examples)**: Combined pattern-based guidance with inline examples
   - Optimized strategy: 60% accuracy, 0.583 F1, LGBTQ+ FNR 11% (excellent)
   - Mexican FNR: Mixed results (focused variant still 83%)
   - Less consistent across strategies

**Decision**: Adopted Option B (few-shot examples) as base for v1 template because:
- **Best overall performance**: Conservative achieved highest accuracy (71%) and F1 (0.667)
- **Consistency**: All strategies achieved 33% Mexican FNR (Option C focused failed at 83%)
- **Interpretability**: Clear hate/normal examples make model behavior more predictable
- **Deterministic synergy**: Conservative strategy (temp=0.0) benefited most from concrete examples

### Key Changes from V0 to V1

1. **Mexican/Latino Section**:
   - **Added**: 5-example few-shot block with explicit HATE/NORMAL labels
   - **Result**: Mexican FNR reduced from 83% → 33% (-50% improvement)

2. **LGBTQ+ Section**:
   - **Added**: "CRITICAL:" prefix in focused variant with concrete example
   - **Increased**: Focused variant tokens from 200 → 256
   - **Result**: Conservative achieved 25%/25% FPR/FNR balance

3. **Strategy Recommendations**:
   - **Updated**: Conservative now RECOMMENDED for production (was third-ranked)
   - **Rationale**: Deterministic sampling + few-shot examples = best performance

### Validation Results Summary

| Strategy | Accuracy | F1 | Mexican FNR | LGBTQ+ FPR/FNR | Recommendation |
|----------|----------|-------|-------------|----------------|----------------|
| **combined_conservative** | **71%** | **0.667** | **33%** | **25% / 25%** | ✅✅✅ **RECOMMENDED** |
| combined_optimized | 58% | 0.571 | 33% | 50% / 33% | ✅ Good balance |
| combined_focused | 52% | 0.400 | 33% | 44% / 89% | ⚠️ LGBTQ+ issues |

---

## Validation Testing

This validation framework follows a two-phase systematic testing approach aligned with the `HyperparameterOptimiser` framework documented in `pipeline/baseline/hyperparam/hyperparameter_optimiser.py`.

### Phase 1: Hyperparameter/Strategy Testing

**Objective**: Evaluate all three strategies on a controlled sample to identify optimal configuration using hybrid scoring (70% performance + 30% bias fairness).

**Dataset**: `canned_100_size_varied` (100 samples with balanced demographic representation)


---

### Phase 2: Production Validation

**Objective**: Validate the optimal strategy (identified in Phase 1) on full production dataset to assess scalability and real-world performance.

**Dataset**: `unified` (1,009 samples with representative demographic distribution)

---
## Usage Example

### Run Optimal Configuration (Recommended)

```bash
# Run combined_optimized strategy on full unified dataset (1,009 samples)
python prompt_runner.py \
  --data-source unified \
  --strategies combined_optimized \
  --output-dir outputs/combined_v1/gptoss \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined_policy_persona_gptoss_v1.json
```

### Run Small-Scale Validation (100 samples)

```bash
# Test on canned_100_size_varied dataset for quick validation
python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_optimized \
  --output-dir outputs/combined_v1/gptoss/validation \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined_policy_persona_gptoss_v1.json
```

### Compare All Three Strategies

```bash
# Run all strategies for comparative analysis
python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_optimized combined_focused combined_conservative \
  --output-dir outputs/combined_v1/gptoss/comparison \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined_policy_persona_gptoss_v1.json
```

---
## References

**[1] X Corp. Hateful Conduct Policy.**  
Platform policy defining prohibited hate speech categories.  
URL: https://help.twitter.com/en/rules-and-policies/hateful-conduct-policy

**[2] Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021).**  
"Calibrate before use: Improving few-shot performance of language models."  
*Proceedings of the 38th International Conference on Machine Learning*, 139, 12697-12706.  
URL: https://proceedings.mlr.press/v139/zhao21c.html

**[3] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019).**  
"The curious case of neural text degeneration."  
*arXiv preprint arXiv:1904.09751*.  
URL: https://arxiv.org/abs/1904.09751

**[4] Barocas, S., Hardt, M., & Narayanan, A. (2023).**  
*Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.  
URL: https://fairmlbook.org/

**[5] Hardt, M., Price, E., & Srebro, N. (2016).**  
"Equality of opportunity in supervised learning."  
*Advances in Neural Information Processing Systems*, 29, 3315-3323.  
URL: https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html

**[6] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021).**  
"A survey on bias and fairness in machine learning."  
*ACM Computing Surveys*, 54(6), 1-35.  
URL: https://doi.org/10.1145/3457607

---

## Cross-Referenced Documentation

**Empirical Validation**: See `gptoss_ift_summary_README.md` for complete hyperparameter optimization results, including 6-strategy comparison, production validation on 1,009 samples, and bias analysis across protected groups.

**Baseline Template**: See `baseline_v1_README.md` for foundational prompt structure and hyperparameter definitions used in empirical testing.

**V1 Enhancement Analysis**: See `OPTION_B_VS_C_RESULTS_ANALYSIS.md` for comprehensive comparison of few-shot examples (Option B) vs. hybrid patterns (Option C) approach, including detailed performance metrics and decision rationale.

**V1 Option B Template**: See `combined_gptoss_v1_optionB_fewshot.json` for the few-shot examples variant that was adopted as the v1 base after validation testing.

**V1 Option C Template**: See `combined_gptoss_v1_optionC_hybrid.json` for the alternative hybrid patterns + examples approach tested during v1 development.

**Original Combined Prompts**: See `all_combined.json` for initial policy-persona integration approaches (baseline, policy, persona, combined, enhanced_combined) that informed this optimized version.

**Optimization Framework**: See `pipeline/baseline/hyperparam/README.md` for HyperparameterOptimiser implementation details and hybrid scoring methodology (70% performance + 30% bias fairness).
