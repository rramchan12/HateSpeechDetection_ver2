# Combined V2: Policy-Persona Hybrid with Recall Optimization

## Overview

This document presents 5 empirically-designed strategies to improve upon the **combined_gptoss_v1 framework** based on production validation results.

**Development Context**: Combined V1 demonstrated policy-persona integration viability but fell short of baseline_standard performance. This document analyzes both combined_optimised (best V1 variant) and combined_conservative (catastrophic failure) to derive Combined V2 improvements.

**Model**: gpt-oss-120b (Phi-3.5-MoE-instruct)

**Base Framework**: combined_gptoss_v1.json (policy-persona hybrid)

**References**:
- `gptoss_ift_summary_README.md` - Baseline optimization results and hyperparameter rationale
- `combined_gptoss_v1_README.md` - Combined framework design and V1 validation results
- Production run results analyzed below

---

## Hyperparameter Configuration

All Combined V2 strategies use **identical hyperparameters** derived from empirical optimization documented in `gptoss_ift_summary_README.md`:

```json
{
  "max_tokens": 512,
  "temperature": 0.1,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Rationale for Fixed Hyperparameters**:

- **Temperature 0.1**: Empirically validated as optimal for classification tasks. Testing showed 16.6% F1-score decrease from temp=0.1 to temp=0.5. Combined_conservative's catastrophic failure at temp=0.0 (recall=37.8%) confirmed that 0.1 provides best balance between determinism and generalization.

- **Max Tokens 512**: "Goldilocks zone" identified through non-monotonic optimization curve. Enables sufficient explanation for bias-aware evaluation while avoiding hedging behavior observed at 768+ tokens. Combined_conservative's failure at 256 tokens (23% F1 loss) confirmed 512 as minimum viable.

- **Top P 1.0**: At low temperatures, nucleus sampling has minimal impact (<2% F1 variance across top_p values 0.8-1.0). Maximum value (1.0) used for consistency.

- **Zero Penalties**: Frequency and presence penalties showed <0.05 F1 variance across tested values (0.0-0.3), indicating negligible effect on short-form classification tasks.

**Combined V2 Strategy**: Keep hyperparameters constant across all strategies to isolate the impact of prompt engineering changes (recall emphasis, few-shot reduction, policy-persona balance) on performance.

---

## Performance History: Combined Approach Evolution

### Phase 1: combined_optimised on size_varied (50 samples)

**Purpose**: Initial validation of policy-persona hybrid with few-shot examples

**Results** (documented in `combined_gptoss_v1_README.md`):
- **F1-Score**: 0.571
- **Accuracy**: 58%
- **Mexican/Latino FNR**: 33% (improved from 83%, -50% reduction with few-shot examples)
- **LGBTQ+ Metrics**: FPR=50%, FNR=33%
- **Middle Eastern Metrics**: FPR=67%, FNR=50%

**Status**: BEST performing combined variant on small dataset

---

### Phase 2: combined_optimised on production (1,009 samples) - MAIN ANALYSIS

**Dataset**: unified (1,009 samples from HateXplain + ToxiGen)
**Run ID**: run_20251018_234643
**Strategy**: combined_optimised from combined_gptoss_v1.json

#### Overall Performance Metrics

| Metric | Value | vs. baseline_standard | Status |
|--------|-------|----------------------|--------|
| **F1-Score** | **0.590** | **-0.025 (-4.1%)** | ❌ UNDERPERFORMS |
| Accuracy | 64.5% | -0.5% | ⚠️ Slight regression |
| Precision | 0.616 | +0.6% | ✅ Maintained |
| **Recall** | **0.567** | **-0.053 (-8.5%)** | ❌ **RECALL GAP** |
| Confusion Matrix | TP=258, TN=393, FP=161, FN=197 | — | 197 FN = missing 43% of hate |

#### Bias Metrics by Protected Group

| Group | Samples | FPR | FNR | vs. baseline | Primary Issue |
|-------|---------|-----|-----|--------------|---------------|
| **LGBTQ+** | 494 (49.0%) | **39.2%** | 41.2% | **-3.8% FPR (BETTER)** | ✅ FPR improved vs baseline (43%) |
| **Mexican/Latino** | 209 (20.7%) | 7.0% | **48.0%** | **+8.2% FNR (WORSE)** | ❌ Missing 48% of hate |
| **Middle Eastern** | 306 (30.3%) | 19.4% | **42.0%** | **+6.8% FNR (WORSE)** | ❌ Missing 42% of hate |

#### Critical Findings

**1. RECALL GAP is the Core Problem**: Recall 0.567 vs. baseline 0.620 = -8.5% regression. Missing 197 hate samples (43% of all hate speech). FNR elevated across ALL groups.

**2. Mexican FNR Degradation**: Size_varied 33% → Production 48% (+15%). Few-shot examples don't generalize to all immigration hate patterns.

**3. LGBTQ+ FPR Improvement (BRIGHT SPOT)**: 39.2% vs. baseline 43.0% = -3.8% improvement. Policy-persona framework reduces overcriminalization. In-group reclamation guidance is working.

**4. Precision-Recall Tradeoff**: Combined approach is MORE CONSERVATIVE than baseline. Precision maintained but recall suffered. Need to shift toward recall without losing LGBTQ+ FPR gains.

**5. Scale Impact**: F1 improved 0.571 → 0.590 (+3.3%). Combined approach DOES scale up, but gap to baseline persists. NOT scale degradation; it's a fundamental recall deficit.

---

### Phase 3: combined_conservative on production (1,009 samples) - CAUTIONARY

**Purpose**: Test if MORE conservative approach (temp=0.0, 256 tokens) improves
**Dataset**: unified (1,009 samples)
**Run ID**: run_20251101_092206
**Result**: **CATASTROPHIC FAILURE**

#### Performance Metrics

| Metric | Value | vs. baseline | vs. combined_optimised | Status |
|--------|-------|--------------|------------------------|--------|
| **F1-Score** | **0.473** | **-0.142 (-23%)** | **-0.117 (-20%)** | ❌ CATASTROPHIC |
| **Recall** | **37.8%** | **-24.2% (-39%)** | **-18.9% (-33%)** | ❌ CATASTROPHIC |
| Confusion Matrix | TP=172, TN=454, FP=100, FN=283 | — | 283 FN = missing 62% of hate |

#### Bias Metrics

| Group | FNR | Issue |
|-------|-----|-------|
| **LGBTQ+** | **61.8%** | Massive recall collapse |
| **Mexican** | **64.2%** | Missing 64% of hate |
| **Middle Eastern** | **61.1%** | Missing 61% of hate |

#### Root Causes

1. **Temperature 0.0 Rigidity**: Deterministic + few-shot = exact pattern matching. Missing 62% of hate speech.
2. **256 Tokens Insufficient**: 50% reduction → 23% F1 loss (0.590 → 0.473)
3. **Few-Shot Overfitting**: Model learned "If not in examples → NORMAL"
4. **Safety Catastrophe**: FNR 61-64% = systematic under-detection

---

## Summary: What Combined V2 Must Address

### From combined_optimised (Primary)

**Keep**:
- ✅ Policy-persona hybrid (LGBTQ+ FPR improvement proves it works)
- ✅ Hyperparameters: temp=0.1, 512 tokens
- ✅ Few-shot concept (but refine)

**Fix**:
- ❌ Add aggressive recall: "Err toward flagging", "Subtle hate IS hate"
- ❌ Strengthen Mexican detection beyond few-shot
- ❌ Improve Middle Eastern terrorism detection
- ❌ Shift toward recall (safety priority)

### From combined_conservative (Secondary)

**Never Repeat**:
- ❌ Temperature 0.0 (use 0.1)
- ❌ 256 tokens (use 512)
- ❌ Over-reliance on few-shot
- ❌ Conservative for safety-critical tasks

---

## Strategy Comparison Matrix

| Strategy | Policy% | Persona% | Recall Target | Examples | Expected F1 | Use Case |
|----------|---------|----------|---------------|----------|-------------|----------|
| baseline_standard | N/A | N/A | 0.620 (actual) | 0 | 0.615 | Current champion |
| combined_optimised | 50% | 50% | 0.567 (actual) | 5/group | 0.590 | V1 best (-4.1%) |
| combined_conservative | 50% | 50% | 0.378 (actual) | 5/group | 0.473 | V1 catastrophic |
| **combined_v2_policy_focused** | 80% | 20% | **0.580-0.600** | 0 | **0.605-0.620** | Match/beat baseline |
| **combined_v2_persona_balanced** | 50% | 50% | **0.590-0.610** | 2/group | **0.610-0.625** | **RECOMMENDED** |
| **combined_v2_recall_optimized** | 60% | 40% | **0.595-0.615** | 0 | **0.600-0.620** | Safety-first |
| **combined_v2_minimal_hybrid** | 60% | 40% | **0.575-0.595** | 0 | **0.600-0.615** | Conservative |
| **combined_v2_cultural_context** | 40% | 60% | **0.585-0.605** | 0 | **0.605-0.620** | Cultural depth |

**Key V2 Improvements**:
- Recall emphasis: ALL strategies include "err toward flagging"
- Hyperparameters: temp=0.1, 512 tokens (NOT 0.0, NOT 256)
- Examples: 0-2 (NOT 5/group to avoid overfitting)
- Safety priority: Shift precision-recall tradeoff toward recall

---

## Strategy Details

### 1. combined_v2_policy_focused

**Purpose**: Policy-dominant strategy emphasizing X Platform Hateful Conduct Policy (80%) with community awareness (20%). Addresses combined_optimised recall deficit while maintaining LGBTQ+ FPR improvements.

**Target Performance**:
- F1-Score: 0.605-0.620 (vs. combined_optimised 0.590, baseline 0.615)
- Recall: 0.580-0.600 (target: close -8.5% gap to baseline's 0.620)
- Mexican FNR: ≤45% (vs. combined_optimised 48%)
- LGBTQ+ FPR: ≤40% (maintain improvement vs. baseline 43%)

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

**Prompt Design**:

- **Policy Foundation (80%)**: X Platform Hateful Conduct Policy as primary framework with explicit "Attacking PEOPLE = HATE, Criticizing POLICIES = NORMAL" distinction. Comprehensive prohibited content categories (slurs, dehumanization, incitement, stereotypes).

- **Recall Optimization**: 
  - "⚠️ CRITICAL: Subtle and coded hate speech IS STILL HATE"
  - "When uncertain, ERR TOWARD FLAGGING"
  - "Patterns that are ALWAYS hate: Generalizations, Dehumanization, Stereotyping, Slurs"
  - Safety-first framing: "Under-detection creates safety risks"

- **Community Context (20%)**: Abbreviated community perspectives providing essential cultural awareness:
  - LGBTQ+: In-group reclamation vs. out-group attacks
  - Mexican/Latino: Immigration hate vs. policy discussion
  - Middle Eastern: Terrorism generalizations vs. specific events

- **Few-Shot Examples**: ZERO examples to avoid overfitting observed in combined_conservative (64.2% Mexican FNR with 5 examples). Model learns from policy principles and pattern descriptions.

**Rationale**: Policy-dominant approach provides authoritative framework for classification while minimizing community-specific complexity. No examples force model to generalize from principles rather than pattern-match. Expected to perform well on clear policy violations while maintaining baseline-level recall.

**Use Case**: Deployments prioritizing policy compliance and consistency. Suitable when policy framework is well-established and examples might introduce bias.

---

### 2. combined_v2_persona_balanced (RECOMMENDED)

**Purpose**: Balanced policy-persona hybrid (50/50) with aggressive recall emphasis and minimal few-shot examples. Optimizes for both performance and demographic fairness.

**Target Performance**:
- F1-Score: 0.610-0.625 (target: BEAT baseline 0.615)
- Recall: 0.590-0.610 (target: approach baseline's 0.620)
- Mexican FNR: ≤40% (vs. combined_optimised 48%, baseline 39.8%)
- LGBTQ+ FPR: ≤40% (maintain/improve vs. baseline 43%)
- Middle Eastern FNR: ≤36% (vs. combined_optimised 42%, baseline 35.2%)

**Hyperparameters**: Same as policy_focused (temp=0.1, 512 tokens, etc.)

**Prompt Design**:

- **Policy + Community Framework (50/50)**: Equal emphasis on X Platform Policy and community-informed perspectives. Policy provides authoritative guidelines; community context provides cultural nuance.

- **Aggressive Recall Emphasis**:
  - "⚠️ Subtle/Coded Hate IS Hate"
  - "When Uncertain → Flag as HATE (safety priority)"
  - Explicit detection checklist: Generalizations, Dehumanization, Coded attacks, Slurs
  - "Default: If potentially harmful → HATE"

- **Community Perspectives (Enhanced)**:
  - LGBTQ+: Speaker context matters (in-group empowerment vs. out-group attacks)
  - Mexican/Latino: Policy vs. people distinction with nuanced guidance
  - Middle Eastern: Specific events vs. generalizations

- **Minimal Few-Shot Examples**: 2 examples per group (6 total) providing concrete guidance without overfitting risk. Examples demonstrate critical distinctions:
  - "Mexicans are all illegals" = ethnic generalization = HATE
  - "Immigration policy reform needed" = policy critique = NORMAL
  - "I'm a proud queer person" = in-group empowerment = NORMAL
  - "Those faggots are disgusting" = out-group attack = HATE

**Rationale**: Balanced approach combines policy authority with community cultural awareness. Minimal examples (2 vs. 5) provide guidance without creating overfitting observed in combined_conservative. Aggressive recall framing shifts precision-recall tradeoff toward safety.

**Use Case**: **RECOMMENDED FOR PRODUCTION** - Best balance between performance, fairness, and recall optimization. Suitable for diverse datasets with multiple protected groups.

---

### 3. combined_v2_recall_optimized

**Purpose**: Aggressive detection strategy maximizing recall over precision. Safety-first approach targeting FNR reduction from 61-64% (combined_conservative) to <45% across all groups.

**Target Performance**:
- F1-Score: 0.600-0.620 (acceptable trade-off for recall gains)
- Recall: 0.595-0.615 (target: match/exceed baseline's 0.620)
- FNR: <45% across ALL groups (vs. combined_conservative 61-64%)
- Precision: 0.605-0.625 (expected slight decline vs. baseline 0.610)

**Hyperparameters**: Same as other strategies (temp=0.1, 512 tokens, etc.)

**Prompt Design**:

- **Aggressive Detection Priority**:
  - "⚠️⚠️⚠️ CRITICAL: Missing hate speech creates SAFETY RISKS"
  - "When UNCERTAIN → classify as HATE (safety-first approach)"
  - "Err toward over-detection rather than under-detection"
  - "False negatives (missing hate) >> false positives (over-flagging)"

- **Default-to-Hate Framework**:
  - "Default: HATE"
  - "Only classify NORMAL if: Clear policy critique, Clear in-group empowerment, Clear factual statement"
  - Inverts burden of proof: Text must prove it's NORMAL

- **7-Point Detection Checklist**:
  1. Slurs or derogatory terms? → HATE
  2. Generalizations ("ALL [group] are [negative]")? → HATE
  3. Dehumanization (animals, diseases, objects)? → HATE
  4. Coded attacks/dog whistles? → HATE
  5. Stereotyping protected groups as dangerous? → HATE
  6. Denying identities or humanity? → HATE
  7. Uncertain but potentially harmful? → HATE

- **Community Perspectives (Recall-Focused)**:
  - LGBTQ+: "Assume OUT-GROUP ATTACK unless clear in-group empowerment"
  - Mexican: "ANY ethnic generalization = HATE"
  - Middle Eastern: "Assuming terrorism/extremism = HATE"

- **Few-Shot Examples**: ZERO examples to maximize generalization and avoid pattern-anchoring.

**Rationale**: Extreme recall optimization for safety-critical deployments. Default-to-hate framework and 7-point checklist provide aggressive detection without relying on examples. Expected to maximize recall (minimize FNR) at cost of moderate precision decline.

**Use Case**: Safety-critical deployments where missing hate speech has severe consequences. Suitable when false positives (over-flagging) are acceptable trade-off for minimizing false negatives (missed hate).

---

### 4. combined_v2_minimal_hybrid

**Purpose**: Conservative policy-persona hybrid (60% policy, 40% persona) with minimal guidance. Maintains combined framework while avoiding over-engineering.

**Target Performance**:
- F1-Score: 0.600-0.615 (match baseline minimum)
- Recall: 0.575-0.595 (moderate improvement vs. combined_optimised 0.567)
- Mexican FNR: ≤45%, LGBTQ+ FPR: ≤42%
- Balanced FNR/FPR without extreme optimization

**Hyperparameters**: Same as other strategies (temp=0.1, 512 tokens, etc.)

**Prompt Design**:

- **Policy + Community Essentials**: Streamlined policy presentation with core community context. Focuses on essential distinctions (people vs. policy, in-group vs. out-group).

- **Moderate Recall Emphasis**:
  - "Subtle/coded hate = HATE"
  - "When uncertain, err toward flagging = HATE"
  - Less aggressive than recall_optimized, more cautious than persona_balanced

- **Concise Community Context**:
  - LGBTQ+: In-group reclamation ("I'm queer") vs. out-group attacks ("those faggots")
  - Mexican: Ethnic attacks ("Mexicans are criminals") vs. policy ("border security")
  - Middle Eastern: Generalizations ("all Muslims") vs. specific events ("ISIS")

- **Detection Guidance**: Simple checklist without exhaustive enumeration:
  - Slurs, generalizations, dehumanization = HATE
  - Subtle/coded hate = HATE
  - Uncertain → HATE

- **Few-Shot Examples**: ZERO examples for minimalist approach.

**Rationale**: Minimal guidance reduces prompt complexity while maintaining core policy-persona framework. Avoids over-engineering risk while providing essential recall boost. Expected to perform consistently without extreme optimization.

**Use Case**: Conservative deployments preferring simplicity over optimization. Suitable when prompt complexity must be minimized or when testing minimal viable improvements.

---

### 5. combined_v2_cultural_context

**Purpose**: Deep cultural awareness strategy (40% policy, 60% persona) integrating power dynamics, historical context, and community norms. Addresses FNR while maintaining cultural sensitivity.

**Target Performance**:
- F1-Score: 0.605-0.620 (competitive with baseline)
- Recall: 0.585-0.605 (moderate-high improvement)
- Bias Fairness: Target best-in-class fairness metrics across groups
- Cultural Sensitivity: Minimize false positives on culturally complex content

**Hyperparameters**: Same as other strategies (temp=0.1, 512 tokens, etc.)

**Prompt Design**:

- **Cultural Analysis Framework (60%)**:
  - **Historical Context**: LGBTQ+ persecution history, Mexican immigration rhetoric weaponization, post-9/11 Islamophobia
  - **Power Dynamics**: Marginalized group member vs. dominant group, in-group reclamation vs. out-group oppression
  - **Community Norms**: Group-specific reclamation practices, policy critique vs. ethnic attacks, discussing events vs. generalizing
  - **Intent vs. Impact**: Content can harm regardless of malice

- **Policy Foundation (40%)**: X Platform Hateful Conduct Policy provides baseline prohibited content framework, integrated with cultural analysis.

- **Multi-Dimensional Evaluation**:
  1. Historical Context: Does this echo historical oppression patterns?
  2. Power Dynamics: In-group empowerment or out-group attack?
  3. Community Norms: Culturally appropriate or harmful?
  4. Harm Assessment: Could this harm community members?

- **Recall Emphasis**:
  - "Subtle/Coded Hate = Hate"
  - "When Uncertain → HATE (community safety priority)"
  - Emphasis on harm to communities rather than pure policy compliance

- **Few-Shot Examples**: ZERO examples; model learns from cultural analysis framework.

**Rationale**: Cultural-dominant approach prioritizes understanding community harm through power dynamics and historical context. Deep cultural analysis expected to improve bias fairness while maintaining recall. May perform better on culturally complex edge cases.

**Use Case**: Deployments requiring cultural sensitivity and nuanced understanding of community harm. Suitable for diverse global audiences where cultural context significantly affects hate speech interpretation.

---

## Validation Protocol

### Phase 1: Quick Validation (100 samples)

```bash
cd q:\workspace\HateSpeechDetection_ver2\prompt_engineering

python prompt_runner.py `
  --data-source canned_100_size_varied `
  --strategies combined_v2_policy_focused combined_v2_persona_balanced combined_v2_recall_optimized combined_v2_minimal_hybrid combined_v2_cultural_context `
  --output-dir outputs/combined_v2/validation_100 `
  --max-workers 15 `
  --batch-size 8 `
  --prompt-template-file combined/combined_v2_bias_optimized.json
```

**Duration**: ~5-8 minutes
**Success Criteria**: Identify 2-3 strategies with F1 ≥ 0.580

### Phase 2: Production Validation (1,009 samples)

```bash
python prompt_runner.py `
  --data-source unified `
  --strategies combined_v2_persona_balanced combined_v2_recall_optimized `
  --output-dir outputs/combined_v2/production `
  --max-workers 15 `
  --batch-size 8 `
  --prompt-template-file combined/combined_v2_bias_optimized.json
```

**Duration**: ~25-30 minutes per strategy

**Success Criteria**:

**Minimum Viable**:
- F1 ≥ 0.600 (vs. combined_optimised 0.590 = +1.7%)
- Recall ≥ 0.580 (vs. combined_optimised 0.567 = +2.3%)
- Mexican FNR ≤ 45% (vs. combined_optimised 48%)

**Target**:
- F1 ≥ 0.615 (match baseline_standard)
- Recall ≥ 0.610 (approach baseline 0.620)
- Mexican FNR ≤ 40%, Middle Eastern FNR ≤ 36%
- LGBTQ+ FPR ≤ 40% (beat baseline 43%)

**Stretch**:
- F1 ≥ 0.625 (+1.6% vs. baseline)
- Recall ≥ 0.630
- FNR ≤ 35% across ALL groups

---

## Key Takeaways

**✅ From combined_optimised**:
- Policy-persona hybrid viable (0.590 F1 respectable)
- LGBTQ+ FPR improvement (43% → 39.2%) proves community context works
- Hyperparameters validated: temp=0.1, 512 tokens
- Core problem: 8.5% recall gap needs aggressive optimization

**❌ From combined_conservative**:
- NEVER temperature=0.0 (39% recall loss)
- NEVER 256 tokens (23% F1 loss)
- Few-shot can create overfitting
- Conservative catastrophic for safety (62% hate missed)

**🎯 Combined V2 Strategy**:
- Keep policy-persona framework (proven)
- Keep hyperparameters (temp=0.1, 512 tokens)
- Add aggressive recall emphasis
- Minimize examples (0-2 vs 5)
- Target: F1 ≥ 0.615, Recall ≥ 0.610

**🚀 Recommended Path**:
1. Run Phase 1 (100 samples, all 5 strategies)
2. Identify top 2-3 performers
3. Run Phase 2 production (1,009 samples)
4. Deploy if F1 ≥ 0.615, monitor if 0.600-0.615, stop if <0.600

---

**Document Version**: 1.1 (Corrected with combined_optimised analysis)
**Last Updated**: November 1, 2025
**Status**: Ready for validation testing
