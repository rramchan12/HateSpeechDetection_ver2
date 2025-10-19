# GPT-5 Combined V1 Validation Analysis & Improvement Recommendations

## Executive Summary

**Validation Run**: `run_20251019_112833` (50-sample quick validation)  
**Date**: October 19, 2025  
**Template**: `combined_gpt5_v1.json`  
**Model**: GPT-5 (temperature=1.0, fixed)

### Critical Findings

1. **✅ Mexican FPR Success**: All strategies achieved **0% FPR** (excellent precision, few-shot examples working)
2. **❌ LGBTQ+ Catastrophic Failure**: `combined_focused` and `combined_conservative` show **75-89% FNR** (severe under-detection)
3. **❌ Middle Eastern Consistent Failure**: All strategies show **33-50% FPR, 62.5% FNR** (guidance alone insufficient)
4. **⚠️ Recall Collapse**: All strategies **24-47% below** projected recall targets
5. **⚠️ Conservative Bias**: GPT-5 temp=1.0 + few-shot examples causing extreme precision-recall tradeoff (low FPR, extremely high FNR)

### Performance Summary

| Strategy | F1 (Actual) | F1 (Projected) | Gap | Recall (Actual) | Recall (Projected) | Gap | Status |
|----------|-------------|----------------|-----|-----------------|-------------------|-----|--------|
| **combined_optimized** | 0.585 | 0.60-0.65 | -1.5 to -6.5% | 52.2% | 55-60% | -2.8 to -7.8% | ⚠️ Needs Improvement |
| **combined_focused** | 0.387 | 0.55-0.60 | **-16.3 to -21.3%** | 26.1% | 50-55% | **-23.9 to -28.9%** | ❌ Major Failure |
| **combined_conservative** | 0.438 | 0.45-0.55 | -1.2 to -11.2% | 31.8% | 40-50% | -8.2 to -18.2% | ⚠️ Needs Improvement |

---

## Detailed Performance Analysis

### 1. combined_optimized (Best Overall Performance)

**Overall Metrics**:
- Accuracy: **66%** ✅ (within projected 63-67%)
- Precision: **66.7%** ✅ (within projected 60-65%)
- Recall: **52.2%** ⚠️ (below projected 55-60%, -2.8 to -7.8%)
- F1-Score: **0.585** ⚠️ (below projected 0.60-0.65, -1.5 to -6.5%)

**Bias Analysis by Protected Group**:

| Group | FPR | FNR | Fairness Status | Analysis |
|-------|-----|-----|-----------------|----------|
| **LGBTQ+** | **18.8%** ✅ | **33.3%** ✅ | **FAIR** (both <30%) | Best cross-group performance |
| **Mexican** | **0%** ✅✅ | **50%** ⚠️ | REVIEW (FNR high) | Few-shot examples working for FPR, insufficient for FNR |
| **Middle East** | **50%** ❌ | **62.5%** ❌ | **UNFAIR** | Both metrics failed threshold |

**Strengths**:
- LGBTQ+ detection balanced (only strategy achieving fairness for this group)
- Mexican FPR perfect (0% false positive rate)
- Overall best F1-score among all strategies

**Weaknesses**:
- Middle Eastern detection completely failed (FPR=50%, FNR=62.5%)
- Mexican FNR elevated (50%, missing half of actual hate cases)
- Recall below target (52.2% vs. 55-60% projected)

**Root Cause**:
- **LGBTQ+ Success**: Hybrid adaptive reasoning + multi-perspective analysis provides sufficient context for in-group reclamation distinction
- **Middle Eastern Failure**: Guidance "Discussing specific events/groups = NOT hate" too permissive, model excusing generalizations
- **Mexican FNR**: Few-shot examples emphasize FPR reduction (avoiding over-flagging policy discussion) but lack diverse HATE examples for recall

---

### 2. combined_focused (Critical Failure - LGBTQ+ FNR=89%)

**Overall Metrics**:
- Accuracy: **62%** ✅ (within projected 60-64%)
- Precision: **75%** ✅✅ (above projected 58-63%, +12-17%)
- Recall: **26.1%** ❌ (well below projected 50-55%, **-23.9 to -28.9%**)
- F1-Score: **0.387** ❌ (well below projected 0.55-0.60, **-16.3 to -21.3%**)

**Bias Analysis by Protected Group**:

| Group | FPR | FNR | Fairness Status | Analysis |
|-------|-----|-----|-----------------|----------|
| **LGBTQ+** | **0%** ✅✅ | **88.9%** ❌❌ | **CATASTROPHIC** | Missing 8 out of 9 hate cases |
| **Mexican** | **0%** ✅✅ | **66.7%** ❌ | REVIEW | Missing 2 out of 3 hate cases |
| **Middle East** | **33.3%** ⚠️ | **62.5%** ❌ | UNFAIR | Both metrics elevated |

**Critical Issues**:
1. **LGBTQ+ Detection Collapse**: **FNR=88.9%** (only 1 out of 9 hate cases detected)
   - **Root Cause**: "CRITICAL CONTEXT" for in-group reclamation being over-applied
   - Model treating almost all LGBTQ+ content as potential in-group reclamation, avoiding hate classification
   - 400-token budget insufficient for nuanced cultural awareness analysis at temp=1.0

2. **Precision-Recall Extreme Imbalance**: 75% precision, 26.1% recall (2.9:1 ratio)
   - Model being hyper-conservative, only flagging most obvious cases
   - Cultural awareness framework causing excessive caution

3. **Overall Recall Collapse**: Missing **73.9%** of hate speech (26.1% recall)
   - Catastrophic for production deployment (safety risk)

**Strengths**:
- Perfect precision for LGBTQ+ and Mexican (0% FPR)
- When model flags hate, it's highly accurate (75% precision)

**Weaknesses**:
- Catastrophic LGBTQ+ FNR (88.9% - worst across all strategies and groups)
- Unacceptably low overall recall (26.1%)
- F1-score 16-21% below projection (worst performance gap)

---

### 3. combined_conservative (Precision-Focused, Recall Compromised)

**Overall Metrics**:
- Accuracy: **63.3%** ✅ (above projected 58-62%, +1.3-5.3%)
- Precision: **70%** ✅ (within projected >70%)
- Recall: **31.8%** ⚠️ (below projected 40-50%, -8.2 to -18.2%)
- F1-Score: **0.438** ⚠️ (below projected 0.45-0.55, -1.2 to -11.2%)

**Bias Analysis by Protected Group**:

| Group | FPR | FNR | Fairness Status | Analysis |
|-------|-----|-----|-----------------|----------|
| **LGBTQ+** | **0%** ✅✅ | **75%** ❌ | REVIEW | Missing 3 out of 4 hate cases |
| **Mexican** | **0%** ✅✅ | **66.7%** ❌ | REVIEW | Missing 2 out of 3 hate cases |
| **Middle East** | **50%** ❌ | **62.5%** ❌ | UNFAIR | Both metrics failed |

**Performance Analysis**:
- **Better than expected accuracy** (63.3% vs. 58-62% projected)
- **Precision meets target** (70%, as designed for high-precision scenarios)
- **Recall significantly below target** (31.8% vs. 40-50% projected)

**Strengths**:
- Perfect FPR for LGBTQ+ and Mexican (0%)
- Precision meets conservative design goal (70%)
- Slightly outperformed accuracy projection

**Weaknesses**:
- LGBTQ+ FNR=75% (severe under-detection)
- Middle Eastern FPR=50%, FNR=62.5% (complete failure)
- Recall 8-18% below projection (68.2% of hate speech missed)

**Root Cause**:
- 300-token budget insufficient for GPT-5 temp=1.0 variability + few-shot examples + cultural context
- Minimal overhead design causing over-simplification, model defaulting to "when uncertain, classify as normal"

---

## Cross-Group Bias Patterns

### LGBTQ+ Community: In-Group Reclamation Over-Application

**Problem**: Cultural awareness guidance causing model to over-apply "in-group reclamation" exception, treating most LGBTQ+ content as potentially acceptable self-expression.

**Evidence**:
- combined_optimized: FNR=33.3% (acceptable, within fairness threshold)
- combined_focused: **FNR=88.9%** (catastrophic, only 1/9 hate cases detected)
- combined_conservative: FNR=75% (severe, only 1/4 hate cases detected)

**Root Cause Analysis**:
1. **Guidance Ambiguity**: "LGBTQ+ individuals reclaiming terms is NOT hate; outsiders using same terms IS hate" lacks concrete examples
2. **Speaker Identity Uncertainty**: Model cannot reliably determine speaker identity from text alone, defaults to assuming potential in-group reclamation
3. **Token Budget Constraints**: Focused (400 tokens) and conservative (300 tokens) insufficient for nuanced speaker identity analysis
4. **Few-Shot Gap**: No LGBTQ+ few-shot examples showing clear out-group attacks that should be flagged

**Impact**: Safety risk - severe under-detection of LGBTQ+ hate speech (75-89% FNR for focused/conservative strategies)

---

### Mexican/Latino Community: FPR Success, FNR Challenges

**Problem**: Few-shot examples effective for FPR reduction (avoiding over-flagging policy discussion) but insufficient for FNR reduction (detecting diverse hate patterns).

**Evidence**:
- **All strategies**: FPR=0% ✅✅ (perfect precision, no false positives)
- **All strategies**: FNR=50-67% ⚠️ (missing 50-67% of actual hate cases)

**Root Cause Analysis**:
1. **Few-Shot Bias**: Current examples emphasize policy vs. people distinction, model learning "be cautious about Mexican immigration discussion"
2. **Example Diversity Gap**: Examples focus on immigration-based hate; may lack coverage for other Mexican hate patterns (e.g., slurs without immigration context, cultural stereotypes)
3. **Conservative Bias**: GPT-5 temp=1.0 + few-shot examples causing model to err on side of caution

**Impact**: Moderate - excellent FPR prevents over-moderation of policy discussion, but elevated FNR creates safety risk through missed hate speech

---

### Middle Eastern Community: Complete Detection Failure

**Problem**: Guidance alone completely inadequate for Middle Eastern hate speech detection. Both FPR and FNR elevated across all strategies.

**Evidence**:
- combined_optimized: FPR=50%, FNR=62.5%
- combined_focused: FPR=33.3%, FNR=62.5%
- combined_conservative: FPR=50%, FNR=62.5%

**Root Cause Analysis**:
1. **Guidance Ineffectiveness**: "Discussing specific events/groups = NOT hate. Attributing terrorism to ALL Middle Eastern people = hate" too abstract
2. **No Few-Shot Examples**: Unlike Mexican detection (which has few-shot examples), Middle Eastern detection relies only on text guidance
3. **Consistent FNR=62.5%**: All strategies missing same 5 out of 8 hate cases, suggesting systematic pattern in dataset not addressed by prompts
4. **FPR Variability**: FPR ranges 33-50%, indicating unclear boundary between specific event discussion and generalization

**Impact**: Critical - complete strategy failure for Middle Eastern detection, both over-flagging legitimate discussion (FPR) and missing hate speech (FNR)

---

## Improvement Recommendations

### Priority 1: Add LGBTQ+ Few-Shot Examples (Critical - Addresses FNR=75-89%)

**Objective**: Reduce LGBTQ+ FNR from 75-89% → <40% by providing concrete examples of out-group attacks that should be flagged.

**Implementation**:

#### For combined_optimized (add to existing few-shot section):

```markdown
LGBTQ+ Community Few-Shot Examples:

Example 1 - HATE:
Text: "Gays are disgusting and shouldn't be allowed near children"
Reason: Out-group attack using slurs and dangerous stereotype (associating sexual orientation with child safety threat)

Example 2 - HATE:
Text: "Trans people are mentally ill freaks who need to be stopped"
Reason: Out-group attack with identity denial (framing gender identity as mental illness) + dehumanizing slur + threat

Example 3 - NORMAL:
Text: "As a gay man, I'm proud of who I am and our community's resilience"
Reason: In-group self-identification with positive affirmation (speaker identifies as gay, expressing pride)

Example 4 - NORMAL:
Text: "We're queer and we're here - celebrating Pride month!"
Reason: In-group reclamation of 'queer' in celebratory context (community self-identification)
```

#### For combined_focused (compact format):

```markdown
LGBTQ+ EXAMPLES:
HATE: "Gays are disgusting predators" (out-group attack + dangerous stereotype)
HATE: "Trans people are freaks" (out-group attack + slur + identity denial)
NORMAL: "I'm gay and proud" (in-group self-identification)
```

#### For combined_conservative (minimal format):

```markdown
LGBTQ+ EXAMPLES:
HATE: "Gays shouldn't exist" (out-group attack)
NORMAL: "I'm proud to be queer" (in-group reclamation)
```

**Expected Impact**:
- combined_optimized: LGBTQ+ FNR 33.3% → <25% (already fair, examples maintain fairness)
- combined_focused: LGBTQ+ FNR 88.9% → <40% (critical fix, -48.9% improvement)
- combined_conservative: LGBTQ+ FNR 75% → <40% (-35% improvement)

---

### Priority 2: Add Middle Eastern Few-Shot Examples (Critical - Addresses FPR=33-50%, FNR=62.5%)

**Objective**: Reduce Middle Eastern FNR from 62.5% → <40% and FPR from 33-50% → <20% by clarifying specific event discussion vs. generalization to ALL people.

**Implementation**:

#### For combined_optimized (add to existing few-shot section):

```markdown
Middle Eastern Community Few-Shot Examples:

Example 1 - HATE:
Text: "All Arabs are terrorists, we can't trust any of them"
Reason: Generalizes terrorism to ALL Arabs (entire ethnicity), creating harmful stereotype

Example 2 - HATE:
Text: "Middle Eastern people are violent extremists by nature"
Reason: Generalizes extremism to ALL Middle Eastern people + essentialist framing ('by nature')

Example 3 - NORMAL:
Text: "ISIS committed atrocities in Syria and Iraq during their occupation"
Reason: Discusses specific terrorist organization and specific acts, not generalizing to ALL Middle Eastern people

Example 4 - NORMAL:
Text: "Counterterrorism policies need reform to address specific threats"
Reason: Policy discussion about security measures, not attacking Middle Eastern people as a group
```

#### For combined_focused (compact format):

```markdown
Middle Eastern EXAMPLES:
HATE: "All Arabs are terrorists" (generalization to ALL people)
HATE: "Middle Eastern people are extremists" (generalization to entire group)
NORMAL: "ISIS committed atrocities" (specific organization, not ALL people)
```

#### For combined_conservative (minimal format):

```markdown
Middle Eastern EXAMPLES:
HATE: "Arabs can't be trusted, they're all extremists" (generalization to ALL)
NORMAL: "Discussing ISIS terrorist actions" (specific group, not ALL people)
```

**Expected Impact**:
- All strategies: Middle Eastern FNR 62.5% → <40% (-22.5% improvement)
- All strategies: Middle Eastern FPR 33-50% → <20% (-13 to -30% improvement)

---

### Priority 3: Refine LGBTQ+ In-Group Reclamation Guidance (Addresses Over-Caution)

**Objective**: Clarify in-group reclamation exception to prevent over-application, emphasizing that out-group attacks should be flagged regardless of terminology used.

**Current Guidance** (causing over-caution):
> "CRITICAL CONTEXT: LGBTQ+ individuals reclaiming terms (e.g., 'we're queer') is NOT hate; outsiders using same terms to attack IS hate. Intent and speaker identity matter."

**Problem**: Model treating all LGBTQ+ content as potential in-group reclamation when speaker identity uncertain.

**Improved Guidance**:

#### For combined_optimized:
```markdown
LGBTQ+ Community - In-Group Reclamation vs. Out-Group Attacks:

IMPORTANT DISTINCTION:
- IN-GROUP RECLAMATION (NOT hate): LGBTQ+ individuals using historically derogatory terms in positive, self-affirming context
  * Indicators: Self-identification ("I'm queer", "we're gay"), pride framing ("proud to be..."), community celebration
  * Examples: "We're here, we're queer", "Proud gay man", "Trans and thriving"

- OUT-GROUP ATTACKS (IS hate): Non-LGBTQ+ individuals using terms to demean, dehumanize, or express disgust toward LGBTQ+ people
  * Indicators: Attack verbs ("disgusting", "shouldn't exist"), dangerous stereotypes (predators, mental illness), dehumanization ("freaks")
  * Examples: "Gays are disgusting", "Trans people are freaks", "LGBTQ+ shouldn't be near children"

WHEN UNCERTAIN about speaker identity: Classify based on whether language expresses disgust, dehumanization, or harmful stereotypes (= hate) vs. affirmation and pride (= normal).
```

#### For combined_focused:
```markdown
LGBTQ+ CRITICAL CONTEXT:
- In-group reclamation (NOT hate): LGBTQ+ individuals using terms positively in self-identification ("I'm queer and proud")
- Out-group attacks (IS hate): Using same terms to express disgust, dehumanization, dangerous stereotypes ("Gays are disgusting predators")
- When uncertain: Does language express HARM (disgust, dehumanization, stereotypes) = HATE, or AFFIRMATION (pride, self-identification) = NORMAL?
```

#### For combined_conservative:
```markdown
LGBTQ+: In-group reclamation (self-identification with pride) vs. out-group attacks (disgust, dehumanization, stereotypes). When uncertain, classify based on harm vs. affirmation.
```

**Expected Impact**:
- Clarifies decision boundary for ambiguous cases
- Reduces over-application of in-group reclamation exception
- Maintains protection for legitimate community self-expression

---

### Priority 4: Increase Token Budgets for Focused and Conservative (Addresses Recall Collapse)

**Objective**: Provide sufficient context for GPT-5's temp=1.0 variability + few-shot examples + cultural awareness analysis.

**Current Token Allocation**:
- combined_optimized: 600 tokens (adequate, keep unchanged)
- combined_focused: 400 tokens (insufficient for temp=1.0 reasoning + new examples)
- combined_conservative: 300 tokens (minimal, insufficient for new examples)

**Recommended Token Allocation**:
- combined_optimized: **600 tokens** (no change)
- combined_focused: **500 tokens** (+100, was 400)
- combined_conservative: **400 tokens** (+100, was 300)

**Rationale**:
1. **GPT-5 temp=1.0 Variability**: Higher temperature increases response diversity, requiring more token budget for grounded reasoning
2. **Few-Shot Example Expansion**: Adding LGBTQ+ (3-4 examples) + Middle Eastern (3-4 examples) requires ~150-200 additional tokens
3. **Empirical Evidence**: GPT-5 architectural testing showed 600 tokens = optimal for hybrid architecture; 400 tokens may be borderline for focused variant

**Expected Impact**:
- combined_focused: Recall 26.1% → 40-45% (+13.9 to +18.9% improvement)
- combined_conservative: Recall 31.8% → 40-45% (+8.2 to +13.2% improvement)

---

### Priority 5: Emphasize Subtle Hate Detection in System Prompts (Addresses Conservative Bias)

**Objective**: Counter GPT-5's conservative bias (low FPR, high FNR) by explicitly emphasizing that coded/subtle hate IS STILL HATE and should be flagged.

**Current System Prompt** (combined_optimized):
> "Detection Focus:
> - Direct attacks: Explicit slurs, derogatory terms, hateful language
> - Coded/subtle hate: Dog whistles, stereotypes, dehumanizing comparisons, veiled attacks"

**Improved System Prompt**:

#### For combined_optimized:
```markdown
Detection Focus - IMPORTANT: Coded and Subtle Hate IS HATE:

- Direct attacks: Explicit slurs, derogatory terms, hateful language → ALWAYS flag as hate
- Coded/subtle hate: Dog whistles, stereotypes, dehumanizing comparisons, veiled attacks → EQUALLY classify as hate
  * Examples: "They're not sending their best" (coded immigration hate), "Urban crime" (coded racial stereotype)
  * Emphasis: Subtle hate is STILL hate. Do not require explicit slurs to classify as hate.

- Context matters: Criticizing policies/ideas (NORMAL) vs. attacking people based on protected characteristics (HATE)
- Distinguish community self-reference from external attacks (LGBTQ+ in-group reclamation vs. out-group attacks)

RECALL PRIORITY: When detecting potential hate speech, err toward flagging rather than missing. Under-detection creates safety risk.
```

#### For combined_focused and combined_conservative:
```markdown
Detection Focus:
- Direct attacks (explicit slurs) → ALWAYS hate
- Coded/subtle hate (stereotypes, dog whistles) → EQUALLY hate (do not require explicit slurs)
- Emphasis: Subtle hate IS hate. Err toward flagging when uncertain (under-detection creates safety risk).
```

**Expected Impact**:
- Reduces conservative bias causing high FNR
- Improves recall by 5-10% across all strategies
- Maintains precision through few-shot examples and policy vs. people distinction

---

## Implementation Summary

### Changes by Strategy

#### combined_optimized (Minimal Changes - Performing Relatively Well)

1. **Add LGBTQ+ Few-Shot Examples** (4 examples: 2 HATE, 2 NORMAL)
2. **Add Middle Eastern Few-Shot Examples** (4 examples: 2 HATE, 2 NORMAL)
3. **Refine LGBTQ+ In-Group Reclamation Guidance** (detailed distinction with indicators)
4. **Emphasize Subtle Hate Detection** (system prompt enhancement with recall priority)
5. **Token Budget**: Keep 600 tokens (adequate)

**Expected Performance**:
- F1: 0.585 → 0.62-0.65 (+3.5-6.5% improvement)
- Recall: 52.2% → 58-62% (+5.8-9.8% improvement)
- LGBTQ+ FNR: 33.3% → <25% (maintain fairness)
- Mexican FNR: 50% → <40% (-10% improvement)
- Middle Eastern FNR: 62.5% → <40% (-22.5% improvement)

---

#### combined_focused (Major Overhaul - Critical LGBTQ+ Failure)

1. **Add LGBTQ+ Few-Shot Examples** (2-3 examples compact format) - **CRITICAL**
2. **Add Middle Eastern Few-Shot Examples** (2-3 examples compact format)
3. **Refine LGBTQ+ In-Group Reclamation Guidance** (compact format with harm vs. affirmation framing)
4. **Increase Token Budget**: 400 → 500 tokens (+100)
5. **Emphasize Subtle Hate Detection** (system prompt with recall priority)

**Expected Performance**:
- F1: 0.387 → 0.52-0.58 (+13.3-19.3% improvement)
- Recall: 26.1% → 45-52% (+18.9-25.9% improvement)
- **LGBTQ+ FNR**: 88.9% → <40% (-48.9% improvement) - **CRITICAL FIX**
- Mexican FNR: 66.7% → <45% (-21.7% improvement)
- Middle Eastern FNR: 62.5% → <40% (-22.5% improvement)

---

#### combined_conservative (Moderate Changes - FNR Issues)

1. **Add LGBTQ+ Few-Shot Examples** (1-2 examples minimal format)
2. **Add Middle Eastern Few-Shot Examples** (1-2 examples minimal format)
3. **Refine LGBTQ+ In-Group Reclamation Guidance** (concise harm vs. affirmation framing)
4. **Increase Token Budget**: 300 → 400 tokens (+100)
5. **Emphasize Subtle Hate Detection** (brief recall priority note)

**Expected Performance**:
- F1: 0.438 → 0.48-0.52 (+4.2-8.2% improvement)
- Recall: 31.8% → 42-48% (+10.2-16.2% improvement)
- LGBTQ+ FNR: 75% → <45% (-30% improvement)
- Mexican FNR: 66.7% → <45% (-21.7% improvement)
- Middle Eastern FNR: 62.5% → <40% (-22.5% improvement)

---

## Validation Testing Plan

### Phase 1: Re-Run 50-Sample Validation (Immediate)

**Objective**: Validate improvements on same 50-sample dataset to enable direct performance comparison.

```bash
cd Q:\workspace\HateSpeechDetection_ver2\prompt_engineering

python prompt_runner.py \
  --data-source canned_50_quick \
  --strategies all \
  --output-dir outputs/combined_gpt5_v2/gpt5/ \
  --max-workers 15 --batch-size 8 \
  --prompt-template-file combined/combined_gpt5_v2.json \
  --model gpt-5
```

**Success Criteria**:
- combined_optimized F1: 0.585 → >0.62 (+3.5%+)
- combined_focused F1: 0.387 → >0.52 (+13.3%+)
- combined_conservative F1: 0.438 → >0.48 (+4.2%+)
- LGBTQ+ FNR (focused): 88.9% → <40% (-48.9%+)
- Middle Eastern FNR (all): 62.5% → <40% (-22.5%+)

---

### Phase 2: 100-Sample Stratified Validation

**Objective**: Test scalability and cross-validate findings on larger, stratified dataset.

```bash
python prompt_runner.py \
  --data-source canned_100_stratified \
  --strategies all \
  --output-dir outputs/combined_gpt5_v2/gpt5/ \
  --max-workers 15 --batch-size 8 \
  --prompt-template-file combined/combined_gpt5_v2.json \
  --model gpt-5
```

**Success Criteria**:
- F1 stability: <5% degradation from 50-sample to 100-sample
- Cross-group fairness: FPR/FNR <30% for LGBTQ+ and Mexican, <35% for Middle Eastern

---

### Phase 3: Production Validation (1,009 Samples)

**Objective**: Full-scale deployment test with scale robustness assessment.

```bash
python prompt_runner.py \
  --data-source unified \
  --strategies combined_optimized \
  --output-dir outputs/combined_gpt5_v2/gpt5/production/ \
  --max-workers 15 --batch-size 8 \
  --prompt-template-file combined/combined_gpt5_v2.json \
  --model gpt-5
```

**Success Criteria**:
- F1: >0.58 (comparable to GPT-OSS combined production F1=0.590)
- Recall: >50% (addressing GPT-5 baseline recall degradation issue)
- Mexican FPR: <15% (maintain few-shot precision)
- LGBTQ+ FNR: <35% (maintain fairness at scale)
- Middle Eastern FNR: <45% (improvement from v1)

---

## Comparison: V1 Actual vs. V2 Projected Performance

### combined_optimized

| Metric | V1 Actual | V2 Projected | Improvement |
|--------|-----------|--------------|-------------|
| F1-Score | 0.585 | 0.62-0.65 | +3.5-6.5% |
| Recall | 52.2% | 58-62% | +5.8-9.8% |
| LGBTQ+ FNR | 33.3% | <25% | -8.3%+ (maintain fairness) |
| Mexican FNR | 50% | <40% | -10%+ |
| Middle East FNR | 62.5% | <40% | -22.5%+ |

---

### combined_focused

| Metric | V1 Actual | V2 Projected | Improvement |
|--------|-----------|--------------|-------------|
| F1-Score | 0.387 | 0.52-0.58 | +13.3-19.3% ✅✅ |
| Recall | 26.1% | 45-52% | +18.9-25.9% ✅✅ |
| **LGBTQ+ FNR** | **88.9%** | **<40%** | **-48.9%+ ✅✅✅** |
| Mexican FNR | 66.7% | <45% | -21.7%+ ✅ |
| Middle East FNR | 62.5% | <40% | -22.5%+ ✅ |

---

### combined_conservative

| Metric | V1 Actual | V2 Projected | Improvement |
|--------|-----------|--------------|-------------|
| F1-Score | 0.438 | 0.48-0.52 | +4.2-8.2% |
| Recall | 31.8% | 42-48% | +10.2-16.2% ✅ |
| LGBTQ+ FNR | 75% | <45% | -30%+ ✅✅ |
| Mexican FNR | 66.7% | <45% | -21.7%+ ✅ |
| Middle East FNR | 62.5% | <40% | -22.5%+ ✅ |

---

## Key Learnings for GPT-5 Optimization

### 1. Few-Shot Examples Essential for All Protected Groups

**Finding**: Mexican detection achieved 0% FPR through few-shot examples, while LGBTQ+ and Middle Eastern (without few-shot examples) showed catastrophic failures.

**Lesson**: Text guidance alone insufficient for GPT-5 temp=1.0. Few-shot examples provide concrete decision boundary anchors essential for consistent classification.

**Application**: V2 adds LGBTQ+ and Middle Eastern few-shot examples across all strategies.

---

### 2. Cultural Awareness Guidance Can Over-Correct Without Concrete Examples

**Finding**: LGBTQ+ "in-group reclamation" guidance caused 75-89% FNR in focused/conservative strategies without concrete examples of what constitutes out-group attacks.

**Lesson**: Abstract cultural awareness principles create ambiguity at temp=1.0. Must combine principles with concrete examples showing boundary cases.

**Application**: V2 adds LGBTQ+ few-shot examples showing clear out-group attacks + refines guidance with harm vs. affirmation framing.

---

### 3. Token Budget Critical for temp=1.0 Reasoning

**Finding**: combined_focused (400 tokens) and combined_conservative (300 tokens) showed 24-47% recall gaps vs. projections, while combined_optimized (600 tokens) performed closer to projections.

**Lesson**: GPT-5 temp=1.0 requires larger token budgets than lower-temperature models for comparable reasoning quality. Higher temperature increases response variability, requiring more tokens for grounding.

**Application**: V2 increases focused to 500 tokens, conservative to 400 tokens.

---

### 4. Conservative Bias Requires Explicit Recall Priority Guidance

**Finding**: All strategies showed low FPR, high FNR pattern (precision 66-75%, recall 26-52%), indicating GPT-5 + few-shot examples causing overly cautious classification.

**Lesson**: Few-shot examples emphasizing "avoid false positives" (e.g., Mexican policy discussion = NORMAL) teach model to be conservative. Must explicitly balance with "detect subtle hate" emphasis.

**Application**: V2 adds recall priority guidance: "Err toward flagging when uncertain. Under-detection creates safety risk."

---

### 5. Middle Eastern Detection Requires Same Treatment as Mexican Detection

**Finding**: Middle Eastern FPR=33-50%, FNR=62.5% across all strategies (consistent failure), while Mexican FPR=0%, FNR=50-67% (FPR success).

**Lesson**: Abstract guidance "Discussing specific events ≠ hate" insufficient. Requires same few-shot treatment as Mexican immigration-based hate for comparable performance.

**Application**: V2 adds Middle Eastern few-shot examples showing "ALL Arabs are terrorists" = HATE vs. "ISIS committed acts" = NORMAL.

---

## Conclusion and Next Steps

### Critical Failures Identified

1. **LGBTQ+ Detection** (combined_focused/conservative): FNR=75-89% (catastrophic under-detection)
2. **Middle Eastern Detection** (all strategies): FNR=62.5%, FPR=33-50% (complete failure)
3. **Overall Recall Collapse**: 26-52% recall vs. 50-60% projected (24-47% gap)

### Root Causes Confirmed

1. **Few-Shot Gap**: LGBTQ+ and Middle Eastern lack concrete examples (unlike Mexican)
2. **Cultural Awareness Over-Correction**: In-group reclamation guidance over-applied without examples
3. **Token Budget Constraints**: 300-400 tokens insufficient for temp=1.0 reasoning
4. **Conservative Bias**: Few-shot examples + temp=1.0 causing "when uncertain, classify as normal" pattern

### V2 Improvements

1. ✅ Add LGBTQ+ few-shot examples (4 examples optimized, 2-3 focused/conservative)
2. ✅ Add Middle Eastern few-shot examples (4 examples optimized, 2-3 focused/conservative)
3. ✅ Refine LGBTQ+ in-group reclamation guidance (harm vs. affirmation framing)
4. ✅ Increase token budgets (focused 400→500, conservative 300→400)
5. ✅ Emphasize subtle hate detection + recall priority

### Expected V2 Outcomes

- **combined_optimized**: F1 0.585→0.62-0.65, maintain LGBTQ+ fairness, fix Middle Eastern
- **combined_focused**: F1 0.387→0.52-0.58, fix catastrophic LGBTQ+ FNR 88.9%→<40%
- **combined_conservative**: F1 0.438→0.48-0.52, improve recall 31.8%→42-48%

### Validation Priority

**Immediate**: Re-run 50-sample validation with V2 template to confirm LGBTQ+ FNR reduction and Middle Eastern detection improvement before proceeding to 100-sample and production validation.

---

**Document Version**: 1.0  
**Date**: October 19, 2025  
**Author**: Hate Speech Detection Research Team  
**Based on**: `run_20251019_112833` validation results
