# Combined V2: Policy-Persona Hybrid with V1 Few-Shot Learnings

## Overview

This document presents 3 revised strategies that incorporate proven V1 few-shot learnings after Combined V2 testing revealed that aggressive recall language contradicts example-based pattern learning.

**Model**: gpt-oss-120b (Phi-3.5-MoE-instruct)

**Base Framework**: combined_gptoss_v1.json (policy-persona hybrid with 5-example few-shot structure)

**Development Context**: Combined V2 testing (100 samples, 5 strategies) validated that V1 few-shot approach is correct. All V2 strategies with reduced examples (0-2) and aggressive "err toward flagging" language underperformed (F1=0.378-0.565 vs. V1s 0.571). Root cause: Conflicting text instructions dominated over few-shot learning signals.

**Key Insight**: Models learn better from **examples than imperatives**. When text says "when uncertain to HATE" but examples show "immigration policy = NORMAL", the model becomes conservative rather than learning the pattern.

---

## V2 Testing Results: What We Learned

### Initial V2 Approach (100 samples, 5 strategies)

**Hypothesis (WRONG)**: Reduce few-shot examples (0-2) to avoid overfitting + add aggressive "err toward flagging" language to boost recall.

**Results (ALL UNDERPERFORMED)**:

| Strategy | Examples | F1 | Recall | Mexican FNR | LGBTQ+ FPR | Verdict |
|----------|----------|-----|--------|-------------|------------|---------|
| cultural_context | 0 | 0.565 | 0.553 | 50% | 43.3% | Best but still below V1 |
| recall_optimized | 0 | 0.557 | 0.574 | 58.3% | 50% | Recall high but F1 suffered |
| policy_focused | 0 | 0.521 | 0.532 | 41.7% | 56.7% | Poor LGBTQ+ performance |
| persona_balanced | 2 | 0.506 | 0.426 | 58.3% | 30% | Low recall despite examples |
| minimal_hybrid | 0 | 0.378 | 0.298 | 75% | 26.7% | Catastrophic failure |

**Comparison to V1**:

- **V1 Best (combined_optimised)**: F1=0.571, Mexican FNR=33% (size_varied, 50 samples)
- **V1 Production**: F1=0.590, Mexican FPR=7% (1,009 samples)
- **V2 Best**: F1=0.565 (-1% vs. V1)
- **V2 Worst**: F1=0.378 (-34% vs. V1)

### Root Cause Analysis: Conflicting Signals

**V2 Prompt Structure (PROBLEMATIC)**:
System Prompt: "CRITICAL: When uncertain, ERR TOWARD FLAGGING" + "Subtle hate IS hate" + "Default: HATE"
User Template: EXAMPLES show "Immigration policy reform = NORMAL" (CONTRADICTION)
Model Behavior: Text imperative dominates -> Conservative classification despite aggressive framing

**Why V2 Failed**:

1. **Insufficient Examples**: 0-2 examples provided weak signal compared to 5 in V1
2. **Conflicting Instructions**: Aggressive text contradicted NORMAL examples
3. **Signal Priority**: Model prioritized text over few-shot learning
4. **Example Visibility**: Examples buried in long guidance text

**V1 Success Formula (PROVEN)**:
5 examples (3 HATE, 2 NORMAL) in dedicated section + balanced guidance
Examples = PRIMARY signal, text = SECONDARY support
Result: Mexican FNR 83% to 33% (-60% improvement)

---

## Corrected Approach: V1 Few-Shot Integration

### Design Principles

1. **Examples = PRIMARY Signal**: Use 2-5 Mexican examples with clear HATE/NORMAL labels
2. **Balanced Guidance**: Remove aggressive "err toward flagging" language
3. **High Visibility**: Dedicated "EXAMPLES" sections (not buried in text)
4. **Harmonious Messaging**: Text supports examples, doesn''t contradict them
5. **Pattern Learning**: Let model learn from examples rather than follow imperatives

### Strategy Comparison

| Strategy | Examples | Structure | Expected Performance |
|----------|----------|-----------|---------------------|
| combined_v2_optimized | 5 Mexican | V1 proven formula | F1=0.571-0.590, Mexican FNR less or equal to 35% |
| combined_v2_focused | 2 Mexican | V1 compact format | F1=0.550-0.570, Efficient |
| combined_v2_cultural_context | 2 Mexican | V1 + cultural analysis | F1=0.560-0.580, Bias-aware |

**Expected Improvements vs. Failed V2**:

- Mexican FNR: 50-75% (failed V2) to 30-40% (V1 formula)
- F1: 0.378-0.565 (failed V2) to 0.550-0.590 (V1 formula)
- Recall: Balanced with precision (not artificially inflated)

---

## Hyperparameter Configuration

All strategies use identical hyperparameters derived from empirical optimization:

```json
{
  "max_tokens": 512,
  "temperature": 0.1,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "response_format": "json_object"
}
```

**Rationale**:

- **Temperature 0.1**: Optimal for classification (16.6% F1 decrease at 0.5)
- **Max Tokens 512**: "Goldilocks zone" for rationale without hedging
- **Top P 1.0**: Minimal impact at low temperature (less than 2% variance)
- **Zero Penalties**: Negligible effect on short-form tasks (less than 0.05 F1 variance)

---

## Strategy Details

### 1. combined_v2_optimized (RECOMMENDED)

**Description**: 50/50 policy-persona hybrid with V1 proven 5-example few-shot structure.

**Few-Shot Structure** (V1 Formula): 5 Mexican examples in dedicated EXAMPLES section with clear HATE/NORMAL labels

**System Prompt**: Balanced guidance - Direct attacks, coded/subtle hate, context matters. Distinguish community self-reference from external attacks. Policy criticism not equal to people attacks.

**Expected Performance**:

- F1: 0.571-0.590 (match/exceed V1)
- Mexican FNR: less or equal to 35% (restore from 50-75% in failed V2)
- LGBTQ+ FPR: less or equal to 40% (maintain improvement)
- Recall: 0.55-0.60 (balanced with precision)

**Use Case**: Primary strategy for production validation. Proven formula from V1 with policy-persona balance.

### 2. combined_v2_focused

**Description**: Streamlined policy-persona hybrid with V1 few-shot structure in compact format.

**Few-Shot Structure**: 2 examples, compact format

**System Prompt**: Concise rules - Hate if violates X policy or targets protected communities (including coded/subtle). Normal if no violation and no community harm.

**Expected Performance**:

- F1: 0.550-0.570 (competitive efficiency)
- Mexican FNR: 35-45% (improvement over failed V2)
- Efficiency: Shorter prompts, faster inference

**Use Case**: Balance between performance and efficiency.

### 3. combined_v2_cultural_context

**Description**: Deep cultural awareness integrating V1 few-shot examples with power dynamics and historical context.

**Few-Shot Structure**: 2 examples + cultural framing

**System Prompt**: Multi-dimensional cultural framework - Historical context, power dynamics, community norms

**Expected Performance**:

- F1: 0.560-0.580 (cultural awareness + pattern learning)
- LGBTQ+ FPR: less or equal to 35% (best in-group/out-group distinction)
- Mexican FNR: 35-45% (cultural context supports pattern)

**Use Case**: Scenarios requiring nuanced cultural understanding. Best for LGBTQ+ in-group reclamation cases.

---

## Validation Plan

### Phase 1: 100-Sample Validation

**Objective**: Verify V1 few-shot formula replicates in Combined V2 context.

**Dataset**: canned_100_size_varied (balanced representation of 3 communities)

**Strategies to Test**: All 3 (combined_v2_optimized, combined_v2_focused, combined_v2_cultural_context)

**Success Criteria**:

- Minimum: F1 greater or equal to 0.571 (match V1 combined_optimised)
- Target: F1 greater or equal to 0.580 (approach V1 production)
- Mexican FNR: less or equal to 35% (restore from 50-75% in failed V2)
- LGBTQ+ FPR: less or equal to 40%
- At least 1 strategy meets minimum criteria

**Command**:

```bash
python prompt_runner.py --data-source canned_100_size_varied --strategies combined_v2_optimized combined_v2_focused combined_v2_cultural_context --output-dir outputs/combined_v2_revised/validation_100 --max-workers 15 --batch-size 8 --prompt-template-file combined/combined_v2_bias_optimized.json
```

### Phase 2: Production Validation (If Phase 1 Successful)

**Objective**: Confirm top-performing strategies from Phase 1 scale to production.

**Dataset**: 1,009 production samples

**Strategies**: Top 1-2 from Phase 1 (expect combined_v2_optimized + one other)

**Success Criteria**:

- Target: F1 greater or equal to 0.615 (match baseline_standard)
- Mexican FPR: less or equal to 10% (V1 achieved 7%)
- LGBTQ+ FPR: less or equal to 40%
- Middle Eastern FNR: less or equal to 40%

---

## Expected Outcomes

### Why This Will Work (V1 Evidence)

**Combined V1 Few-Shot Success**:

- Size_varied (50 samples): Mexican FNR 83% to 33% (-60% improvement)
- Production (1,009 samples): Mexican FPR 17% to 7% (-76% improvement)
- Structure: 5 examples in dedicated section + balanced guidance
- Result: F1=0.571 (size_varied), F1=0.590 (production)

**Combined V2 Revised Incorporates**:

- 5-example structure (combined_v2_optimized) from V1
- Balanced guidance (no aggressive contradictions)
- High visibility examples (dedicated sections)
- Harmonious text (supports examples, doesn''t contradict)

### Failure Scenarios and Mitigation

**Scenario 1: F1 less than 0.571 (Worse than V1)**
Possible causes: Prompt corruption, example visibility reduced, text still conflicting
Diagnostic steps: Manual review of labels, compare to V1 structure, test with exact V1 system prompt

**Scenario 2: Mexican FNR greater than 40% (Pattern Not Learning)**
Possible causes: Examples not prominent, model ignoring examples, policy framing overwhelming
Diagnostic steps: Increase to 7-9 examples, reorder examples to top, simplify text

**Scenario 3: LGBTQ+ FPR greater than 50% (Over-Correction)**
Possible causes: In-group logic too permissive, missing slur examples, cultural context overriding policy
Diagnostic steps: Add LGBTQ+ few-shot examples, tighten in-group logic, test focused vs cultural_context

---

## Implementation Notes

### Few-Shot Example Selection Criteria

**HATE Examples**: Generalizations ("ALL group are negative trait"), dehumanization (animals/diseases comparisons), coded attacks ("not sending their best", "go back")

**NORMAL Examples**: Policy critique (system-level discussion without people attacks), in-group reclamation (LGBTQ+ empowerment language), specific events (not generalizing ALL people)

**Balance**: 3 HATE, 2 NORMAL per group (60/40 ratio matches dataset distribution)

### Prompt Maintenance Guidelines

**DO**:

- Keep examples in dedicated, high-visibility sections
- Use clear HATE/NORMAL labels
- Write balanced guidance that supports examples
- Test changes on size_varied before production

**DON''T**:

- Add aggressive "err toward flagging" imperatives
- Bury examples in long guidance paragraphs
- Create conflicting signals (text vs. examples)
- Reduce examples below 2 without explicit testing

---

## References

- **V1 Success**: combined_gptoss_v1_README.md - Documented Mexican FNR 83% to 33% improvement
- **V2 Failure**: Combined V2 validation results (100 samples) - F1=0.378-0.565, all underperformed
- **Hyperparameters**: gptoss_ift_summary_README.md - Empirical temp=0.1, 512 tokens rationale
- **Baseline**: baseline_standard - F1=0.615 target for production

---

## Conclusion

Combined V2 testing validated that V1 few-shot approach is correct. The issue was not overfitting but rather insufficient examples + conflicting text instructions. By restoring V1 proven 5-example structure with balanced guidance, we expect to:

1. Match/exceed V1 performance: F1=0.571-0.590
2. Restore Mexican FNR: 30-40% (from 50-75% in failed V2)
3. Maintain LGBTQ+ improvements: FPR less or equal to 40%
4. Scale to production: Competitive with baseline_standard (F1=0.615)

**Next Step**: Run Phase 1 validation (100 samples) to confirm V1 formula replicates in Combined V2 context.
