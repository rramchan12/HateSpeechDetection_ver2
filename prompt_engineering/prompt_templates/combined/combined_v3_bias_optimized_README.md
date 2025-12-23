# Combined V3: Return to V1 Few-Shot Success with V2 Hyperparameter Improvements

## Overview

This document presents 3 refined strategies that restore V1's proven 5-example few-shot approach while incorporating V2's hyperparameter optimizations.

**Model**: gpt-oss-120b (Phi-3.5-MoE-instruct)

**Base Framework**: combined_gptoss_v1.json (policy-persona hybrid with 5-example few-shot structure)

**Development Context**: Combined V2 testing (100 samples, 5 strategies) revealed that reducing examples to 0-2 was a mistake. V2 best (F1=0.565) still underperformed V1 (F1=0.571 small, F1=0.590 production). Critical finding: **2 examples = WORST performance (F1=0.506)**, worse than 0 examples (F1=0.565). V1's 5-example approach achieved F1=0.667 (BEST EVER) but used suboptimal hyperparameters (temp=0.0, 256 tokens).

**V3 Hypothesis**: V1's 5-example approach was correct. The issue wasn't the examples—it was the hyperparameters (temp=0.0 too rigid, 256 tokens insufficient). V3 tests V1's proven few-shot structure with V2's optimized hyperparameters.

---

## Performance History: What We Learned

### V1 Results (100 samples - run_20251018_232343)

**combined_optimized** (5 examples, temp=0.1, 512 tokens):
- **F1-Score**: 0.614
- **Accuracy**: 61%
- **Recall**: 0.660
- **Mexican FNR**: 58.3%
- **LGBTQ+ FPR/FNR**: 56.7%/21.1%

**combined_conservative** (5 examples, temp=0.0, 256 tokens):
- **F1-Score**: 0.500
- **Accuracy**: 56%
- **Recall**: 0.468
- **Mexican FNR**: 58.3%
- **LGBTQ+ FPR/FNR**: 43.3%/47.4%

**V1 Production Results** (1,009 samples - combined_optimized):
- **F1-Score**: 0.590 (-2.4% from 100-sample)
- **Recall**: 0.567
- **Mexican FPR**: 7.0% (excellent), FNR: 48.0%
- **LGBTQ+ FPR**: 39.2%, FNR: 41.2%

### V2 Results (100 samples) - ALL UNDERPERFORMED

| Strategy | Examples | F1 | Recall | Mexican FNR | LGBTQ+ FPR | Issue |
|----------|----------|-----|--------|-------------|------------|-------|
| cultural_context | 0 | **0.565** | 0.553 | 50% | 43.3% | Best V2, still below V1 |
| recall_optimized | 0 | 0.557 | 0.574 | 58.3% | 50% | Moderate |
| policy_focused | 0 | 0.521 | 0.532 | 41.7% | 56.7% | LGBTQ+ overcriminalization |
| **persona_balanced** | **2** | **0.506** | 0.426 | 58.3% | 30% | **WORST: 2 examples = confusion** |
| minimal_hybrid | 0 | 0.378 | 0.298 | 75% | 26.7% | Catastrophic failure |

**Comparison to V1**:

- V1 Best (100 samples): F1=0.614 with **5 examples** (combined_optimized)
- V1 Production: F1=0.590 with **5 examples** (1,009 samples)
- V2 Best: F1=0.565 with **0 examples** (-4.3% vs V1 100-sample, -4% vs V1 production)
- V2 Recommended: F1=0.506 with **2 examples** (-17.6% vs V1 100-sample, -14% vs V1 production)

### Critical V2 Learnings

**1. Few-Shot Pattern Analysis**:

- **5 examples** (V1): F1=0.614 (100 samples)  SUCCESS - enough patterns to learn
- **2 examples** (V2): F1=0.506  WORST - insufficient learning + confusion
- **0 examples** (V2): F1=0.557-0.565  MODERATE - pure instruction-following

**Conclusion**: It's not overfitting, it's **UNDERFITTING**. 2 examples creates worst-case scenario: not enough to learn patterns, but enough to create confusion. 0 examples forces pure instruction-following. 5 examples provides sufficient patterns for learning.

**2. V2's Mistake**: Abandoned what worked (5 examples) without testing it with improved hyperparameters (temp=0.1, 512 tokens).

**3. Scale Hypothesis Untested**: V1's combined_conservative (temp=0.0, 256 tokens) achieved F1=0.500 on 100 samples, suggesting hyperparameters were the issue, not the examples. Was the problem the examples OR the hyperparameters? V2 never tested this.

**4. V2's Success Elements**:

-  Temperature 0.1 (vs V1's 0.0 for conservative)
-  512 tokens (vs V1's 256 for conservative)
-  Balanced guidance (cultural_context worked best)
-  Reduced examples to 0-2 (unnecessary abandonment)

---

## V3 Strategy: Best of Both Worlds

### Design Principles

**From V1 (Keep)**:
1.  **5 examples per group** (15 total) - proven success formula
2.  **Examples as PRIMARY signal** - dedicated sections, high visibility
3.  **Pattern learning** - let model learn from examples, not imperatives
4.  **Balanced guidance** - no aggressive contradictions

**From V2 (Incorporate)**:
1.  **Temperature 0.1** (NOT 0.0) - determinism + generalization
2.  **512 tokens** (NOT 256) - sufficient rationale space
3.  **Cultural context** (from best V2 strategy)
4.  **Structured evaluation frameworks**

**V3 Innovation**:

- Test if V1's 5-example approach scales with V2's improved hyperparameters
- 3 strategies varying guidance style, all with 5 examples
- Target: Match/exceed V1's F1=0.614 at 100-sample scale and improve production beyond 0.590

---

## Hyperparameter Configuration

All V3 strategies use **V2's optimized hyperparameters**:

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
- **Temperature 0.1**: Balances determinism with generalization (V1's 0.0 too rigid)
- **Max Tokens 512**: Sufficient for rationale (V1's 256 too limiting)
- **Top P 1.0**: Minimal impact at low temperature
- **Zero Penalties**: Negligible effect on classification

**Critical Change from V1**: V1's combined_conservative used temp=0.0 + 256 tokens (F1=0.500 on 100 samples). V1's combined_optimized used temp=0.1 + 512 tokens (F1=0.614 on 100 samples, 0.590 on production). V3 uses the optimized hyperparameters across all strategies.

---

## Strategy Comparison Matrix

| Strategy | Policy% | Persona% | Examples | Guidance Style | Target F1 | Use Case |
|----------|---------|----------|----------|----------------|-----------|----------|
| **Baseline** | N/A | N/A | 0 | Policy-only | 0.615 | Current champion |
| **V1 Best** | 50% | 50% | 5/group | Balanced | 0.614 (100) / 0.590 (1009) | V1 optimized |
| **V2 Best** | 40% | 60% | 0 | Cultural | 0.565 | V2 winner (still below V1) |
| **combined_v3_optimized** | 50% | 50% | **5/group** | Balanced | **0.600-0.650** | **RECOMMENDED** |
| **combined_v3_recall_focused** | 60% | 40% | **5/group** | Recall-priority | **0.590-0.640** | Safety-first |
| **combined_v3_cultural_aware** | 40% | 60% | **5/group** | Cultural-deep | **0.600-0.650** | Bias-aware |

**V3 Advantages**:

- Restores V1's proven 5-example structure
- Uses V2's optimized hyperparameters (temp=0.1, 512 tokens)
- Tests scale hypothesis: Examples + better hyperparameters = production success?
- Expected: Match/exceed V1's F1=0.614 (100 samples) with better generalization

---

## Strategy Details

### 1. combined_v3_optimized (RECOMMENDED)

**Description**: 50/50 policy-persona hybrid restoring V1's proven 5-example structure with V2's optimized hyperparameters. Direct successor to V1's combined_optimized (F1=0.614 on 100 samples, 0.590 on production).

**Few-Shot Structure**: 5 examples per group (LGBTQ+, Mexican, Middle Eastern) in dedicated EXAMPLES sections with clear HATE/NORMAL labels.

**System Prompt**: Balanced policy-persona guidance emphasizing:

- X Platform Hateful Conduct Policy (attacks on people = hate)
- Community perspectives (in-group vs out-group, policy vs people)
- Examples as PRIMARY learning signal
- Pattern recognition over imperatives

**Expected Performance**:

- **F1-Score**: 0.600-0.650 (target: match/exceed V1's 0.614)
- **Mexican FNR**: ≤35% (restore from V2's 50-75%)
- **LGBTQ+ FPR**: ≤40% (maintain improvement)
- **Recall**: 0.580-0.620 (balanced with precision)

**Rationale**: V1 proved 5 examples work (F1=0.614 on 100 samples). V3 maintains the successful V1 optimized configuration while testing if it can beat V2's best (0.565) and approach baseline (0.615).

**Use Case**: Primary production candidate. Best balance of V1's example-based learning and V2's hyperparameter optimization.

---

### 2. combined_v3_recall_focused

**Description**: Recall-optimized strategy with V1's 5-example structure and moderate "err toward flagging" guidance (less aggressive than V2's failed attempts).

**Few-Shot Structure**: 5 examples per group with emphasis on subtle/coded hate patterns.

**System Prompt**: Policy-dominant (60%) with community awareness (40%):
- "Subtle and coded hate IS hate" (moderate framing, not aggressive)
- Examples demonstrate subtle patterns (dog whistles, generalizations)
- Safety-aware but not "default to HATE" (V2's recall_optimized mistake)

**Expected Performance**:
- **F1-Score**: 0.590-0.640
- **Recall**: 0.600-0.640 (prioritized)
- **Mexican FNR**: ≤40%
- **Precision**: 0.580-0.620 (acceptable trade-off)

**Rationale**: V2's aggressive recall strategies (default to HATE, err toward flagging) performed poorly when combined with 0-2 examples. With 5 examples providing strong patterns, moderate recall emphasis should succeed without overcriminalization.

**Use Case**: Safety-critical deployments where missing hate has higher cost than over-flagging.

---

### 3. combined_v3_cultural_aware

**Description**: Deep cultural awareness integrating V1's 5-example structure with V2's best-performing cultural context framework (F1=0.565).

**Few-Shot Structure**: 5 examples per group embedded in cultural analysis framework:
- Historical context (persecution, immigration rhetoric, Islamophobia)
- Power dynamics (in-group vs out-group)
- Community norms (reclamation, policy critique)

**System Prompt**: Persona-dominant (60%) with policy foundation (40%):
- Multi-dimensional cultural analysis
- Examples demonstrate cultural nuances
- Pattern learning through cultural lens

**Expected Performance**:
- **F1-Score**: 0.600-0.650
- **LGBTQ+ FPR**: ≤35% (best in-group/out-group distinction)
- **Mexican FNR**: ≤40%
- **Middle Eastern FNR**: ≤36%

**Rationale**: V2's cultural_context was BEST strategy (F1=0.565) with 0 examples. Adding V1's 5 examples to this framework should amplify success by combining cultural awareness with concrete pattern learning.

**Use Case**: Scenarios requiring nuanced cultural understanding. Best for diverse datasets with multiple protected groups and complex in-group dynamics.

---

## Validation Plan

### Phase 1: 100-Sample Validation (CRITICAL TEST)

**Objective**: Verify V1's 5-example formula succeeds with V2's hyperparameters at 100-sample scale.

**Dataset**: canned_100_size_varied (same as V2 for direct comparison)

**Strategies to Test**: All 3 (combined_v3_optimized, combined_v3_recall_focused, combined_v3_cultural_aware)

**Success Criteria**:

- **Minimum**: F1 ≥ 0.565 (beat V2 best)
- **Target**: F1 ≥ 0.600 (approach V1's 0.614)
- **Stretch**: F1 ≥ 0.620 (beat V1's 0.614)
- **Mexican FNR**: ≤40% (improve from V2's 50-75%)
- **LGBTQ+ FPR**: ≤40% (maintain)
- **At least 1 strategy meets target criteria**

**Command**:

```bash
python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_v3_optimized combined_v3_recall_focused combined_v3_cultural_aware \
  --output-dir outputs/combined_v3/validation_100 \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v3_bias_optimized.json
```

**Duration**: ~10-15 minutes

---

### Phase 2: Production Validation (If Phase 1 Successful)

**Objective**: Confirm top-performing V3 strategies scale to production (1,009 samples).

**Dataset**: unified (1,009 production samples)

**Strategies**: Top 1-2 from Phase 1 (expect combined_v3_optimized + one other)

**Success Criteria**:
- **Target**: F1 ≥ 0.615 (match/beat baseline_standard)
- **Stretch**: F1 ≥ 0.620 (beat baseline by 5%)
- **Mexican FPR**: ≤10% (V1 achieved 7%)
- **Mexican FNR**: ≤40% (improve from V1's 48%)
- **LGBTQ+ FPR**: ≤40%
- **Middle Eastern FNR**: ≤36%

**Scale Test**: Does V1's 5-example approach maintain F1=0.600+ at 10x scale (100→1,009)?

---

## Expected Outcomes

### Why V3 Will Succeed Where V2 Failed

**V1 Evidence (5 Examples Work)**:

- combined_optimized: F1=0.614 (100 samples), F1=0.590 (production)
- Mexican FPR: 20%→7% at scale (improvement)
- Examples were KEY to success, not a problem

**V2 Evidence (Wrong Number of Examples)**:

- 0 examples: F1=0.565 (moderate, instruction-following)
- 2 examples: F1=0.506 (WORST, confusion)
- Never tested 5 examples with improved hyperparameters

**V3 Approach (Fix the Gap)**:

- Restore 5 examples (V1 proven)
- Use temp=0.1 + 512 tokens (V2 proven, already used by V1 optimized)
- Test hypothesis: Can we beat V2's 0.565 and approach baseline 0.615?

**Expected Improvements vs V2**:

1. **F1**: 0.565 (V2 best) → 0.600-0.650 (V3 target)
2. **Mexican FNR**: 50-75% (V2) → 30-40% (V3)
3. **Recall**: 0.553-0.574 (V2) → 0.580-0.620 (V3)
4. **Consistency**: V3 strategies should cluster 0.600-0.650 (vs V2's 0.378-0.565 spread)

---

## Failure Scenarios and Mitigation

### Scenario 1: F1 < 0.565 (Worse than V2 Best)

**Possible Causes**:
- 5 examples truly overfit at 100+ sample scale
- Examples conflict with hyperparameters
- Prompt structure issues

**Diagnostic Steps**:
1. Compare to V2 cultural_context (0 examples, F1=0.565)
2. Manual review: Are examples being followed?
3. Test with 3-4 examples (between V2's 2 and V1's 5)

**Mitigation**: If 5 examples fail, try combined_v3.5 with 3 examples per group (9 total) as middle ground.

---

### Scenario 2: F1 0.565-0.600 (Better than V2, Worse than V1 100-sample)

**Interpretation**: Scale IS the issue. 5 examples work at 100 samples (F1=0.614 in V1) but degrade with different prompt structure.

**Diagnostic Steps**:

1. Compare Mexican FNR: Is improvement vs V2 (50-75%) achieved?
2. Test on production (1,009): Does degradation continue or stabilize?

**Next Steps**: If stable at 0.580-0.600, acceptable for production. If degrading, consider:

- Hybrid approach: 3 examples + enhanced guidance
- Example diversity: Add more varied examples

---

### Scenario 3: F1 ≥ 0.600 (SUCCESS)

**Validation**: V1's 5-example approach scales with proper hyperparameters.

**Production Test**: Deploy top strategy on 1,009 samples.

**Target**: F1 ≥ 0.615 (match/beat baseline_standard).

**Deployment Decision**: If production F1 ≥ 0.615 with better bias metrics (Mexican FNR <40%, LGBTQ+ FPR <40%), recommend for production over baseline.

---

## Key Takeaways

**What V2 Taught Us**:

1.  2 examples = worst case (insufficient + confusion)
2.  0 examples = moderate (instruction-following only)
3. ❓ 5 examples at scale with better hyperparameters = UNTESTED (V3's goal)
4.  Temperature 0.1 + 512 tokens = optimal hyperparameters

**What V1 Proved**:

1.  5 examples = F1=0.614 (100 samples), F1=0.590 (production)
2.  Examples improve Mexican FPR: 20%→7% at scale
3.  Temperature 0.1 + 512 tokens works (combined_optimized)
4.  Temperature 0.0 + 256 tokens = poor performance (F1=0.500 on 100 samples)

**V3's Hypothesis**:

- V1 was RIGHT about examples (5 = optimal)
- V1 already used optimal hyperparameters (temp=0.1, 512 tokens in combined_optimized)
- V2 was WRONG to abandon examples (0-2 underperformed)
- **V3 tests**: Can V1's approach beat V2's best (0.565) and approach baseline (0.615)?

**Success Definition**:

- Phase 1: F1 ≥ 0.600 on 100 samples (beat V2 by 6%+)
- Phase 2: F1 ≥ 0.615 on production (match/beat baseline)
- Bias: Mexican FNR ≤40%, LGBTQ+ FPR ≤40%

---

## References

- **V1 Success**: gpt_oss_combined_ift_summary_README.md - F1=0.614 (100 samples), F1=0.590 (production) with 5 examples
- **V1 Configuration**: combined_gptoss_v1.json - Original 5-example template
- **V2 Testing**: evaluation_report_20251101_125228.txt - All strategies F1=0.378-0.565
- **V2 Best**: cultural_context F1=0.565 (0 examples)
- **V2 Worst**: persona_balanced F1=0.506 (2 examples)
- **Baseline**: baseline_standard F1=0.615 (production target)

---

## Conclusion

Combined V2 testing revealed that reducing examples to 0-2 was premature. V1's 5-example approach achieved F1=0.614 on 100 samples and F1=0.590 on production (1,009 samples), demonstrating that examples work at scale. V2 proved that different guidance structures affect performance, but never tested the V1 approach that already used optimal hyperparameters (temp=0.1, 512 tokens in combined_optimized).

**V3 fills this critical gap**: Test V1's proven few-shot approach with refined guidance from V2's learnings.

**Expected Outcomes**:

1. Phase 1: F1=0.600-0.650 on 100 samples (beat V2's 0.565)
2. Phase 2: F1≥0.615 on production (match/beat baseline)
3. Bias: Mexican FNR restored to ≤40%, LGBTQ+ FPR maintained ≤40%
4. Validation: Examples + refined guidance = production-ready

**Next Step**: Run Phase 1 validation (100 samples) to test hypothesis.
