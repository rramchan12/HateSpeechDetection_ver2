# Combined V4: Baseline-Enhanced Strategies for Hate Speech Detection

## Executive Summary

**Objective**: Beat baseline_standard's F1=0.615 (production, 1,009 samples) by adding MINIMAL enhancements to the proven baseline structure.

**Critical Context**: V1, V2, and V3 combined approaches ALL failed to beat baseline:
- **V1 combined_optimized**: F1=0.590 (production) - BELOW baseline by 2.5%
- **V2 best (cultural_context)**: F1=0.565 (100 samples) - BELOW baseline by 5%
- **V3 best (recall_focused)**: F1=0.559 (100 samples) - BELOW baseline by 5.6%
- **V3 worst (optimized)**: F1=0.438 (100 samples) - CATASTROPHIC 28.6% degradation

**Key Lesson Learned**: Adding complexity (verbose prompts, 15 examples, structured frameworks) consistently DEGRADES performance. V3's attempt to "improve" V1 with more structure resulted in catastrophic recall collapse (0.340 vs V1's 0.660).

**V4 Hypothesis**: Baseline structure works (F1=0.615). Add ONLY minimal, essential elements from V1/V2/V3 learnings without verbosity. Test where the "goldilocks zone" exists between baseline simplicity and combined complexity.

## Baseline Performance to Beat

From `gptoss_ift_summary_README.md`:

**baseline_standard Configuration**:
```json
{
  "max_tokens": 512,
  "temperature": 0.1,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Production Performance (1,009 samples)**:
- **F1-Score**: 0.615 ← **TARGET TO BEAT**
- **Accuracy**: 65.0%
- **Precision**: 0.610
- **Recall**: 0.620
- **Confusion Matrix**: 282 TP, 373 TN, 180 FP, 173 FN

**Bias Metrics**:
- LGBTQ+ (494 samples): FPR=43.0%, FNR=39.4%
- Mexican/Latino (209 samples): FPR=8.1%, FNR=39.8%
- Middle Eastern (306 samples): FPR=23.6%, FNR=35.2%

**Key Strengths**:
- Simple, direct prompts (no verbosity)
- Optimal hyperparameters (temp=0.1, 512 tokens from systematic optimization)
- Balanced precision/recall (0.610/0.620)
- Excellent generalization (F1=0.626 on 100 samples → 0.615 on 1,009 samples, only 1.1% degradation)

**Key Weaknesses**:
- No community-specific guidance (generic "hateful language and social norms")
- No examples to demonstrate subtle hate patterns
- High FPR disparity across groups (LGBTQ+ 43% vs Mexican 8.1%)

## What Went Wrong in V1, V2, V3

### V1 Failure Pattern (F1=0.590 production, -2.5% vs baseline)

**What V1 Did**:
- Added 5 examples per group (15 total)
- Verbose system prompts with explicit policy
- Structured user templates with "EVALUATION FRAMEWORK"
- Total prompt length: LONG (extensive examples + policy + persona)

**Why It Failed**:
- Too many examples (15) may confuse rather than clarify
- Verbose structure made model over-cautious
- Production degradation: F1=0.614 (100 samples) → 0.590 (production) = 2.4% drop
- Still underperformed baseline_standard's F1=0.615

### V2 Failure Pattern (F1=0.565 best, -5% vs baseline)

**What V2 Did**:
- Tested 0-example strategies (cultural_context best at F1=0.565)
- Tested 2-example strategy (persona_balanced worst at F1=0.506)
- Removed examples to reduce verbosity
- Finding: 2 examples WORSE than 0 examples (underfitting)

**Why It Failed**:
- 0 examples: Model has no pattern reference (underperforms)
- 2 examples: Not enough to establish patterns, causes confusion
- Hypothesis: There's a "valley of confusion" between 0-4 examples

### V3 Catastrophic Failure (F1=0.438-0.559, up to -28.6% vs baseline)

**What V3 Did**:
- Restored V1's 5-example approach
- Added MORE structure and verbosity
- Emphasized "Examples as PRIMARY learning signal"
- Created "EVALUATION FRAMEWORK" sections with numbered steps

**Why It Catastrophically Failed**:
- **Over-engineering**: Made prompts TOO verbose and structured
- **Model conservatism**: High precision (0.615) but catastrophically low recall (0.340)
- **Missing hate speech**: FNR 66-75% (model missed 2/3 to 3/4 of hate posts)
- **Key insight**: V1 was already too verbose; V3 made it worse

**Critical Finding**: Simple beats complex. V3's attempt to "improve" V1 with more structure caused recall to collapse from V1's 0.660 to V3's 0.340 (-48.5%).

## V4 Strategy: Minimal Baseline Enhancements

### Core Principle

**"Add the minimum effective dose, not the maximum tolerable dose"**

- Keep baseline's simple, direct structure
- Use baseline's proven hyperparameters (temp=0.1, 512 tokens)
- Add ONLY essential elements that previous versions showed potential
- Test multiple minimal variations to find optimal balance

### V4 Strategies

#### 1. **combined_v4_minimal_examples** (Minimal Examples Approach)

**Rationale**: 
- V1's 15 examples (5 per group) were too many → verbosity overhead
- V2's 0-2 examples were too few → insufficient pattern learning
- Hypothesis: 6 examples (1 HATE + 1 NORMAL per group) = optimal balance

**What's Added to Baseline**:
- 6 concise examples (2 per group)
- Examples show HATE vs NORMAL pattern for each community
- No policy verbosity, no evaluation framework
- Total addition: ~150 words vs baseline's ~80 words

**Expected Performance**: F1=0.620-0.630
- Examples provide pattern reference without overwhelming model
- Demonstrates in-group vs out-group usage (LGBTQ+)
- Shows generalization vs policy critique distinction (Mexican/Middle Eastern)

**Risk**: Low - minimal addition, proven pattern (examples help)

---

#### 2. **combined_v4_subtle_emphasis** (Detection Emphasis Approach)

**Rationale**:
- Baseline lacks guidance on subtle/coded hate
- V2's recall_optimized showed emphasis can help
- Add single sentence without verbosity

**What's Added to Baseline**:
- One sentence in system prompt: "Pay special attention to coded and subtle hate speech (dog whistles, generalizations, dehumanization) which are hate even without explicit slurs."
- No examples, no policy, no persona
- Total addition: ~30 words vs baseline's ~80 words

**Expected Performance**: F1=0.618-0.625
- Improves recall by highlighting subtle patterns
- Minimal addition means low risk of over-engineering
- Addresses baseline's weakness (missing coded hate)

**Risk**: Very Low - single sentence addition

---

#### 3. **combined_v4_community_aware** (Cultural Context Approach)

**Rationale**:
- Baseline lacks community-specific understanding
- V2's cultural_context achieved F1=0.565 (best V2 but below baseline)
- Add BRIEF context (2-3 sentences per group) without V2's verbosity

**What's Added to Baseline**:
- 3 concise community context statements (one per group)
- Explains key patterns: in-group reclamation (LGBTQ+), generalization vs policy (Mexican), terrorism stereotyping (Middle Eastern)
- No examples, no policy documents, no evaluation framework
- Total addition: ~100 words vs baseline's ~80 words

**Expected Performance**: F1=0.615-0.625
- Cultural awareness without verbose persona instructions
- May reduce FPR disparity (LGBTQ+ 43% → closer to balanced)
- Addresses baseline's generic approach

**Risk**: Low-Medium - more addition than emphasis approach but still concise

---

#### 4. **combined_v4_balanced_lite** (Hybrid Minimal Approach)

**Rationale**:
- Combines two proven elements: minimal examples + brief context
- Tests if combination beats individual approaches
- Still maintains brevity (no V1/V3 verbosity)

**What's Added to Baseline**:
- 1 example per group (3 HATE + 3 NORMAL = 6 total)
- Brief community context (one sentence per group)
- Combined addition: ~150 words vs baseline's ~80 words

**Expected Performance**: F1=0.625-0.635 (HIGHEST EXPECTED)
- Examples provide pattern learning
- Context provides cultural understanding
- Still significantly shorter than V1's verbose prompts
- Hypothesis: Combination effect may beat individual strategies

**Risk**: Medium - largest addition but still minimal vs V1/V3

## Hyperparameter Configuration (All V4 Strategies)

All V4 strategies use **baseline_standard's proven configuration**:

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

**Rationale**: These hyperparameters were empirically optimized through systematic testing (see `gptoss_ift_summary_README.md`):
- **temp=0.1**: Optimal for classification (16.6% F1 range from temp effects)
- **512 tokens**: Goldilocks zone for bias-fairness trade-off
- **top_p=1.0**: Low impact at low temperature
- **Penalties=0.0**: Negligible effect on classification tasks

## Success Criteria

### Phase 1: Small-Sample Validation (100 samples)

**Primary Goal**: At least ONE V4 strategy beats baseline on 100-sample test
- **Target**: F1 ≥ 0.626 (baseline's small-sample performance)
- **Minimum**: F1 ≥ 0.620 (showing improvement potential)

**Secondary Goals**:
- Identify best-performing V4 strategy for production testing
- Verify recall improvement without precision collapse (avoid V3 failure pattern)
- Compare FPR/FNR patterns across protected groups

**Test Command**:
```bash
python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_v4_minimal_examples combined_v4_subtle_emphasis combined_v4_community_aware combined_v4_balanced_lite \
  --output-dir outputs/combined_v4/gptoss/validation_100 \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v4_baseline_enhanced.json
```

**Decision Criteria**:
- **If F1 ≥ 0.626**: Proceed to Phase 2 with best strategy
- **If 0.620 ≤ F1 < 0.626**: Promising but needs refinement
- **If F1 < 0.620**: Strategy failed, analyze why minimal additions didn't help

### Phase 2: Production Validation (1,009 samples)

**Only proceed if Phase 1 succeeds (F1 ≥ 0.626 on 100 samples)**

**Primary Goal**: Beat baseline_standard on full dataset
- **Target**: F1 > 0.615 (baseline's production performance)
- **Stretch**: F1 ≥ 0.630 (meaningful 1.5% improvement)

**Secondary Goals**:
- Verify generalization (small-sample to production degradation < 2%)
- Reduce FPR disparity across groups (LGBTQ+ 43% → closer to 25-30%)
- Maintain balanced precision/recall (no V3-style recall collapse)

**Test Command** (replace `<best_strategy>` with Phase 1 winner):
```bash
python prompt_runner.py \
  --data-source unified \
  --strategies <best_strategy> \
  --output-dir outputs/combined_v4/gptoss/production \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v4_baseline_enhanced.json
```

**Success Metrics**:
- **F1-score**: > 0.615 (baseline)
- **Generalization**: Degradation < 2% from Phase 1 to Phase 2
- **Bias fairness**: FPR disparity reduced (LGBTQ+ FPR from 43% toward 30%)
- **Balanced performance**: Precision ≥ 0.600, Recall ≥ 0.600

## Expected Outcomes & Ranking

Based on analysis of V1/V2/V3 failures and baseline strengths:

### Most Likely to Succeed (Rank 1)
**combined_v4_balanced_lite** (F1=0.625-0.635 expected)
- Combines proven elements: minimal examples + brief context
- Examples provide pattern reference, context provides cultural awareness
- Still maintains baseline's simplicity advantage
- May capture synergy between example-based learning and context-based understanding

### Strong Contender (Rank 2)
**combined_v4_minimal_examples** (F1=0.620-0.630 expected)
- Examples help pattern recognition (proven in literature)
- 6 examples = optimal between V2's 0-2 (too few) and V1's 15 (too many)
- Direct pattern demonstration without verbosity

### Promising (Rank 3)
**combined_v4_subtle_emphasis** (F1=0.618-0.625 expected)
- Single-sentence addition = lowest risk
- Addresses baseline's weakness (subtle hate detection)
- May improve recall without harming precision

### Baseline+ (Rank 4)
**combined_v4_community_aware** (F1=0.615-0.625 expected)
- Cultural context without verbosity
- V2's cultural_context underperformed (F1=0.565) but that version was more verbose
- May reduce FPR disparity but uncertain F1 improvement

## Analysis Framework

### If V4 Succeeds (F1 > 0.615)

**Key Questions**:
1. Which minimal addition worked? (examples, emphasis, context, or combination?)
2. Does success replicate V1's pattern learning or baseline's simplicity?
3. Is improvement due to better recall, precision, or both?
4. Does bias fairness improve (FPR disparity reduction)?

**Next Steps**:
1. Deploy winning strategy to production
2. Document what made the difference (examples vs context vs emphasis)
3. Consider V5 with targeted refinements to winning approach

### If V4 Fails (F1 ≤ 0.615)

**Key Questions**:
1. Why did minimal additions not improve baseline?
2. Is baseline_standard already optimal for this task/model?
3. Did we add wrong elements or still too much complexity?
4. Should we try even MORE minimal additions (single example, single context)?

**Possible Conclusions**:
1. **Baseline is optimal**: Deploy baseline_standard, accept F1=0.615 as ceiling
2. **Wrong additions**: Try different minimal elements (different example types, different emphasis)
3. **Model limitation**: gpt-oss-120b may need fine-tuning, not prompt engineering
4. **V5 ultra-minimal**: Test single-example or single-context approaches

## Risk Assessment

### Low-Risk Strategies
- **combined_v4_subtle_emphasis**: Single sentence addition, very low risk of over-engineering
- **combined_v4_minimal_examples**: Small example set, proven pattern

### Medium-Risk Strategies
- **combined_v4_balanced_lite**: Larger addition but still minimal vs V1/V3
- **combined_v4_community_aware**: Context may introduce unwanted verbosity

### What Could Go Wrong
1. **Minimal additions aren't enough**: Improvements too small to beat F1=0.615
2. **Still too much complexity**: Even minimal additions degrade performance (unlikely given V1/V2/V3 analysis)
3. **Wrong elements added**: Examples/context don't address baseline's actual weaknesses
4. **Model saturation**: gpt-oss-120b already at prompt engineering limit

## Theoretical Foundation

### Why Minimal Enhancements May Work

**1. Information Theory Perspective**:
- Baseline lacks information about community-specific patterns → low recall on subtle hate
- Adding minimal information (examples, context) provides pattern reference
- But excessive information (V1's 15 examples, V3's verbosity) introduces noise
- Optimal: Just enough information to establish patterns without confusion

**2. Cognitive Load Theory**:
- Simple prompts (baseline) enable clear classification decisions
- Moderate additions (V4) provide guidance without overwhelming model
- Excessive complexity (V1/V3) causes decision paralysis → conservatism (high FNR)

**3. Few-Shot Learning Literature**:
- 1-5 examples often optimal for LLM pattern recognition
- 0 examples = no pattern reference (V2's underperformance)
- 2-4 examples = insufficient pattern establishment ("valley of confusion")
- 5+ examples = diminishing returns, potential confusion
- V4's 6 examples (1 per group × 2 per category) may hit sweet spot

**4. Prompt Engineering Best Practices**:
- "Be clear and specific" (baseline succeeds)
- "Provide examples" (V4 adds minimal examples)
- "Give the model time to think" (512 tokens enables reasoning)
- "Avoid unnecessary complexity" (V4 stays concise vs V1/V3)

## Comparison to Previous Versions

| Aspect | Baseline | V1 | V2 | V3 | V4 |
|--------|----------|----|----|----|----|
| **Examples per group** | 0 | 5 (15 total) | 0-2 | 5 (15 total) | 1 (6 total) |
| **Prompt verbosity** | Minimal | High | Medium | Very High | Minimal+ |
| **Community context** | None | Policy+Persona | Persona | Policy+Persona+Cultural | Brief context |
| **Structure complexity** | Simple | Structured | Moderate | Very Structured | Simple+ |
| **Hyperparameters** | Optimized | Borrowed | Borrowed | Borrowed | Same as baseline |
| **Production F1** | 0.615 | 0.590 (-2.5%) | 0.565 (-5%) | Not tested | **TBD (Target: >0.615)** |
| **Key weakness** | No patterns | Over-engineering | Insufficient examples | Catastrophic verbosity | Unknown |

**V4's Unique Position**: 
- Only version that MINIMALLY enhances baseline
- All others made LARGE additions that degraded performance
- Tests if "goldilocks zone" exists between baseline simplicity and combined complexity

## References

**Baseline Performance**: `gptoss_ift_summary_README.md`
- baseline_standard: F1=0.615 (production), optimal hyperparameters

**V1 Analysis**: `gpt_oss_combined_ift_summary_README.md`
- combined_optimized: F1=0.614 (100 samples), F1=0.590 (production)
- Lesson: Verbose prompts with 15 examples underperformed baseline

**V2 Analysis**: `combined_v2_bias_optimized_README.md`
- cultural_context: F1=0.565 (best V2, but below baseline)
- Lesson: 0-2 examples insufficient; all strategies underperformed

**V3 Analysis**: `combined_v3_bias_optimized_README.md`
- recall_focused: F1=0.559 (best V3, below baseline)
- optimized: F1=0.438 (catastrophic -28.6% vs V1)
- Lesson: Over-engineering with verbose structure collapsed recall

## Validation Plan

### Step 1: Small-Sample Test (100 samples)
```bash
cd prompt_engineering

python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_v4_minimal_examples combined_v4_subtle_emphasis combined_v4_community_aware combined_v4_balanced_lite \
  --output-dir outputs/combined_v4/gptoss/validation_100 \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v4_baseline_enhanced.json
```

**Expected Duration**: ~5-7 minutes for 100 samples × 4 strategies = 400 classifications

### Step 2: Analyze Results
```bash
# Review evaluation report
cat outputs/combined_v4/gptoss/validation_100/run_*/evaluation_report_*.txt

# Check metrics
cat outputs/combined_v4/gptoss/validation_100/run_*/performance_metrics_*.csv
cat outputs/combined_v4/gptoss/validation_100/run_*/bias_metrics_*.csv
```

**Look for**:
- F1 ≥ 0.626 (baseline's small-sample performance)
- Balanced precision/recall (both ≥ 0.600)
- Reduced FPR disparity across groups
- No V3-style recall collapse

### Step 3: Production Test (if Phase 1 succeeds)
```bash
# Test winning strategy on full dataset
python prompt_runner.py \
  --data-source unified \
  --strategies <best_v4_strategy> \
  --output-dir outputs/combined_v4/gptoss/production \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v4_baseline_enhanced.json
```

**Expected Duration**: ~20 minutes for 1,009 samples

### Step 4: Production Decision

**If F1 > 0.615 on production**:
- ✅ SUCCESS - V4 beats baseline
- Deploy to production
- Document winning approach

**If F1 ≤ 0.615 on production**:
- ❌ BASELINE REMAINS OPTIMAL
- Use baseline_standard for production
- Consider alternative approaches (model fine-tuning, different base model)

## Key Insights & Predictions

### Critical Hypothesis
**"The optimal prompt exists between baseline's simplicity and V1's complexity"**

- Baseline (F1=0.615): Too simple, lacks pattern guidance
- V1 (F1=0.590): Too complex, verbosity degrades performance
- V4: Targets the "goldilocks zone" with minimal additions

### Expected Best Performer
**combined_v4_balanced_lite** because:
1. Combines two proven elements (examples + context)
2. Examples provide concrete pattern learning
3. Context provides cultural understanding
4. Still maintains baseline's simplicity advantage
5. Synergy effect: combination > sum of parts

### Fallback Position
If V4 fails, **baseline_standard remains production choice**:
- Proven F1=0.615 (production)
- Excellent generalization (1.1% degradation from 100 to 1,009 samples)
- Simple, maintainable, reproducible

### Long-Term Strategy
If prompt engineering hits ceiling (V4 fails):
1. **Fine-tuning**: Train gpt-oss-120b on hate speech dataset
2. **Model upgrade**: Test larger/newer models (GPT-4, Claude 3.5)
3. **Ensemble**: Combine baseline with specialized detectors
4. **Human-in-loop**: Deploy baseline with human review for borderline cases

## Conclusion

V4 represents a **fundamental strategy shift**: instead of adding more complexity (V1/V2/V3), we're testing if minimal, targeted additions can beat baseline's F1=0.615.

**Key Innovation**: Recognizing that all previous "improvements" degraded performance, V4 stays close to what works (baseline) while testing minimal enhancements.

**Expected Outcome**: At least one V4 strategy beats baseline (F1 > 0.615), with **combined_v4_balanced_lite** most likely to succeed.

**If V4 succeeds**: We've found the optimal balance between simplicity and guidance.

**If V4 fails**: Baseline is optimal; further improvement requires model fine-tuning or architecture changes, not prompt engineering.
