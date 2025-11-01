# V4 and Beyond: Additional Approaches to Consider

## Current Status Summary

**Baseline to Beat**: F1=0.615 (baseline_standard, production)

**V4 Strategies Created** (4 minimal baseline enhancements):
1. `combined_v4_minimal_examples` - 6 examples total
2. `combined_v4_subtle_emphasis` - Single sentence emphasis  
3. `combined_v4_community_aware` - Brief community context
4. `combined_v4_balanced_lite` - Hybrid approach (RECOMMENDED)

## If V4 Succeeds → Further Refinements (V5 Ideas)

### Approach 1: Optimize Example Count
**Based on V4's best performer, fine-tune example quantity**

If `combined_v4_minimal_examples` (6 examples) works best:
- Test 3, 4, 5, 6, 7, 8 examples systematically
- Find optimal count in the "goldilocks zone"
- Expected sweet spot: 4-6 examples

**Test**:
```json
"combined_v5_examples_3": {}, // 3 total (1 per group)
"combined_v5_examples_4": {}, // 4 total (alternating groups)
"combined_v5_examples_8": {}  // 8 total (2-3 per group)
```

---

### Approach 2: Example Quality over Quantity
**If examples help, optimize WHICH examples to show**

Test different example types:
- **Most subtle**: Show hardest-to-detect hate patterns
- **Most common**: Show most frequent hate patterns in data
- **Most confusing**: Show borderline cases (policy vs attack)
- **Balanced**: Mix of obvious + subtle

**Example Selection Strategy**:
```
Subtle Focus: Dog whistles, coded language, generalizations
Common Focus: Most frequent slurs, stereotypes in dataset
Confusing Focus: "Build the wall" vs "Border policy reform"
```

---

### Approach 3: Dynamic Context Length
**Test if different community groups need different guidance amounts**

Hypothesis: Some groups need more context than others

**Observation from baseline bias**:
- LGBTQ+ FPR=43.0% (highest) → May need more context on in-group reclamation
- Mexican FPR=8.1% (lowest) → Current understanding sufficient
- Middle Eastern FPR=23.6% → Moderate guidance needed

**V5 Strategy**:
```
LGBTQ+: 2 examples + detailed context (high confusion)
Mexican: 1 example + brief context (low confusion)  
Middle Eastern: 1 example + moderate context
```

---

### Approach 4: Temperature Micro-Optimization
**If V4 beats baseline, test temp variations around 0.1**

Baseline uses temp=0.1 (optimal from hyperparameter optimization)

Test narrow range:
- temp=0.08
- temp=0.09
- temp=0.10 (current)
- temp=0.11
- temp=0.12

Expected: Minimal impact (<1% F1 change) but worth testing

---

### Approach 5: Precision vs Recall Tuning
**Add bias toward precision or recall based on production needs**

**For higher recall** (catch more hate, accept more false positives):
```
"When borderline, classify as hate to prevent harm"
```

**For higher precision** (fewer false positives, accept more misses):
```
"Only classify as hate when clearly violating policy"
```

**Balanced** (current approach):
```
"Apply consistent standards across all cases"
```

---

## If V4 Fails → Alternative Directions

### Approach 6: Ultra-Minimal (Less Than V4)
**Test if even V4's minimal additions were too much**

**combined_v5_single_example**:
- Just 1 example total (most representative)
- No context, no emphasis
- Absolute minimum addition

**combined_v5_two_sentence_context**:
- 2 sentences only: "LGBTQ+ may reclaim slurs. Generalizing groups is hate."
- No examples
- Test if examples unnecessary

---

### Approach 7: Baseline Variations (No Additions)
**Test if baseline's PROMPT can be tweaked (not just additions)**

Modify baseline's existing wording:
```
Current: "Base your decision on general understanding of hateful language"

Alternative 1: "Base your decision on whether content harms protected groups"
Alternative 2: "Base your decision on whether content attacks people vs policies"  
Alternative 3: "Base your decision on platform hate speech policies"
```

Keep everything else identical, just test wording variations.

---

### Approach 8: Format Experiments
**Test if JSON structure or instruction format affects performance**

**Current format**:
```json
{"classification": "hate", "rationale": "brief explanation"}
```

**Alternative formats**:
```json
// Option 1: Confidence scoring
{"classification": "hate", "confidence": 0.9, "rationale": "..."}

// Option 2: Multi-label
{"classification": "hate", "type": "generalization", "rationale": "..."}

// Option 3: Detailed reasoning
{"classification": "hate", "violates": "targets protected group", "evidence": "..."}
```

May help model reason better, but risks over-complication.

---

### Approach 9: Chain-of-Thought Reasoning
**Add explicit reasoning steps before classification**

**System prompt addition**:
```
"Before classifying, consider:
1. Does it attack people or critique policies?
2. Does it generalize about protected groups?  
3. Would the targeted community view this as harmful?

Then provide your classification."
```

Literature shows CoT improves reasoning, but adds tokens/latency.

---

### Approach 10: Ensemble Approach
**Combine baseline with specialized detectors**

**Strategy**:
1. Run baseline_standard (F1=0.615)
2. Run specialized detectors for each group:
   - LGBTQ+ specialist (optimized for FPR=43% issue)
   - Mexican specialist (optimized for FNR=39.8% issue)
   - Middle Eastern specialist
3. Combine predictions (voting or weighted)

**Expected**: Higher F1 but more complex pipeline

---

### Approach 11: Few-Shot Learning with Retrieved Examples
**Dynamically select examples similar to input text**

Instead of fixed examples, use embedding similarity:
1. Embed input text
2. Retrieve 2-3 most similar examples from labeled dataset
3. Include in prompt
4. Classify

**Pros**: Examples always relevant to input
**Cons**: Requires embedding model, adds latency

---

### Approach 12: Model Fine-Tuning
**If prompt engineering hits ceiling, fine-tune the model**

Create fine-tuning dataset:
- Use unified dataset (1,009 samples) as training data
- Format: {prompt, completion} pairs
- Fine-tune gpt-oss-120b on hate speech classification

**Expected**: F1=0.65-0.70+ (significant improvement)
**Cost**: Requires fine-tuning infrastructure and time

---

### Approach 13: Different Base Model
**Test if gpt-oss-120b is the limiting factor**

Compare baseline_standard on different models:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Llama 3.1 70B/405B
- Mistral Large

**Hypothesis**: Larger/newer models may beat F1=0.615 with same prompt

---

### Approach 14: Adversarial Testing & Refinement
**Identify specific failure cases and target them**

**Process**:
1. Run baseline on full dataset
2. Analyze 173 FN + 180 FP errors
3. Find common patterns in failures
4. Create targeted guidance for those patterns

**Example failure patterns**:
- Missing dog whistles → Add dog whistle detection emphasis
- Misclassifying policy debate → Add policy vs attack distinction
- LGBTQ+ in-group confusion → Add reclamation context

---

### Approach 15: Human-in-the-Loop Hybrid
**Combine model with human review for borderline cases**

**Strategy**:
1. Model provides classification + confidence score
2. If confidence < threshold (e.g., 0.7), flag for human review
3. Human provides final decision
4. Use human decisions to improve model over time

**Expected**: Higher accuracy but requires human resources

---

## Recommended Testing Sequence

### Phase 1: V4 Test (CURRENT)
Test 4 minimal baseline enhancements

**If V4 SUCCEEDS** (F1 > 0.615):
1. Deploy winning V4 strategy
2. Consider Approach 1 (optimize example count) for V5
3. Consider Approach 3 (dynamic context length) for fairness

**If V4 MODERATELY SUCCEEDS** (F1 = 0.610-0.615):
1. Try Approach 2 (example quality)
2. Try Approach 7 (baseline variations)
3. Try Approach 4 (temperature micro-optimization)

**If V4 FAILS** (F1 < 0.610):
1. Deploy baseline_standard (F1=0.615)
2. Try Approach 6 (ultra-minimal)
3. Consider Approach 12 (fine-tuning) or Approach 13 (different model)

---

## Quick Reference: Approach Selection Matrix

```
┌─────────────────────────┬──────────────────────┬──────────────────┐
│ If V4 Result...         │ Try These Next       │ Long-term Option │
├─────────────────────────┼──────────────────────┼──────────────────┤
│ V4 BEATS baseline       │ • Optimize examples  │ • Fine-tuning    │
│ (F1 > 0.615)            │ • Dynamic context    │ • Ensemble       │
│                         │ • Refine winner      │                  │
├─────────────────────────┼──────────────────────┼──────────────────┤
│ V4 MATCHES baseline     │ • Example quality    │ • Different model│
│ (F1 = 0.610-0.615)      │ • Baseline variants  │ • Fine-tuning    │
│                         │ • Temp optimization  │                  │
├─────────────────────────┼──────────────────────┼──────────────────┤
│ V4 FAILS                │ • Ultra-minimal      │ • Fine-tuning    │
│ (F1 < 0.610)            │ • Format experiments │ • New base model │
│                         │ • Deploy baseline    │ • Human-in-loop  │
└─────────────────────────┴──────────────────────┴──────────────────┘
```

---

## Complexity vs Effort vs Expected Gain

```
LOW EFFORT, HIGH EXPECTED GAIN:
✓ Approach 1: Optimize example count (if V4 examples work)
✓ Approach 2: Example quality selection
✓ Approach 6: Ultra-minimal testing
✓ Approach 7: Baseline wording variations

MEDIUM EFFORT, MEDIUM EXPECTED GAIN:
○ Approach 3: Dynamic context length
○ Approach 4: Temperature micro-optimization  
○ Approach 5: Precision/recall tuning
○ Approach 8: Format experiments
○ Approach 9: Chain-of-thought

HIGH EFFORT, HIGH EXPECTED GAIN:
△ Approach 10: Ensemble approach
△ Approach 11: Retrieved examples
△ Approach 12: Model fine-tuning (BEST long-term)
△ Approach 13: Different base model

HIGH EFFORT, UNCERTAIN GAIN:
✗ Approach 14: Adversarial refinement
✗ Approach 15: Human-in-the-loop
```

---

## Key Decision Points

### Decision 1: After V4 Small-Sample Test
```
IF F1 ≥ 0.626:
  → Run V4 production test (best strategy)

IF 0.620 ≤ F1 < 0.626:
  → Try Approach 2 or 7 (quick tweaks)
  → Then re-test

IF F1 < 0.620:
  → Try Approach 6 (ultra-minimal)
  → If still fails, consider Approach 12/13
```

### Decision 2: After V4 Production Test
```
IF F1 > 0.615:
  ✓ DEPLOY V4 WINNER
  → Consider Approach 1 or 3 for V5 refinement

IF F1 ≤ 0.615:
  ✓ DEPLOY baseline_standard (F1=0.615)
  → Consider Approach 12 (fine-tuning) for major upgrade
```

### Decision 3: Long-Term Investment
```
IF prompt engineering plateaus at F1=0.615-0.625:
  → Invest in Approach 12 (fine-tuning)
  → Expected: F1=0.65-0.70+
  → Or accept current performance for production
```

---

## Final Recommendations

### Immediate (This Week):
1. ✅ **Test V4** (4 minimal baseline enhancements) - PRIORITY 1
2. Analyze V4 results and select best performer
3. If V4 succeeds → production test
4. If V4 fails → try Approach 6 (ultra-minimal)

### Short-Term (Next 2 Weeks):
1. If V4 wins → refine with Approach 1 or 2
2. If V4 fails → test Approach 7 (baseline variations)
3. Consider Approach 4 (temperature optimization)

### Long-Term (1-2 Months):
1. If stuck at F1=0.615-0.625 → **Approach 12** (fine-tuning)
2. Or **Approach 13** (test GPT-4o/Claude 3.5)
3. Or accept baseline_standard for production

### Moonshot:
- **Approach 10** (ensemble) for maximum F1
- Expected: F1=0.65-0.68
- Cost: Complex pipeline, higher latency/cost

---

## Summary: The Path Forward

**V4 is the right next step** because:
1. Tests minimal baseline enhancements (learned from V1/V2/V3 failures)
2. Low risk (4 strategies, small additions)
3. Fast to test (5-7 minutes for 100 samples)
4. Clear success/failure criteria (F1 vs 0.615)

**If V4 succeeds**: You've beaten baseline, refine the winner

**If V4 fails**: Baseline remains optimal; consider fine-tuning or new model

**Bottom line**: V4 determines if prompt engineering can beat F1=0.615, or if we've hit the ceiling for this model/approach.
