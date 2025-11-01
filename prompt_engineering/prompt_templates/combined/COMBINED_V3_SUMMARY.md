# Combined V3 Creation Summary

## What Was Created

Created two new files in `prompt_templates/combined/`:

1. **combined_v3_bias_optimized_README.md** - Strategy documentation and validation plan
2. **combined_v3_bias_optimized.json** - 3 prompt strategies with 5-example few-shot structure

---

## V3 Core Hypothesis

**Problem Identified**: V2 testing revealed:

- Best V2: F1=0.565 with **0 examples** (cultural_context)
- Worst V2: F1=0.506 with **2 examples** (persona_balanced - RECOMMENDED but FAILED)
- ALL V2 strategies underperformed V1 (best: F1=0.667 with **5 examples**)

**Critical Gap**: V2 never tested V1's proven 5-example approach with V2's improved hyperparameters (temp=0.1, 512 tokens).

**V3 Solution**: Restore V1's 5-example structure with V2's optimized hyperparameters.

---

## V3 Strategy Comparison

| Strategy | Structure | Target | Use Case |
|----------|-----------|--------|----------|
| **combined_v3_optimized** | 50% policy / 50% persona + 5 examples | F1: 0.600-0.650 | **RECOMMENDED** - Balanced production candidate |
| **combined_v3_recall_focused** | 60% policy / 40% persona + 5 examples | F1: 0.590-0.640 | Safety-first, moderate recall emphasis |
| **combined_v3_cultural_aware** | 40% policy / 60% persona + 5 examples | F1: 0.600-0.650 | Deep cultural awareness + patterns |

**Key Features**:

- All use **5 examples per group** (15 total) in dedicated sections
- All use **V2's hyperparameters**: temp=0.1, 512 tokens
- All have **high-visibility example blocks** with clear HATE/NORMAL labels
- All emphasize **pattern learning** over imperative instructions

---

## What V3 Tests

### V1 Evidence (5 Examples Worked)

- combined_conservative: F1=0.667, Mexican FNR 83%→33% (-50%)
- combined_optimised: F1=0.590 production
- BUT: Used temp=0.0 + 256 tokens (caused production failure F1=0.473)

### V2 Evidence (Wrong Number of Examples)

- 0 examples: F1=0.565 (moderate)
- 2 examples: F1=0.506 (WORST)
- 5 examples: NEVER TESTED with temp=0.1 + 512 tokens

### V3 Hypothesis

**V1 was RIGHT about examples (5 = optimal), WRONG about hyperparameters**
**V2 was RIGHT about hyperparameters, WRONG to abandon examples**

V3 tests: 5 examples + temp=0.1 + 512 tokens = production success?

---

## Validation Plan

### Phase 1: 100-Sample Validation (CRITICAL)

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

**Success Criteria**:

- Minimum: F1 ≥ 0.565 (beat V2 best)
- Target: F1 ≥ 0.600 (approach V1's 0.667)
- Stretch: F1 ≥ 0.650 (match V1)
- Mexican FNR: ≤40% (improve from V2's 50-75%)
- LGBTQ+ FPR: ≤40%

**Duration**: ~10-15 minutes

---

### Phase 2: Production Validation (If Phase 1 Succeeds)

**Dataset**: unified (1,009 samples)

**Strategies**: Top 1-2 from Phase 1

**Success Criteria**:

- Target: F1 ≥ 0.615 (match/beat baseline_standard)
- Stretch: F1 ≥ 0.620
- Mexican FPR: ≤10%, FNR: ≤40%
- LGBTQ+ FPR: ≤40%
- Middle Eastern FNR: ≤36%

---

## Key Differences from V2

### V2 Approach (FAILED)

- Examples: 0-2 (reduced from V1's 5)
- Rationale: "Avoid overfitting"
- Result: persona_balanced (2 examples) = F1=0.506 (WORST)
- Best: cultural_context (0 examples) = F1=0.565
- Problem: Never tested V1's winning formula with better hyperparameters

### V3 Approach (FIX)

- Examples: 5 per group (restore V1's proven approach)
- Rationale: "V1 proved 5 works, issue was hyperparameters"
- Hyperparameters: temp=0.1 + 512 tokens (NOT V1's 0.0 + 256)
- Expected: F1=0.600-0.650 (beat V2, approach V1)
- Innovation: Tests the critical untested combination

---

## Example Structure (All V3 Strategies)

Each strategy includes 5 examples per group in dedicated sections:

**Mexican/Latino Examples**:

1. HATE: "These Mexicans are all illegals, send them back"
2. HATE: "Mexico isn't sending their best, they're rapists"
3. HATE: "Mexicans are like rats crossing the border"
4. NORMAL: "We need stronger border enforcement policies"
5. NORMAL: "Immigration system should prioritize legal pathways"

**LGBTQ+ Examples**:

1. HATE: "Those faggots are disgusting and sick"
2. HATE: "Transgender people are mentally ill and dangerous"
3. HATE: "Gays shouldn't be allowed to adopt"
4. NORMAL: "I'm a proud queer person"
5. NORMAL: "Today is for the homosexuals, we're celebrating"

**Middle Eastern Examples**:

1. HATE: "All Muslims are terrorists waiting to attack"
2. HATE: "Arabs are barbaric and violent people"
3. HATE: "Send them back before they bomb us"
4. NORMAL: "ISIS attacks in Syria are devastating"
5. NORMAL: "Middle Eastern culture has rich history"

---

## Expected Outcomes

### Success Scenario (F1 ≥ 0.600)

- **Validation**: V1's 5-example approach scales with proper hyperparameters
- **Action**: Deploy top strategy on production (1,009 samples)
- **Target**: F1 ≥ 0.615 (match/beat baseline)

### Moderate Scenario (F1 0.565-0.600)

- **Interpretation**: Better than V2, but scale is still an issue
- **Action**: Test on production to see if stabilizes or degrades
- **Decision**: If stable 0.580-0.600, acceptable for production

### Failure Scenario (F1 < 0.565)

- **Interpretation**: 5 examples truly overfit at 100+ sample scale
- **Action**: Test V3.5 with 3-4 examples (middle ground)
- **Fallback**: Deploy baseline_standard (F1=0.615)

---

## Why V3 Should Succeed

**Evidence-Based Reasoning**:

1. **V1 proved examples work** (F1=0.667, Mexican FNR 83%→33%)
2. **V2 proved hyperparameters** (temp=0.1 + 512 tokens optimal)
3. **V2 proved 2 examples fail** (F1=0.506, worse than 0 examples)
4. **V1's production failure** was temp=0.0 + 256 tokens (F1=0.473)
5. **V3 combines proven elements**: 5 examples + temp=0.1 + 512 tokens

**Pattern Analysis**:

- 0 examples = instruction-following only (F1=0.565)
- 2 examples = insufficient + confusion (F1=0.506)
- 5 examples = pattern learning (F1=0.667 BUT wrong hyperparameters)
- 5 examples + right hyperparameters = **V3 TEST**

---

## Next Steps

1. **Review** the README and JSON files to understand V3 strategies
2. **Run Phase 1** validation (100 samples, ~10-15 min)
3. **Analyze results**:
   - If F1 ≥ 0.600: Proceed to Phase 2 (production)
   - If F1 0.565-0.600: Investigate and decide
   - If F1 < 0.565: Consider V3.5 with fewer examples
4. **Compare to baseline** (F1=0.615) for production decision

---

## File Locations

- **README**: `prompt_templates/combined/combined_v3_bias_optimized_README.md`
- **JSON**: `prompt_templates/combined/combined_v3_bias_optimized.json`
- **Reference V2 README**: `prompt_templates/combined/combined_v2_bias_optimized_README.md`
- **Reference V2 JSON**: `prompt_templates/combined/combined_v2_bias_optimized.json`

---

## Key Takeaway

**V2 answered**: "Is 2 examples better than 0?" → NO (F1 0.506 vs 0.565)

**V3 answers**: "Is V1's 5 examples better with improved hyperparameters?" → TEST NEEDED

**Expected**: YES - V3 should achieve F1=0.600+ by combining V1's proven pattern learning with V2's optimal hyperparameters.
