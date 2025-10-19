# Option B vs Option C Comparison

**Date**: 2025-01-18  
**Goal**: Fix Mexican detection regression (V3: 83% FNR) while maintaining Middle Eastern improvements and restoring LGBTQ+ context

---

## Approaches

### Option B: Few-Shot Examples
**Strategy**: Provide explicit examples showing hate vs. normal for Mexican/Latino content

**Key Changes**:
1. **Mexican Section - Added Examples Block**:
   ```
   EXAMPLES - Mexican/Latino Hate vs. Policy Discussion:
   HATE: "These Mexicans are all illegals, send them back" (generalizes ALL Mexicans as illegal)
   HATE: "Mexico isn't sending their best, they're rapists and criminals" (attacks ALL Mexicans)
   HATE: "Mexicans are like rats crossing the border" (dehumanization)
   NORMAL: "We need stronger border enforcement policies" (policy discussion, not attacking people)
   NORMAL: "Immigration system should prioritize legal pathways" (policy critique)
   ```

2. **LGBTQ+ Context Restoration**:
   - **Optimized**: Maintained full context (already adequate)
   - **Focused**: Restored full in-group reclamation explanation with "CRITICAL:" emphasis
   - **Focused tokens**: Increased from 200 → 256 to accommodate context

3. **Conservative**: Added compact examples

**Rationale**: 
- Direct examples show model exactly what patterns to detect
- Contrast between HATE and NORMAL clarifies the PEOPLE vs. policy distinction
- Follows classic few-shot learning approach
- May improve precision by reducing ambiguity

---

### Option C: Hybrid (Patterns + Critical Examples)
**Strategy**: Combine pattern-based guidance with selective critical examples embedded in flow

**Key Changes**:
1. **Mexican Section - Hybrid Format**:
   ```
   Mexican/Latino Community:
   - Explicit slurs: beaners, wetbacks, anchor babies, illegals (when dehumanizing)
   - Immigration-based hate PATTERN: Generalizing ALL Mexicans/Latinos as 'illegal', 
     'criminals', 'invaders', 'rapists', 'job stealers', 'disease carriers'
   - Coded hate examples: "They're not sending their best" (implies ALL Mexicans are 
     inferior), "go back" (targeting ethnicity), "taking our jobs" (scapegoating 
     entire group)
   - Dehumanization: Comparing to animals (rats, vermin), diseases (infestations), objects
   - KEY DISTINCTION: Discussing immigration policy = NORMAL. Attacking ALL Mexican/Latino 
     people = HATE.
     - HATE: "These Mexicans are all illegals" / "Mexico sends criminals and rapists"
     - NORMAL: "We need stronger border policies" / "Immigration reform is needed"
   ```

2. **LGBTQ+ Context Restoration**: Same as Option B

3. **Focused tokens**: Increased from 200 → 256

**Rationale**:
- Maintains pattern-based teaching from V3 (which worked for Middle Eastern)
- Adds explicit slur list to catch obvious cases
- Embeds critical examples inline with explanations
- More compact than Option B while providing guidance
- Emphasizes the "generalizing ALL" detection principle

---

## Comparison Matrix

| Aspect | Option B (Few-Shot) | Option C (Hybrid) |
|--------|---------------------|-------------------|
| **Approach** | Separate examples block | Inline examples with patterns |
| **Mexican Section Length** | Medium (examples block) | Longer (comprehensive) |
| **Pattern Emphasis** | Moderate | Strong |
| **Example Count** | 5 examples (3 hate, 2 normal) | 4 examples (2 hate, 2 normal) inline |
| **Slur List** | Implicit in patterns | Explicit list provided |
| **Teaching Method** | Show examples, learn by analogy | Teach pattern + examples |
| **Token Efficiency** | More efficient | Less efficient |
| **LGBTQ+ Restoration** | Full (256 tokens focused) | Full (256 tokens focused) |
| **Complexity** | Lower (clearer structure) | Higher (more comprehensive) |

---

## Expected Outcomes

### Option B Predictions:
✅ **Mexican FNR**: Should improve significantly (83% → 40-50%) - examples directly show pattern  
✅ **LGBTQ+ FPR/FNR**: Should return to v2 levels (FPR ~19%, FNR ~33%) due to restored context  
✅ **Middle Eastern**: Should maintain v3 improvements (FPR 17%, FNR 38%)  
✅ **Overall clarity**: Higher - examples are unambiguous  
⚠️ **Precision risk**: May reduce precision if model over-relies on exact example matching  

### Option C Predictions:
✅ **Mexican FNR**: Should improve moderately (83% → 50-60%) - comprehensive guidance  
✅ **LGBTQ+ FPR/FNR**: Should return to v2 levels due to restored context  
✅ **Middle Eastern**: Should maintain v3 improvements  
✅ **Robustness**: Higher - teaches pattern generalization not just examples  
⚠️ **Complexity risk**: More information may cause overload for conservative variant  

---

## Key Differences

### Mexican Detection Approach:

**Option B** (Learning by Example):
```
EXAMPLES - Mexican/Latino Hate vs. Policy Discussion:
HATE: "These Mexicans are all illegals, send them back"
NORMAL: "We need stronger border enforcement policies"
```
→ Clear contrast, easy to understand, may not generalize well to variations

**Option C** (Learning by Pattern + Example):
```
Immigration-based hate PATTERN: Generalizing ALL Mexicans/Latinos as 'illegal'...
Coded hate examples: "They're not sending their best" (implies ALL Mexicans are inferior)
KEY DISTINCTION: Policy = NORMAL. Attacking ALL people = HATE.
  HATE: "These Mexicans are all illegals"
  NORMAL: "We need stronger border policies"
```
→ Teaches underlying principle, examples reinforce pattern, better generalization

### LGBTQ+ Context (Both Options):

**Focused Variant Improvement**:
- **V3 (failed)**: "LGBTQ+: Slurs/denying identity/dangerous stereotypes. Context: In-group reclamation vs. out-group attacks."
- **Options B & C**: "LGBTQ+: Slurs targeting orientation/identity, denying identities, dangerous stereotypes. CRITICAL: LGBTQ+ individuals reclaiming terms (e.g., 'we're queer') is NOT hate; outsiders using same terms to attack IS hate."

Added:
- "CRITICAL:" emphasis
- Concrete example ('we're queer')
- Explicit NOT hate vs. IS hate distinction
- Increased tokens (200 → 256)

---

## Validation Plan

### Commands:

**Option B:**
```bash
cd Q:\workspace\HateSpeechDetection_ver2\prompt_engineering
python prompt_runner.py --data-source canned_50_quick --strategies all --output-dir outputs/optionB_fewshot/gptoss/ --max-workers 15 --batch-size 8 --prompt-template-file combined/combined_gptoss_v1_optionB_fewshot.json
```

**Option C:**
```bash
cd Q:\workspace\HateSpeechDetection_ver2\prompt_engineering
python prompt_runner.py --data-source canned_50_quick --strategies all --output-dir outputs/optionC_hybrid/gptoss/ --max-workers 15 --batch-size 8 --prompt-template-file combined/combined_gptoss_v1_optionC_hybrid.json
```

### Success Criteria:

| Metric | V3 Baseline | Target |
|--------|-------------|--------|
| **Mexican FNR (optimized)** | 83% ❌ | <50% ✅ |
| **LGBTQ+ FPR (focused)** | 25% ⚠️ | <20% ✅ |
| **LGBTQ+ FNR (focused)** | 44% ⚠️ | <35% ✅ |
| **Middle Eastern FPR (optimized)** | 17% ✅ | Maintain <25% ✅ |
| **Overall F1 (focused)** | 0.619 ✅ | Maintain ≥0.60 ✅ |
| **Overall Accuracy** | 66-68% | Maintain ≥65% ✅ |

---

## Decision Criteria

**Choose Option B if**:
- Mexican FNR improves to <40% (strong example-based learning)
- LGBTQ+ metrics return to v2 levels
- Overall F1 maintains ≥0.60
- Precision doesn't degrade significantly

**Choose Option C if**:
- Mexican FNR improves to <50% (good generalization)
- Better robustness across all communities
- Higher recall (catches more variations)
- Teaches pattern better than pure examples

**Choose Hybrid (mix both) if**:
- Option B better for Mexican detection
- Option C better for overall balance
- Can combine best elements from each

---

## Implementation Notes

### Both Options Include:

1. **Restored LGBTQ+ Context**:
   - All variants have full in-group reclamation explanation
   - Focused variant increased from 200 → 256 tokens
   - Added "CRITICAL:" emphasis and concrete example

2. **Maintained Middle Eastern Improvements**:
   - Keep V3's simplified terrorism distinction
   - No changes to Middle Eastern section (it worked well)

3. **Same Hyperparameters**:
   - optimized: temp=0.1, tokens=512
   - focused: temp=0.05, tokens=256 (increased from 200)
   - conservative: temp=0.0, tokens=256

### File Locations:

- Option B: `prompt_templates/combined/combined_gptoss_v1_optionB_fewshot.json`
- Option C: `prompt_templates/combined/combined_gptoss_v1_optionC_hybrid.json`
- This comparison: `prompt_templates/combined/OPTION_B_VS_C_COMPARISON.md`
