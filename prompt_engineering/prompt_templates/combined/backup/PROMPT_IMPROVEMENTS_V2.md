# Combined Policy-Persona Prompts v2 - Improvement Summary

## Performance Analysis from v1 (Run 20251018_185205)

### Critical Issues Identified:

| Strategy | F1 Score | Recall | Issue |
|----------|----------|--------|-------|
| combined_optimized | 0.432 | 0.348 | **Missing 65% of hate speech** - severe undercriminalization |
| combined_focused | **0.571** | **0.609** | **Best performer** - but 66.7% FPR for Middle East |
| combined_conservative | 0.316 | 0.261 | **Missing 74% of hate speech** - worst recall |

### Key Findings:

1. **Undercriminalization Crisis**: All strategies had high false negative rates (FNR 33-83%), meaning they missed most hate speech
2. **Token Length Paradox**: combined_focused (200 tokens) outperformed combined_optimized (512 tokens) by 32% - longer prompts caused overthinking
3. **Group-Specific Bias**:
   - Mexican/Latino: 0% FPR (good) but 33-83% FNR (missing hate)
   - Middle East: 67% FPR in focused variant (overcriminalization)
   - LGBTQ+: Moderate issues across all strategies

---

## Improvements Implemented in v2

### 1. **combined_optimized** (Renamed from "optimized" to "improved sensitivity")

**Changes:**
- ✅ **Reduced tokens**: 512 → 256 (eliminate overthinking)
- ✅ **Optimized temperature**: 0.1 → 0.05 (match best performer)
- ✅ **Direct language**: Removed hedging phrases like "reasonably find harmful"
- ✅ **Explicit examples**: Added clear hate speech indicators upfront
- ✅ **Assertive framing**: "If ANY hate speech indicator appears... classify as 'hate'"
- ✅ **Sensitivity instruction**: "Err on the side of detecting hate speech"

**Before (v1):**
```
Analyze this text systematically:
1. Does it contain slurs or direct attacks on protected characteristics?
2. Does it dehumanize or stereotype any protected community?
3. Would members of affected communities reasonably find this harmful?
```

**After (v2):**
```
If ANY hate speech indicator appears OR if text attacks/demeans protected groups, 
classify as 'hate'. Err on the side of detecting hate speech to protect vulnerable communities.
```

**Expected Impact:** Increase recall from 34.8% to 50%+ by reducing false negatives

---

### 2. **combined_focused** (Calibrated for Middle East FPR)

**Changes:**
- ✅ **Kept winning config**: Maintained temp=0.05, 200 tokens (best F1 score)
- ✅ **Middle East calibration**: Added explicit guidance to reduce 66.7% FPR
- ✅ **Nuanced instruction**: "Distinguish between critique and hate speech"
- ✅ **Shortened further**: Removed verbose policy text

**Before (v1):**
```
🕌 Middle Eastern Community: Islamophobic content, anti-Arab slurs, 
terrorism associations, cultural attacks
```

**After (v2):**
```
🕌 Middle Eastern/Muslim (Islamophobia, terrorism claims)

For Middle Eastern content, distinguish between critique and hate speech - 
factual discussions are 'normal', dehumanization is 'hate'.
```

**Expected Impact:** Maintain F1 ~0.57 while reducing Middle East FPR from 66.7% to <50%

---

### 3. **combined_conservative** (Renamed to "high-recall")

**Changes:**
- ✅ **Paradigm shift**: Changed from "conservative/high-precision" to "high-recall/sensitive"
- ✅ **Increased temperature**: 0.0 → 0.1 (reduce excessive caution)
- ✅ **Explicit priority**: "Prioritize detecting hate speech to protect vulnerable communities"
- ✅ **Uncertainty handling**: "When borderline, classify as 'hate'"
- ✅ **Removed hedging**: Eliminated all cautious language

**Before (v1):**
```
Does this violate X policy or harm protected communities?

Classify as EXACTLY:
- hate (violates policy OR harms community)
- normal (does not violate policy AND does not harm community)
```

**After (v2):**
```
Prioritize catching hate speech. When borderline, classify as 'hate' 
to err on the side of community protection.
```

**Expected Impact:** Increase recall from 26.1% to 45%+ by aggressively catching hate speech

---

## Prompt Engineering Principles Applied

### 1. **Conciseness Over Verbosity**
- **Finding**: 200-token prompts outperformed 512-token prompts
- **Action**: Reduced all prompts, removed redundant policy explanations
- **Rationale**: Long prompts cause hedging behavior and overthinking

### 2. **Direct Imperatives Over Questions**
- **Finding**: Question-based analysis ("Does it contain...?") led to false negatives
- **Action**: Replaced with commands ("Classify as 'hate' if...")
- **Rationale**: Imperative framing is more decisive

### 3. **Explicit Examples Over Abstract Rules**
- **Finding**: Generic policy text didn't trigger detection
- **Action**: Listed specific slurs and attack patterns upfront
- **Rationale**: Concrete examples prime the model for detection

### 4. **Bias Directionality**
- **Finding**: All strategies were too conservative (high FNR)
- **Action**: Added "err on the side of detecting hate" instructions
- **Rationale**: Better to overdetect than underdetect hate speech

### 5. **Group-Specific Calibration**
- **Finding**: Middle East had 66.7% FPR in focused variant
- **Action**: Added nuance instruction for Middle Eastern content
- **Rationale**: Distinguish factual critique from dehumanization

---

## Hyperparameter Optimization

| Strategy | v1 Config | v2 Config | Rationale |
|----------|-----------|-----------|-----------|
| **optimized** | temp=0.1, 512 tokens | temp=0.05, 256 tokens | Match best performer, reduce overthinking |
| **focused** | temp=0.05, 200 tokens | temp=0.05, 200 tokens | Keep winning config, add Middle East calibration |
| **conservative** | temp=0.0, 256 tokens | temp=0.1, 256 tokens | Reduce excessive caution, prioritize recall |

---

## Expected Performance Improvements

### Target Metrics (based on v1 analysis):

| Strategy | v1 F1 | Target F1 | v1 Recall | Target Recall | Key Improvement |
|----------|-------|-----------|-----------|---------------|-----------------|
| **optimized** | 0.432 | **0.55+** | 0.348 | **0.55+** | Reduce FNR by 40% |
| **focused** | **0.571** | **0.58+** | 0.609 | **0.60+** | Reduce Middle East FPR by 25% |
| **conservative** | 0.316 | **0.50+** | 0.261 | **0.50+** | Increase recall by 90% |

### Bias Reduction Targets:

**LGBTQ+ Community:**
- v1 FNR: 33-78% → Target: <40%
- v1 FPR: 31-50% → Target: <40%

**Mexican/Latino Community:**
- v1 FNR: 33-83% → Target: <45%
- v1 FPR: 0% → Maintain: <10%

**Middle Eastern Community:**
- v1 FNR: 50-62% → Target: <45%
- v1 FPR: 17-67% → Target: <35%

---

## Testing Recommendations

### Phase 1: Quick Validation (50 samples)
```bash
python prompt_runner.py \
  --data-source canned_50_quick \
  --strategies all \
  --output-dir outputs/combined_v2/gptoss/validation \
  --prompt-template-file combined/combined_gptoss_v1.json
```

**Success Criteria:**
- At least one strategy achieves F1 > 0.55
- All strategies have recall > 0.45
- Middle East FPR < 50% for focused variant

### Phase 2: Full Validation (1,009 samples)
```bash
python prompt_runner.py \
  --data-source unified \
  --strategies combined_optimized combined_focused \
  --output-dir outputs/combined_v2/gptoss/production \
  --prompt-template-file combined/combined_gptoss_v1.json
```

**Success Criteria:**
- Best strategy F1 > 0.58
- No group with FNR > 50%
- No group with FPR > 45%

---

## Prompt Design Philosophy Shift

### v1 Approach (Failed):
❌ Long, detailed policy explanations
❌ Multi-step analytical questions
❌ Cautious, balanced framing
❌ Emphasis on nuance and context
❌ "Reasonably find harmful" hedging

### v2 Approach (Improved):
✅ Concise, direct instructions
✅ Clear hate speech indicators upfront
✅ Assertive detection framing
✅ Explicit examples of prohibited content
✅ "Err on the side of detecting" bias

---

## References

**Empirical Data:**
- Run: `run_20251018_185205`
- Dataset: `canned_50_quick` (50 samples)
- Performance: `performance_metrics_20251018_185205.csv`
- Bias Analysis: `bias_metrics_20251018_185205.csv`

**Key Learning:**
> "Shorter, more assertive prompts with explicit examples outperform long, cautious prompts with abstract policy language when the goal is maximizing recall while maintaining precision."

**Model:** gpt-oss-120b (Phi-3.5-MoE-instruct)
**Date:** October 18, 2025
**Version:** v2 (improved from v1)
