# Examples V5 Prompt Template

## Overview
The V5 Examples template builds directly on the breakthrough insights from V4 Simple, adding **concrete examples** to all strategies based on the Policy strategy's massive success (+27.2% recall improvement).

## V4 Analysis & V5 Design Rationale

### V4 Results Analysis
| Strategy | V3 Recall | V4 Recall | Change | Root Cause |
|----------|-----------|-----------|--------|------------|
| **Baseline** | 39.4% | **45.5%** | +6.1% | ✅ Simple + Temperature worked |
| **Policy** | 15.2% | **42.4%** | +27.2% | ✅ Examples + Temperature = SUCCESS |
| **Persona** | 21.2% | 15.6% | -5.6% | ❌ No examples, just questions |
| **Combined** | 15.2% | 18.2% | +3.0% | ❌ No examples, just questions |

### Key Success Pattern Identified
**Examples + Higher Temperature = Dramatic Improvement**

The Policy strategy's massive recovery (15.2% → 42.4% recall) was due to:
1. **Concrete examples** of hate speech categories in the system prompt
2. **High temperature (0.4)** to overcome conservative bias
3. **Ultra-simple binary decision** format
4. **Specific guidance** rather than abstract reasoning

## V5 Strategy: Universal Example-Driven Approach

### Core Design Principles
1. **Add concrete examples** to ALL strategies, especially Persona and Combined
2. **Maintain high temperatures** (0.3-0.4) that proved successful
3. **Keep ultra-simple formats** that worked in V4
4. **Provide specific guidance** rather than abstract questions

### Expected Improvements

#### Persona Strategy (Current: 15.6% recall)
**V5 Changes:**
- Added specific examples of harm to each target group:
  - LGBTQ+: anti-gay slurs, trans hatred
  - Hispanic/Latino: "Mexicans are criminals", ethnic slurs
  - Middle Eastern: "Muslims are terrorists", religious hatred
- Maintained 0.35 temperature
- **Expected:** 25-35% recall (following Policy pattern)

#### Combined Strategy (Current: 18.2% recall)
**V5 Changes:**
- Added concrete examples for each detection question:
  - Attack examples: "All [group] are [negative trait]"
  - Slur examples: racial, religious, gender, sexuality slurs
  - Stereotype examples: "[Group] people always [negative behavior]"
- Maintained 0.4 temperature
- **Expected:** 30-40% recall (following Policy pattern)

#### Baseline Strategy (Current: 45.5% recall - BEST)
**V5 Changes:**
- Added general hate speech examples to system prompt
- Maintained successful 0.3 temperature
- **Expected:** Maintain or improve 45-50% recall

#### Policy Strategy (Current: 42.4% recall - RECOVERED)
**V5 Changes:**
- Enhanced examples with more specific patterns
- Maintained successful 0.4 temperature and format
- **Expected:** Maintain or improve 40-45% recall

## Temperature Strategy
- **Baseline:** 0.3 (proven successful)
- **Policy:** 0.4 (proven successful)  
- **Persona:** 0.35 (increased from 0.15 original)
- **Combined:** 0.4 (increased from 0.05 original)

Higher temperatures appear crucial for overcoming the model's conservative bias in hate speech detection.

## Validation Hypothesis
**If examples + temperature = success**, then V5 should show:
1. **Persona recall:** 15.6% → 25-35% (+10-20 percentage points)
2. **Combined recall:** 18.2% → 30-40% (+12-22 percentage points)
3. **Baseline recall:** Maintain 45%+ (currently best performing)
4. **Policy recall:** Maintain 40%+ (recovered from disaster)

## Success Metrics
- **Primary Goal:** Achieve 30%+ recall across ALL strategies
- **Stretch Goal:** Achieve 40%+ recall for at least 3/4 strategies
- **Validation:** If Persona/Combined don't improve significantly, the model may be fundamentally limited for these approaches

## Next Steps
1. Test V5 template on same 100-sample dataset
2. Compare results with V4 to validate example-driven hypothesis
3. If successful, consider V6 with even more specific examples
4. If unsuccessful, explore alternative model architectures or completely different prompt strategies

---
*Created: September 21, 2025*  
*Based on: V4 Simple analysis and Policy strategy success pattern*  
*Hypothesis: Examples are the missing ingredient for Persona and Combined strategy success*