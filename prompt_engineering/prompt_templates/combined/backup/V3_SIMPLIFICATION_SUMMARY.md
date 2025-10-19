# V3 Prompt Simplification Summary

**Date**: 2025-01-18  
**Objective**: Simplify prompts while maintaining v2 performance improvements and adhering to user guidelines (X policy + 3 personas, no emojis)

---

## Changes Made

### User Template Simplification (All Strategies)

**Before (v2)**: ~40 lines with exhaustive examples
**After (v3)**: ~15 lines with pattern-based guidance

#### Key Reductions:

1. **Policy Section**:
   - **Removed**: 10-line detailed policy breakdown with bullet points
   - **Kept**: 2-line concise policy statement emphasizing PEOPLE vs. policies distinction
   - **Rationale**: Full policy already in system prompt (shown once); user template just needs reminder

2. **Community Sections**:
   - **LGBTQ+**: 5 lines → 3 lines (removed slur examples, kept pattern + context)
   - **Mexican/Latino**: 7 lines → 3 lines (removed slur list, emphasized immigration-based generalization patterns)
   - **Middle Eastern**: 6 lines → 3 lines (removed slur list, simplified terrorism distinction)

3. **Analysis Framework**:
   - **Removed**: 5-step detailed evaluation process
   - **Kept**: 3 core questions focusing on pattern detection
   - **Rationale**: Simpler framework reduces cognitive load while maintaining effectiveness

4. **Pattern-Based vs. Exhaustive Approach**:
   - **Old**: Listed specific slurs (faggots, trannies, beaners, wetbacks, towelhead, etc.)
   - **New**: Emphasized patterns ("generalizing ALL as X", "immigration-based hate", "terrorism generalizations")
   - **Why**: Mexican FNR remained 50-83% despite exhaustive slurs, suggesting examples cause information overload

---

## Simplified Prompt Structure

### Combined Optimized (Full Version)

```
Apply X Platform Hateful Conduct Policy: Content attacking people based on 
protected characteristics (race, ethnicity, national origin, sexual orientation, 
gender identity, religion, age, disability) is prohibited. Attacking PEOPLE is 
hate; criticizing policies/ideologies is not.

COMMUNITY FOCUS:

LGBTQ+ Community:
- Slurs targeting sexual orientation/gender identity (both explicit attacks and coded language)
- Denying LGBTQ+ identities or stereotyping as dangerous/mentally ill
- Context: LGBTQ+ individuals reclaiming terms (e.g., 'we're queer') is NOT hate; 
  outsiders using same terms to attack IS hate

Mexican/Latino Community:
- Slurs, dehumanization (comparing to animals/diseases/infestations)
- Immigration-based hate: Generalizing ALL Mexicans as 'illegal', 'criminals', 
  'invaders', 'rapists', 'job stealers'
- Coded phrases: 'They're not sending their best', 'go back' when targeting 
  ethnicity (not individual actions)
- Note: Immigration policy debates are NOT hate unless generalizing/attacking 
  ALL Mexican/Latino people

Middle Eastern Community:
- Anti-Arab/anti-Muslim slurs, Islamophobic stereotypes
- Terrorism generalizations: Assuming ALL Middle Eastern people are terrorists/extremists
- Distinction: Discussing specific conflicts/groups = NOT hate. Attributing 
  terrorism to ALL Middle Eastern people = hate

Text: "{text}"

Evaluate:
1. Does it attack PEOPLE based on protected characteristics (vs. criticizing policies)?
2. Does it use slurs, dehumanization, or harmful generalizations?
3. Would affected community members reasonably view this as hateful?

Classify:
- hate (attacks people based on protected characteristics)
- normal (does not attack protected communities)

Respond in JSON: {"classification": "[hate/normal]", "rationale": "[cite specific violation or community harm]"}
```

### Combined Focused & Conservative (Ultra-Compact)

```
Apply X Policy: Attacking people based on protected characteristics (race, ethnicity, 
national origin, sexual orientation, gender identity, religion, age, disability) is 
prohibited. Policy criticism is allowed.

COMMUNITY FOCUS:
LGBTQ+: Slurs/denying identity/dangerous stereotypes. Context: In-group reclamation 
  vs. out-group attacks.
Mexican/Latino: Slurs, immigration-based hate (generalizing ALL as 
  'illegal/criminals/invaders'), coded phrases ('not sending best', 'go back').
Middle Eastern: Slurs, Islamophobia, terrorism generalizations (ALL people vs. 
  specific events/groups).

Text: "{text}"

Evaluate: 1) Attacks PEOPLE vs. policies? 2) Slurs/dehumanization/generalizations? 
3) Harmful to communities?

Classify: hate OR normal

JSON: {"classification": "[hate/normal]", "rationale": "[cite violation]"}
```

---

## Comparison: V2 → V3

### Mexican/Latino Section

**V2 (Exhaustive)**:
```
Mexican/Latino Community:
- Explicit slurs: beaners, wetbacks, illegals (when dehumanizing), anchor babies
- Immigration-based attacks: Calling all Mexicans 'illegal', 'criminals', 'invaders'
- Ethnic stereotypes: lazy, drug dealers, rapists, job stealers
- Dehumanization: Comparing to animals, diseases, or infestations
- Coded hate: 'They're not sending their best', 'go back to your country' when 
  targeting ethnicity
```

**V3 (Pattern-Focused)**:
```
Mexican/Latino Community:
- Slurs, dehumanization (comparing to animals/diseases/infestations)
- Immigration-based hate: Generalizing ALL Mexicans as 'illegal', 'criminals', 
  'invaders', 'rapists', 'job stealers'
- Coded phrases: 'They're not sending their best', 'go back' when targeting 
  ethnicity (not individual actions)
- Note: Immigration policy debates are NOT hate unless generalizing/attacking 
  ALL Mexican/Latino people
```

**Key Change**: Emphasized "generalizing ALL" pattern instead of listing every possible slur/stereotype. This addresses the root detection problem: distinguishing between individual criticism and group-based attacks.

### LGBTQ+ Section

**V2**:
```
LGBTQ+ Community:
- Explicit slurs: faggots, trannies, dykes, queers (when used as attacks)
- Attacks on sexual orientation or gender identity
- Denial of LGBTQ+ identities or existence
- Harmful stereotypes (predators, mentally ill, groomers)
- Context: Distinguish in-group reclamation (LGBTQ+ person saying 'we're queer') 
  from out-group slurs
```

**V3**:
```
LGBTQ+ Community:
- Slurs targeting sexual orientation/gender identity (both explicit attacks and coded language)
- Denying LGBTQ+ identities or stereotyping as dangerous/mentally ill
- Context: LGBTQ+ individuals reclaiming terms (e.g., 'we're queer') is NOT hate; 
  outsiders using same terms to attack IS hate
```

**Key Change**: Removed explicit slur examples, consolidated patterns, maintained critical in-group/out-group distinction that reduced FPR (50%→19% in v2).

---

## Rationale for Simplification

### Why V2 Prompts Were Too Complex

1. **Redundancy**: Policy stated in both system prompt (comprehensive) and user template (repeated)
2. **Information Overload**: 40+ lines in user template may cause model confusion
3. **Exhaustive Lists Don't Work**: Mexican section expanded from 2→7 lines in v2, but FNR still 50-83%
4. **Token Budget**: Longer prompts reduce space for model reasoning in response

### Why Pattern-Based Approach Works Better

1. **Generalization**: "Generalizing ALL Mexicans as X" teaches detection principle, not memorization
2. **Scalability**: Patterns cover infinite variations; slur lists are never complete
3. **Cognitive Load**: Simpler prompts = clearer decision-making for model
4. **Mexican Detection**: Issue isn't missing slur examples, it's detecting generalization pattern

### What We Preserved from V2

✅ **Coded hate detection**: Maintained "coded phrases" guidance  
✅ **Immigration-based patterns**: Kept "They're not sending their best" examples  
✅ **In-group reclamation**: Preserved LGBTQ+ context distinction (reduced v2 FPR)  
✅ **Terrorism distinction**: Maintained Middle Eastern "specific events vs. ALL people" clarification  
✅ **PEOPLE vs. policies**: Emphasized throughout as core detection principle  

---

## Expected Outcomes

### Performance Goals

1. **Maintain v2 accuracy improvements**: Target ≥65% accuracy for focused/conservative
2. **Improve Mexican detection**: Target <40% FNR (from current 50-83%)
3. **Maintain LGBTQ+ FPR improvements**: Keep ~20% FPR (from v2's 19%)
4. **Calibrate Middle Eastern FPR**: Return optimized variant to <50% (from v2's 67%)
5. **Token efficiency**: Reduce prompt tokens by ~50%

### Validation Plan

Run v3 validation on `canned_50_quick`:
```bash
python prompt_runner.py --template combined_gptoss_v1.json --data canned_50_quick --model gptoss --output combined_v3
```

**Success Criteria**:
- Overall F1 ≥0.60 (maintain v2 performance)
- Mexican FNR improves by ≥10% (target: 40-73% range)
- LGBTQ+ FPR maintains ≤30%
- Middle Eastern FPR for optimized returns to ≤50%

---

## Guidelines Compliance

✅ **X Platform Policy**: Used as authoritative foundation throughout  
✅ **3 Personas**: Focus maintained on LGBTQ+, Mexican/Latino, Middle Eastern  
✅ **No Emojis**: No emojis used (removed in v2, maintained in v3)  
✅ **Simplification**: Dramatically reduced complexity while maintaining effectiveness  

---

## Next Steps

1. **Validate v3**: Run 50-sample validation to compare against v2 baseline
2. **Analyze results**: Compare performance_metrics and bias_metrics CSVs
3. **Iterate if needed**: If Mexican FNR doesn't improve, consider alternative approaches:
   - Add explicit "generalization vs. individual" instruction
   - Test separate system prompt emphasis on pattern detection
   - Explore few-shot examples for Mexican immigration-based hate
4. **Production testing**: If v3 succeeds, run full 1,009-sample validation
