# Combined V5: Noise-Reduced Policy-Persona Strategies

## Hypothesis: Noise Reduction Over Verbosity

**Learning from V1-V4 failures**: All verbose, structured prompts degraded performance. Hypothesis: The problem wasn't policy/persona *concepts*, but **noise from excessive text**.

**V5 Strategy**: Test noise-reduction techniques that preserve policy-persona guidance while minimizing token count and structural complexity.

---

## Why V5 is Different from V1-V4

### What Failed in V1-V4

```
V1-V4 Pattern: More text = worse performance

Baseline:         80 words → F1 = 0.615 ✓
V4 minimal:      230 words → F1 = 0.589 ❌
V1 optimized:    500 words → F1 = 0.590 ❌
V3 optimized:    700 words → F1 = 0.438 ❌

Problem: Verbose explanations, structured sections, lengthy examples
```

### V5 Approach: Compress, Don't Remove

**Key insight from noise-reduction principles**:
- ❌ Don't add verbose policy explanations
- ✅ Do compress policy into compact signals
- ❌ Don't structure into frameworks
- ✅ Do encode patterns efficiently
- ❌ Don't explain examples extensively
- ✅ Do show raw pattern examples

---

## V5 Strategies (5 Noise-Reduction Approaches)

### 1. **combined_v5_compressed_tokens** (Token Compression)

**Approach**: Replace verbose instructions with compact token-style markers

**Traditional (V1-V4 verbose)**:
```
"You are an expert content moderation assistant trained to classify 
social media posts as either 'hate' or 'normal' using X platform's 
hateful conduct policy and community-informed perspectives..."
(200+ words of explanation)
```

**V5 Compressed**:
```
[Policy: Protected characteristics]
[Task: Binary classification]
[Format: JSON strict]
[Context: LGBTQ+, Mexican/Latino, Middle Eastern]
[Focus: Coded language, generalizations, dehumanization]
```

**Word count**: 80 words (baseline-equivalent) vs V1's 500 words  
**Expected**: Preserves policy guidance without noise

---

### 2. **combined_v5_implicit_examples** (Implicit Policy Encoding)

**Approach**: Policy encoded through examples only, zero explanatory text

**Traditional (V4 balanced_lite)**:
```
LGBTQ+: LGBTQ+ individuals may reclaim terms as empowerment; 
outsiders using same terms to attack is hate.
HATE Example: "Those faggots are disgusting"
NORMAL Example: "I'm queer and proud"
(Repeated with explanations for each group)
```

**V5 Implicit**:
```
Examples:

LGBTQ+:
"Those faggots are disgusting" → hate
"I'm queer and proud" → normal

Mexican/Latino:
"Mexicans are all illegals" → hate
"Border policy needs reform" → normal
(No explanatory text)
```

**Word count**: 60 words vs V4's 180 words  
**Expected**: Pattern learning without explanation overhead

---

### 3. **combined_v5_chain_of_thought** (Reasoning Before Output)

**Approach**: Ask for step-by-step reasoning, let model work through detection

**Traditional (V3 structured)**:
```
=== EVALUATION FRAMEWORK ===
Text: "{text}"

Evaluate:
1. Does it attack PEOPLE based on protected characteristics?
2. Does it use slurs, dehumanization, or harmful generalizations?
3. Would affected community members reasonably view this as hateful?
(Verbose evaluation sections)
```

**V5 Chain-of-Thought**:
```
Use chain-of-thought reasoning:
Step 1: Identify if text attacks people based on protected characteristics
Step 2: Check for coded language, generalizations, dehumanization
Step 3: Distinguish policy critique vs people attack
Step 4: Output classification

Text: "{text}"
Analyze and classify:
```

**Word count**: 90 words vs V3's 350 words  
**Expected**: Reasoning improves nuance detection without structure noise

---

### 4. **combined_v5_minimal_signal** (Single-Line Enhancement)

**Approach**: Baseline + one sentence defining hate with key patterns

**Traditional (V4 policy_lite)**:
```
Policy: Content attacking people based on protected characteristics 
(race, ethnicity, national origin, sexual orientation, gender identity, 
religion) is hate speech. Criticizing policies/ideologies without 
attacking people is not hate.

Key Patterns:
LGBTQ+: Out-group slurs/attacks = HATE | In-group reclamation = NORMAL
(Multiple pattern explanations)
```

**V5 Minimal**:
```
Hate = attacking people based on protected characteristics (race, 
ethnicity, national origin, sexual orientation, gender identity, 
religion) including generalizations, coded language, dehumanization.
```

**Word count**: 50 words vs V4 policy_lite's 200 words  
**Expected**: Essential definition without noise

---

### 5. **combined_v5_example_only** (Pure Pattern Learning)

**Approach**: Examples only, zero explanatory text or instructions

**Traditional (All previous versions)**:
```
System: "You are an expert content moderation assistant..."
User: "Classify the following text... [context] [examples] [instructions]"
```

**V5 Example-Only**:
```
System: "Classify as 'hate' or 'normal'. Output JSON."

User:
"Those faggots are disgusting" → hate
"I'm queer and proud" → normal
"Mexicans are all illegals" → hate
"Border policy needs reform" → normal
"All Muslims are terrorists" → hate
"ISIS commits terrorist acts" → normal

"{text}" →
```

**Word count**: 35 words (MINIMAL) vs V1's 500 words  
**Expected**: Pure pattern matching without any instructional overhead

---

## V5 Strategy Comparison

| Strategy | Word Count | Examples | Policy | Reasoning | Innovation |
|----------|------------|----------|--------|-----------|------------|
| **Baseline** | ~80 | 0 | Generic | No | N/A |
| **compressed_tokens** | ~80 | 0 | Compact tokens | No | Token compression |
| **implicit_examples** | ~60 | 6 | Via examples | No | Implicit encoding |
| **chain_of_thought** | ~90 | 0 | Minimal | Yes | Explicit steps |
| **minimal_signal** | ~50 | 0 | One sentence | No | Essential only |
| **example_only** | ~35 | 6 | None | No | Pure patterns |

---

## Expected Performance vs V1-V4

### Hypothesis

```
Noise Reduction → Performance Recovery

V3 optimized:  700 words → F1 = 0.438 (catastrophic)
V1 optimized:  500 words → F1 = 0.590 (-4%)
V4 balanced:   230 words → F1 = 0.571 (-7%)
Baseline:       80 words → F1 = 0.615 ✓

V5 strategies:  35-90 words → F1 = 0.600-0.630? (hypothesis)
```

**If hypothesis correct**: V5 strategies should outperform V1-V4 by staying closer to baseline's token economy while adding minimal, compressed guidance.

**If hypothesis wrong**: Even compressed additions still introduce noise, confirming baseline is absolute ceiling.

---

## Testing Strategy

### Test Command

```bash
cd prompt_engineering

python prompt_runner.py \
  --data-source canned_100_size_varied \
  --strategies combined_v5_compressed_tokens combined_v5_implicit_examples combined_v5_chain_of_thought combined_v5_minimal_signal combined_v5_example_only \
  --output-dir outputs/combined_v5/gptoss/validation_100 \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v5_noise_reduced.json
```

**Duration**: ~7 minutes for 500 classifications (100 samples × 5 strategies)

---

## Success Criteria

### Primary Goal
**Beat baseline F1=0.615** or at minimum match it (F1≥0.615)

### Strategy-Specific Predictions

| Strategy | Expected F1 | Rationale |
|----------|-------------|-----------|
| **compressed_tokens** | 0.600-0.620 | Token compression preserves guidance without noise |
| **implicit_examples** | 0.595-0.615 | Examples without explanations reduce noise |
| **chain_of_thought** | 0.590-0.625 | Reasoning may help nuance OR add length overhead |
| **minimal_signal** | 0.605-0.620 | Single sentence optimal? (least addition) |
| **example_only** | 0.590-0.610 | Pure patterns, but may lack context |

**Best bet**: `minimal_signal` or `compressed_tokens` (closest to baseline word count)

---

## What This Tests

### Core Questions

1. **Is verbosity the problem?**
   - If V5 succeeds → Yes, noise from text length
   - If V5 fails → No, ANY addition degrades (confirm baseline ceiling)

2. **Can policy be compressed effectively?**
   - compressed_tokens vs baseline
   - If succeeds → Token-style markers work
   - If fails → Policy guidance inherently harmful

3. **Do examples need explanations?**
   - implicit_examples vs V4 balanced_lite (same examples, less text)
   - If succeeds → Explanations were noise
   - If fails → Examples themselves are noise

4. **Does reasoning help nuance?**
   - chain_of_thought vs baseline
   - If succeeds → Reasoning improves detection
   - If fails → Adds token overhead without benefit

5. **What's the absolute minimum?**
   - example_only and minimal_signal
   - Tests floor of what additions might work

---

## Comparison to Previous Versions

### Word Count Analysis

```
Version Evolution (System + User prompts):

Baseline:              ~80 words  → F1 = 0.615 ✓
V5 example_only:       ~35 words  → F1 = ???
V5 minimal_signal:     ~50 words  → F1 = ???
V5 implicit_examples:  ~60 words  → F1 = ???
V5 compressed_tokens:  ~80 words  → F1 = ???
V5 chain_of_thought:   ~90 words  → F1 = ???
V4 minimal_examples:  ~230 words  → F1 = 0.589 ❌
V4 balanced_lite:     ~230 words  → F1 = 0.571 ❌
V1 optimized:         ~500 words  → F1 = 0.590 ❌
V3 optimized:         ~700 words  → F1 = 0.438 ❌
```

**V5's unique position**: All strategies stay ≤90 words (at or below baseline length)

---

## Risk Assessment

### Low Risk Strategies
- **minimal_signal**: Smallest addition (1 sentence), very low noise risk
- **compressed_tokens**: Same word count as baseline, just reformatted
- **example_only**: Minimal text, pure patterns

### Medium Risk Strategies
- **implicit_examples**: Examples may still introduce noise (V4 showed this)
- **chain_of_thought**: Reasoning adds tokens, may hurt despite utility

### What Could Go Wrong

**Scenario 1: All V5 fail (most likely)**
- Confirms: ANY addition degrades performance
- Learning: Baseline is absolute ceiling for this model
- Action: Deploy baseline, stop prompt engineering

**Scenario 2: One V5 beats baseline (hopeful)**
- Confirms: Noise reduction works
- Learning: Compression enables guidance without overhead
- Action: Deploy winner, refine approach

**Scenario 3: V5 matches baseline (interesting)**
- Confirms: No harm, but no gain
- Learning: Model already optimal, additions neutral
- Action: Deploy baseline (simpler)

---

## Decision Tree After V5

```
Run V5 Test (100 samples)
         │
         ▼
    ┌────┴────┐
    │ Results │
    └────┬────┘
         │
    ┌────┴────────────────┐
    │                     │
F1 > 0.615           F1 ≤ 0.615
    │                     │
    ▼                     ▼
✓ SUCCESS          ❌ FAILURE
    │                     │
    ├─ Test winner on     ├─ Baseline confirmed
    │  production (1,009)  │  optimal
    │                     │
    ├─ If prod > 0.615:   ├─ Deploy baseline_standard
    │  Deploy V5 winner   │  F1 = 0.615
    │                     │
    └─ If prod ≤ 0.615:   └─ Stop prompt engineering
       Deploy baseline        Move to LoRA
```

---

## V5 vs LoRA: Complementary Tests

### If V5 Succeeds
- **V5**: Proves noise reduction works for prompt engineering
- **LoRA**: Still pursue for +5-10% gains beyond optimized prompts
- **Path**: Deploy V5 short-term, LoRA long-term

### If V5 Fails
- **V5**: Confirms prompt engineering ceiling (F1=0.615)
- **LoRA**: Only path forward for improvement
- **Path**: Deploy baseline, prioritize LoRA development

**Both valuable**: V5 tests prompt optimization ceiling, LoRA enables breakthrough

---

## Key Innovation: Testing Compression Hypothesis

**Previous versions tested**:
- V1: Full policy + persona (verbose)
- V2: Less examples (still verbose)
- V3: More structure (MORE verbose)
- V4: Minimal additions (still added 150-230 words)

**V5 uniquely tests**:
- Compression techniques (tokens, implicit encoding)
- Absolute minimum additions (35-90 words vs baseline's 80)
- Pure pattern learning (examples without explanations)
- Reasoning-based detection (chain-of-thought)

**Novel hypothesis**: Maybe V1-V4 had the RIGHT ideas (policy, examples, context) but WRONG implementation (too verbose). V5 tests if compressed versions work.

---

## Expected Outcomes (Probabilistic)

### Most Likely (70% probability): All V5 Fail
```
Best V5: F1 = 0.590-0.610 (still below baseline 0.615)
Conclusion: Baseline is ceiling, ANY addition harmful
Action: Deploy baseline, move to LoRA
```

### Possible (25% probability): One V5 Matches Baseline
```
Best V5: F1 = 0.610-0.615 (approaches baseline)
Conclusion: Compression reduces harm but doesn't beat baseline
Action: Deploy baseline (simpler), consider LoRA
```

### Hopeful (5% probability): One V5 Beats Baseline
```
Best V5: F1 = 0.620-0.630 (beats baseline!)
Conclusion: Noise reduction enables prompt optimization
Action: Test winner on production, deploy if validated
```

---

## Evaluation Metrics to Track

### Performance Metrics
- **F1 Score**: Primary metric (target > 0.615)
- **Precision**: Maintain ≥ 0.600 (avoid false positives)
- **Recall**: Maintain ≥ 0.600 (avoid V3/V4 collapse)

### Noise Impact Indicators
- **Token count correlation**: Does less text = better F1?
- **Structure impact**: Does chain-of-thought help or hurt?
- **Example efficiency**: Do raw examples beat explained examples?

### Bias Fairness
- **LGBTQ+ FPR**: Target < 43% (baseline's weakness)
- **Mexican FNR**: Target < 40% (missing coded hate)
- **Balance**: All groups FPR/FNR < 35%

---

## Documentation Cross-Reference

### Related Files
- **V4 Results**: `gpt_oss_combined_ift_4Iter_summary.md` (why V1-V4 failed)
- **Baseline**: `gptoss_ift_summary_README.md` (F1=0.615 performance)
- **LoRA Rationale**: `gpt_oss_combined_ift_4Iter_summary.md` (Section: "Beyond Prompt Engineering")

### This Approach
- **Template**: `combined/combined_v5_noise_reduced.json`
- **README**: `combined/combined_v5_noise_reduced_README.md` (this file)
- **Runs**: `outputs/combined_v5/gptoss/validation_100/`

---

## Summary: The Noise-Reduction Experiment

**What we're testing**: Can compressed policy/persona guidance beat baseline where verbose versions failed?

**Why it matters**: 
- If YES → Noise reduction is key to prompt optimization
- If NO → Confirms baseline ceiling, validates LoRA path

**Expected result**: Likely failure (baseline still best), but worth testing to confirm compression hypothesis before abandoning prompt engineering.

**Time investment**: 10 minutes (7 min run + 3 min analysis)

**Value**: Definitive answer on whether prompt engineering has ANY remaining optimization potential for this task.

---

**Ready to test**: Run the command above and compare V5 results to baseline F1=0.615 🎯
