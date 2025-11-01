# V4 Strategy Visualization: Minimal Baseline Enhancement

## Performance History

```
Baseline Performance (Target to Beat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
baseline_standard: F1 = 0.615 (production, 1,009 samples)
                   Configuration: temp=0.1, 512 tokens
                   ╰─ Optimal hyperparameters from systematic testing
```

```
Combined Approaches History (All Failed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V1: 15 examples + verbose policy/persona
    combined_optimized: F1 = 0.590 (-2.5% vs baseline) ❌
    ╰─ Too many examples, verbosity overhead

V2: 0-2 examples + cultural context  
    cultural_context: F1 = 0.565 (-5% vs baseline) ❌
    ╰─ Insufficient examples, underperformed

V3: 5 examples + very verbose structure
    recall_focused: F1 = 0.559 (-5.6% vs baseline) ❌
    optimized: F1 = 0.438 (-28.6% vs baseline) 💀 CATASTROPHIC
    ╰─ Over-engineering collapsed recall (0.340 vs V1's 0.660)
```

## The Problem Pattern

```
Complexity vs Performance Curve

F1 Score
   ↑
0.65│
    │
0.62│              ┌─── Baseline (0.615) ◄─── Current Best
    │              │
0.61│              │
    │              │
0.60│         V1 (0.590)
    │              
0.59│              
    │         
0.57│    V2 (0.565)
    │         
0.56│    V3 (0.559)
    │
    │                           ┌─── Hypothesis: 
0.44│                                "Goldilocks Zone"
    │                                exists here
0.43│                      V3_optimized (0.438) 💀
    │
    └──────────────────────────────────────────────────► Complexity
       Simple          Moderate              Very Complex
```

**Key Insight**: Adding complexity CONSISTENTLY degraded performance

## V4's Strategic Position

```
Finding the Goldilocks Zone

Too Simple          GOLDILOCKS ZONE         Too Complex
   ↓                      ↓                      ↓
Baseline          ╔═══════════════╗         V1/V2/V3
F1=0.615         ║   V4 TESTS    ║         F1=0.438-0.590
                 ║   THIS ZONE   ║
No examples      ║               ║         5-15 examples
No context       ║  1-6 examples ║         Verbose prompts
Generic          ║  Brief context║         Structured frameworks
                 ╚═══════════════╝

Expected: F1 = 0.620-0.635 (beat baseline)
```

## V4 Strategy Comparison

```
Strategy 1: combined_v4_minimal_examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline + 6 examples (1 HATE + 1 NORMAL per group)

Prompt Size: ████████ (small increase)
Expected F1: 0.620-0.630
Risk Level:  LOW ✓

Rationale: V1's 15 examples too many, V2's 0-2 too few
           6 examples = optimal balance


Strategy 2: combined_v4_subtle_emphasis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline + single sentence on coded hate

Prompt Size: ██ (minimal increase)
Expected F1: 0.618-0.625
Risk Level:  VERY LOW ✓✓

Rationale: Addresses baseline's weakness (missing subtle hate)
           Minimal addition = lowest over-engineering risk


Strategy 3: combined_v4_community_aware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline + brief community context (no examples)

Prompt Size: ██████ (moderate increase)
Expected F1: 0.615-0.625
Risk Level:  LOW-MEDIUM ⚠️

Rationale: Cultural awareness without V2's verbosity
           May reduce FPR disparity (LGBTQ+ 43% → lower)


Strategy 4: combined_v4_balanced_lite ⭐ RECOMMENDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline + 1 example per group + brief context

Prompt Size: ████████████ (moderate increase, still <V1)
Expected F1: 0.625-0.635 (HIGHEST)
Risk Level:  MEDIUM ⚠️

Rationale: Combines two proven elements
           Examples teach patterns, context teaches culture
           Synergy effect may beat individual approaches
```

## Addition Comparison Table

```
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│                     │ Baseline │   V1     │   V3     │   V4     │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Examples per group  │    0     │    5     │    5     │   0-1    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Total examples      │    0     │   15     │   15     │   0-6    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Community context   │   None   │ Verbose  │ Verbose  │  Brief   │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Policy guidance     │   None   │ Verbose  │ Verbose  │  None    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Structure           │  Simple  │ Complex  │Very Cmplx│ Simple+  │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Prompt word count   │   ~80    │  ~500    │  ~700    │ ~100-230 │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Production F1       │  0.615   │  0.590   │   N/A    │   TBD    │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Status              │ Current  │ Failed   │Catastroph│ Testing  │
│                     │  Best    │  (-2.5%) │(-28.6%)  │          │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

## Decision Tree

```
                    Run V4 Test (100 samples)
                            │
                            ▼
            ┌───────────────┴───────────────┐
            │                               │
      F1 ≥ 0.626                      F1 < 0.620
         ✓ SUCCESS                      ✗ FAILURE
            │                               │
            ▼                               ▼
    Run Production Test             Deploy baseline_standard
    (1,009 samples)                      F1 = 0.615
            │                               │
            ▼                               ▼
    ┌───────┴────────┐              Consider alternatives:
    │                │              • Model fine-tuning
F1 > 0.615      F1 ≤ 0.615         • Different base model
 ✓ BEAT          ✗ FAILED          • Ensemble approach
BASELINE        BASELINE            • Human-in-loop
    │                │
    ▼                ▼
Deploy V4      Deploy baseline
to production      (0.615)
```

## Testing Timeline

```
Day 1: Small-Sample Test (100 samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

09:00 │ Run V4 test (all 4 strategies)
      │ Duration: ~5-7 minutes
      │
09:10 │ Analyze results
      │ • Check F1 scores
      │ • Compare bias metrics
      │ • Identify best performer
      │
09:30 │ Decision Point
      ├─ If F1 ≥ 0.626: Proceed to Day 2
      └─ If F1 < 0.620: V4 failed, use baseline


Day 2: Production Test (1,009 samples) - IF Phase 1 succeeds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

10:00 │ Run production test (best V4 strategy)
      │ Duration: ~20 minutes
      │
10:30 │ Analyze production results
      │ • F1 score vs baseline (0.615)
      │ • Generalization (100 → 1,009 samples)
      │ • Bias fairness metrics
      │
11:00 │ Final Decision
      ├─ If F1 > 0.615: Deploy V4 ✓
      └─ If F1 ≤ 0.615: Deploy baseline (0.615) ✓
```

## Expected Outcomes (Ranked by Probability)

```
Outcome 1: V4 BEATS BASELINE (60% probability)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Most likely: combined_v4_balanced_lite
Expected F1: 0.620-0.635
Action: Deploy to production ✓

Reasoning:
• Minimal additions address baseline weaknesses
• Examples provide pattern learning
• Context provides cultural awareness
• No verbosity overhead (unlike V1/V3)


Outcome 2: V4 MATCHES BASELINE (25% probability)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected F1: 0.610-0.620
Action: Deploy baseline_standard (0.615) ✓

Reasoning:
• Minimal additions not enough to improve
• Baseline already near-optimal for this model
• Prompt engineering may have hit ceiling


Outcome 3: V4 FAILS (15% probability)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected F1: < 0.610
Action: Deploy baseline_standard (0.615) ✓

Reasoning:
• Even minimal additions harmful
• Model architecture limitation
• Consider fine-tuning or model upgrade
```

## Key Metrics to Watch

```
Performance Metrics
━━━━━━━━━━━━━━━━━━
• F1-Score:      Target > 0.615 (baseline)
• Precision:     Maintain ≥ 0.600
• Recall:        Maintain ≥ 0.600 (avoid V3's 0.340 collapse)
• Accuracy:      Target > 65.0%


Bias Fairness Metrics
━━━━━━━━━━━━━━━━━━━━
• LGBTQ+ FPR:    Target < 43.0% (baseline's weakness)
• Mexican FPR:   Maintain ≈ 8.1% (baseline's strength)
• Middle East FPR: Target < 23.6%
• FNR Balance:   All groups < 40%


Generalization Metrics
━━━━━━━━━━━━━━━━━━━━
• Degradation:   Target < 2% (100 → 1,009 samples)
                 (baseline achieved 1.1%)
```

## Success Definition

```
BEAT BASELINE = TRUE if ALL conditions met:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. F1 > 0.615 (production)               ✓ Required
2. Generalization < 2% degradation       ✓ Required  
3. No recall collapse (recall ≥ 0.600)   ✓ Required
4. FPR disparity reduced                 ○ Nice-to-have
5. Balanced precision/recall             ✓ Required

If ANY required condition fails:
  └─> Deploy baseline_standard (F1=0.615)
```

## The V4 Bet

```
HYPOTHESIS: "Minimal additions can beat baseline"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evidence FOR:
✓ Baseline lacks pattern guidance (no examples)
✓ Baseline lacks cultural awareness (generic)
✓ Literature shows 1-5 examples help few-shot learning
✓ V1/V2/V3 failed due to OVER-engineering, not concept

Evidence AGAINST:
✗ Three attempts (V1/V2/V3) all failed to beat baseline
✗ Baseline may already be optimal for this model
✗ Adding ANYTHING might introduce noise

Conclusion: Worth testing, but baseline remains strong fallback
```
