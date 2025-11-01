# Combined V4: Quick Reference Guide

## Mission
**Beat baseline_standard's F1=0.615** with minimal baseline enhancements

## Why V4 is Different

**ALL previous combined approaches FAILED to beat baseline:**
- V1 (15 examples, verbose): F1=0.590 production ❌
- V2 (0-2 examples): F1=0.565 best ❌  
- V3 (5 examples, very verbose): F1=0.438-0.559 ❌ CATASTROPHIC

**V4 Strategy:** Add MINIMAL enhancements to baseline (no verbosity)

## V4 Strategies

### 1. combined_v4_minimal_examples
- **What**: Baseline + 6 examples (1 HATE + 1 NORMAL per group)
- **Expected F1**: 0.620-0.630
- **Risk**: Low

### 2. combined_v4_subtle_emphasis  
- **What**: Baseline + single sentence on coded hate
- **Expected F1**: 0.618-0.625
- **Risk**: Very Low

### 3. combined_v4_community_aware
- **What**: Baseline + brief community context (no examples)
- **Expected F1**: 0.615-0.625
- **Risk**: Low-Medium

### 4. combined_v4_balanced_lite ⭐ RECOMMENDED
- **What**: Baseline + 1 example per group + brief context
- **Expected F1**: 0.625-0.635
- **Risk**: Medium

## Quick Test (100 samples)

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

**Duration**: ~5-7 minutes

## Success Criteria

**Phase 1 (100 samples):**
- ✅ F1 ≥ 0.626 → Proceed to production test
- ⚠️ F1 ≥ 0.620 → Promising, needs refinement
- ❌ F1 < 0.620 → Failed

**Phase 2 (1,009 samples, if Phase 1 succeeds):**
- ✅ F1 > 0.615 → BEAT BASELINE, deploy to production
- ❌ F1 ≤ 0.615 → Baseline remains optimal

## Production Test (if Phase 1 succeeds)

```bash
python prompt_runner.py \
  --data-source unified \
  --strategies <best_v4_strategy> \
  --output-dir outputs/combined_v4/gptoss/production \
  --max-workers 15 \
  --batch-size 8 \
  --prompt-template-file combined/combined_v4_baseline_enhanced.json
```

**Duration**: ~20 minutes

## Expected Best Performer

**combined_v4_balanced_lite** (F1=0.625-0.635 expected)
- Combines minimal examples + brief context
- Synergy effect: examples teach patterns, context teaches culture
- Still maintains baseline simplicity

## Key Hypothesis

**"The optimal prompt exists between baseline's simplicity and V1's complexity"**

V4 targets the "goldilocks zone" with minimal additions.

## Files

- **Template**: `combined/combined_v4_baseline_enhanced.json`
- **Detailed README**: `combined/combined_v4_baseline_enhanced_README.md`
- **Baseline Reference**: `gptoss_ift_summary_README.md`

## What If V4 Fails?

- Deploy **baseline_standard** (F1=0.615, proven)
- Consider model fine-tuning or architecture changes
- Prompt engineering may have hit ceiling for this model
