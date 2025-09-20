# Improved Combined Prompt Template Analysis

## Performance Issues Analysis

Based on the confusion matrix results, there's a performance paradox where the simplest strategy (Baseline) achieves the best performance while the most sophisticated strategy (Combined) achieves the worst performance. Initial analysis showed the original prompts were over-engineering the problem and making models too conservative.

**Critical Discovery:** The first "improved" version successfully enhanced individual strategies (Baseline: 57.0% → 61.0% accuracy, Policy: 56.0% → 62.0%, Persona: 56.0% → 66.7%) but severely degraded the Combined strategy (recall dropped from 24.2% to 15.2%, missing even more hate speech). This revealed that complexity in combined approaches can backfire.

## Root Cause Analysis

### Current Prompt Issues

1. **Policy Strategy**: Too narrow definition (only "race, religion, gender, sexual orientation") - FIXED in v1
2. **Persona Strategy**: Limited perspectives (only Middle Eastern + Conservative) - FIXED in v1  
3. **Combined Strategy**: Contradictory instructions and over-conservative logic - CRITICAL ISSUE in v1, ADDRESSED in v2
4. **All Strategies**: Extremely high false negative rates (62-78% missed hate speech) - PARTIALLY ADDRESSED

**Version 1 Results:** Individual strategies improved significantly, but Combined strategy became even more conservative.
**Version 2 Focus:** Simplify Combined strategy while maintaining fusion benefits.

## Improved Prompt Template Strategy

The improved template focuses on fixing the performance inversion by addressing each strategy's specific weaknesses while maintaining clear differentiation between approaches.

## Detailed Rationale for Each Change

### 1. Baseline Strategy - Made Simpler & More Direct

**Changes:**
- Removed complex instructions about "careful analysis"
- Simplified to basic mean/attacking judgment
- Reduced temperature from 0.3 to 0.1 for more consistent responses
- Shortened max_tokens to 256 to force conciseness

**Rationale:** Current baseline is paradoxically outperforming sophisticated strategies, suggesting simpler approaches are more effective. Made it even more basic to establish true baseline performance.

### 2. Policy Strategy - Expanded & More Aggressive

**Changes:**
- Expanded protected characteristics beyond just "race, religion, gender, sexual orientation"
- Added specific policy violations (dehumanizing language, harmful stereotypes, calls for exclusion)
- Explicit instruction to "err on the side of user safety"
- Structured enumeration of policy rules for clarity
- Slightly increased temperature to 0.15 for nuanced policy application

**Rationale:** Current policy is too narrow. Real hate speech policies cover broader characteristics and behaviors. More comprehensive coverage leads to better detection.

### 3. Persona Strategy - Multi-Perspective & Safety-First

**Changes:**
- Expanded from single Middle Eastern + Conservative to 4 diverse perspectives
- Added LGBTQ+, Hispanic/Latino, and intersectional viewpoints
- Clear safety instruction: classify as hate if ANY perspective finds it harmful
- Explicit vulnerability prioritization
- Kept low temperature (0.1) for consistent perspective application

**Rationale:** Single perspective was too limiting. Multiple perspectives catch hate speech that single viewpoints miss. Safety-first approach reduces false negatives.

### 4. Combined Strategy - Simplified Fusion with Aggressive Detection

**Changes:**
- Eliminated complex two-step process that caused decision paralysis
- Simplified to clear bullet-point checks with explicit YES/NO decision rule
- Made safety bias crystal clear: "When uncertain, choose 'hate' to protect communities"
- Streamlined OR logic: "Classify as 'hate' if you answer YES to ANY check above"
- Reduced temperature from 0.1 to 0.05 for more consistent responses
- Reduced max_tokens from 1024 to 768 to reduce cognitive load
- Direct harm-focused questions instead of abstract perspective analysis

**Rationale:** Performance analysis revealed the original "improved" combined strategy became too conservative, missing more hate speech (recall dropped from 0.242 to 0.152). The complex structure confused the model. This revision simplifies the logic while maintaining fusion of policy and perspective analysis, with explicit instructions to prioritize hate detection over avoiding false positives.

## Expected Performance Improvements

### Version 1 Results (Verified)
**Individual Strategies SUCCESS:**
- **Baseline**: 57.0% → 61.0% accuracy (+4.0%)
- **Policy**: 56.0% → 62.0% accuracy (+6.0%), false positives reduced 48%  
- **Persona**: 56.0% → 66.7% accuracy (+10.7%), precision improved 39%

**Combined Strategy FAILURE:**
- **Recall**: 24.2% → 15.2% (-37%), missing even more hate speech
- **Root Cause**: Over-complexity caused decision paralysis

### Version 2 Targets for Combined Strategy
- **Recall**: Target 30%+ (at least match original 24.2%)
- **True Positives**: Target 10+ (vs current 5)
- **False Negatives**: Target <25 (vs current 28)
- **Approach**: Simplify while maintaining fusion benefits

### Predicted Final Ranking (Best to Worst)
1. **Combined** (should now properly outperform all others)
2. **Persona** (multi-perspective safety-first - proven effective)
3. **Policy** (comprehensive but single-method - proven effective)
4. **Baseline** (simple but limited - proven effective)

### Key Improvements
- **Reduced False Negatives**: Safety-first approach across all strategies
- **Better Recall**: More aggressive hate detection reduces missed cases
- **Clearer Instructions**: Removes contradictory or vague guidance
- **Proper Strategy Differentiation**: Each strategy now has distinct advantages

### JSON Response Preservation
- All strategies maintain strict JSON format requirements
- Binary classification (hate/normal) preserved
- Response format parameter maintained for all strategies

This improved template should fix the performance inversion where Combined now leverages the best of both Policy and Persona approaches with clear safety prioritization.

## Implementation Notes

The improved_combined.json file contains two iterations of improvements:

**Version 1 (Initial Improvements):** Successfully enhanced individual strategies but over-complicated the Combined strategy, making it too conservative and missing more hate speech.

**Version 2 (Combined Strategy Fix):** Maintained successful individual strategy improvements while completely redesigning the Combined strategy to be simpler, more direct, and focused on hate detection rather than over-analysis.

### Key Lessons Learned

1. **Simplicity Often Outperforms Complexity:** The baseline strategy consistently performs well because it's direct and unambiguous.

2. **Complex Instructions Can Backfire:** The first "improved" Combined strategy became too conservative because complex multi-step instructions confused the model.

3. **Safety Bias Must Be Explicit:** Vague instructions like "err on the side of protecting communities" can be misinterpreted. Clear, actionable safety instructions work better.

4. **OR Logic Needs Crystal Clarity:** When combining multiple detection methods, the fusion logic must be impossible to misunderstand.

5. **Temperature Matters for Consistency:** Lower temperatures (0.05-0.1) provide more consistent responses for safety-critical decisions.

The final version strikes a balance between sophistication and clarity, with explicit safety prioritization to reduce the high false negative rates while maintaining the benefits of multi-faceted analysis.