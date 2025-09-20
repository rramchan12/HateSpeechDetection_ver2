# Aggressive V4 Prompt Template - Safety-First Approach

## Critical V3 Failure Analysis

### Performance Disaster in V3
**Combined Strategy:** 37.5% → 15.2% recall (-59% decline!)
**Policy Strategy:** 27.3% → 12.5% recall (-54% decline!)
**Persona Strategy:** 15.2% → 18.8% recall (minimal improvement)

### Root Cause of V3 Failures
1. **Over-Optimization for Precision**: Added "balance" and "avoid over-interpretation" guidance
2. **Conservative Temperature Changes**: Made responses too rigid and conservative
3. **False Priority**: Focused on reducing false positives instead of protecting users
4. **Safety Compromise**: All strategies became dangerously conservative for user safety

## V4 Aggressive Safety-First Design

### Core Philosophy: User Protection Over False Positive Avoidance
**Key Principle**: Missing hate speech harms real people. False positives are preferable to missing actual hate speech that targets vulnerable communities.

### Strategy-Specific Improvements

#### 1. Baseline Strategy - Maintained Stability
- **No Changes**: Kept successful simple structure as performance anchor
- **Rationale**: Baseline consistently performs well across all versions

#### 2. Policy Strategy - Aggressive Detection Focus
**Critical Changes:**
- **Removed "Decisive" Language**: Replaced with explicit safety priority
- **Increased Temperature**: 0.12 → 0.20 for more flexible responses
- **Enhanced Policy Scope**: Added coded language and stereotype detection
- **Safety Instruction**: "Protecting users from hate speech is more important than avoiding false positives"

**Expected Impact**: Restore recall from disastrous 12.5% back to 25%+ range

#### 3. Persona Strategy - Maximum Community Protection
**Critical Changes:**
- **Warning Symbols**: Added 🚨 to emphasize harm detection priority
- **Increased Temperature**: 0.08 → 0.15 for better sensitivity
- **Explicit Safety Rule**: "Classify as 'hate' if ANY community perspective identifies potential harm"
- **Enhanced Sensitivity**: Added guidance for "subtle discrimination, coded language, and implied hostility"

**Expected Impact**: Significant improvement from poor 18.8% recall to 30%+ range

#### 4. Combined Strategy - Return to V2 Success Formula
**Critical Changes:**
- **Removed All "Balance" Language**: Eliminated the guidance that caused V3 failure
- **Back to 0.05 Temperature**: Returned to V2 setting that achieved 37.5% recall
- **Maximum Sensitivity Focus**: "Analyze with maximum sensitivity"
- **Clear Safety Priority**: "When uncertain, choose 'hate' to protect vulnerable communities"
- **Comprehensive Checklist**: 6 specific indicators to check

**Expected Impact**: Restore to V2 performance level (35%+ recall) or better

### Technical Optimizations

#### Temperature Strategy
- **Policy**: 0.12 → 0.20 (more flexible for nuanced policy application)
- **Persona**: 0.08 → 0.15 (better sensitivity to community harm)
- **Combined**: Back to 0.05 (proven successful in V2)
- **Baseline**: 0.1 unchanged (stable anchor)

#### Instruction Design
- **Eliminated**: All "balance," "decisive," and "avoid over-interpretation" language
- **Emphasized**: User safety, community protection, harm prevention
- **Added**: Explicit safety-first decision rules
- **Enhanced**: Sensitivity to coded language and subtle discrimination

#### Token Allocation
- **Combined**: Maintained 512 tokens (efficient but comprehensive)
- **Policy/Persona**: Maintained 768 tokens (complex analysis required)
- **Baseline**: Maintained 256 tokens (simple and effective)

## Expected V4 Performance Targets

### Primary Success Metrics
1. **Combined Strategy**: 35%+ recall (restore V2 success)
2. **Policy Strategy**: 25%+ recall (recover from 12.5% disaster)
3. **Persona Strategy**: 30%+ recall (significant improvement from 18.8%)
4. **Overall Safety**: Prioritize catching hate speech over precision metrics

### Ranking Prediction
1. **Combined** (aggressive fusion with proven V2 base)
2. **Persona** (enhanced community protection focus)
3. **Policy** (comprehensive aggressive detection)
4. **Baseline** (stable reliable anchor)

### Critical Success Indicators
- **No strategy below 20% recall** (user safety threshold)
- **Combined strategy outperforms individual strategies** (fusion benefit)
- **Significant improvement from V3 disaster** (recovery validation)
- **Balanced approach to user protection** (safety without excessive false positives)

## Safety-First Design Principles

### 1. User Harm Prevention Priority
Missing hate speech that targets vulnerable communities causes real psychological and social harm. This outweighs the inconvenience of reviewing additional content flagged as potentially hateful.

### 2. Vulnerable Community Protection
LGBTQ+, racial/ethnic minorities, religious groups, and other marginalized communities face disproportionate online harassment. Detection systems must be sensitive to their experiences and perspectives.

### 3. Coded Language Recognition
Modern hate speech often uses subtle, coded language that may not be obviously hateful but still causes harm. Detection systems must be trained to recognize these patterns.

### 4. Multiple Perspective Validation
What seems acceptable from one viewpoint may be harmful from another. Comprehensive detection requires considering multiple community perspectives.

This V4 template abandons the failed "balance" approach of V3 and returns to aggressive, safety-first hate speech detection designed to protect vulnerable users.