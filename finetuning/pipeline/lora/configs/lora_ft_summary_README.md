# LoRA Fine-Tuning Experiment Summary

This document tracks the results of different LoRA fine-tuning configurations and their performance.

---

## Phase 1: Train with Default Configuration - COMPLETED

**Date**: October 26, 2025  
**Config**: `default.json`  
**Output Directory**: `finetuning/models/lora_default`  
**Baseline Results**: `finetuning/outputs/gptoss/post_finetune/lora_default`

### Configuration Details

```json
{
  "model_name_or_path": "openai/gpt-oss-20b",
  "lora_r": 32,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "lora_target_modules": ["q_proj", "v_proj"],
  "learning_rate": 2e-4,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "lr_scheduler_type": "cosine",
  "early_stopping_patience": 2,
  "early_stopping_threshold": 0.01
}
```

### Training Statistics

- **Training samples**: 2,686
- **Validation samples**: 55
- **Effective batch size**: 64 (4 per device × 4 GPUs × 4 grad accumulation)
- **Steps per epoch**: 42
- **Total training steps**: 126 (3 epochs)
- **Trainable parameters**: 7,962,624 (0.038% of 20.9B total)

### Training Results

| Metric | Value |
|--------|-------|
| **Final Epoch** | 3.0 |
| **Final Eval Loss** | 0.5280 |
| **Final Train Loss** | 1.5196 |
| **Training Runtime** | 825.39 seconds (~13.8 minutes) |
| **Samples/Second** | 9.763 |

### Training Progression

| Checkpoint | Epoch | Step | Eval Loss | Improvement from Previous |
|------------|-------|------|-----------|---------------------------|
| checkpoint-42 | 1.0 | 42 | 1.3383 | Baseline |
| checkpoint-84 | 2.0 | 84 | 0.5454 | Down 59.3% (Major improvement) |
| checkpoint-126 (BEST) | 3.0 | 126 | 0.5280 | Down 3.1% (Marginal improvement) |

**Best Model**: checkpoint-126 (automatically loaded to root directory)

### Validation Results

**Performance Metrics**:

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5152 (51.52%) |
| **Precision** | 0.4881 (48.81%) |
| **Recall** | 0.8913 (89.13%) |
| **F1 Score** | 0.6308 |
| **True Positive** | 41 |
| **True Negative** | 10 |
| **False Positive** | 43 |
| **False Negative** | 5 |

**Bias Metrics**:

| Target Group | FPR | FNR | Samples |
|--------------|-----|-----|---------|
| **LGBTQ+** | 93.33% | 15.79% | 49 |
| **Mexican** | 50.0% | 9.09% | 22 |
| **Middle East** | 76.92% | 6.25% | 29 |

**Analysis**: High false positive rates across all groups, especially LGBTQ+ (93.33%), indicating the model is over-predicting hate speech for these target groups.

---

## Phase 2: Train with High Capacity Configuration - COMPLETED

**Date**: October 26, 2025  
**Config**: `high_capacity.json`  
**Output Directory**: `finetuning/models/high_capacity`  
**Baseline Results**: `finetuning/outputs/gptoss/post_finetune/high_capacity`

### Configuration Details

```json
{
  "model_name_or_path": "openai/gpt-oss-20b",
  "lora_r": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "lora_target_modules": ["q_proj", "v_proj"],
  "learning_rate": 2e-4,
  "num_train_epochs": 5,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "lr_scheduler_type": "cosine",
  "early_stopping_patience": 2,
  "early_stopping_threshold": 0.01
}
```

### Training Statistics

- **Training samples**: 2,686
- **Validation samples**: 55
- **Effective batch size**: 64 (4 per device × 4 GPUs × 4 grad accumulation)
- **Steps per epoch**: 42
- **Total training steps**: 210 (5 epochs)
- **Trainable parameters**: 15,925,248 (0.076% of 20.9B total)

### Training Results

| Metric | Value |
|--------|-------|
| **Final Epoch** | 5.0 |
| **Final Eval Loss** | 0.5088 |
| **Final Train Loss** | 0.9915 |
| **Training Runtime** | 1,388.71 seconds (~23.1 minutes) |
| **Samples/Second** | 9.671 |

### Training Progression

| Checkpoint | Epoch | Step | Eval Loss | Improvement from Previous |
|------------|-------|------|-----------|---------------------------|
| checkpoint-42 | 1.0 | 42 | 0.7652 | Baseline |
| checkpoint-84 | 2.0 | 84 | 0.5379 | Down 29.7% (Major improvement) |
| checkpoint-126 | 3.0 | 126 | 0.5150 | Down 4.3% |
| checkpoint-168 | 4.0 | 168 | 0.5108 | Down 0.8% (Minimal) |
| checkpoint-210 (BEST) | 5.0 | 210 | 0.5088 | Down 0.4% (Marginal) |

**Best Model**: checkpoint-210 (automatically loaded to root directory)

### Validation Results

**Performance Metrics**:

| Metric | Value | Change from Phase 1 |
|--------|-------|---------------------|
| **Accuracy** | 0.6531 (65.31%) | +27.2% |
| **Precision** | 0.5968 (59.68%) | +22.2% |
| **Recall** | 0.8043 (80.43%) | -9.7% |
| **F1 Score** | 0.6852 | +8.6% |
| **True Positive** | 37 | -4 |
| **True Negative** | 27 | +17 |
| **False Positive** | 25 | -18 |
| **False Negative** | 9 | +4 |

**Bias Metrics**:

| Target Group | FPR | FNR | Samples | Change from Phase 1 (FPR) |
|--------------|-----|-----|---------|---------------------------|
| **LGBTQ+** | 62.07% | 21.05% | 49 | -33.5% (Major improvement) |
| **Mexican** | 30.0% | 25.0% | 22 | -40.0% (Major improvement) |
| **Middle East** | 30.77% | 13.33% | 29 | -60.0% (Major improvement) |

**Analysis**: 
- Dramatic FPR improvements across all target groups
- LGBTQ+ FPR: 93.33% → 62.07% (still high but much better)
- Model is significantly less biased but still over-predicts for LGBTQ+ group
- Trade-off: Higher precision, lower recall (model more conservative)

---

## Phase 3: K-Projection with Full Capacity - COMPLETED

**Date**: October 26, 2025  
**Config**: `k_proj_full_capacity.json`  
**Output Directory**: `finetuning/models/k_proj_full_capacity`  
**Baseline Results**: `finetuning/outputs/gptoss/post_finetune/k_proj_full_capacity`

### Hypothesis
Adding the key projection (k_proj) to target modules with full capacity (r=64) should improve attention mechanism modeling and potentially achieve F1 ≥ 0.70.

### Configuration Details

```json
{
  "model_name_or_path": "openai/gpt-oss-20b",
  "lora_r": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "lora_target_modules": ["q_proj", "v_proj", "k_proj"],
  "learning_rate": 2e-4,
  "num_train_epochs": 4,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "lr_scheduler_type": "cosine",
  "early_stopping_patience": 2,
  "early_stopping_threshold": 0.01
}
```

### Training Statistics

- **Training samples**: 2,686
- **Validation samples**: 55
- **Effective batch size**: 64 (4 per device × 4 GPUs × 4 grad accumulation)
- **Steps per epoch**: 42
- **Total training steps**: 168 (4 epochs)
- **Trainable parameters**: ~23,887,872 (0.114% of 20.9B total)

### Training Results

| Metric | Value |
|--------|-------|
| **Final Epoch** | 4.0 |
| **Final Eval Loss** | 0.5098 |
| **Final Train Loss** | 1.0768 |
| **Training Runtime** | 1,101.20 seconds (~18.4 minutes) |
| **Samples/Second** | 9.757 |

### Training Progression

| Checkpoint | Epoch | Step | Eval Loss | Improvement from Previous |
|------------|-------|------|-----------|---------------------------|
| checkpoint-42 | 1.0 | 42 | 0.6882 | Baseline |
| checkpoint-84 | 2.0 | 84 | 0.5337 | Down 22.4% (Major improvement) |
| checkpoint-126 | 3.0 | 126 | 0.5138 | Down 3.7% |
| checkpoint-168 (BEST) | 4.0 | 168 | 0.5098 | Down 0.8% (Minimal) |

**Best Model**: checkpoint-168 (automatically loaded to root directory)

### Validation Results

**Performance Metrics**:

| Metric | Value | Change from Phase 2 |
|--------|-------|---------------------|
| **Accuracy** | 0.5918 (59.18%) | -9.4% |
| **Precision** | 0.5522 (55.22%) | -7.5% |
| **Recall** | 0.7872 (78.72%) | -2.1% |
| **F1 Score** | 0.6491 | -5.3% |
| **True Positive** | 37 | 0 |
| **True Negative** | 21 | -6 |
| **False Positive** | 30 | +5 |
| **False Negative** | 10 | +1 |

**Bias Metrics**:

| Target Group | FPR | FNR | Samples | Change from Phase 2 (FPR) |
|--------------|-----|-----|---------|---------------------------|
| **LGBTQ+** | 70.0% | 21.05% | 49 | +12.8% (Worse) |
| **Mexican** | 22.22% | 25.0% | 22 | -25.9% (Better) |
| **Middle East** | 58.33% | 18.75% | 29 | +89.6% (Worse) |

### Phase 3 Analysis

**What Worked**:
1. **Training Efficiency**: 18.4 minutes (20% faster than Phase 2's 23.1 min)
2. **Eval Loss Competitive**: 0.5098 vs 0.5088 (Phase 2) - only 0.2% worse
3. **Mexican FPR Improved**: 30% → 22.22% (-26% improvement)

**What Didn't Work**:
1. **F1 Score Dropped**: 0.6852 → 0.6491 (-5.3%)
2. **Accuracy Decreased**: 65.31% → 59.18% (-9.4%)
3. **LGBTQ+ FPR Increased**: 62.07% → 70% (+12.8%)
4. **Middle East FPR Increased**: 30.77% → 58.33% (+89.6%)
5. **False Positives Increased**: 25 → 30 (+20%)

**Key Findings**:
- Adding k_proj with full capacity (r=64) **did NOT improve** overall performance
- Despite having 50% more trainable parameters (23.9M vs 15.9M), F1 decreased
- Training was faster but model performance regressed
- k_proj addition may introduce complexity that doesn't benefit this specific task
- The query (q_proj) and value (v_proj) projections appear sufficient for hate speech detection

**Conclusion**: 
The experiment demonstrates that **more parameters ≠ better performance**. Phase 2 (r=64, 2 modules) remains the best configuration with F1=0.6852.

---

## Comparison Summary

### Training Efficiency

| Phase | Config | Rank | Modules | Trainable Params | Runtime | Eval Loss | F1 Score | Status |
|-------|--------|------|---------|------------------|---------|-----------|----------|--------|
| Phase 1 | default | 32 | q_proj, v_proj | 7.96M (0.038%) | 13.8 min | 0.5280 | 0.6308 | Complete |
| Phase 2 | high_capacity | 64 | q_proj, v_proj | 15.9M (0.076%) | 23.1 min | 0.5088 | **0.6852** | Complete |
| Phase 3 | k_proj_full | 64 | q_proj, v_proj, k_proj | 23.9M (0.114%) | 18.4 min | 0.5098 | 0.6491 | Complete |

### Performance Metrics Comparison

| Phase | F1 Score | Precision | Recall | Accuracy | LGBTQ FPR | Mexican FPR | Middle East FPR |
|-------|----------|-----------|--------|----------|-----------|-------------|-----------------|
| **Phase 1 (r=32, 2 modules)** | 0.6308 | 0.4881 | 0.8913 | 0.5152 | 93.33% | 50.0% | 76.92% |
| **Phase 2 (r=64, 2 modules)** | **0.6852** | **0.5968** | 0.8043 | **0.6531** | **62.07%** | **30.0%** | **30.77%** |
| **Phase 3 (r=64, 3 modules)** | 0.6491 | 0.5522 | **0.7872** | 0.5918 | 70.0% | 22.22% | 58.33% |

### Key Insights

1. **Best Overall Model**: Phase 2 (high_capacity)
   - Highest F1 score: 0.6852
   - Best precision: 59.68%
   - Best accuracy: 65.31%
   - Lowest LGBTQ+ FPR: 62.07%
   - Lowest Middle East FPR: 30.77%

2. **Phase 3 Learnings**:
   - Adding k_proj doesn't improve hate speech detection
   - Query and Value projections are sufficient
   - More parameters can hurt if they don't address task requirements
   - Training efficiency improved but model quality decreased

3. **Bias Patterns**:
   - All models struggle most with LGBTQ+ false positives (62-93% FPR)
   - Phase 2 achieved best balance across all demographic groups
   - Mexican FPR improved in Phase 3 but other groups worsened

4. **Target Achievement**:
   - **F1 ≥ 0.615 target**: ACHIEVED in Phase 2 (0.6852 = +11.4% above target)
   - **F1 ≥ 0.70 target**: NOT ACHIEVED (Phase 2: 0.6852, Phase 3: 0.6491)

---

## Recommendations

### Immediate Action
**Use Phase 2 (high_capacity) model** for production/further experiments:
- Best F1 score: 0.6852
- Best overall bias metrics
- Proven stable and reliable

### Future Exploration Options

#### Option A: Threshold Tuning (No Retraining Required)
**Purpose**: Optimize precision/recall balance on Phase 2 model  
**Method**: Adjust classification threshold to reduce false positives  
**Expected**: F1 improvement of 1-2%, LGBTQ+ FPR reduction to ~55%  
**Time**: < 1 hour (inference only)

#### Option B: Data Augmentation for LGBTQ+ Bias
**Purpose**: Address persistent 62% LGBTQ+ FPR  
**Method**: Add balanced non-hate LGBTQ+ examples to training data  
**Expected**: Reduce LGBTQ+ FPR to < 40% while maintaining F1  
**Time**: Data collection + retraining (~2-3 days)

#### Option C: Alternative Architecture Exploration
**Purpose**: Test if other projection modules help  
**Method**: Try o_proj (output projection) instead of k_proj  
**Expected**: Uncertain - exploratory experiment  
**Time**: ~20 minutes training + validation

#### Option D: Fine-tune Phase 2 with Lower Learning Rate
**Purpose**: Squeeze out marginal F1 improvements  
**Method**: Continue training Phase 2 with LR=1e-4 for 2 more epochs  
**Expected**: F1 improvement of 0.5-1% (0.6852 → 0.69+)  
**Time**: ~10 minutes

---

## Experimental Conclusions

### What We Learned

1. **LoRA Rank Matters**: r=64 significantly better than r=32 (+8.6% F1)
2. **Module Selection Critical**: q_proj + v_proj sufficient; adding k_proj hurts performance
3. **Diminishing Returns**: Phase 1→2 gave +8.6% F1, Phase 2→3 gave -5.3% F1
4. **Bias is Challenging**: All models over-predict hate speech for LGBTQ+ content
5. **Training Efficiency**: More parameters don't always mean longer training

### Best Practices Identified

- **Start simple**: 2 target modules (q_proj, v_proj) work well
- **Scale capacity first**: Increase rank before adding modules
- **Monitor bias metrics**: F1 alone doesn't tell the full story
- **Use early stopping**: Prevents wasted compute on marginal improvements
- **Validate assumptions**: Phase 3 proved k_proj hypothesis wrong

### Open Questions

1. Would o_proj or gate_proj work better than k_proj?
2. Can threshold tuning push F1 above 0.70 without retraining?
3. Is 62% LGBTQ+ FPR the best achievable with current data?
4. Would a smaller model (7B) with full fine-tuning outperform LoRA on 20B?

---

## Notes

- All experiments use the same training/validation split (2,686 train / 55 val samples)
- Base model: `openai/gpt-oss-20b` (pre-quantized Mxfp4, dequantized to bf16)
- Hardware: 4× A100 80GB GPUs with DDP training
- Early stopping enabled: patience=2 epochs, threshold=1% improvement
- TensorBoard logs available in respective `{output_dir}/runs/` directories
- Validation performed on 100-sample test set (canned_100_size_varied)
- All configurations use gradient checkpointing and bf16 precision

**Last Updated**: October 26, 2025
