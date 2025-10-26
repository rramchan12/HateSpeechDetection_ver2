# LoRA Training Configurations

This directory contains configuration files for LoRA fine-tuning with different parameter settings.

> **Note**: These are pure JSON files (no comments allowed in JSON). For parameter explanations and theoretical justification, see [`lora_ft_approach.md`](../../baseline/templates/lora_ft_approach.md).

## Available Configurations

### `default.json` (Recommended)
**Standard training configuration** based on [`lora_ft_approach.md`](../../baseline/templates/lora_ft_approach.md) recommendations.

- **LoRA Rank**: 32 (balanced capacity)
- **Epochs**: 3 (sufficient for convergence)
- **Learning Rate**: 2e-4 (LoRA-optimized)
- **Batch Size**: 4 per GPU, effective 64 with accumulation
- **Use Case**: Production training, baseline experiments

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json
```

### `high_capacity.json`
**Higher rank variant** for tasks requiring more adapter capacity.

- **LoRA Rank**: 64 (2× capacity, 2× parameters)
- **Changes**: Doubled rank and alpha (r=64, α=64)
- **Use Case**: Complex classification tasks, when default config underperforms

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/high_capacity.json
```

**When to use**: 
- Validation loss plateaus early with default config
- Task requires nuanced distinction learning
- You have sufficient VRAM (should still fit on 4× A100 80GB)

### `memory_efficient.json`
**Memory-constrained variant** for limited VRAM scenarios.

- **Batch Size**: 2 per GPU (vs. 4 in default)
- **Gradient Accumulation**: 8 steps (vs. 4 in default)
- **Sequence Length**: 256 tokens (vs. 512 in default)
- **Use Case**: Training on GPUs with <80GB VRAM, OOM errors

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/memory_efficient.json
```

**Trade-offs**:
- Effective batch size maintained (2 × 4 GPUs × 8 = 64)
- Training time increased ~30% (more accumulation steps)
- Shorter sequences may truncate longer examples

### `quick_test.json`
**Fast testing variant** for development and debugging.

- **Epochs**: 1 (vs. 3 in default)
- **LoRA Rank**: 16 (vs. 32 in default)
- **Warmup**: 50 steps (vs. 100 in default)
- **Use Case**: Code testing, pipeline validation, quick experiments

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/quick_test.json
```

**Expected duration**: ~30-40 minutes (vs. 2-3 hours for full training)

## Configuration Structure

All configs follow the same section-based structure:

1. **Model Configuration** - Model loading and 4-bit quantization (QLoRA)
2. **Data Configuration** - Training/validation paths, sequence length
3. **LoRA Configuration** - Rank, alpha, dropout, target modules
4. **Training Hyperparameters** - LR, batch size, epochs, warmup
5. **Optimization Configuration** - Precision, gradient checkpointing, optimizer
6. **Logging and Checkpointing** - Output directory, evaluation strategy

## Creating Custom Configurations

To create a custom configuration:

```bash
# Copy a base config
cp finetuning/pipeline/lora/configs/default.json \
   finetuning/pipeline/lora/configs/my_experiment.json

# Edit the parameters
nano finetuning/pipeline/lora/configs/my_experiment.json

# Run with your config
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/my_experiment.json
```

## Parameter Justification

For **detailed theoretical justification** of all hyperparameters, see:
- **Primary Reference**: [`lora_ft_approach.md`](../../baseline/templates/lora_ft_approach.md)
- **Section**: "Hyperparameter Specifications" (#1-#15)

Each parameter value is backed by:
- Academic research (LoRA, QLoRA, AdamW papers)
- Empirical validation (baseline F1=0.615 results)
- Best practices from literature

## Quick Parameter Reference

| Parameter | Default | High Cap | Mem Eff | Test | Justification Link |
|-----------|---------|----------|---------|------|-------------------|
| `lora_r` | 32 | **64** | 32 | **16** | [Link](../../baseline/templates/lora_ft_approach.md#9-lora-rank-r-32) |
| `num_train_epochs` | 3 | 3 | 3 | **1** | [Link](../../baseline/templates/lora_ft_approach.md#2-number-of-epochs-3) |
| `per_device_train_batch_size` | 4 | 4 | **2** | 4 | [Link](../../baseline/templates/lora_ft_approach.md#3-batch-size-4-per-gpu-effective-16-with-4-gpus) |
| `gradient_accumulation_steps` | 4 | 4 | **8** | 4 | [Link](../../baseline/templates/lora_ft_approach.md#4-gradient-accumulation-steps-4) |
| `max_seq_length` | 512 | 512 | **256** | 512 | [Link](../../baseline/templates/lora_ft_approach.md#15-maximum-sequence-length-512) |
| `learning_rate` | 2e-4 | 2e-4 | 2e-4 | 2e-4 | [Link](../../baseline/templates/lora_ft_approach.md#1-learning-rate-2e-4) |

## Configuration Selection Guide

Choose configuration based on your scenario:

| Scenario | Config | Rationale |
|----------|--------|-----------|
| **Standard training** | `default.json` | Research-backed baseline, proven effective |
| **First production run** | `default.json` | Start with validated parameters |
| **Model underfitting** | `high_capacity.json` | Increase adapter capacity (rank 64) |
| **OOM errors** | `memory_efficient.json` | Reduce memory footprint |
| **Testing pipeline** | `quick_test.json` | Fast iteration (30-40 min) |
| **Long sequences** | `default.json` | Supports up to 512 tokens |
| **Short sequences only** | `memory_efficient.json` | 256 tokens sufficient, saves memory |

## Notes

- All configurations use **4-bit quantization (QLoRA)** for memory efficiency
- All configurations target **effective batch size = 64** (optimal per literature)
- All configurations use **AdamW optimizer** with **cosine annealing**
- All configurations enable **bf16 precision** and **gradient checkpointing**

## Support

For questions about configuration parameters:
1. Check [`lora_ft_approach.md`](../../baseline/templates/lora_ft_approach.md) for detailed explanations
2. Review [`VALIDATION_GUIDE.md`](../../../VALIDATION_GUIDE.md) Phase 5
3. See [`README.md`](../README.md) for usage examples
