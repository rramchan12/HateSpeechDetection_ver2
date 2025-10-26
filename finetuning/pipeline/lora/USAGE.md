# LoRA Fine-Tuning Command-Line Reference

Complete command-line reference for LoRA fine-tuning implementation.

> **📖 For detailed theoretical justification of all hyperparameters**, see [`lora_ft_approach.md`](configs/lora_ft_approach.md) - Section "Hyperparameter Specifications".

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Methods](#training-methods)
3. [Configuration Files](#configuration-files)
4. [Command-Line Arguments](#command-line-arguments)
5. [Example Workflows](#example-workflows)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Environment Variables](#environment-variables)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Easiest Method (Recommended)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run automated quickstart script
bash finetuning/pipeline/lora/quickstart.sh
```

The script will:
- ✓ Check prerequisites (venv, accelerate, data)
- ✓ Auto-detect number of GPUs
- ✓ Launch training with optimal settings
- ✓ Show next steps after completion

### Standard Method

```bash
# Using default configuration
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json
```

---

## Training Methods

### Method 1: Using Config File (Recommended)

**Advantages**: 
- Organized parameter management
- Easy to version control
- Reusable configurations
- Clear parameter grouping

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json
```

**Available configs**:
- `default.json` - Standard training (r=32, 3 epochs, lr=2e-4)
- `high_capacity.json` - Higher capacity (r=64)
- `memory_efficient.json` - Reduced memory (bs=2, seq=256)
- `quick_test.json` - Fast testing (r=16, 1 epoch)

See [`configs/README.md`](configs/README.md) for detailed comparison.

### Method 2: Config File + Command-Line Overrides

**Advantages**:
- Use base config as template
- Override specific parameters for experiments
- Maintain config file for documentation

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --learning_rate 3e-4 \
    --num_train_epochs 5 \
    --lora_r 64 \
    --output_dir ./finetuning/models/lora_experiment
```

### Method 3: Full Command-Line (No Config File)

**Advantages**:
- Complete control via command line
- Useful for automated scripts
- No config file needed

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --model_name_or_path openai/gpt-oss-20b \
    --train_file ./finetuning/data/ft_prompts/train.jsonl \
    --validation_file ./finetuning/data/ft_prompts/validation.jsonl \
    --output_dir ./finetuning/models/lora_checkpoints \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_r 32 \
    --lora_alpha 32 \
    --load_in_4bit \
    --bf16
```

---

## Configuration Files

### Using Different Configurations

```bash
# Standard training (recommended for production)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# High capacity for complex tasks
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/high_capacity.json

# Memory efficient for limited VRAM
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/memory_efficient.json

# Quick test for development
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/quick_test.json
```

### Creating Custom Configuration

```bash
# Copy a base config
cp finetuning/pipeline/lora/configs/default.json \
   finetuning/pipeline/lora/configs/my_experiment.json

# Edit parameters
nano finetuning/pipeline/lora/configs/my_experiment.json

# Run with custom config
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/my_experiment.json
```

---

## Command-Line Arguments

### Complete Argument Reference

For **theoretical justification** of each parameter value, see [`lora_ft_approach.md`](configs/lora_ft_approach.md).

#### Model Arguments

| Argument | Type | Default | Description | Reference |
|----------|------|---------|-------------|-----------|
| `--model_name_or_path` | str | openai/gpt-oss-20b | HuggingFace model ID or path | - |
| `--cache_dir` | str | ./data/models | Directory to cache models | - |
| `--load_in_4bit` | flag | true | Enable 4-bit quantization (QLoRA) | [Link](configs/lora_ft_approach.md#14-quantization-4-bit-nf4) |
| `--bnb_4bit_quant_type` | str | nf4 | Quantization type (nf4/fp4) | [Link](configs/lora_ft_approach.md#14-quantization-4-bit-nf4) |
| `--bnb_4bit_compute_dtype` | str | bfloat16 | Compute dtype (bfloat16/float16/float32) | [Link](configs/lora_ft_approach.md#14-quantization-4-bit-nf4) |

#### Data Arguments

| Argument | Type | Default | Description | Reference |
|----------|------|---------|-------------|-----------|
| `--train_file` | str | finetuning/data/ft_prompts/train.jsonl | Training data path (JSONL) | [Link](configs/lora_ft_approach.md#training-data-configuration) |
| `--validation_file` | str | finetuning/data/ft_prompts/validation.jsonl | Validation data path (JSONL) | [Link](configs/lora_ft_approach.md#training-data-configuration) |
| `--max_seq_length` | int | 512 | Maximum sequence length (tokens) | [Link](configs/lora_ft_approach.md#15-maximum-sequence-length-512) |

#### LoRA Arguments

| Argument | Type | Default | Description | Reference |
|----------|------|---------|-------------|-----------|
| `--lora_r` | int | 32 | LoRA rank (8-64) | [Link](configs/lora_ft_approach.md#9-lora-rank-r-32) |
| `--lora_alpha` | int | 32 | LoRA alpha scaling (typically = r) | [Link](configs/lora_ft_approach.md#10-lora-alpha-32) |
| `--lora_dropout` | float | 0.05 | LoRA dropout (0.0-0.1) | [Link](configs/lora_ft_approach.md#11-lora-dropout-005) |
| `--lora_target_modules` | str | q_proj,v_proj | Target modules (comma-separated) | [Link](configs/lora_ft_approach.md#12-target-modules-q_proj-v_proj) |
| `--lora_bias` | str | none | Bias strategy (none/all/lora_only) | [Link](configs/lora_ft_approach.md#13-lora-bias-none) |
| `--early_stopping_patience` | int | 2 | Stop if no improvement for N epochs (0=disabled) | - |
| `--early_stopping_threshold` | float | 0.01 | Minimum improvement required (e.g., 0.01 = 1%) | - |

#### Training Hyperparameters

| Argument | Type | Default | Description | Reference |
|----------|------|---------|-------------|-----------|
| `--learning_rate` | float | 2e-4 | Peak learning rate | [Link](configs/lora_ft_approach.md#1-learning-rate-2e-4) |
| `--num_train_epochs` | int | 3 | Number of training epochs | [Link](configs/lora_ft_approach.md#2-number-of-epochs-3) |
| `--per_device_train_batch_size` | int | 4 | Batch size per GPU | [Link](configs/lora_ft_approach.md#3-batch-size-4-per-gpu-effective-16-with-4-gpus) |
| `--per_device_eval_batch_size` | int | 4 | Eval batch size per GPU | - |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps | [Link](configs/lora_ft_approach.md#4-gradient-accumulation-steps-4) |
| `--warmup_steps` | int | 100 | Learning rate warmup steps | [Link](configs/lora_ft_approach.md#5-warmup-steps-100) |
| `--weight_decay` | float | 0.01 | Weight decay (L2 regularization) | [Link](configs/lora_ft_approach.md#6-weight-decay-001) |
| `--max_grad_norm` | float | 1.0 | Gradient clipping threshold | [Link](configs/lora_ft_approach.md#7-max-gradient-norm-10) |
| `--lr_scheduler_type` | str | cosine | LR scheduler (cosine/linear/constant) | [Link](configs/lora_ft_approach.md#8-learning-rate-scheduler-cosine) |

#### Optimization Configuration

| Argument | Type | Default | Description | Reference |
|----------|------|---------|-------------|-----------|
| `--bf16` | flag | true | Use bfloat16 precision | - |
| `--gradient_checkpointing` | flag | true | Enable gradient checkpointing | - |
| `--optim` | str | adamw_torch | Optimizer (adamw_torch/adamw_hf) | [Link](configs/lora_ft_approach.md#3-adamw-optimizer-and-weight-decay) |

#### Logging and Checkpointing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Output directory for checkpoints |
| `--logging_dir` | str | {output_dir}/logs | TensorBoard logs directory |
| `--logging_steps` | int | 10 | Log every N steps |
| `--eval_strategy` | str | epoch | When to evaluate (epoch/steps/no) |
| `--save_strategy` | str | epoch | When to save checkpoints (epoch/steps/no) |
| `--save_total_limit` | int | 3 | Max checkpoints to keep |
| `--load_best_model_at_end` | bool | true | Load best checkpoint at end of training |
| `--metric_for_best_model` | str | eval_loss | Metric to determine best model |
| `--greater_is_better` | bool | false | Whether higher metric is better |
| `--report_to` | str | tensorboard | Logging integration (tensorboard/none) |

#### Data Loading Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataloader_num_workers` | int | 4 | Number of data loading workers |
| `--remove_unused_columns` | bool | false | Remove unused dataset columns |

---

## Example Workflows

### 1. Standard Production Training

**Use case**: Default training with validated hyperparameters

```bash
# Standard 3-epoch training with rank 32
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# Monitor progress
tail -f finetuning/models/lora_checkpoints/training.log
```

**Expected**: 2-3 hours on 4× A100 80GB GPUs

### 2. High Capacity Experiment

**Use case**: Task requires more adapter capacity

```bash
# Rank 64 for increased capacity
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/high_capacity.json

# Or override default config
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --lora_r 64 \
    --lora_alpha 64 \
    --output_dir ./finetuning/models/lora_r64
```

### 3. Memory-Constrained Training

**Use case**: Limited VRAM or OOM errors

```bash
# Reduced batch size and sequence length
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/memory_efficient.json
```

**Changes from default**:
- Batch size: 4 → 2
- Gradient accumulation: 4 → 8 (maintains effective batch=64)
- Sequence length: 512 → 256

### 4. Quick Development Test

**Use case**: Testing pipeline, debugging code

```bash
# Single epoch, small rank for fast iteration
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/quick_test.json
```

**Expected**: 30-40 minutes (vs. 2-3 hours for full training)

### 5. Custom Learning Rate Experiment

**Use case**: Exploring different learning rates

```bash
# Try higher learning rate (3e-4)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --learning_rate 3e-4 \
    --output_dir ./finetuning/models/lora_lr3e4

# Try lower learning rate (1e-4)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --learning_rate 1e-4 \
    --output_dir ./finetuning/models/lora_lr1e4
```

### 6. Extended Training

**Use case**: Model not fully converged after 3 epochs

```bash
# 5 epochs for additional convergence
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --num_train_epochs 5 \
    --output_dir ./finetuning/models/lora_5epochs

# Or disable early stopping to force full training
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --early_stopping_patience 0 \
    --num_train_epochs 5 \
    --output_dir ./finetuning/models/lora_5epochs_no_early_stop
```

**Note**: With early stopping enabled (default), training may stop before reaching `num_train_epochs` if the model converges.

### 7. Post-Training Validation

**Use case**: Evaluate fine-tuned model on test set

```bash
# Validate with LoRA adapters
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --prompt_template ./prompt_engineering/prompt_templates/baseline_v1.json \
    --strategy baseline_standard \
    --max_samples 100 \
    --output_dir ./finetuning/outputs/gptoss/post_finetune

# Compare with baseline
cd finetuning/outputs/gptoss
BASELINE_F1=$(awk -F',' 'NR==2 {print $5}' baseline/run_*/performance_metrics_*.csv | head -1)
POSTFT_F1=$(awk -F',' 'NR==2 {print $5}' post_finetune/run_*/performance_metrics_*.csv | head -1)
echo "Baseline F1: $BASELINE_F1"
echo "Post-FT F1:  $POSTFT_F1"
```

**Success criterion**: Post-FT F1 ≥ 0.615 (baseline F1 with sophisticated prompts)

### 8. Merge and Deploy

**Use case**: Create standalone model for deployment

```bash
# Merge LoRA adapters with base model
python -m finetuning.pipeline.lora.merge \
    --base_model openai/gpt-oss-20b \
    --adapter_path ./finetuning/models/lora_checkpoints \
    --output_dir ./finetuning/models/merged_model

# Validate merged model
python -m finetuning.pipeline.baseline.runner \
    --model_name ./finetuning/models/merged_model \
    --data_file unified \
    --max_samples 100
```

---

## Monitoring and Debugging

### Early Stopping

The training script supports automatic early stopping to save computation when the model has converged:

```bash
# Default: Early stopping enabled (patience=2, threshold=0.01)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# Customize early stopping
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.005

# Disable early stopping
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --early_stopping_patience 0
```

**Parameters**:
- `early_stopping_patience`: Number of epochs with no improvement before stopping (default: 2, set to 0 to disable)
- `early_stopping_threshold`: Minimum improvement in validation loss required (default: 0.01 = 1%)

**Behavior**:
- Evaluates validation loss after each epoch
- Stops training if improvement < threshold for `patience` consecutive epochs
- Automatically loads the best checkpoint at the end
- Logs early stopping triggers to training.log

### Real-Time Log Monitoring

```bash
# Watch training log
tail -f finetuning/models/lora_checkpoints/training.log

# Follow with auto-scroll
less +F finetuning/models/lora_checkpoints/training.log
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir finetuning/models/lora_checkpoints/logs

# Access in browser
# http://localhost:6006
```

**Available metrics**:
- Training loss per step
- Validation loss per epoch
- Learning rate schedule
- Gradient norm

### GPU Monitoring

```bash
# Real-time GPU usage
nvidia-smi -l 1

# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
```

### Accelerate Configuration

```bash
# Show current Accelerate config
accelerate env

# Test Accelerate setup
accelerate test

# Reconfigure Accelerate
accelerate config
```

### Training Progress

```bash
# Count completed steps
grep "Training" finetuning/models/lora_checkpoints/training.log | wc -l

# View final metrics
cat finetuning/models/lora_checkpoints/all_results.json | python3 -m json.tool

# Check checkpoint sizes
du -sh finetuning/models/lora_checkpoints/checkpoint-*
```

### Verify Training Data

```bash
# Check first training sample
head -1 finetuning/data/ft_prompts/train.jsonl | python3 -m json.tool

# Count samples
wc -l finetuning/data/ft_prompts/train.jsonl
wc -l finetuning/data/ft_prompts/validation.jsonl

# Verify format
head -5 finetuning/data/ft_prompts/train.jsonl | python3 -m json.tool
```

---

## Environment Variables

### HuggingFace Authentication

```bash
# For private models
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Verify authentication
huggingface-cli whoami
```

### CUDA Device Selection

```bash
# Use specific GPUs (e.g., only GPU 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --num_processes 2 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json
```

### Cache Directories

```bash
# Set HuggingFace cache location
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Verify cache location
echo $HF_HOME
```

### Accelerate Environment

```bash
# Set mixed precision
export ACCELERATE_MIXED_PRECISION=bf16

# Set number of processes
export ACCELERATE_NUM_PROCESSES=4
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory`, training crashes

**Solutions**:

```bash
# Option 1: Use memory efficient config
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/memory_efficient.json

# Option 2: Reduce batch size further
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16

# Option 3: Reduce sequence length
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --max_seq_length 256
```

### Slow Training

**Symptoms**: Training taking >5 hours for 3 epochs

**Diagnostics**:

```bash
# Check GPU utilization
nvidia-smi -l 1
# Should see ~80-100% utilization on all GPUs

# Verify Accelerate is using all GPUs
accelerate env
# Should show num_processes=4

# Check data loading
# If CPU bottleneck, reduce workers
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --dataloader_num_workers 2
```

### Model Not Learning

**Symptoms**: Validation loss not decreasing, poor performance

**Solutions**:

```bash
# Option 1: Increase learning rate
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --learning_rate 3e-4

# Option 2: Increase LoRA rank for more capacity
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/high_capacity.json

# Option 3: Train longer (disable early stopping)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --num_train_epochs 5 \
    --early_stopping_patience 0
```

### Data Format Errors

**Symptoms**: `KeyError: 'messages'`, data loading failures

**Verification**:

```bash
# Check training data format
head -1 finetuning/data/ft_prompts/train.jsonl | python3 -m json.tool

# Should have structure:
# {
#   "messages": [
#     {"role": "system", "content": "..."},
#     {"role": "user", "content": "..."},
#     {"role": "assistant", "content": "..."}
#   ]
# }

# Regenerate data if needed
python -m finetuning.ft_prompt_generator.cli \
    --unified_dir ./data/processed/unified \
    --output_dir ./finetuning/data/ft_prompts \
    --strategy combined_optimized
```

### Checkpoint Loading Failures

**Symptoms**: Cannot load saved checkpoint

**Verification**:

```bash
# Check adapter config exists
ls -l finetuning/models/lora_checkpoints/adapter_config.json

# View LoRA configuration
cat finetuning/models/lora_checkpoints/adapter_config.json | python3 -m json.tool

# Verify adapter weights exist
ls -lh finetuning/models/lora_checkpoints/adapter_model.safetensors
```

### Accelerate Configuration Issues

**Symptoms**: Multi-GPU not working, process hanging

**Reset configuration**:

```bash
# Reconfigure Accelerate
accelerate config

# Answer questions:
# - Compute environment: This machine
# - Number of processes: 4
# - Use DeepSpeed: No
# - Use FullyShardedDataParallel: No
# - GPU IDs: 0,1,2,3
# - Mixed precision: bf16

# Test configuration
accelerate test

# View current config
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

---

## Performance Optimization

### Optimize for Speed

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8
```

**Trade-offs**: Requires more VRAM, may reduce generalization

### Optimize for Memory

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/memory_efficient.json
```

**Trade-offs**: Slower training (30% increase), shorter sequences

### Optimize for Accuracy

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --lora_r 64 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --warmup_steps 200
```

**Trade-offs**: Longer training time, more parameters

---

## Additional Resources

### Documentation

- **Primary Reference**: [`lora_ft_approach.md`](configs/lora_ft_approach.md) ⭐
  - Detailed theoretical justification for all hyperparameters
  - Research citations (LoRA, QLoRA, AdamW papers)
  - Empirical validation results
  
- **Configuration Guide**: [`configs/README.md`](configs/README.md)
  - Configuration selection criteria
  - Comparison of all config variants
  - Custom configuration examples

- **Main README**: [`README.md`](README.md)
  - Installation instructions
  - Quick start guide
  - Troubleshooting tips

- **Validation Guide**: [`../../VALIDATION_GUIDE.md`](../../VALIDATION_GUIDE.md)
  - Phase 5: Fine-Tuning with LoRA
  - Phase 6: Post-Fine-Tuning Validation
  - Success criteria

### Research Papers

1. **LoRA**: Hu et al. (2021) - https://arxiv.org/abs/2106.09685
2. **QLoRA**: Dettmers et al. (2023) - https://arxiv.org/abs/2305.14314
3. **AdamW**: Loshchilov & Hutter (2019) - https://arxiv.org/abs/1711.05101
4. **Cosine Annealing**: Loshchilov & Hutter (2017) - https://arxiv.org/abs/1608.03983
5. **Gradient Clipping**: Pascanu et al. (2013) - https://arxiv.org/abs/1211.5063

### Tools

```bash
# Training script
python -m finetuning.pipeline.lora.train --help

# Merge script
python -m finetuning.pipeline.lora.merge --help

# Validation script
python -m finetuning.pipeline.baseline.runner --help

# Data generator
python -m finetuning.ft_prompt_generator.cli --help
```

---

## Quick Reference Card

### Most Common Commands

```bash
# Standard training
accelerate launch --num_processes 4 -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# Monitor training
tail -f finetuning/models/lora_checkpoints/training.log

# Check GPU usage
nvidia-smi -l 1

# View TensorBoard
tensorboard --logdir finetuning/models/lora_checkpoints/logs

# Validate post-training
accelerate launch --num_processes 4 -m finetuning.pipeline.baseline.runner \
    --use_accelerate --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified --max_samples 100
```

### Key Parameters

| To Change | Parameter | Typical Values |
|-----------|-----------|----------------|
| Adapter capacity | `--lora_r` | 16, 32, 64 |
| Training duration | `--num_train_epochs` | 1, 3, 5 |
| Learning speed | `--learning_rate` | 1e-4, 2e-4, 3e-4 |
| Memory usage | `--per_device_train_batch_size` | 1, 2, 4, 8 |
| Sequence length | `--max_seq_length` | 256, 512, 1024 |

---

**Note**: For comprehensive theoretical justification of all parameters, always refer to [`lora_ft_approach.md`](configs/lora_ft_approach.md). This is the authoritative source for understanding hyperparameter choices and their research backing.
