# LoRA Fine-Tuning Pipeline

Complete implementation of LoRA fine-tuning for hate speech detection based on [`lora_ft_approach.md`](../baseline/templates/lora_ft_approach.md).

## Directory Structure

```
finetuning/pipeline/lora/
├── __init__.py          # Package initialization
├── configs/
│   └── default.json     # Hyperparameter configuration (QLoRA)
├── train.py            # Main training script with Accelerate
├── merge.py            # Merge LoRA adapters with base model
├── quickstart.sh       # Automated quick start script
├── README.md           # This file
└── USAGE.md            # Command-line reference guide
```

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `peft>=0.7.0` - LoRA implementation
- `bitsandbytes>=0.41.0` - 4-bit quantization
- `accelerate>=0.25.0` - Multi-GPU training
- `transformers>=4.36.0` - HuggingFace models
- `datasets>=2.14.0` - Dataset loading
- `torch>=2.1.0` - PyTorch
- `tensorboard>=2.14.0` - Training monitoring

### 2. Configure Accelerate

```bash
accelerate config
```

Select:
- Compute environment: This machine
- Number of processes: 4 (for 4 GPUs)
- Use mixed precision: bf16

### 3. Prepare Training Data

Generate fine-tuning data using `ft_prompt_generator`:

```bash
python -m finetuning.ft_prompt_generator.cli \
    --unified_dir ./data/processed/unified \
    --output_dir ./finetuning/data/ft_prompts \
    --template combined/combined_gptoss_v1.json \
    --strategy combined_optimized
```

Expected output:
- `finetuning/data/ft_prompts/train.jsonl` (10,453 samples)
- `finetuning/data/ft_prompts/validation.jsonl` (2,595 samples)

## Usage

### Option 1: Using Config File (Recommended)

Configure training via [`configs/default.json`](configs/default.json), then run:

```bash
# Activate environment
source .venv/bin/activate

# Launch multi-GPU training with Accelerate
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# Monitor training in another terminal
tail -f finetuning/models/lora_checkpoints/training.log

# Or use TensorBoard
tensorboard --logdir finetuning/models/lora_checkpoints/logs
```

**Expected duration**: 2-3 hours for 3 epochs on 10,453 samples (4x A100 80GB GPUs)

### Option 2: Command-Line Arguments

Override config file settings:

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json \
    --learning_rate 3e-4 \
    --num_train_epochs 5 \
    --lora_r 64
```

### Option 3: Full Command-Line (No Config File)

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
    --lora_alpha 32
```

## Configuration Details

All hyperparameters are defined in [`configs/default.json`](configs/default.json) with detailed theoretical justifications in [`lora_ft_approach.md`](../baseline/templates/lora_ft_approach.md).

### Configuration Structure

The config file is organized into sections:

1. **Model Configuration** - Model loading and 4-bit quantization (QLoRA)
2. **Data Configuration** - Training/validation data paths and sequence length
3. **LoRA Configuration** - Low-rank adaptation parameters (r, alpha, dropout, target modules)
4. **Training Hyperparameters** - Learning rate, batch size, epochs, warmup, weight decay
5. **Optimization Configuration** - Precision (bf16), gradient checkpointing, optimizer
6. **Logging and Checkpointing** - Output directory, evaluation strategy, TensorBoard

### Key Hyperparameters Summary

For **detailed theoretical justification** of each parameter value, see:
- **Primary Reference**: [`lora_ft_approach.md`](../baseline/templates/lora_ft_approach.md) - Section "Hyperparameter Specifications"
- **Quick Reference**: Table below with links to detailed explanations

| Parameter | Value | Reference |
|-----------|-------|-----------|
| `learning_rate` | 2e-4 | [LoRA allows 2-5× higher LR](../baseline/templates/lora_ft_approach.md#1-learning-rate-2e-4) |
| `num_train_epochs` | 3 | [Sufficient for LoRA convergence](../baseline/templates/lora_ft_approach.md#2-number-of-epochs-3) |
| `per_device_train_batch_size` | 4 | [Memory constraint (A100 80GB)](../baseline/templates/lora_ft_approach.md#3-batch-size-4-per-gpu-effective-16-with-4-gpus) |
| `gradient_accumulation_steps` | 4 | [Effective batch size = 64](../baseline/templates/lora_ft_approach.md#4-gradient-accumulation-steps-4) |
| `warmup_steps` | 100 | [20.5% of total steps](../baseline/templates/lora_ft_approach.md#5-warmup-steps-100) |
| `weight_decay` | 0.01 | [AdamW decoupled regularization](../baseline/templates/lora_ft_approach.md#6-weight-decay-001) |
| `max_grad_norm` | 1.0 | [Gradient clipping for stability](../baseline/templates/lora_ft_approach.md#7-max-gradient-norm-10) |
| `lr_scheduler_type` | cosine | [Smooth decay](../baseline/templates/lora_ft_approach.md#8-learning-rate-scheduler-cosine) |
| `lora_r` | 32 | [Rank for adapter capacity](../baseline/templates/lora_ft_approach.md#9-lora-rank-r-32) |
| `lora_alpha` | 32 | [Scaling factor (α=r)](../baseline/templates/lora_ft_approach.md#10-lora-alpha-32) |
| `lora_dropout` | 0.05 | [Light regularization](../baseline/templates/lora_ft_approach.md#11-lora-dropout-005) |
| `lora_target_modules` | q_proj, v_proj | [Query & Value projections](../baseline/templates/lora_ft_approach.md#12-target-modules-q_proj-v_proj) |
| `load_in_4bit` | true | [Memory efficiency (QLoRA)](../baseline/templates/lora_ft_approach.md#14-quantization-4-bit-nf4) |
| `bnb_4bit_quant_type` | nf4 | [NormalFloat4 optimal for LLMs](../baseline/templates/lora_ft_approach.md#14-quantization-4-bit-nf4) |
| `early_stopping_patience` | 2 | Stop training if no improvement for N epochs |
| `early_stopping_threshold` | 0.01 | Minimum improvement required (1%) |

### Creating Custom Configurations

To create a custom configuration:

```bash
# Copy default config
cp finetuning/pipeline/lora/configs/default.json \
   finetuning/pipeline/lora/configs/experimental.json

# Edit parameters (e.g., higher rank for more capacity)
# Then run with custom config
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/experimental.json
```

**Example configurations**:
- `default.json` - Standard training (r=32, 3 epochs, lr=2e-4, early stopping enabled)
- `high_capacity.json` - Higher rank (r=64) for complex tasks
- `memory_efficient.json` - Smaller batch size (bs=2) for limited VRAM
- `quick_test.json` - Single epoch for testing (1 epoch, r=16, early stopping disabled)

## Monitoring Training

### Early Stopping

Training will automatically stop if the model converges early, saving computation time. The default configuration uses:
- **Patience**: 2 epochs (stops if no improvement for 2 consecutive epochs)
- **Threshold**: 0.01 (requires at least 1% improvement in validation loss)

To disable early stopping, set `early_stopping_patience` to 0 in your config or add `--early_stopping_patience 0` to the command line.

### Log Files

- **Training log**: `finetuning/models/lora_checkpoints/training.log`
- **TensorBoard logs**: `finetuning/models/lora_checkpoints/logs/`

### TensorBoard

```bash
tensorboard --logdir finetuning/models/lora_checkpoints/logs
```

Open browser to `http://localhost:6006`

### Real-time Log Monitoring

```bash
tail -f finetuning/models/lora_checkpoints/training.log
```

## Post-Training Validation

### 1. Validate with Simple Prompts

Test if model internalized task without complex prompting:

```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --prompt_template ./prompt_engineering/prompt_templates/baseline_v1.json \
    --strategy baseline_standard \
    --max_samples 100 \
    --output_dir ./finetuning/outputs/gptoss/post_finetune
```

### 2. Compare Performance

**Success Criterion**: Post-FT F1 (simple prompts) ≥ 0.615 (Pre-FT baseline with sophisticated prompts)

```bash
# Extract metrics
cd finetuning/outputs/gptoss

# Baseline (sophisticated prompts)
BASELINE_F1=$(awk -F',' 'NR==2 {print $5}' baseline/run_*/performance_metrics_*.csv | head -1)

# Post-FT (simple prompts)
POSTFT_F1=$(awk -F',' 'NR==2 {print $5}' post_finetune/run_*/performance_metrics_*.csv | head -1)

echo "Baseline F1 (sophisticated): $BASELINE_F1"
echo "Post-FT F1 (simple):        $POSTFT_F1"
```

### 3. Bias Metrics Analysis

Check fairness across target groups:

```bash
# View bias metrics
cat finetuning/outputs/gptoss/post_finetune/run_*/bias_metrics_*.csv
```

Target thresholds:
- **LGBTQ+ FPR**: ≤ 40.0% (baseline: 43.0%)
- **Mexican FPR**: ≤ 10.0% (baseline: 8.1%)
- **Middle East FPR**: ≤ 25.0% (baseline: 23.6%)

## Merging LoRA Adapters (Optional)

Merge adapters into base model for deployment without PEFT:

```bash
python -m finetuning.pipeline.lora.merge \
    --base_model openai/gpt-oss-20b \
    --adapter_path ./finetuning/models/lora_checkpoints \
    --output_dir ./finetuning/models/merged_model
```

**Note**: 
- Merged model: ~78GB (full model size)
- Adapter-only: ~32MB (load dynamically with PEFT)
- Merging is optional - adapters work fine for inference

### Using Merged Model

```bash
python -m finetuning.pipeline.baseline.runner \
    --model_name ./finetuning/models/merged_model \
    --data_file unified \
    --max_samples 100
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA OOM errors during training

### Solutions**:
1. Reduce `per_device_train_batch_size` from 4 to 2 in config
2. Increase `gradient_accumulation_steps` to 8 to maintain effective batch size
3. Gradient checkpointing already enabled in default config
4. Reduce `max_seq_length` from 512 to 256 if needed

Edit `configs/default.json`:
```json
{
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 256
}
```

### Slow Training

**Symptoms**: Training taking >5 hours for 3 epochs

**Solutions**:
1. Verify Accelerate is using all 4 GPUs: Check `accelerate env`
2. Check GPU utilization: `nvidia-smi -l 1`
3. Reduce `dataloader_num_workers` if CPU bottleneck
4. Enable gradient checkpointing if not already enabled

### Model Not Learning

**Symptoms**: Validation loss not decreasing

**Solutions**:
1. Increase `learning_rate` from 2e-4 to 3e-4 or 5e-4
2. Increase `lora_r` from 32 to 64 for more capacity
3. Check training data format (must have 'messages' field)
4. Verify data preprocessing with sample inspection

### Checkpoint Loading Fails

**Symptoms**: Cannot load saved checkpoint

**Solutions**:
1. Check `adapter_config.json` exists in checkpoint directory
2. Use `PeftModel.from_pretrained()` not `AutoModel.from_pretrained()`
3. Verify base model name matches training configuration

## Output Files

After training, the following files are created:

```
finetuning/models/lora_checkpoints/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # LoRA weights (~32MB)
├── training_args.bin             # Training arguments
├── trainer_state.json            # Trainer state
├── training.log                  # Complete training log
├── all_results.json              # Final metrics
├── train_results.json            # Training metrics
├── eval_results.json             # Evaluation metrics
├── checkpoint-163/               # Epoch 1 checkpoint
├── checkpoint-326/               # Epoch 2 checkpoint
└── logs/                         # TensorBoard logs
    └── events.out.tfevents.*
```

## References

1. **LoRA Paper**: Hu et al. (2021) - https://arxiv.org/abs/2106.09685
2. **QLoRA Paper**: Dettmers et al. (2023) - https://arxiv.org/abs/2305.14314
3. **AdamW Paper**: Loshchilov & Hutter (2019) - https://arxiv.org/abs/1711.05101
4. **Cosine Annealing**: Loshchilov & Hutter (2017) - https://arxiv.org/abs/1608.03983
5. **Gradient Clipping**: Pascanu et al. (2013) - https://arxiv.org/abs/1211.5063
6. **Approach Document**: [`lora_ft_approach.md`](../baseline/templates/lora_ft_approach.md)

## Support

For issues or questions:
1. Check [`VALIDATION_GUIDE.md`](../../VALIDATION_GUIDE.md) Phase 5: Fine-Tuning with LoRA
2. Review [`lora_ft_approach.md`](../baseline/templates/lora_ft_approach.md) for theoretical foundations
3. Consult training logs in `finetuning/models/lora_checkpoints/training.log`
