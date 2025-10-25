# Baseline Pipeline - Validation Runner

Command-line tool for baseline model validation with comprehensive evaluation metrics. Supports both single-GPU and multi-GPU execution via Accelerate.

---

## Command Reference

### Mode Selection
- `--test_connection` - Test model connection and exit
- `--use_accelerate` - Enable multi-GPU inference (requires `accelerate launch`)

### Model Configuration
- `--model_name` - HuggingFace model ID (default: `openai/gpt-oss-20b`)
- `--cache_dir` - Model cache directory (default: `data/models`)

### Data Configuration
- `--data_file` - Data source options:
  - `'unified'` - Full unified test dataset (~545 samples)
  - `'canned_50_quick'` - Quick test dataset (50 samples, default)
  - `'canned_100_stratified'` - Stratified sample (100 samples)
  - `'canned_100_size_varied'` - Size-varied sample (100 samples)
  - `'./path/to/file.jsonl'` - Direct JSONL file path
- `--max_samples` - Maximum samples to process (default: 5)

### Prompt Configuration
- `--prompt_template` - Path to prompt template JSON
- `--strategy` - Strategy name from template (default: `baseline_conservative`)

### Output Configuration
- `--output_dir` - Base output directory (default: `./finetuning/outputs/baseline`)
- `--debug` - Enable debug logging

### Generation Parameters
- `--max_length` - Maximum input token length (default: 512)
- `--max_new_tokens` - Maximum generated tokens (default: 100)
- `--temperature` - Sampling temperature (default: 0.1)

### Help
```bash
python -m finetuning.pipeline.baseline.runner --help
```

---

## Single GPU Usage

### Test Connection
```bash
# Test default model
python -m finetuning.pipeline.baseline.runner --test_connection

# Test different model
python -m finetuning.pipeline.baseline.runner \
    --test_connection \
    --model_name microsoft/phi-2

# Test private model (requires HF token)
HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner \
    --test_connection \
    --model_name meta-llama/Llama-3.2-3B
```

### Run Validation
```bash
# Quick test (5 samples)
python -m finetuning.pipeline.baseline.runner

# Medium test with canned data
python -m finetuning.pipeline.baseline.runner \
    --data_file canned_50_quick \
    --max_samples 10

# Full validation with unified dataset
python -m finetuning.pipeline.baseline.runner \
    --data_file unified \
    --max_samples 100

# Use fine-tuning validation data
python -m finetuning.pipeline.baseline.runner \
    --data_file ./finetuning/data/prepared/validation.jsonl \
    --max_samples 10

# Custom prompt template and strategy
python -m finetuning.pipeline.baseline.runner \
    --data_file canned_50_quick \
    --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --max_samples 10
```

---

## Multi-GPU Usage with Accelerate

### First-Time Setup
```bash
# Install Accelerate
pip install accelerate

# Configure for your GPU setup
accelerate config
```

**Configuration prompts**:
1. This machine
2. No distributed training (multi-GPU on single machine)
3. NO to DeepSpeed
4. NO to Megatron-LM
5. Number of GPUs: 4 (or your GPU count)
6. NO to FSDP
7. Mixed precision: bf16

**Verify configuration**:
```bash
accelerate env
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

**Expected config**:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: 4
use_cpu: false
```

### Run Validation
```bash
# Quick test with 2 GPUs
accelerate launch --num_processes 2 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file canned_50_quick \
    --max_samples 10

# Full run with 4 GPUs
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file unified \
    --max_samples 100

# With custom prompt template (4 GPUs)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file canned_100_stratified \
    --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --max_samples 100
```

**Performance**:
- 2 GPUs: ~1.9x speedup
- 4 GPUs: ~3.8x speedup

### How Multi-GPU Works
- Dataset automatically split across GPUs
- Each GPU processes its subset in parallel
- Results automatically gathered from all GPUs
- Only main process saves final results
- No manual dataset splitting required

### Troubleshooting
```bash
# Check Accelerate availability
python -c "from finetuning.pipeline.baseline.runner import ACCELERATE_AVAILABLE; print(f'Accelerate available: {ACCELERATE_AVAILABLE}')"

# Verify GPU visibility
nvidia-smi

# Test Accelerate configuration
accelerate test
```

**Common Issues**:
- **Accelerate not installed**: `pip install accelerate`
- **Multi-GPU not working**: Run `accelerate config` and ensure `num_machines: 1`
- **Out of memory**: Reduce `--max_samples` or use fewer GPUs

---

## Output Files

Each validation run creates a timestamped directory:

```
finetuning/outputs/baseline/run_YYYYMMDD_HHMMSS/
├── validation_log_YYYYMMDD_HHMMSS.log          # Detailed execution log
├── evaluation_report_YYYYMMDD_HHMMSS.txt       # Human-readable summary
├── strategy_unified_results_YYYYMMDD_HHMMSS.csv # Full inference results
├── performance_metrics_YYYYMMDD_HHMMSS.csv     # Aggregated metrics
├── bias_metrics_YYYYMMDD_HHMMSS.csv            # Per-group performance
└── test_samples_YYYYMMDD_HHMMSS.csv            # Test samples used
```

---

## Features

- Model caching for fast reloads
- Multiple data sources (unified, canned, JSONL)
- Custom prompt templates and strategies
- Rich metrics (performance, bias analysis)
- Comprehensive logging
- Multi-GPU support via Accelerate
- Scalable (1-4+ GPUs without code changes)

---

## Documentation

- **Accelerate Integration**: `connector/INTEGRATION_STATUS.md`
- **Quick Reference**: `../ACCELERATE_QUICK_REFERENCE.md`
- **Accelerate Docs**: https://huggingface.co/docs/accelerate
