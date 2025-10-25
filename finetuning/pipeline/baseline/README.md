# Baseline Pipeline - Unified Runner

**Single command-line tool for testing, baseline validation, and fine-tuning.**

## 🎯 Three Operational Modes

1. **Test Connection** - Verify model loads correctly
2. **Run Baseline** - Evaluate pre-trained model performance
3. **Run Fine-tuning** - Train model with LoRA configuration (scaffolding)

---

## 🚀 Quick Commands

### Test Model Connection

```bash
# Test default model (openai/gpt-oss-20b)
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

**Time**: ~5 seconds (cached) | ~5-10 minutes (first download)  
**Memory**: ~42GB VRAM (gpt-oss-20b)

---

### Run Baseline Validation

```bash
# Quick test (3 samples) - unified dataset
python -m finetuning.pipeline.baseline.runner --max_samples 3

# Use canned dataset (50 samples)
python -m finetuning.pipeline.baseline.runner \
    --data_source canned_50_quick \
    --max_samples 10

# Use canned stratified dataset
python -m finetuning.pipeline.baseline.runner \
    --data_source canned_100_stratified \
    --max_samples 50

# Full baseline with unified dataset (545 samples)
python -m finetuning.pipeline.baseline.runner

# With specific strategy
python -m finetuning.pipeline.baseline.runner \
    --strategy combined_optimized \
    --data_source canned_100_size_varied \
    --max_samples 50
```

**Data Sources**:
- `unified` (default) - Full validation dataset (545 samples)
- `canned_50_quick` - Quick test dataset (50 samples)
- `canned_100_stratified` - Stratified sample (100 samples)
- `canned_100_size_varied` - Size-varied sample (100 samples)

**Time**: 3 samples (~30s) | 50 samples (~10min) | 545 samples (~100min)

---

### Run Fine-tuning (Scaffolding)

```bash
# Use example template with unified dataset
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json

# Use canned dataset for quick training test
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json \
    --finetune_data_source canned_100_stratified

# Specify model, output directory, and data source
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json \
    --model_name microsoft/phi-2 \
    --finetune_output_dir ./finetuning/outputs/models/phi2_finetuned \
    --finetune_data_source canned_50_quick

# Custom template with unified dataset
python -m finetuning.pipeline.baseline.runner \
    --finetune my_custom_config.json \
    --model_name openai/gpt-oss-20b \
    --finetune_output_dir ./finetuning/outputs/models/custom_model
```

**Data Sources for Fine-tuning**:
- `unified` (default) - Uses `train.jsonl` and `validation.jsonl` from `finetuning/data/prepared/`
- `canned_50_quick` - Small canned dataset for quick experiments
- `canned_100_stratified` - Stratified sample for balanced training
- `canned_100_size_varied` - Size-varied sample

**Status**: Scaffolding - loads and validates configuration

---

## 📂 Directory Structure

```
finetuning/pipeline/baseline/
├── runner.py                   # Main CLI entry point
├── README.md                   # This file
├── templates/                  # Fine-tuning configuration templates
│   └── example_template.json  # Example LoRA configuration
├── model_loader/
│   └── loader.py              # Model loading with caching
└── metrics/
    └── calculator.py          # Metrics calculation

finetuning/outputs/
├── baseline/                  # Baseline validation results
│   └── run_YYYYMMDD_HHMMSS/
└── models/                    # Fine-tuned models
    └── finetuned_model/       # Default output location

data/models/                   # Model cache
├── openai--gpt-oss-20b/
│   └── .meta                  # Model metadata
└── models--openai--gpt-oss-20b/  # HuggingFace cache
```

---

## 🔧 Command-Line Options

### Mode Selection (mutually exclusive)
- `--test_connection` - Test model and exit
- `--finetune TEMPLATE` - Run fine-tuning with template

### Model Configuration
- `--model_name` - HuggingFace model ID (default: `openai/gpt-oss-20b`)
- `--cache_dir` - Model cache directory (default: `data/models`)

### Baseline Options
- `--data_source` - Data source: 'unified' (default) or canned file name (e.g., 'canned_50_quick')
- `--output_dir` - Baseline results directory (default: `./finetuning/outputs/baseline`)
- `--prompt_template` - Template path (default: `combined/combined_gptoss_v1.json`)
- `--strategy` - Strategy name (default: `combined_optimized`)
- `--max_samples` - Max samples to process (default: all)

### Fine-tuning Options
- `--finetune_data_source` - Data source: 'unified' (default) or canned file name
- `--finetune_output_dir` - Fine-tuned model directory (default: `./finetuning/outputs/models/finetuned_model`)

---

## � Logging

All modes automatically create detailed log files for debugging and analysis.

### Log File Location

**Baseline mode**:
```
finetuning/outputs/baseline/run_TIMESTAMP/validation_log_TIMESTAMP.log
```

**Fine-tuning mode**:
```
finetuning/outputs/models/finetuned_model/run_TIMESTAMP/validation_log_TIMESTAMP.log
```

### Log Contents

The log files capture:
- **Configuration**: Model, data source, template, strategy
- **Model loading**: Loading time and success/failure
- **Sample processing**: For each sample:
  - Input text and true label
  - Target group
  - System and user prompts (truncated)
  - Model parameters (temperature, max_tokens, etc.)
  - Response time
  - Raw model response
  - Predicted label and rationale
  - Match result (✓ or ✗)
- **Metrics**: Accuracy, F1 score, precision, recall
- **File operations**: Saved reports and CSV files

### Example Log Entry

```
2025-10-22 09:29:34,599 - baseline_runner - INFO - ==================================================
2025-10-22 09:29:34,599 - baseline_runner - INFO - Processing sample from validation
2025-10-22 09:29:34,599 - baseline_runner - INFO - Strategy: combined_optimized
2025-10-22 09:29:34,599 - baseline_runner - INFO - Input text: example text here
2025-10-22 09:29:34,599 - baseline_runner - INFO - True label: hate
2025-10-22 09:29:34,599 - baseline_runner - INFO - Target group: lgbtq
2025-10-22 09:29:34,599 - baseline_runner - INFO - System prompt: You are an expert content moderation assistant...
2025-10-22 09:29:34,599 - baseline_runner - INFO - User prompt: Apply X Platform Hateful Conduct Policy...
2025-10-22 09:29:34,599 - baseline_runner - INFO - Model parameters: {'max_tokens': 512, 'temperature': 0.1...}
2025-10-22 09:29:40,102 - baseline_runner - INFO - Response time: 5.50s
2025-10-22 09:29:40,102 - baseline_runner - INFO - Raw response: analysis... classification: hate...
2025-10-22 09:29:40,102 - baseline_runner - INFO - Predicted label: hate
2025-10-22 09:29:40,102 - baseline_runner - INFO - Rationale: text parse
2025-10-22 09:29:40,102 - baseline_runner - INFO - Match: ✓
```

### Benefits

- **Debugging**: Full request/response pairs for error analysis
- **Reproducibility**: Complete record of model behavior
- **Performance tracking**: Response times per sample
- **Audit trail**: Timestamped record of all operations

---

## �🛠️ Fine-tuning Template Format

Templates in `templates/` directory contain **only configuration metadata**:

```json
{
  "name": "example_lora_config",
  "description": "LoRA fine-tuning configuration",
  "version": "1.0",
  "training": {
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 8
  },
  "lora": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"]
  }
}
```

**Note**: Model name, data files, and output directories are **command-line arguments**, not in templates.

---

## 🎯 Typical Workflow

```bash
## 🎯 Typical Workflow

```bash
# 1. Test connection
python -m finetuning.pipeline.baseline.runner --test_connection

# 2. Quick baseline with canned data
python -m finetuning.pipeline.baseline.runner \
    --data_source canned_50_quick \
    --max_samples 10

# 3. Full baseline with unified dataset
python -m finetuning.pipeline.baseline.runner

# 4. Compare strategies with canned data
python -m finetuning.pipeline.baseline.runner \
    --data_source canned_100_stratified \
    --strategy combined_focused \
    --max_samples 50

# 5. Fine-tune with canned data (quick test)
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json \
    --finetune_data_source canned_100_stratified \
    --model_name openai/gpt-oss-20b

# 6. Fine-tune with unified dataset (full training)
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json \
    --finetune_data_source unified \
    --finetune_output_dir ./finetuning/outputs/models/my_model
```
```

---

## 📞 Help

```bash
# Show all options
python -m finetuning.pipeline.baseline.runner --help

# List templates
ls -1 finetuning/pipeline/baseline/templates/
```

---

## 🚧 Roadmap

- [x] Test connection mode
- [x] Baseline validation mode
- [x] Fine-tuning scaffolding
- [x] Templates in pipeline/baseline/templates/
- [x] Command-line model/output configuration
- [ ] Implement LoRA fine-tuning
- [ ] Model comparison tools
