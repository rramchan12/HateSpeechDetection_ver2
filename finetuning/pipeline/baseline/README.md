# Baseline Validation Pipeline

Baseline inference pipeline for GPT-OSS models to establish performance baseline on hate speech detection.

## Structure

```
baseline/
├── runner.py              # CLI entry point
├── inference.py           # Inference execution
├── metrics/               # Metrics calculation package
│   ├── __init__.py
│   └── calculator.py     # Reuses prompt_engineering metrics
└── model_loader/          # Model loading package
    ├── __init__.py
    └── loader.py          # HuggingFace model loader
```

## Dependencies

Install finetuning-specific dependencies:
```bash
pip install -r finetuning/requirements.txt
```

## Usage

### Test Connection (Recommended First Step)

Before running full validation, test that the model loads and responds correctly:

```bash
python -m finetuning.pipeline.baseline.runner --test_connection
```

This will:
- Load the model
- Run a simple test prompt
- Verify the model generates a response
- Exit without running full validation

**Example Output:**
```
============================================================
CONNECTION TEST MODE
============================================================
Model: gpt-oss-20b
============================================================

Loading model: gpt-oss-20b
This may take 5-10 minutes...
[OK] Model loaded successfully!
  Parameters: 20.0B
  Memory: 10.42 GB

Test prompt: Classify this text as 'hate' or 'not hate': Hello, how are you?
### Classification:

Generating response...

============================================================
RESPONSE:
============================================================
Classify this text as 'hate' or 'not hate': Hello, how are you?
### Classification: not hate
============================================================

[SUCCESS] Connection test passed!
Model is loaded and responding correctly.
```

### Quick Test (50 samples)
```bash
python -m finetuning.pipeline.baseline.runner --max_samples 50
```

### Full Validation (all samples)
```bash
python -m finetuning.pipeline.baseline.runner
```

### Custom Configuration
```bash
python -m finetuning.pipeline.baseline.runner \
    --model_name gpt-oss-20b \
    --data_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --temperature 0.1
```

## Command Line Arguments

- `--model_name`: HuggingFace model ID (default: `gpt-oss-20b`)
- `--data_file`: Path to validation data (default: `./data/validation.jsonl`)
- `--output_dir`: Output directory (default: `./outputs`)
- `--max_samples`: Max samples to process (default: None = all)
- `--max_length`: Max input tokens (default: 512)
- `--max_new_tokens`: Max output tokens (default: 10)
- `--temperature`: Sampling temperature (default: 0.1)
- `--test_connection`: Test model connection and exit (no validation data needed)

## Output Files

Saved with timestamps in `./outputs/`:

- `baseline_results_YYYYMMDD_HHMMSS.json` - All predictions
- `baseline_metrics_YYYYMMDD_HHMMSS.json` - Metrics (matches prompt_engineering format)

## Metrics Format

Output matches `prompt_engineering/metrics/` format:
```json
{
  "strategy": "baseline",
  "accuracy": 0.65,
  "precision": 0.61,
  "recall": 0.62,
  "f1_score": 0.615,
  "false_positive_rate": 0.18,
  "false_negative_rate": 0.16,
  "confusion_matrix": {
    "true_positive": 106,
    "false_positive": 69,
    "true_negative": 315,
    "false_negative": 52
  }
}
```

## Typical Execution Time

- 50 samples: 5-10 minutes
- 545 samples: 45-60 minutes

## Troubleshooting

**Connection issues**: Run `--test_connection` first to isolate model loading problems

**CUDA out of memory**: Run quick test first with `--max_samples 50`

**Model not found**: Check internet connection and HuggingFace model ID

**Module not found**: Ensure you're in project root with venv activated

