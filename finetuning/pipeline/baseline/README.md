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

Install all project dependencies (from project root):
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The unified `requirements.txt` includes all dependencies for:
- Core ML frameworks (PyTorch, Transformers)
- Fine-tuning pipeline (PEFT, bitsandbytes, accelerate)
- Metrics and evaluation (scikit-learn, scipy)
- Data processing (datasets, pandas, numpy)
- Development tools (jupyter, pytest)

## Usage

### Test Connection (Recommended First Step)

Before running full validation, test that the model loads and responds correctly:

```bash
# Test with default model (openai/gpt-oss-20b)
python -m finetuning.pipeline.baseline.runner --test_connection

# Test with a different model
python -m finetuning.pipeline.baseline.runner \
    --model_name microsoft/phi-2 \
    --test_connection

# Test with private model (requires HF token)
HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner \
    --model_name meta-llama/Llama-3.2-3B \
    --test_connection
```

This will:
- Load the model from HuggingFace Hub
- Run a simple test prompt
- Verify the model generates a response
- Exit without running full validation

**Example Output:**
```
============================================================
CONNECTION TEST MODE
============================================================
Model: openai/gpt-oss-20b
============================================================

Loading model: openai/gpt-oss-20b
This may take 5-10 minutes...
  Using dtype: bfloat16 (GPU supports it)
[OK] Model loaded successfully!
  Parameters: 20.9B
  Memory: 41.83 GB

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

### Explore Available Models

See suggested models for hate speech detection:

```bash
python -m finetuning.model_download.hf_model_downloader --suggest
```

Verify a specific model is accessible:

```bash
python -m finetuning.model_download.hf_model_downloader --verify microsoft/phi-2
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
    --model_name openai/gpt-oss-20b \
    --data_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --temperature 0.1

# With a smaller/faster model
python -m finetuning.pipeline.baseline.runner \
    --model_name microsoft/phi-2 \
    --max_samples 50

# With private model
HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner \
    --model_name meta-llama/Llama-3.2-3B \
    --max_samples 50
```

## Model Selection

**Default Model**: `openai/gpt-oss-20b` (20.9B parameters, public, no token required)

**Recommended Models**:
- **openai/gpt-oss-20b** - OSS flagship, 20.9B params, high quality ✅ Default
- **openai/gpt-oss-120b** - Best quality, 120B params (requires ~80GB VRAM)
- **microsoft/phi-2** - Fast & efficient, 2.8B params
- **google/flan-t5-large** - Excellent for fine-tuning, 780M params
- **meta-llama/Llama-3.2-3B** - High quality, requires HF token

See all suggestions:
```bash
python -m finetuning.model_download.hf_model_downloader --suggest
```

## HuggingFace Authentication

For private models (like Llama), set your HF token:

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Option 2: Login once (token stored permanently)
huggingface-cli login

# Then use the model
python -m finetuning.pipeline.baseline.runner \
    --model_name meta-llama/Llama-3.2-3B \
    --test_connection
```

## Command Line Arguments

- `--model_name`: HuggingFace model ID (default: `openai/gpt-oss-20b`)
- `--data_file`: Path to validation data (default: `./data/validation.jsonl`)
- `--output_dir`: Output directory (default: `./outputs`)
- `--max_samples`: Max samples to process (default: None = all)
- `--max_length`: Max input tokens (default: 512)
- `--max_new_tokens`: Max output tokens (default: 10)
- `--temperature`: Sampling temperature (default: 0.1)
- `--test_connection`: Test model connection and exit (no validation data needed)

**Environment Variables**:
- `HF_TOKEN`: HuggingFace API token for private models

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

**Model not found**: The model doesn't exist on HuggingFace Hub
```bash
# See suggested models
python -m finetuning.model_download.hf_model_downloader --suggest

# Verify a specific model
python -m finetuning.model_download.hf_model_downloader --verify MODEL_NAME
```

**Authentication required**: Model is private and needs a HF token
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# Or login once: huggingface-cli login
```

**Connection issues**: Run `--test_connection` first to isolate model loading problems

**CUDA out of memory**: Run quick test first with `--max_samples 50`

**Module not found**: Ensure you're in project root with .venv activated
```bash
source .venv/bin/activate
```

