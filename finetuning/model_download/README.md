# HuggingFace Model Downloader

Utilities for downloading and managing models from HuggingFace Hub with authentication support.

## Features

- ✅ Verify model access before downloading
- ✅ Support for private models via HF tokens
- ✅ List and search available models
- ✅ Suggest alternative models for hate speech detection
- ✅ Automatic token detection from environment

## Quick Start

### 1. Set Your HuggingFace Token (if needed)

For private models like Llama, set your token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Or login once:
```bash
huggingface-cli login
```

### 2. Verify Model Access

Check if a model exists and is accessible:

```bash
python -m finetuning.model_download.hf_model_downloader --verify microsoft/phi-2
```

### 3. Download a Model

```bash
python -m finetuning.model_download.hf_model_downloader --download microsoft/phi-2
```

### 4. Get Model Suggestions

See recommended open-source models for hate speech detection:

```bash
python -m finetuning.model_download.hf_model_downloader --suggest
```

## Suggested Models

### Small & Fast (< 1GB)
- **google/flan-t5-base** (220M params) - Very fast, good for fine-tuning
- **google/flan-t5-large** (780M params) - Good performance

### Medium (1-5GB)  
- **microsoft/phi-2** (2.7B params) - Efficient, good for classification ✅ Recommended
- **microsoft/Phi-3-mini-4k-instruct** (3.8B params) - Instruction-tuned

### Large (5GB+) - Requires HF Token
- **meta-llama/Llama-3.2-1B** (1B params) - Fast
- **meta-llama/Llama-3.2-3B** (3B params) - Good balance
- **mistralai/Mistral-7B-v0.1** (7B params) - High quality

## Usage Examples

### Verify Access to Private Model
```bash
HF_TOKEN=hf_xxx python -m finetuning.model_download.hf_model_downloader --verify meta-llama/Llama-3.2-3B
```

### Download with Custom Cache Directory
```bash
python -m finetuning.model_download.hf_model_downloader \
    --download microsoft/phi-2 \
    --cache-dir ./models
```

### List Available Models
```bash
# Search for Llama models
python -m finetuning.model_download.hf_model_downloader \
    --list \
    --search "llama" \
    --tags text-generation

# List classification models
python -m finetuning.model_download.hf_model_downloader \
    --list \
    --search "classification" \
    --tags text-classification
```

## Integration with Baseline Pipeline

The baseline pipeline automatically uses models from HuggingFace:

```bash
# Test with default model (microsoft/phi-2)
python -m finetuning.pipeline.baseline.runner --test_connection

# Use a different model
python -m finetuning.pipeline.baseline.runner \
    --model_name google/flan-t5-large \
    --test_connection

# Use private model with token
HF_TOKEN=hf_xxx python -m finetuning.pipeline.baseline.runner \
    --model_name meta-llama/Llama-3.2-3B \
    --test_connection
```

## Environment Variables

- **HF_TOKEN**: HuggingFace API token for authentication
- **HUGGING_FACE_HUB_TOKEN**: Alternative name for HF token
- **HF_HOME**: Custom cache directory (default: ~/.cache/huggingface)

## Command-Line Options

```bash
python -m finetuning.model_download.hf_model_downloader [OPTIONS]

Options:
  --verify MODEL        Verify access to a model
  --download MODEL      Download a model from HuggingFace
  --list                List available models
  --suggest             Suggest models for hate speech detection
  --search QUERY        Search query for listing models
  --tags TAG [TAG ...]  Filter models by tags
  --cache-dir DIR       Cache directory for downloaded models
  --token TOKEN         HuggingFace token (or use HF_TOKEN env)
  --force               Force re-download even if cached
```

## Troubleshooting

### Model Not Found

```
[FAILED] Model not found: gpt-oss-20b
```

**Solution**: The model doesn't exist. Use `--suggest` to see available models:
```bash
python -m finetuning.model_download.hf_model_downloader --suggest
```

### Authentication Required

```
[FAILED] Authentication required for: meta-llama/Llama-3.2-3B
```

**Solution**: Set your HuggingFace token:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# Or login once:
huggingface-cli login
```

### Connection Issues

If downloads are slow or timing out:
1. Check internet connection
2. Try a different model (smaller ones download faster)
3. Use `--resume` to continue interrupted downloads

## Note on Model Selection

For hate speech detection:
- **Start with phi-2** - Good balance of size and performance
- **For fine-tuning** - flan-t5-base/large are excellent choices
- **For production** - Consider larger models like Mistral or Llama
- **For testing** - Use phi-2 or flan-t5-base (fast downloads)
