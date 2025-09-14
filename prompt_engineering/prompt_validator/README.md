# Prompt Validator for GPT-OSS-20B

A simplified prompt validation framework for testing connection to GPT-OSS-20B with scaffolding for future prompt strategy development.

## Overview

This validator focuses on:
- ✅ **Connection Testing**: Validate Azure AI endpoint connectivity
- 🏗️ **Scaffolding**: Empty framework for prompt strategy development
- 📋 **Clean Output**: Professional logging without emojis
- 🔧 **Modular Design**: Separate components for easy extension

## Quick Start

### 1. Environment Setup

Set required environment variables:

```bash
# Required: Azure AI Inference endpoint and key
export AZURE_AI_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
export AZURE_AI_KEY="your-azure-ai-key"
```

**Windows PowerShell:**
```powershell
$env:AZURE_AI_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
$env:AZURE_AI_KEY="your-azure-ai-key"
```

### 2. Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### 3. Test Connection

```bash
cd prompt_engineering/prompt_validator
python runner.py --test-connection
```

## Usage

### Connection Testing
```bash
python runner.py --test-connection
```

### View Data Format
```bash
python runner.py --show-format
```

### Run Scaffolding Demo
```bash
python runner.py --demo --sample-size 5
```

### Default Mode (Connection + Demo)
```bash
python runner.py
```

## Project Structure

```
prompt_validator/
├── core_validator.py          # Main validator (connection + scaffolding)
├── strategy_templates.py      # Prompt strategy templates (scaffolding)
├── evaluation_metrics.py      # Metrics calculation (scaffolding)
├── runner.py                  # CLI interface
├── examples.py               # Usage examples (scaffolding)
└── README.md                 # This file

# Complex versions (for reference)
├── core_validator_complex.py     # Full implementation
├── strategy_templates_complex.py # Full templates
├── evaluation_metrics_complex.py # Full metrics
└── runner_complex.py            # Full CLI
```

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_AI_ENDPOINT` | ✅ Yes | Azure AI Inference endpoint | `https://your-endpoint.services.ai.azure.com/models` |
| `AZURE_AI_KEY` | ✅ Yes | Azure AI authentication key | `your-32-character-key` |

## Scaffolding Features

### Current Implementation ✅
- Connection validation to GPT-OSS-20B
- Basic prompt template structure
- Minimal evaluation metrics framework
- CLI interface for testing

### Future Development 🏗️
- **Strategy Validation**: Policy, Persona, Combined, Baseline
- **Batch Processing**: Multiple texts and strategies
- **Metrics Calculation**: Accuracy, precision, recall, F1
- **Report Generation**: CSV and text output
- **Custom Strategies**: User-defined prompt templates
- **Parameter Optimization**: Automatic strategy tuning

## API Reference

### PromptValidator
```python
from core_validator import PromptValidator

# Initialize with environment variables
validator = PromptValidator()

# Test connection
if validator.validate_connection():
    print("Connected successfully!")

# Scaffolding methods (placeholders)
validator.validate_strategies(sample_size=25)
validator.test_single_strategy("policy", "test text")
validator.batch_validate(["text1", "text2"])
```

### Strategy Templates
```python
from strategy_templates import create_strategy_templates

# Get scaffolding templates
strategies = create_strategy_templates()
# Returns: {"policy": PromptStrategy, "persona": PromptStrategy, ...}
```

### Evaluation Metrics
```python
from evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()
# Scaffolding methods
result = metrics.parse_prediction("model response")
scores = metrics.calculate_metrics(predictions, true_labels)
```

## Development

### Adding New Strategies
1. Edit `strategy_templates.py`
2. Add strategy to `create_strategy_templates()`
3. Update validation logic in `core_validator.py`

### Extending Metrics
1. Edit `evaluation_metrics.py`
2. Implement calculation methods
3. Update report generation

### Testing
```bash
# Test connection only
python runner.py --test-connection

# Full scaffolding demo
python runner.py --demo --sample-size 10
```

## Troubleshooting

### Connection Issues
- ✅ Verify environment variables are set
- ✅ Check Azure AI endpoint URL format
- ✅ Confirm API key is valid
- ✅ Test network connectivity

### Import Errors
- ✅ Install dependencies: `pip install -r requirements.txt`
- ✅ Check Python path includes project root
- ✅ Verify all modules are in correct locations

### Empty Responses
- ✅ Normal behavior for some model configurations
- ✅ Connection is working if status is 200
- ✅ Check model deployment status in Azure

## Next Steps

1. **Implement Strategy Logic**: Replace scaffolding with real implementations
2. **Add Comprehensive Metrics**: Full evaluation suite
3. **Build Report Generation**: Professional output formats
4. **Add Batch Processing**: Efficient multi-sample validation
