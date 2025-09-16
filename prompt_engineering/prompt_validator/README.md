# Prompt Validator for GPT-OSS-20B

A comprehensive prompt validation framework for testing and evaluating different prompt strategies with GPT-OSS-20B, featuring detailed analytics and performance comparison with clean, professional output.

## Overview

This validator provides:
- **Connection Testing**: Validate Azure AI endpoint connectivity
- **Strategy Testing**: Four different prompt strategies (Baseline, Policy, Persona, Combined)
- **Performance Analytics**: Detailed accuracy, timing, and response analysis
- **Results Storage**: Comprehensive output files with timestamped results
- **Clean Output**: Professional logging and reporting without emojis
- **Modular Design**: Separate components for easy extension and customization

## Features

### **Four Prompt Strategies (JSON-Configurable)**
All strategies are now loaded from `strategy_templates.json` for easy customization and enforce structured JSON response format with rationale:

1. **Baseline**: Simple classification without additional context
2. **Policy**: Platform guidelines-based content moderation  
3. **Persona**: Multi-perspective evaluation with role-based analysis
4. **Combined**: Policy + persona approach with comprehensive bias detection

### **JSON Response Format with Rationale**
All strategies enforce structured JSON format responses:
- **Expected Output**: `{"classification": "hate", "rationale": "explanation"}` or `{"classification": "normal", "rationale": "explanation"}`
- **Label Standardization**: Only `hate` and `normal` labels (no `not_hate`, `hateful`, etc.)
- **Robust Parsing**: Automatic fallback to text parsing if JSON fails
- **Label Normalization**: Converts variations to standard binary labels
- **Rationale Tracking**: Model explanations captured and saved in output files

### **JSON Configuration**
Strategy templates are stored in `strategy_templates.json`:
```json
{
  "baseline": {
    "name": "baseline", 
    "description": "Baseline classification approach",
    "system_prompt": "You are a content moderation assistant...",
    "user_template": "Post: \"{text}\"",
    "response_format": {"type": "json_object"},
    "parameters": {"temperature": 0.1}
  }
}
```

### **Comprehensive Analytics**
- Individual sample performance tracking
- Strategy comparison with accuracy metrics
- Response time analysis
- Error pattern identification  
- Confusion matrix and F1-score calculations
- Clean, professional terminal output without emojis

### **Output Files Generated**

- **Detailed Results CSV**: Individual sample results with predictions, response times, and rationale
- **Test Samples CSV**: All test samples with true labels
- **Performance Metrics CSV**: Combined accuracy scores and confusion matrix for all strategies
- **Evaluation Report TXT**: Human-readable summary with test samples and strategy performance

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

### **Available Commands**

#### Connection Testing Only
```bash
python runner.py --test-connection
```

#### Quick Strategy Testing (Single Sample)
```bash
# Test specific strategy with one canned sample
python runner.py --test-prompt baseline
python runner.py --test-prompt persona
python runner.py --test-prompt policy
python runner.py --test-prompt combined
```

#### Comprehensive Strategy Evaluation
```bash
# Test all strategies with detailed metrics and output files
python runner.py --test-strategy all
python runner.py --test-strategy baseline persona --sample-size 10
```

#### Default Mode (Comprehensive Evaluation)
```bash
# Run comprehensive evaluation of all strategies (default: 20 samples)
python runner.py
```

### **Output Files**

All test runs generate timestamped files in `validation_outputs/`:

1. **`strategy_unified_results_TIMESTAMP.csv`**
   - Individual sample results for each strategy
   - Columns: strategy, sample_id, text, true_label, predicted_label, response_time, rationale

2. **`test_samples_TIMESTAMP.csv`**
   - Test samples used for validation
   - Columns: text, label_binary, target_group_norm, persona_tag, source_dataset

3. **`performance_metrics_TIMESTAMP.csv`** 
   - Combined accuracy scores and confusion matrices for all strategies
   - Contains: strategy, accuracy, precision, recall, f1_score, true_positive, true_negative, false_positive, false_negative

4. **`evaluation_report_TIMESTAMP.txt`**
   - Human-readable summary report with test samples and strategy performance analysis

## Project Structure

```
prompt_validator/
├── core_validator.py          # Main validator (connection + validation logic)
├── strategy_templates.py      # Prompt strategy loader (loads from JSON)
├── strategy_templates.json    # Strategy configuration (JSON)
├── canned_samples.json        # Test samples in unified format
├── evaluation_metrics.py      # Metrics calculation
├── runner.py                  # CLI interface (dynamic strategy loading)
├── requirements.txt           # Dependencies
├── validation_outputs/        # Generated result files
└── README.md                 # This file
```
## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_AI_ENDPOINT` | Yes | Azure AI Inference endpoint | `https://your-endpoint.services.ai.azure.com/models` |
| `AZURE_AI_KEY` | Yes | Azure AI authentication key | `your-32-character-key` |

## Current Implementation

### **Connection and Strategy Testing**

- **Connection Validation**: Verify GPT-OSS-20B model connectivity and environment setup
- **Strategy Testing**: Comprehensive evaluation of all 4 prompt strategies (Policy, Persona, Combined, Baseline)
- **Performance Analytics**: Accuracy, precision, recall, F1-score, and response time analysis
- **Output File Generation**: Timestamped CSV and TXT reports for all test runs
- **Interactive CLI**: Command-line interface with multiple testing options
- **Sample Data Integration**: 5 test samples from unified hate speech dataset with true labels
- **Clean Output**: Professional terminal output without emojis for production environments

### **Supported Strategies**

1. **Baseline**: Direct hate speech classification without specialized prompting
2. **Persona**: Persona-based prompting incorporating target group identity and bias awareness  
3. **Policy**: Policy-based prompting with platform community standards and hate speech definitions
4. **Combined**: Fusion of persona and policy strategies for comprehensive bias-aware classification

## CLI Reference

### Main Commands

```bash
# Test connection only
python runner.py --test-connection

# Quick single-sample test
python runner.py --test-prompt baseline

# Comprehensive evaluation (default)
python runner.py
python runner.py --test-strategy all --sample-size 10
```

### Command Line Options

- `--test-connection`: Test Azure AI endpoint connectivity
- `--test-prompt <strategy>`: Test specific strategy with one canned sample
- `--test-strategy <strategies>`: Run comprehensive evaluation with metrics
- `--sample-size <n>`: Number of samples to test (default: 5 for comprehensive, 20 for default mode)

## API Reference

### Core Classes

#### PromptValidator

```python
from core_validator import PromptValidator

# Initialize with environment variables
validator = PromptValidator()

# Test connection
is_connected = validator.validate_connection()

# Test single strategy
result = validator.test_single_strategy("baseline", "Sample text", "normal")
```

#### Strategy Templates

```python
from strategy_templates import load_strategy_templates

# Load all strategies from JSON
templates = load_strategy_templates()

# Get specific strategy
baseline_strategy = templates['baseline']
```

#### Evaluation Metrics

```python
from evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()

# Calculate basic metrics
result = metrics.calculate_metrics(predictions, true_labels)
```

## Development

### Adding New Strategies

1. Edit `strategy_templates.json`
2. Add new strategy configuration following existing patterns
3. Test with `python runner.py --test-prompt <new_strategy>`
4. Validate with comprehensive evaluation

### Extending Metrics

1. Edit `evaluation_metrics.py`
2. Add new calculation methods
3. Update runner to use new metrics
4. Validate with test data

### Testing

```bash
# Run comprehensive strategy testing
python runner.py --test-strategy all

# Test specific functionality
python runner.py --test-connection
python runner.py --test-prompt baseline
```

## Troubleshooting

### Connection Issues

- Verify environment variables are set (`AZURE_AI_ENDPOINT`, `AZURE_AI_KEY`)
- Check Azure AI endpoint accessibility
- Validate authentication key format
- Test with `python runner.py --test-connection`

### Import Errors

- Install all dependencies: `pip install -r requirements.txt`
- Verify Python environment is activated
- Check file paths and module structure

### Performance Issues

- Monitor response times in output files
- Consider adjusting model parameters in strategy templates
- Review prompt complexity for timeout issues