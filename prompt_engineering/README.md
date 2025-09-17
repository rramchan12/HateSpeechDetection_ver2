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
- **Flexible Data Loading**: Support for both canned samples and unified dataset
- **Robust Metrics**: Fixed metrics calculation handling None predictions

## Features

### **Four Prompt Strategies (JSON-Configurable)**
All strategies are loaded from `prompt_templates/all_combined.json` for easy customization and enforce structured JSON response format with rationale:

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
Strategy templates are stored in `prompt_templates/all_combined.json`:
```json
{
  "strategies": {
    "baseline": {
      "name": "baseline", 
      "description": "Baseline classification approach",
      "system_prompt": "You are a content moderation assistant...",
      "user_template": "Classify the following text: \"{text}\"",
      "parameters": {
        "max_tokens": 512,
        "temperature": 0.1,
        "response_format": "json_object"
      }
    }
  }
}
```
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
export AZURE_INFERENCE_SDK_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
export AZURE_INFERENCE_SDK_KEY="your-azure-ai-key"
```

**Windows PowerShell:**
```powershell
$env:AZURE_INFERENCE_SDK_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
$env:AZURE_INFERENCE_SDK_KEY="your-azure-ai-key"
```

### 2. Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### 3. Test Connection

```bash
cd prompt_engineering
python runner.py --test-connection
```

## Usage

### **Available Commands**

#### Connection Testing Only
```bash
python prompt_runner.py --test-connection
```

#### Quick Strategy Testing (Canned Samples)
```bash
# Test specific strategy with canned samples
python prompt_runner.py --dataset-type canned --num-samples 2 --strategy baseline
python prompt_runner.py --dataset-type canned --num-samples 5 --strategy persona
python prompt_runner.py --dataset-type canned --num-samples all --strategy policy
```

#### Comprehensive Strategy Evaluation (Unified Dataset)
```bash
# Test all strategies with unified dataset
python prompt_runner.py --dataset-type unified --num-samples 25 --strategy all
python prompt_runner.py --dataset-type unified --num-samples 100 --strategy baseline policy
```

#### Combined Strategy Testing
```bash
# Test specific strategies on different datasets
python prompt_runner.py --dataset-type canned --num-samples all --strategy combined
python prompt_runner.py --dataset-type unified --num-samples 50 --strategy persona combined
```

### **Output Files**

All test runs generate timestamped files in `outputs/`:

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
prompt_engineering/
├── prompts_validator.py              # Main validator (connection + validation logic)
├── strategy_templates_loader.py      # Prompt strategy loader (loads from JSON)
├── evaluation_metrics_calc.py        # Metrics calculation and result structures
├── persistence_helper.py             # Output file management and saving
├── azureai_mi_connector_wrapper.py   # Azure AI SDK connection wrapper
├── unified_dataset_loader.py         # Flexible dataset loading (canned + unified)
├── prompt_runner.py                  # CLI interface with unified dataset support
├── prompt_templates/
│   └── all_combined.json             # Strategy configuration (all 4 strategies)
├── data_samples/
│   └── canned_basic_all.json         # Test samples in unified format
├── outputs/                          # Generated result files
├── README.md                         # This file
├── STRATEGY_TEST_RESULTS.md          # Test results documentation
└── DEBUG.md                          # Debugging guide
```

### **Flexible Dataset Loading**

The system supports two dataset types:

1. **Canned Dataset** (`data_samples/canned_basic_all.json`):
   - 5 carefully curated samples
   - Covers hate/normal labels for LGBTQ, Mexican, and Middle Eastern groups
   - Perfect for quick testing and validation

2. **Unified Dataset** (from `../../../data/processed/unified/unified_test.json`):
   - Full unified test dataset with 12,589+ samples
   - Filtered to LGBTQ, Mexican, and Middle East target groups
   - Comprehensive evaluation capability

### **CLI Arguments**

```bash
python prompt_runner.py --dataset-type [canned|unified] --num-samples [N|all] --strategy [baseline|policy|persona|combined|all]
```
## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_INFERENCE_SDK_ENDPOINT` | Yes | Azure AI Inference endpoint | `https://your-endpoint.services.ai.azure.com/models` |
| `AZURE_INFERENCE_SDK_KEY` | Yes | Azure AI authentication key | `your-32-character-key` |

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

### **Strategy Configuration**

Each strategy is defined in `strategy_templates.json` with configurable model parameters:

```json
{
  "strategies": {
    "baseline": {
      "name": "baseline",
      "description": "...",
      "system_prompt": "...",
      "user_template": "...",
      "parameters": {
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "response_format": "json_object"
      }
    }
  }
}
```

**Configurable Parameters:**
- `max_tokens`: Maximum response length (default: 512)
- `temperature`: Response randomness (0.0-2.0, default: 0.1)
- `top_p`: Nucleus sampling threshold (0.0-1.0, default: 1.0)
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0, default: 0.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0, default: 0.0)
- `response_format`: Response format ("json_object" or null, default: "json_object")

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
from prompts_validator import PromptValidator

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

- Verify environment variables are set (`AZURE_INFERENCE_SDK_ENDPOINT`, `AZURE_INFERENCE_SDK_KEY`)
- Check Azure AI endpoint accessibility
- Validate authentication key format
- Test with `python prompt_runner.py --test-connection`

### Import Errors

- Install all dependencies: `pip install -r requirements.txt`
- Verify Python environment is activated
- Check file paths and module structure

### Performance Issues

- Monitor response times in output files
- Consider adjusting model parameters in strategy templates
- Review prompt complexity for timeout issues

## Recent Improvements

### Directory Reorganization (September 2025)
- **Moved strategy templates**: `strategy_templates.json` → `prompt_templates/all_combined.json`
- **Moved test samples**: `canned_samples.json` → `data_samples/canned_basic_all.json`  
- **Renamed output directory**: `validation_outputs/` → `outputs/`
- **Updated all code references**: Complete refactoring to use new paths

### Metrics Calculation Fix
- **Fixed None prediction handling**: Robust metrics calculation that filters out None predictions
- **Improved error handling**: Better fallback mechanisms for failed JSON parsing
- **Enhanced debugging**: Clear error messages for "Fallback due to JSON parsing error"

### Enhanced CLI Interface  
- **Unified dataset support**: Choose between canned samples and full unified dataset
- **Flexible sample sizes**: Support for specific counts or "all" samples
- **Strategy selection**: Test individual strategies or all together
- **Clean argument structure**: `--dataset-type`, `--num-samples`, `--strategy`

### System Reliability
- **Modular design**: Separated concerns across multiple specialized modules
- **Comprehensive testing**: All components tested and validated
- **Production ready**: Robust error handling and logging throughout