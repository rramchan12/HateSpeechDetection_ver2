# Prompt Validator for GPT-OSS-20B

A comprehensive prompt validation framework for testing and evaluating different prompt strategies with GPT-OSS-20B, featuring detailed analytics and performance comparison.

## Overview

This validator provides:
- ✅ **Connection Testing**: Validate Azure AI endpoint connectivity
- 🧠 **Strategy Testing**: Four different prompt strategies (Baseline, Policy, Persona, Combined)
- 📊 **Performance Analytics**: Detailed accuracy, timing, and response analysis
- 💾 **Results Storage**: Comprehensive output files with timestamped results
- 📋 **Clean Output**: Professional logging and reporting
- 🔧 **Modular Design**: Separate components for easy extension and customization

## Features

### **Four Prompt Strategies**
1. **Baseline**: Simple classification without additional context
2. **Policy**: Platform guidelines-based content moderation
3. **Persona**: Multi-perspective evaluation with role-based analysis
4. **Combined**: Policy + persona approach with comprehensive bias detection

### **Comprehensive Analytics**
- Individual sample performance tracking
- Strategy comparison with accuracy metrics
- Response time analysis
- Error pattern identification
- Visual indicators for correct/incorrect predictions

### **Output Files Generated**
- **Detailed Results CSV**: Individual sample results with predictions and response times
- **Summary Metrics CSV**: Strategy performance comparison table
- **Detailed Report TXT**: Human-readable analysis with test samples and individual results

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

#### Strategy Performance Testing
```bash
# Test all 4 strategies with 5 samples and generate detailed reports
python runner.py --test-strategies
```

#### Validation Demo
```bash
# Run comprehensive validation demonstration
python runner.py --demo --sample-size 10
```

#### Default Mode (Connection + Demo)
```bash
# Run both connection test and validation demo
python runner.py
```

### **Output Files**

All test runs generate timestamped files in `validation_outputs/`:

1. **`strategy_test_results_TIMESTAMP.csv`**
   - Individual sample results for each strategy
   - Columns: timestamp, strategy, sample_id, input_text, predicted_label, true_label, response_time, response_text, correct

2. **`strategy_summary_TIMESTAMP.csv`**
   - Strategy performance comparison
   - Columns: strategy, accuracy, correct_predictions, total_samples, avg_response_time

3. **`strategy_report_TIMESTAMP.txt`**
   - Human-readable detailed report
   - Test samples with true labels
   - Individual prediction results with ✓/✗ indicators
   - Strategy performance summary and best performer identification

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

## Features

### **Current Implementation** ✅

- **Connection Validation**: Verify GPT-OSS-20B model connectivity and environment setup
- **Strategy Testing**: Comprehensive evaluation of all 4 prompt strategies (Policy, Persona, Combined, Baseline)
- **Performance Analytics**: Accuracy, prediction correctness, and response time analysis
- **Output File Generation**: Timestamped CSV and TXT reports for all test runs
- **Interactive CLI**: Command-line interface with multiple testing options and demos
- **Sample Data Integration**: 5 test samples from unified hate speech dataset with true labels

### **Supported Strategies**

1. **Baseline**: Direct hate speech classification without specialized prompting
2. **Persona**: Persona-based prompting incorporating target group identity and bias awareness  
3. **Policy**: Policy-based prompting with platform community standards and hate speech definitions
4. **Combined**: Fusion of persona and policy strategies for comprehensive bias-aware classification

## Scaffolding Features

### Current Implementation ✅
- Connection validation to GPT-OSS-20B
- Four prompt strategy templates (Policy, Persona, Combined, Baseline)
- Interactive CLI with demonstration and testing capabilities
- Clean, emoji-free output for professional validation

### Future Development 🏗️
- **Strategy Validation**: Policy, Persona, Combined, Baseline
- **Comprehensive Analytics**: Performance comparison across all strategies
- **Evaluation Metrics**: Accuracy, response time, and bias detection
- **Batch Processing**: Large-scale validation with the unified hate speech dataset
- **Results Documentation**: Automated reporting and visualization
- **Configuration Management**: Flexible model and endpoint configuration

## API Reference

### Core Classes

#### PromptValidator

```python
from core_validator import PromptValidator

# Initialize with environment variables
validator = PromptValidator()

# Test connection
is_connected = validator.test_connection()

# Validate with different strategies
result = validator.validate_with_strategy("Sample text", "baseline")
result = validator.validate_with_strategy("Sample text", "persona")
result = validator.validate_with_strategy("Sample text", "policy")
result = validator.validate_with_strategy("Sample text", "combined")
```

#### Strategy Templates

```python
from strategy_templates import StrategyTemplates

templates = StrategyTemplates()

# Get formatted prompts
baseline_prompt = templates.get_baseline_prompt("Sample text")
persona_prompt = templates.get_persona_prompt("Sample text")
policy_prompt = templates.get_policy_prompt("Sample text")
combined_prompt = templates.get_combined_prompt("Sample text")
```

#### Evaluation Metrics

```python
from evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()

# Calculate performance metrics
accuracy = metrics.calculate_accuracy(predictions, true_labels)
response_times = metrics.calculate_response_times(start_times, end_times)
```

## Development

### Adding New Strategies

1. Edit `strategy_templates.py`
2. Add new prompt method following existing patterns
3. Update `core_validator.py` to support new strategy
4. Test with `python runner.py --test-strategies`

### Extending Metrics

1. Edit `evaluation_metrics.py`
2. Add new calculation methods
3. Update runner to use new metrics
4. Validate with test data

### Testing

```bash
# Run comprehensive strategy testing
python runner.py --test-strategies

# Test specific functionality
python runner.py --test-connection
python runner.py --demo
```

## Troubleshooting

### Connection Issues

- ✅ Verify environment variables are set (`AZURE_AI_ENDPOINT`, `AZURE_AI_KEY`)
- ✅ Check Azure AI endpoint accessibility
- ✅ Validate authentication key format
- ✅ Test with `python runner.py --test-connection`

### Import Errors

- ✅ Install all dependencies: `pip install -r requirements.txt`
- ✅ Verify Python environment is activated
- ✅ Check file paths and module structure

### Performance Issues

- ✅ Monitor response times in output files
- ✅ Consider adjusting model parameters
- ✅ Review prompt complexity for timeout issues