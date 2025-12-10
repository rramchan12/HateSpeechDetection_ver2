# Hate Speech Detection Prompt Validation Framework

A comprehensive, production-ready prompt validation framework for testing and evaluating different hate speech detection strategies with multiple AI models. Built with a modular package architecture featuring YAML-based configuration, incremental storage, detailed analytics, and flexible data source management.

## Overview

This framework provides:

- **Modular Package Architecture**: Organized into `connector`, `loaders`, and `metrics` packages
- **Multi-Model Support**: Switch between configured models via YAML configuration
- **Strategy Testing**: Five prompt strategies (Baseline, Policy, Persona, Combined, Enhanced Combined)
- **Flexible Data Sources**: Unified dataset sampling and canned datasets
- **Concurrent Processing**: Multi-threaded execution with configurable settings
- **Performance Analytics**: Accuracy, timing, and response analysis with runId organization
- **YAML Configuration**: Model configuration with environment variable support

## Main Entry Point

**Use `prompt_runner.py` as the primary CLI interface:**

```bash
# Get help and see all available options
python prompt_runner.py --help

# Basic validation with canned dataset
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Run all strategies on specific canned dataset with sampling
python prompt_runner.py --data-source canned_100_stratified --strategies all --sample-size 10

# Extract samples from unified dataset
python prompt_runner.py --data-source unified --sample-size 50 --strategies policy persona

# Test connection to Azure AI endpoint
python prompt_runner.py --test-connection

# Recalculate metrics from previous run
python prompt_runner.py --metrics-only --run-id run_20250920_015821

# Use custom prompt template file
python prompt_runner.py --prompt-template-file custom_prompts.json --strategies baseline policy

# Concurrent processing with custom settings
python prompt_runner.py --data-source unified --sample-size 100 --strategies all --max-workers 10 --batch-size 20
```

## Data Sources

### **Flexible Dataset Selection**

- **Unified Dataset**: `--data-source unified` - Large comprehensive dataset with configurable sampling
- **Canned Datasets**: Curated, ready-to-use test sets:
  - `--data-source canned_50_quick` - Quick test samples (50 samples)
  - `--data-source canned_100_size_varied` - Size-varied text samples (100 samples)
  - `--data-source canned_100_stratified` - Diverse stratified samples (100 samples)

### **Sample Size Control**

The `--sample-size` parameter now works with ALL data sources:

```bash
# Sample 25 items from unified dataset
python prompt_runner.py --data-source unified --sample-size 25 --strategies baseline

# Sample 5 items from canned dataset  
python prompt_runner.py --data-source canned_100_stratified --sample-size 5 --strategies all

# Use full canned dataset (omit --sample-size)
python prompt_runner.py --data-source canned_50_quick --strategies policy
```

## Package Usage

### **Direct Package Imports**

The framework is organized into specialized packages that can be used independently:

```python
# Import from main package (recommended)
from prompt_engineering import StrategyTemplatesLoader, EvaluationMetrics, AzureAIConnector

# Import from specific packages (for advanced usage)
from prompt_engineering.loaders import UnifiedDatasetLoader, load_dataset
from prompt_engineering.metrics import PersistenceHelper, ValidationResult
from prompt_engineering.connector import AzureAIConnector, ModelConfigLoader

# Example: Load and analyze data programmatically
loader = UnifiedDatasetLoader()
samples = loader.load_samples("unified", num_samples=10)

# Example: Calculate metrics from stored results  
from prompt_engineering.metrics import calculate_metrics_from_runid
results = calculate_metrics_from_runid("run_20251008_210824")
```

### **Package Structure Benefits**

- **Modular Development**: Each package can be developed and tested independently
- **Clean Imports**: Import only what you need for specific functionality
- **Standard Conventions**: Follows Python packaging best practices
- **Easy Extension**: Add new functionality to appropriate packages without affecting others

## Configuration

### **YAML-Based Model Configuration**

Models are configured in `model_connection.yaml` with support for environment variables:

```yaml
models:
  gpt-oss-20b:
    model_name: "GPT-OSS-20B"
    endpoint: "https://your-endpoint.azure.com/models"
    model_deployment: "gpt-oss-120b"
    provider: "azure_ai_inference"
    description: "Azure AI hosted GPT-OSS-20B model for hate speech detection"
    default_parameters:
      max_tokens: 512
      temperature: 0.1
      response_format: "json_object"
  
  gpt-5:
    model_name: "GPT-5"
    endpoint: "${GPT5_ENDPOINT}"  # Environment variable
    api_key: "${GPT5_API_KEY}"    # Environment variable
    model_deployment: "gpt-5"
    provider: "azure_ai_inference"
    description: "Azure AI hosted GPT-5 model for advanced text analysis"
    default_parameters:
      max_tokens: 1024
      temperature: 0.2
      response_format: "json_object"
```

### **Environment Variables**

Set these environment variables or update the YAML file directly:

- `AZURE_AI_ENDPOINT`: Default Azure AI endpoint (for GPT-OSS-20B)
- `AZURE_AI_KEY`: Default Azure AI key (for GPT-OSS-20B)
- `GPT5_ENDPOINT`: GPT-5 specific endpoint
- `GPT5_API_KEY`: GPT-5 specific API key

## Features

### **Multiple Prompt Strategies (JSON-Configurable)**

All strategies are loaded from `prompt_templates/all_combined.json` or `prompt_templates/combined/combined_v5.json` for easy customization and enforce structured JSON response format with rationale.

**Available Strategies**:
1. **Baseline**: Simple classification without additional context
2. **Policy**: Platform guidelines-based content moderation  
3. **Persona**: Multi-perspective evaluation with role-based analysis
4. **Combined**: Policy + persona approach with comprehensive bias detection
5. **Enhanced Combined**: Advanced fusion with enhanced reasoning
6. **V5 Noise-Reduced Strategies** (5 variants): Compressed prompts (60-90 words) with pattern demonstration
   - `combined_v5_implicit_examples`: Contrasting examples without explanations
   - `combined_v5_chain_of_thought`: 4-step structured reasoning
   - `combined_v5_compressed_tokens`: Token-efficient policy
   - `combined_v5_minimal_signal`: Ultra-minimal approach
   - `combined_v5_example_only`: Pure examples without framing

**For Performance Results**: See `prompt_templates/combined/gpt_oss_combined_ift_5iter_summary.md` for comprehensive 5-iteration analysis and performance metrics.

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

### **Incremental Storage and Performance Analytics**

- Response time analysis
- Error pattern identification  
- Confusion matrix and F1-score calculations
- Clean, professional terminal output without emojis
- Memory-efficient incremental CSV writing during validation
- Metrics calculation from stored results, not in-memory data

### **Output Files Generated (runId-based Organization)**

All outputs are organized in unique timestamped folders (`outputs/run_YYYYMMDD_HHMMSS/`):

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
python prompt_runner.py --test-connection
```

## Usage

### **Available Commands**

#### Connection Testing and Metrics Only

```bash
# Test model connection
python prompt_runner.py --test-connection

# Recalculate metrics from previous run  
python prompt_runner.py --metrics-only --run-id run_20250920_015821
```

#### Quick Strategy Testing (Canned Samples)

```bash
# Test specific strategy with canned samples
python prompt_runner.py --data-source canned_basic_all --strategies baseline

# Test multiple strategies with sampling
python prompt_runner.py --data-source canned_100_all --strategies baseline policy --sample-size 5
```

#### Comprehensive Strategy Evaluation (Unified Dataset)

```bash
# Test all strategies with unified dataset
python prompt_runner.py --data-source unified --strategies all --sample-size 50

# Use specific strategies with custom sampling
python prompt_runner.py --data-source unified --strategies policy persona --sample-size 25
```

#### Advanced Usage

```bash
# Use different model configuration
python prompt_runner.py --model gpt-5 --data-source unified --strategies baseline

# Enable debug logging with file output in runID folder
python prompt_runner.py --debug --data-source canned_basic_all --strategies baseline

# Custom random seed for reproducible sampling
python prompt_runner.py --data-source unified --sample-size 20 --random-seed 42 --strategies all

# High-performance concurrent processing
python prompt_runner.py --data-source unified --sample-size 500 --strategies all --max-workers 15 --batch-size 25

# Custom prompt template with sequential processing
python prompt_runner.py --prompt-template-file experimental.json --data-source canned_100_all --strategies policy --sequential
```

### **CLI Arguments**

- `--data-source`: Choose data source (`unified`, `canned_basic_all`, `canned_100_all`)
- `--strategies`: Select strategies (`baseline`, `policy`, `persona`, `combined`, `all`)
- `--sample-size`: Number of samples to process (optional, uses full dataset if omitted)
- `--model`: Model configuration to use (default: `gpt-oss-20b`)
- `--prompt-template-file`: Prompt template file to use (default: `all_combined.json`)
- `--test-connection`: Test Azure AI endpoint connectivity
- `--metrics-only`: Recalculate metrics from existing results
- `--run-id`: Specify runId for metrics recalculation
- `--random-seed`: Seed for reproducible sampling (default: 42)
- `--debug`: Enable detailed debug logging
- `--max-workers`: Maximum concurrent threads for processing (default: 5)
- `--batch-size`: Number of samples per batch in concurrent processing (default: 10)
- `--sequential`: Force sequential processing instead of concurrent (slower but more reliable)

### **Output Files**

All test runs generate files in timestamped runId directories (`outputs/run_YYYYMMDD_HHMMSS/`):

1. **`validation_results.csv`** (or `strategy_unified_results_[timestamp].csv`)
   - Individual sample results for each strategy
   - Columns: strategy, sample_id, text, true_label, predicted_label, response_time, rationale

2. **`test_samples_[timestamp].csv`**
   - Test samples used for validation
   - Columns: text, label_binary, target_group_norm, persona_tag, source_dataset

3. **`performance_metrics_[timestamp].csv`**
   - Combined accuracy scores and confusion matrices for all strategies
   - Contains: strategy, accuracy, precision, recall, f1_score, true_positive, true_negative, false_positive, false_negative

4. **`evaluation_report_[timestamp].txt`**
   - Human-readable summary report with test samples and strategy performance analysis
   - Includes model metadata, prompt template file, data source, and command line used
   - Shows important sample predictions with clear indicators
   - Comprehensive execution details and performance metrics

5. **`validation_log_[timestamp].log`**
   - Complete execution log with debug information
   - Rate limiting monitoring and performance metrics
   - Azure AI request/response details and timing
   - Error handling and retry logic status

## Project Structure

```text
prompt_engineering/
├── prompt_runner.py                  # Main CLI orchestrator (primary entry point)
├── dataset_sampler.py                # Dataset sampling utilities
├── connector/                        # Azure AI connection package
│   ├── __init__.py                   # Package initialization with exports
│   ├── azureai_connector.py          # Azure AI SDK connection wrapper
│   ├── model_connection.yaml         # YAML-based model configuration
│   └── README.md                     # Connector package documentation
├── loaders/                          # Data and template loading utilities package
│   ├── __init__.py                   # Package initialization with exports
│   ├── strategy_templates_loader.py  # Strategy template management and loading
│   └── unified_dataset_loader.py     # Flexible dataset loading (canned + unified)
├── metrics/                          # Evaluation and persistence utilities package
│   ├── __init__.py                   # Package initialization with exports
│   ├── evaluation_metrics_calc.py    # Metrics calculation and result structures
│   ├── persistence_helper.py         # Output file management and saving
│   └── outputs/                      # Package-specific output directory
├── prompt_templates/                 # Strategy configuration files
│   ├── all_combined.json             # Main strategy configuration (5 strategies)
│   ├── baseline_v1.json              # Legacy baseline strategy
│   └── baseline_v1_README.md         # Legacy baseline documentation
├── data_samples/                     # Test datasets and samples
│   ├── canned_50_quick.json          # Quick test samples (50 samples)
│   ├── canned_100_size_varied.json   # Size-varied samples (100 samples)
│   ├── canned_100_stratified.json    # Stratified diverse samples (100 samples)
│   └── README.md                     # Data samples documentation
├── outputs/                          # Generated result files (organized by runId)
│   ├── run_20251008_210824/          # Example run with results
│   └── run_20251008_213409/          # Another example run
├── __pycache__/                      # Python compiled bytecode cache
├── README.md                         # This file (framework documentation)
└── DEBUG.md                          # Debugging and troubleshooting guide
```

## Architecture Highlights

### **Modular Package Design**

- **Main Entry Point**: `prompt_runner.py` - CLI orchestrator coordinating all packages with concurrent processing
- **Connector Package** (`connector/`): Azure AI SDK integration with YAML configuration
  - `AzureAIConnector`: Model connection management with environment variable support
  - `ModelConfigLoader`: YAML-based model configuration loading
- **Loaders Package** (`loaders/`): Data and template management utilities
  - `StrategyTemplatesLoader`: Encapsulates strategy template logic and prompt extraction
  - `UnifiedDatasetLoader`: Flexible data loading with sample size support for all sources
- **Metrics Package** (`metrics/`): Evaluation and persistence functionality
  - `EvaluationMetrics`: Comprehensive metrics calculation from stored results
  - `PersistenceHelper`: File I/O and output organization in runId folders with incremental storage

### **Concurrent Processing & Performance**

- Multi-threaded execution with configurable worker pools and batch sizes
- Intelligent rate limiting with exponential backoff and retry logic
- Real-time progress monitoring and performance metrics
- Memory-efficient incremental storage for large datasets
- Adaptive throttling detection and response time monitoring

### **Incremental Storage**

- Results are written incrementally during validation to avoid memory issues
- Each sample result is immediately saved to CSV
- Metrics are calculated from stored results, not in-memory data
- Memory-efficient for large dataset processing with thousands of samples

### **File Logging & Audit Trail**

- Complete execution logs written to runID folders
- Rate limit header monitoring and Azure AI request tracking
- Detailed error handling and retry logic status
- No console output except runID for clean CI/CD integration

### **Robust Error Handling**

- Connection validation before processing
- Graceful handling of JSON parsing failures with fallback mechanisms
- Comprehensive logging with debug mode support
- Critical failures log and exit cleanly

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_INFERENCE_SDK_ENDPOINT` | Yes | Azure AI Inference endpoint | `https://your-endpoint.services.ai.azure.com/models` |
| `AZURE_INFERENCE_SDK_KEY` | Yes | Azure AI authentication key | `your-32-character-key` |
| `GPT5_ENDPOINT` | No | GPT-5 specific endpoint | `https://gpt5-endpoint.azure.com/models` |
| `GPT5_API_KEY` | No | GPT-5 specific API key | `gpt5-specific-key` |

## CLI Reference

### Main Commands

```bash
# Test connection
python prompt_runner.py --test-connection

# Quick validation with canned data
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Comprehensive evaluation
python prompt_runner.py --data-source unified --strategies all --sample-size 100

# Recalculate metrics from existing run
python prompt_runner.py --metrics-only --run-id run_20250920_015821
```

### Strategy Options

**Available Strategies**:
- `baseline`: Direct hate speech classification
- `policy`: Platform guidelines-based moderation
- `persona`: Multi-perspective evaluation with bias awareness
- `combined`: Policy + persona fusion approach
- `enhanced_combined`: Advanced fusion with enhanced reasoning
- `combined_v5_implicit_examples`: V5 pattern demonstration (contrasting examples)
- `combined_v5_chain_of_thought`: V5 4-step structured reasoning
- `combined_v5_compressed_tokens`: V5 token-efficient policy
- `combined_v5_minimal_signal`: V5 ultra-minimal approach
- `combined_v5_example_only`: V5 pure examples (no framing)
- `all`: Execute all available strategies

**Usage Examples**:
```bash
# Test specific strategy
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Test multiple strategies
python prompt_runner.py --data-source unified --sample-size 50 --strategies baseline policy combined_v5_implicit_examples

# Test all strategies
python prompt_runner.py --data-source unified --sample-size 100 --strategies all
```

**For Performance Comparison**: See `prompt_templates/combined/gpt_oss_combined_ift_5iter_summary.md` for detailed strategy performance analysis.

### Data Source Options

- `unified`: Large unified dataset with configurable sampling
- `canned_50_quick`: 50 quick test samples for rapid validation
- `canned_100_size_varied`: 100 samples with varied text lengths
- `canned_100_stratified`: 100 diverse stratified samples for comprehensive testing

## Development

### Adding New Strategies

1. Edit `prompt_templates/all_combined.json`
2. Add new strategy configuration following existing patterns
3. Test with `python prompt_runner.py --data-source canned_50_quick --strategies <new_strategy>`
4. Validate with comprehensive evaluation

### Extending Metrics

1. Edit `metrics/evaluation_metrics_calc.py`
2. Add new calculation methods to `EvaluationMetrics` class
3. Update `PerformanceMetrics` dataclass if needed
4. Test with existing runs using `--metrics-only` flag

### Adding New Models

1. Edit `connector/model_connection.yaml`
2. Add new model configuration with required parameters
3. Test connection with `python prompt_runner.py --model <new_model> --test-connection`
4. Validate with strategy testing

### Package Development

1. **Connector Package**: Add new model providers or connection types in `connector/`
2. **Loaders Package**: Extend data sources or template formats in `loaders/`
3. **Metrics Package**: Add new evaluation metrics or output formats in `metrics/`
4. Each package has its own `__init__.py` with proper exports for clean imports

## Results Documentation

For comprehensive performance analysis and iteration history:

- **5-Iteration Summary**: `prompt_templates/combined/gpt_oss_combined_ift_5iter_summary.md`
  - Complete performance timeline across all iterations
  - All strategies tested and ranked with metrics
  - Key findings and production recommendations
  
- **Policy-Persona Analysis**: `prompt_templates/combined/gpt_oss_iter5_policy_persona_coverage.md`
  - Detailed analysis of V5 noise-reduction approach
  - Explicit vs implicit encoding comparison
  - Information density and signal-to-noise analysis

- **V5 Templates**: `prompt_templates/combined/combined_v5.json`
  - Production-ready configurations for noise-reduced strategies

- **Results Data**: `outputs/combined_v5/gptoss/production/run_20251102_191102/`
  - Complete validation and production run results
  - Confusion matrices, bias analysis, and detailed metrics

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
- Use smaller sample sizes for testing
- Enable debug logging to identify bottlenecks

### Metrics Calculation Issues

- Ensure runId exists in outputs directory
- Verify `validation_results.csv` contains valid data
- Check for None predictions in results
- Use `--debug` flag for detailed error information

## Recent Improvements

### **Performance & Concurrency (September 2025)**

- **Concurrent Processing**: Multi-threaded execution with configurable worker pools and batch sizes
- **Rate Limiting**: Intelligent retry with exponential backoff and rate limit detection
- **Performance Monitoring**: Real-time progress tracking and response time analysis
- **Adaptive Throttling**: Automatic detection and handling of API rate limits

### **Logging & Audit Trail**

- **RunID-based Logging**: All logs written to timestamped runID folders
- **Azure AI Monitoring**: Rate limit headers and request/response tracking
- **Clean Console Output**: Only runID printed to console for CI/CD integration
- **Complete Audit Trail**: Full execution logs with detailed error handling

### **Modular Package Architecture (October 2025)**

- **Package Organization**: Separated into specialized packages for better maintainability
- **Connector Package**: Azure AI integration with YAML configuration management
- **Loaders Package**: Data and template loading utilities with proper path handling
- **Metrics Package**: Evaluation metrics and persistence functionality
- **Clean Imports**: Each package exports classes via `__init__.py` for easy importing
- **Standard Python Conventions**: Follows established packaging patterns for professional development

### **Incremental Storage and Memory Efficiency**

- **Incremental CSV Writing**: Results saved during validation, not at the end
- **Memory Optimization**: No in-memory accumulation of large result sets
- **Metrics from Storage**: Calculate metrics from saved CSV files, not memory
- **Separation of Concerns**: File I/O in PersistenceHelper, metrics in EvaluationMetricsCalc within dedicated packages

### **Enhanced CLI and Data Source Support**

- **Universal Sample Size**: `--sample-size` works with all data sources (unified and canned)
- **Canned Dataset Selection**: Pick specific canned files by name
- **Strategy Selection**: Use `--strategies all` to execute all strategies
- **Metrics Recalculation**: Operate on stored results with `--metrics-only` and `--run-id`
- **Custom Prompt Templates**: CLI argument for selecting different template files

### **Rich Evaluation Reports**

- **Metadata Integration**: Model name, prompt template file, data source, and command line included
- **Concise Sample Display**: Only important predictions shown for readability
- **Execution Context**: Complete configuration and performance details
- **Professional Formatting**: Clean, structured output for analysis and documentation

### **Architecture Refactoring**

- **Modular Package Structure**: Organized into `connector`, `loaders`, and `metrics` packages
- **Single Orchestrator**: `prompt_runner.py` is the only entry point
- **Clean Package Separation**: Each package handles specific functionality with proper `__init__.py` exports
- **Encapsulated Strategy Logic**: All template logic in `StrategyTemplatesLoader` class within `loaders` package
- **Robust Error Handling**: Critical failures log and exit cleanly
- **YAML Configuration**: Multi-model support with environment variable substitution in `connector` package
- **Standard Python Conventions**: Follows established packaging patterns for better maintainability

### **Output Organization**

- **runId-based Folders**: All outputs organized in timestamped directories
- **Consistent File Names**: Standardized naming across all output files
- **Complete Documentation**: Updated README, DEBUG.md, and STRATEGY_TEST_RESULTS.md
