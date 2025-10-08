# Debug Guide for Hate Speech Detection Framework

## Simple VS Code Debugging Setup

### How It Works

1. Open├── connector/                        # Azure AI connection package
│   ├── azureai_connector.py          # Azure AI - debug connection issues
│   └── model_connection.yaml         # Config - check YAML structureprompt_runner.py` (the main entry point)
2. Click line number to set breakpoints where needed
3. Press `F5` - VS Code will debug the file you have open
4. When prompted, enter your CLI arguments

### That's It

- **No complex configuration needed**
- **No need to map every method**  
- **Just open the file you want to debug and press F5**

## Quick Debug Examples

### Basic Testing

```bash
# Debug basic canned dataset validation
# Open prompt_runner.py, press F5, enter: --data-source canned_basic_all --strategies baseline

# Debug all strategies with sampling
# Open prompt_runner.py, press F5, enter: --data-source canned_100_all --strategies all --sample-size 5

# Debug unified dataset sampling
# Open prompt_runner.py, press F5, enter: --data-source unified --sample-size 10 --strategies policy
```

### Connection and Metrics Testing

```bash
# Debug connection issues
# Open prompt_runner.py, press F5, enter: --test-connection

# Debug metrics recalculation
# Open prompt_runner.py, press F5, enter: --metrics-only --run-id run_20250920_015821
```

## Common Debug Scenarios

### 1. JSON Parsing Issues

**Problem**: "Fallback due to JSON parsing error" messages

**Debug Steps**:

1. Open `prompt_runner.py`
2. Set breakpoint in `run_validation` method where model responses are processed
3. Run with problematic sample
4. Inspect the raw `model_response` to see malformed JSON
5. Check if model is returning proper JSON format

### 2. Strategy Loading Problems

**Problem**: Strategy templates not loading correctly

**Debug Steps**:
1. Open `loaders/strategy_templates_loader.py`
2. Set breakpoint in `load_all_strategy_templates` method
3. Inspect `self.templates_path` and verify JSON file exists
4. Check JSON structure matches expected format

### 3. Dataset Loading Issues

**Problem**: Data sources not loading or sampling issues

**Debug Steps**:
1. Open `loaders/unified_dataset_loader.py`
2. Set breakpoint in `load_dataset_by_filename` method
3. Check file paths and data structure
4. Verify sample_size logic for different data sources

### 4. Metrics Calculation Errors

**Problem**: Metrics calculation fails or returns unexpected results

**Debug Steps**:
1. Open `metrics/evaluation_metrics_calc.py`
2. Set breakpoint in `calculate_metrics_from_runid` method
3. Verify CSV file exists and has correct structure
4. Check data type conversions and None handling

### 5. File Output Issues

**Problem**: Output files not generated or incorrect format

**Debug Steps**:
1. Open `metrics/persistence_helper.py`
2. Set breakpoint in `save_validation_results` or related methods
3. Check runId generation and folder creation
4. Verify CSV writing and file permissions

### 6. Model Connection Problems

**Problem**: Azure AI connection failures

**Debug Steps**:
1. Open `connector/azureai_connector.py`
2. Set breakpoint in `test_connection` method
3. Check environment variables and YAML configuration
4. Verify endpoint URLs and authentication

## Architecture-Specific Debugging

### Current File Structure for Debugging

```text
prompt_engineering/
├── prompt_runner.py                  # Main CLI - primary debug entry point
├── connector/                        # Azure AI connection package
│   ├── azureai_connector.py          # Azure AI - debug connection issues
│   └── model_connection.yaml         # Config - check YAML structure  
├── loaders/                          # Data and template loading utilities
│   ├── strategy_templates_loader.py  # Strategy management - debug template loading
│   └── unified_dataset_loader.py     # Data loading - debug sampling and file access
├── metrics/                          # Evaluation and persistence utilities
│   ├── evaluation_metrics_calc.py    # Metrics - debug calculation logic
│   └── persistence_helper.py         # File I/O - debug output generation
```

### Key Debug Points by Component

#### PromptRunner (prompt_runner.py)
- **CLI argument parsing**: Check argument validation and default values
- **Orchestration flow**: Verify component initialization and method calls
- **Error handling**: Check exception catching and logging
- **runId generation**: Verify timestamp format and folder creation

#### StrategyTemplatesLoader (loaders/strategy_templates_loader.py)
- **JSON loading**: Verify template file parsing
- **Strategy extraction**: Check individual strategy access
- **Prompt building**: Verify SystemMessage and UserMessage creation

#### UnifiedDatasetLoader (loaders/unified_dataset_loader.py)
- **File resolution**: Check path construction for different data sources
- **Sampling logic**: Verify random sampling and seed handling
- **Data filtering**: Check sample size application

#### EvaluationMetricsCalc (metrics/evaluation_metrics_calc.py)
- **CSV reading**: Verify result file parsing
- **Metrics calculation**: Check accuracy, precision, recall computations
- **Data cleaning**: Verify None/invalid prediction handling

#### PersistenceHelper (metrics/persistence_helper.py)
- **Folder creation**: Check runId directory generation
- **CSV writing**: Verify incremental result saving
- **File formatting**: Check column headers and data types

## Debug Environment Variables

Set these for debugging Azure AI connection issues:

```bash
# Windows PowerShell
$env:AZURE_INFERENCE_SDK_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
$env:AZURE_INFERENCE_SDK_KEY="your-azure-ai-key"

# Linux/Mac bash
export AZURE_INFERENCE_SDK_ENDPOINT="https://your-endpoint.services.ai.azure.com/models"
export AZURE_INFERENCE_SDK_KEY="your-azure-ai-key"
```

## Performance Debugging

### Memory Usage

Monitor memory with large datasets:
1. Set breakpoints before and after dataset loading
2. Check memory usage in VS Code's debug console
3. Verify incremental CSV writing is working

### Response Time Analysis

Debug slow model responses:
1. Set breakpoints around model inference calls in `prompt_runner.py`
2. Time individual operations
3. Check if timeout issues occur

## Common Error Patterns

### 1. "FileNotFoundError" for Data Sources

**Cause**: Incorrect path resolution for canned files or unified dataset

**Solution**: 
- Check `loaders/unified_dataset_loader.py` path construction
- Verify file existence in `data_samples/` or `../data/processed/unified/`

### 2. "KeyError" in Strategy Templates

**Cause**: Strategy name not found in `all_combined.json`

**Solution**:
- Verify strategy names match JSON keys exactly
- Check for typos in CLI arguments vs. template file

### 3. "ValueError" in Metrics Calculation

**Cause**: Invalid data types or None values in results

**Solution**:
- Check CSV data integrity in `validation_results.csv`
- Verify label normalization in metrics calculation

### 4. "ConnectionError" with Azure AI

**Cause**: Environment variables not set or incorrect endpoint

**Solution**:
- Verify environment variables with `echo $AZURE_INFERENCE_SDK_ENDPOINT`
- Test connection with `--test-connection` flag

## Advanced Debugging Tips

### 1. Use Debug Console

Access VS Code's debug console to inspect variables:
```python
# In debug console, inspect loaded data
len(dataset)
dataset[0]

# Check strategy configuration
strategy_templates['baseline']

# Verify model configuration
connector.config
```

### 2. Conditional Breakpoints

Set breakpoints that only trigger on specific conditions:
- Right-click breakpoint → "Edit Breakpoint" → "Expression"
- Example: `strategy == 'baseline'` to debug only baseline strategy

### 3. Log Point Debugging

Add log points instead of breakpoints for continuous monitoring:
- Right-click line → "Add Logpoint"
- Example: `Processing sample {sample_id} with strategy {strategy}`

This simplified debugging approach works with the current modular architecture and makes troubleshooting much more straightforward!