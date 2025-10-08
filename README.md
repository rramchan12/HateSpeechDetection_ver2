# Hate Speech Detection Framework

A hate speech detection project that unifies HateXplain and ToxiGen datasets into a consistent schema. The project includes data processing pipelines and a prompt validation framework for testing hate speech detection strategies with Azure AI models.

## Project Overview

- **Dataset Unification**: Combines HateXplain and ToxiGen datasets into unified schema
- **Target Groups**: Focuses on LGBTQ, Mexican, and Middle East demographics
- **Data Processing Pipeline**: End-to-end data collection, processing, and unification
- **Prompt Engineering Framework**: Multi-model validation system with modular architecture
- **Test Coverage**: Unit tests for core unification logic

## Prompt Engineering Framework

Located in `prompt_engineering/` - a modular framework for validating hate speech detection strategies:

- **Package Architecture**: Organized into `connector`, `loaders`, and `metrics` packages
- **Multi-Model Support**: YAML configuration for different Azure AI models
- **Strategy Testing**: Five prompt strategies (Baseline, Policy, Persona, Combined, Enhanced Combined)
- **Data Sources**: Unified dataset sampling and canned test datasets
- **Output Organization**: Timestamped runId folders with comprehensive results

See [`prompt_engineering/README.md`](prompt_engineering/README.md) for detailed framework documentation.

## Project Structure

```
HateSpeechDetection_ver2/
├── data/
│   ├── hatexplain/           # HateXplain raw dataset
│   ├── toxigen/              # ToxiGen raw dataset  
│   └── processed/            # Processed & unified data
│       └── unified/          # Final unified dataset
├── data_collection/          # Dataset downloaders and validators
├── data_preparation/         # Data processing pipeline
├── tests/                    # Unit test suite
├── eda/                      # Exploratory Data Analysis
│   └── unified_dataset_eda.ipynb    # EDA notebook
├── prompt_engineering/       # Prompt Validation Framework
│   ├── prompt_runner.py              # Main CLI entry point
│   ├── dataset_sampler.py            # Dataset sampling utilities
│   ├── connector/                    # Azure AI connection package
│   │   ├── azureai_connector.py      # Model connection management
│   │   └── model_connection.yaml     # YAML model configuration
│   ├── loaders/                      # Data and template loading utilities
│   │   ├── strategy_templates_loader.py  # Strategy template management
│   │   └── unified_dataset_loader.py     # Dataset loading with sampling  
│   ├── metrics/                      # Evaluation and persistence utilities
│   │   ├── evaluation_metrics_calc.py    # Metrics calculation
│   │   └── persistence_helper.py         # Output file management
│   ├── prompt_templates/             # Strategy configuration files
│   │   └── all_combined.json         # Main strategy definitions
│   ├── data_samples/                 # Test datasets
│   │   ├── canned_50_quick.json      # Quick test samples (50)
│   │   ├── canned_100_size_varied.json   # Size-varied samples (100)
│   │   └── canned_100_stratified.json    # Stratified samples (100)
│   ├── outputs/                      # Generated results (runId organized)
│   ├── README.md                     # Framework documentation
│   └── DEBUG.md                      # Debugging guide
├── requirements.txt          # Project dependencies
├── pyproject.toml           # Project configuration
├── run_tests.py             # Test runner
└── README.md                # This file
```

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Download HateXplain dataset
python -m data_collection.hatexplain_downloader

# Download ToxiGen dataset  
python -m data_collection.toxigen_downloader
```

### 3. Process Datasets

```bash
# Process HateXplain data
python -m data_preparation.data_preparation_hatexplain

# Process ToxiGen data
python -m data_preparation.data_preparation_toxigen
```

### 4. Create Unified Dataset

```bash
# Unify both datasets (filtered to LGBTQ, Mexican, Middle East)
python -m data_preparation.data_unification
```

### 5. Prompt Engineering Framework

```bash
# Navigate to prompt engineering directory
cd prompt_engineering

# Test connection to Azure AI models
python prompt_runner.py --test-connection

# Quick validation with canned samples
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Run all strategies with stratified samples
python prompt_runner.py --data-source canned_100_stratified --strategies all --sample-size 10
python prompt_runner.py --prompt-template-file experimental.json --data-source canned_100_all --strategies policy persona

# Recalculate metrics from previous run
python prompt_runner.py --metrics-only --run-id run_20250920_015821

# Use unified dataset for comprehensive evaluation
python prompt_runner.py --data-source unified --sample-size 100 --strategies all
```

For detailed framework documentation, see [`prompt_engineering/README.md`](prompt_engineering/README.md).

The unified dataset contains 64,321 entries across 3 target groups:

- **LGBTQ**: 22,785 entries (35.4%)
- **Mexican**: 20,632 entries (32.1%)  
- **Middle East**: 20,904 entries (32.5%)

Dataset characteristics:

- Original target group identities preserved (e.g., "Arab" → `target_group_norm: "middle_east"`, `persona_tag: "arab"`)
- Standardized binary and multiclass labels
- Combined real and synthetic data sources

## 📊 Unified Dataset Features

### Target Groups & Mapping

| **Group** | **HateXplain Source** | **ToxiGen Source** | **Persona Tags** | **Final Count** |
|-----------|----------------------|-------------------|-----------------|-----------------|
| **LGBTQ** | Homosexual, Gay | lgbtq | homosexual, gay, lgbtq | 22,785 |
| **Mexican** | Hispanic, Latino | mexican | hispanic, latino, mexican | 20,632 |
| **Middle East** | Arab | middle_east | arab, middle_east | 20,904 |

**Persona Tag Preservation**: Original target group identities are preserved in `persona_tag` field while `target_group_norm` provides normalized grouping for consistent analysis.

### Label Distribution

- **Binary Labels**: 46.9% hate, 53.1% normal (balanced for training)
- **Multiclass Labels**: hate, offensive, normal, toxic_implicit, benign_implicit
- **Sources**: 95.8% ToxiGen (synthetic), 4.2% HateXplain (real social media)
- **Rationale Coverage**: 3.2% of entries include human explanations

### Unified Schema (12 Fields)

| **Field** | **Purpose** | **Example Values** |
|-----------|-------------|-------------------|
| `text` | Input text | "This is offensive content..." |
| `label_binary` | Binary classification | "hate", "normal" |
| `label_multiclass` | Multi-class labels | "hatespeech", "toxic_implicit", "benign_implicit" |
| `target_group_norm` | Normalized group | "lgbtq", "mexican", "middle_east" |
| `persona_tag` | Original identity preserved | "homosexual", "arab", "hispanic" |
| `source_dataset` | Data provenance | "hatexplain", "toxigen" |
| `rationale_text` | Label explanation | Human rationale (HateXplain only) |
| `is_synthetic` | Generated flag | true (ToxiGen), false (HateXplain) |
| `fine_tuning_embedding` | Model features | "[PERSONA:ARAB] text [POLICY:HATE_SPEECH_DETECTION]" |
| `original_id` | Source identifier | Original dataset ID |
| `split` | Data split | "train", "val", "test" |

## Prompt Engineering & Validation

The project includes a prompt validation system for testing hate speech detection strategies with Azure AI models:

### Available Strategies

- **Baseline**: Direct text classification
- **Policy**: Classification with hate speech policy guidelines
- **Persona**: Classification using persona tags (Arab, LGBTQ, etc.)
- **Combined**: Policy + Persona integration
- **Enhanced Combined**: Advanced fusion approach

### Framework Components

- Multi-threaded execution with configurable settings
- YAML-based model configuration
- Multiple data sources (unified dataset and canned samples)
- runId-based output organization
- Incremental result storage
- Connection testing and error handling

### Usage Examples

```bash
cd prompt_engineering

# Test Azure AI connection
python prompt_runner.py --test-connection

# Quick validation with canned samples
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Run all strategies with stratified samples
python prompt_runner.py --data-source canned_100_stratified --strategies all --sample-size 10

# Use unified dataset
python prompt_runner.py --data-source unified --sample-size 50 --strategies policy persona

# Recalculate metrics from previous run
python prompt_runner.py --metrics-only --run-id run_20250920_015821
```

For detailed documentation, see [`prompt_engineering/README.md`](prompt_engineering/README.md).

## Testing & Validation

The project includes unit tests for data unification components.

### Unit Test Coverage

Data Unification Tests (36 tests):

- Class Initialization (3 tests)
- Target Group Normalization (8 tests)
- Persona Tag Extraction (7 tests)
- Label Mapping (2 tests)
- Fine-Tuning Embeddings (5 tests)
- Entry Unification (9 tests)
- Dataset Loading (3 tests)
- Dataset Analysis (3 tests)

Coverage: 63% of core unification logic (273 lines tested)

### Quick Test Commands

```bash
# Run all tests with coverage
python run_tests.py all

# Run only fast unit tests  
python run_tests.py unit

# Run data validation tests
python run_tests.py data

# Run without coverage (fastest)
python run_tests.py fast
```

### Test Categories

**Data Collection Tests:**

- HateXplain downloader validation
- ToxiGen downloader validation  
- Data presence verification
- File integrity checks

**Data Preparation Tests:**

- HateXplain processing pipeline
- ToxiGen processing pipeline
- Label mapping validation
- Feature extraction verification

**Unification Tests:**

- ✅ **Schema Consistency**: 12-field unified schema validation
- ✅ **Target Group Filtering**: Only LGBTQ, Mexican, Middle East included
- ✅ **Persona Tag Preservation**: Original identities maintained (e.g., "Arab" → "arab")
- ✅ **Label Distribution**: Binary/multiclass mapping verification
- ✅ **Synthetic Flag Handling**: Proper is_synthetic field management
- ✅ **Fine-Tuning Embeddings**: Persona placeholder generation
- ✅ **Split Ratio Validation**: Train/val/test distribution checks

### Coverage Reports

```bash
# Generate and view coverage report
python run_tests.py coverage
# Open htmlcov/index.html in browser
```

### Manual pytest Usage

```bash
# All tests with detailed output
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m "unit"           # Unit tests only
pytest tests/ -v -m "data"           # Data validation tests  
pytest tests/ -v -m "integration"    # Integration tests

# Run specific test files
pytest tests/test_data_unification.py -v      # 36 unification unit tests
pytest tests/test_hatexplain_downloader.py -v
pytest tests/test_toxigen_data_preparation.py -v
```

## 📈 Dataset Information

### HateXplain Dataset

**Source**: Real social media posts from Twitter and Gab  
**Size**: ~20K posts with human annotations  
**Splits**: train/val/test (80/10/10)  
**Labels**: hate, offensive, normal  
**Target Groups**: 13+ demographic groups (filtered to 3 for this project)

**Key Fields:**

- `post_text` - Original social media text
- `majority_label` - Human-annotated labels  
- `target` - Target demographic groups
- `rationale_text` - Human explanations for labels

### ToxiGen Dataset  

**Source**: Machine-generated synthetic text  
**Size**: ~250K generated examples  
**Splits**: train/val/test (70/10/20)  
**Labels**: toxic/benign based on prompts  
**Target Groups**: 13 minority demographic groups

**Key Fields:**

- `generation` - Machine-generated text
- `prompt_label` - Binary toxicity label (1=toxic, 0=benign)
- `group` - Target demographic group  
- `roberta_prediction` - RoBERTa toxicity score

### Unified Dataset Output

After processing and unification:

- **Total Entries**: 64,321 (filtered from ~270K original)
- **Target Groups**: 3 (LGBTQ, Mexican, Middle East)  
- **Label Balance**: 47% hate, 53% normal
- **Data Split**: 80% train, 10% val, 10% test
- **File Format**: JSON with unified 12-field schema

## 🔧 Development & Extension

### Recent Framework Improvements (September 2025)

**🚀 Performance & Concurrency:**
- Multi-threaded processing with configurable worker pools (default: 5 workers)
- Intelligent batching for optimal throughput (default: 10 samples per batch)
- Real-time progress monitoring and performance metrics
- Adaptive rate limiting with exponential backoff and intelligent retry logic

**📁 File Logging & Audit Trail:**
- Complete execution logs written to runID folders
- Azure AI request/response monitoring with rate limit headers
- Detailed error handling and retry logic status
- Clean console output with only runID for CI/CD integration

**🎨 Enhanced User Experience:**
- Custom prompt template file selection via CLI (`--prompt-template-file`)
- Rich evaluation reports with model metadata, command line, and execution context
- Sample size control for all data sources (unified and canned datasets)
- Comprehensive debug logging with file output

### Adding New Target Groups

1. Update `VALID_TARGET_GROUPS` in `data_unification.py`
2. Add mapping rules in `TARGET_GROUP_NORMALIZATION`  
3. Update persona tags in `TOP_PERSONA_TAGS`
4. Re-run unification pipeline

### Modifying Label Mapping

**HateXplain**: Edit `map_hatexplain_labels()` in `data_unification.py`  
**ToxiGen**: Edit `map_toxigen_labels()` in `data_unification.py`

### Custom Processing

Extend the data processors:

- `data_preparation_hatexplain.py` - HateXplain-specific processing
- `data_preparation_toxigen.py` - ToxiGen-specific processing  
- `data_unification.py` - Cross-dataset unification logic

## 📋 Dependencies

**Core Libraries:**

- `datasets` - Hugging Face datasets for loading
- `pandas` - Data manipulation and analysis
- `pyarrow` - Efficient Parquet file processing
- `numpy` - Numerical computing support

**Testing Libraries:**

- `pytest` - Test framework with fixtures
- `pytest-cov` - Coverage reporting and analysis  
- `pytest-mock` - Mocking utilities for isolation

**Prompt Engineering Libraries:**

- `azure-ai-inference` - Azure AI model integration with rate limiting support
- `python-dotenv` - Environment variable management
- `PyYAML` - YAML configuration file parsing for multi-model setup
- `scikit-learn` - Machine learning metrics and evaluation
- `concurrent.futures` - Multi-threaded concurrent processing support

**See `requirements.txt` for complete dependency list with versions.**

## 📚 Additional Documentation

**📖 Prompt Engineering Framework Documentation:**

- **[`prompt_engineering/README.md`](prompt_engineering/README.md)** - **Production prompt validation framework documentation**
  - Multi-model Azure AI configuration and usage with YAML support
  - Concurrent processing with configurable worker pools and batch sizes
  - Rate limiting intelligence with exponential backoff and retry logic
  - File logging with complete audit trails in runID folders
  - Comprehensive CLI reference and examples with performance tuning
  - Incremental storage and runId organization for memory efficiency
  - Strategy configuration (Policy, Persona, Combined, Baseline) with custom template support
  - Rich evaluation reports with metadata integration

- **[`prompt_engineering/DEBUG.md`](prompt_engineering/DEBUG.md)** - **Debugging guide for prompt validation framework**
  - VS Code debugging setup and workflow
  - Component-specific troubleshooting guidance
  - Common error patterns and solutions
  - Performance monitoring and rate limiting diagnostics

- **[`prompt_engineering/STRATEGY_TEST_RESULTS.md`](prompt_engineering/STRATEGY_TEST_RESULTS.md)** - **Test results and analysis framework**
  - Strategy performance evaluation templates with concurrent processing results
  - Metrics analysis and comparison guidelines
  - Usage patterns and best practices for production deployments
  - Performance benchmarking and optimization recommendations

**📊 Dataset & Project Documentation:**

- **`UNIFICATION_GUIDE.md`** - Detailed unification process and schema documentation
- **`eda/unified_dataset_eda.ipynb`** - Exploratory data analysis notebook
- **`htmlcov/index.html`** - Test coverage reports (generated after running tests)
- **`data/processed/unified/unified_dataset_stats.json`** - Dataset statistics and distributions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality  
4. Ensure test coverage remains above 70%
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project processes publicly available datasets:

- **HateXplain**: MIT License  
- **ToxiGen**: Apache License 2.0

Please cite the original papers when using this unified dataset.
