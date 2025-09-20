# Hate Speech Detection: Production-Ready Framework with Unified Dataset

A comprehensive hate speech detection project that unifies **HateXplain** and **ToxiGen** datasets into a single, consistent schema optimized for training robust hate speech detection models. The project focuses specifically on **LGBTQ**, **Mexican**, and **Middle East** target groups with comprehensive testing, validation, and **a production-ready prompt engineering framework for Azure AI models**.

## 🚀 Prompt Engineering & Validation Framework

The project includes a **production-ready prompt validation framework** for Azure AI models with advanced capabilities:

### Framework Highlights

- **🎯 Multi-Model Support**: Switch between GPT-OSS-20B, GPT-5, and custom models via YAML configuration
- **📊 Flexible Data Sources**: Unified dataset sampling + curated canned datasets (`canned_basic_all`, `canned_100_all`)
- **⚡ Concurrent Processing**: Multi-threaded execution with configurable worker pools and batch sizes
- **🔄 Rate Limiting & Retry Logic**: Intelligent retry with exponential backoff and rate limit detection  
- **📁 File Logging**: Complete audit trail with logs written to runID folders
- **💾 Incremental Storage**: Memory-efficient processing with real-time CSV writing
- **📁 runId Organization**: Timestamped output folders (`outputs/run_YYYYMMDD_HHMMSS/`)
- **🔧 Sample Size Control**: Configurable sampling that works with all data sources
- **📈 Robust Metrics**: Comprehensive evaluation calculated from stored results, not memory
- **🎨 Custom Prompt Templates**: CLI support for selecting different prompt template files
- **📊 Rich Evaluation Reports**: Detailed reports with model metadata, command line, and execution context

## 🎯 Project Overview

This project provides:
- **Dual Dataset Integration**: Combines real social media data (HateXplain) with synthetic data (ToxiGen)
- **Filtered Target Groups**: Focuses on 3 specific demographics for targeted analysis
- **Unified Schema**: Consistent 12-field schema with preserved persona identities
- **Robust Pipeline**: End-to-end data collection, processing, and unification
- **Comprehensive Testing**: 36 unit tests with 63%+ coverage for core unification logic
- **Persona Preservation**: Original target group identities preserved as persona tags
- **Production Prompt Framework**: Multi-model Azure AI validation system with incremental storage, YAML configuration, and runId-based organization

## 📁 Project Structure

```
HateSpeechDetection_ver2/
├── data/
│   ├── hatexplain/           # HateXplain raw dataset
│   ├── toxigen/              # ToxiGen raw dataset  
│   └── processed/            # Processed & unified data
│       └── unified/          # Final unified dataset (64,321 entries)
├── data_collection/          # Dataset downloaders and validators
├── data_preparation/         # Data processing pipeline
├── tests/                    # Comprehensive test suite (36 unit tests)
├── eda/                      # Exploratory Data Analysis
│   └── unified_dataset_eda.ipynb    # Comprehensive EDA notebook
├── prompt_engineering/       # 🎯 Production-Ready Prompt Validation Framework
│   ├── prompt_runner.py              # Main CLI orchestrator (single entry point)
│   ├── strategy_templates_loader.py  # Strategy template management
│   ├── evaluation_metrics_calc.py    # Metrics calculation from stored results
│   ├── persistence_helper.py         # runId-based output organization
│   ├── azureai_mi_connector_wrapper.py # YAML-based multi-model Azure AI connection
│   ├── unified_dataset_loader.py     # Flexible data loading with sampling
│   ├── model_connection.yaml         # Multi-model configuration
│   ├── prompt_templates/             # Strategy templates (JSON configurable)
│   ├── data_samples/                 # Curated test datasets
│   │   ├── canned_basic_all.json     # 5 basic test samples
│   │   └── canned_100_all.json       # 100 diverse stratified samples
│   ├── outputs/                      # runId-organized validation results
│   ├── README.md                     # 📖 Detailed framework documentation
│   ├── DEBUG.md                      # 🔧 Debugging guide
│   └── STRATEGY_TEST_RESULTS.md      # 📊 Test results and analysis
├── requirements.txt          # Project dependencies
├── pyproject.toml           # Project configuration
├── run_tests.py             # Test runner
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Download both HateXplain and ToxiGen datasets:

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

### 5. Prompt Engineering & Validation

The project includes a **production-ready prompt validation framework** for Azure AI models with comprehensive features:

```bash
# Navigate to prompt engineering directory
cd prompt_engineering

# Test connection to Azure AI models
python prompt_runner.py --test-connection

# Quick validation with curated samples
python prompt_runner.py --data-source canned_basic_all --strategies all

# Comprehensive evaluation with unified dataset and concurrent processing
python prompt_runner.py --data-source unified --sample-size 100 --strategies all --max-workers 10 --batch-size 20

# Custom prompt template with specific strategies
python prompt_runner.py --prompt-template-file experimental.json --data-source canned_100_all --strategies policy persona

# Recalculate metrics from previous run
python prompt_runner.py --metrics-only --run-id run_20250920_015821

# High-performance concurrent processing for large datasets
python prompt_runner.py --data-source unified --sample-size 500 --strategies all --debug
```

**📖 For comprehensive documentation, see [`prompt_engineering/README.md`](prompt_engineering/README.md)** which covers:

- **Multi-Model Support**: YAML-based configuration for GPT-OSS-20B, GPT-5, and custom models
- **Concurrent Processing**: Multi-threaded execution with configurable worker pools and batch sizes
- **Flexible Data Sources**: Unified dataset sampling + multiple canned datasets by name
- **Strategy Configuration**: Four comprehensive approaches (Policy, Persona, Combined, Baseline)
- **Rate Limiting & Retry Logic**: Intelligent retry with exponential backoff and rate limit detection
- **File Logging**: Complete audit trail with logs written to runID folders
- **Incremental Storage**: Memory-efficient processing with real-time result saving
- **runId Organization**: Timestamped output folders for easy comparison and analysis
- **Sample Size Control**: Configurable sampling that works with all data sources
- **Custom Prompt Templates**: CLI support for selecting different prompt template files
- **Rich Evaluation Reports**: Detailed reports with model metadata, command line, and execution context
- **Azure AI Integration**: Environment variable + YAML configuration support
- **Robust Metrics**: Comprehensive evaluation calculated from stored results
- **Debugging Guide**: [`prompt_engineering/DEBUG.md`](prompt_engineering/DEBUG.md) for troubleshooting
- **Test Results**: [`prompt_engineering/STRATEGY_TEST_RESULTS.md`](prompt_engineering/STRATEGY_TEST_RESULTS.md) for analysis framework

**🎯 Key Framework Features:**

- **Single Orchestrator**: `prompt_runner.py` as the only entry point
- **YAML Configuration**: Multi-model support with environment variable substitution
- **Concurrent Processing**: Multi-threaded execution with intelligent rate limiting
- **Incremental Processing**: Real-time CSV writing for memory efficiency
- **Sample Size Flexibility**: Works with unified dataset and all canned files
- **Metrics Recalculation**: Operate on stored results, not in-memory data
- **Performance Monitoring**: Real-time progress tracking and Azure AI rate limit monitoring

This creates the unified dataset with **64,321 entries** across 3 target groups:

- **LGBTQ**: 22,785 entries (35.4%)
- **Mexican**: 20,632 entries (32.1%)  
- **Middle East**: 20,904 entries (32.5%)

**Key Features:**

- **Persona Preservation**: Original target group identities preserved (e.g., "Arab" → `target_group_norm: "middle_east"`, `persona_tag: "arab"`)
- **Label Consistency**: Standardized binary and multiclass labels with implicit toxicity markers
- **Synthetic Augmentation Ready**: Future-proofed for synthetic data generation with HateXplain

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

## � Prompt Engineering & Validation

The project includes a comprehensive **GPT-OSS-20B prompt validation system** for testing hate speech detection strategies:

### Available Strategies

- **Baseline**: Direct text classification without additional context
- **Policy**: Context-aware classification with hate speech policy guidelines
- **Persona**: Identity-informed classification using persona tags (Arab, LGBTQ, etc.)
- **Combined**: Policy + Persona integration for comprehensive analysis

### Key Features

- **Concurrent Processing**: Multi-threaded execution with configurable worker pools and batch sizes
- **Rate Limiting Intelligence**: Exponential backoff with retry logic and rate limit detection
- **File Logging**: Complete audit trail with logs written to runID folders for each execution
- **Flexible Dataset Support**: Test with curated samples or full unified dataset with configurable sampling
- **Custom Prompt Templates**: CLI support for selecting different prompt template files
- **Rich Evaluation Reports**: Detailed reports with model metadata, command line, and execution context
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score with detailed reporting
- **Azure AI Integration**: Seamless connection to GPT-OSS-20B and GPT-5 via Azure AI endpoints
- **Production Ready**: Robust error handling, logging, and output management
- **Memory Efficiency**: Incremental result storage for large dataset processing
- **CLI Interface**: Comprehensive command-line tools for validation workflows

### Quick Start Examples

```bash
cd prompt_engineering

# Test Azure AI connection
python prompt_runner.py --test-connection

# Quick validation with basic samples
python prompt_runner.py --data-source canned_basic_all --strategies all

# Comprehensive evaluation with concurrent processing
python prompt_runner.py --data-source canned_100_all --strategies all --sample-size 10 --max-workers 10

# Large-scale unified dataset testing with performance monitoring
python prompt_runner.py --data-source unified --sample-size 50 --strategies policy persona --debug

# Custom prompt template testing
python prompt_runner.py --prompt-template-file experimental.json --data-source canned_basic_all --strategies baseline policy

# Recalculate metrics from previous run
python prompt_runner.py --metrics-only --run-id run_20250920_015821
```

### Architecture Features

- **Single Entry Point**: All functionality through `prompt_runner.py`
- **YAML Configuration**: Multi-model setup in `model_connection.yaml`
- **Modular Design**: Separate components for templates, metrics, persistence, and connection
- **Concurrent Processing**: Multi-threaded execution with intelligent batching and rate limiting
- **Error Handling**: Robust connection testing and graceful failure handling
- **Memory Efficiency**: Incremental result storage during validation
- **File Logging**: Complete execution logs with Azure AI monitoring in runID folders

**📖 For comprehensive documentation, see [`prompt_engineering/README.md`](prompt_engineering/README.md)**

## �🧪 Testing & Validation

This project includes comprehensive testing for all components with detailed coverage analysis.

### Unit Test Coverage

**Data Unification Tests (36 tests)**:

- ✅ **Class Initialization** (3 tests): Basic setup, default directories, constants validation
- ✅ **Target Group Normalization** (8 tests): Valid/invalid groups, None/empty handling, whitespace processing
- ✅ **Persona Tag Extraction** (7 tests): Identity preservation, case handling, validation logic
- ✅ **Label Mapping** (2 tests): HateXplain and ToxiGen label transformations
- ✅ **Fine-Tuning Embeddings** (5 tests): Persona placeholders, rationale handling, template formatting
- ✅ **Entry Unification** (9 tests): Valid entries, invalid filtering, synthetic flag handling
- ✅ **Dataset Loading** (3 tests): File I/O, missing files, partial data handling
- ✅ **Dataset Analysis** (3 tests): Statistics generation, error conditions

**Coverage Statistics**:

- **Core Unification Logic**: 63% coverage (273 lines tested)
- **All Tests Combined**: 36 tests passing with comprehensive edge case handling

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
