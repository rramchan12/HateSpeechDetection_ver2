# Hate Speech Detection: HateXplain + ToxiGen Unified Dataset

A comprehensive hate speech detection project that unifies **HateXplain** and **ToxiGen** datasets into a single, consistent schema optimized for training robust hate speech detection models. The project focuses specifically on **LGBTQ**, **Mexican**, and **Middle East** target groups with comprehensive testing and validation.

## 🎯 Project Overview

This project provides:
- **Dual Dataset Integration**: Combines real social media data (HateXplain) with synthetic data (ToxiGen)
- **Filtered Target Groups**: Focuses on 3 specific demographics for targeted analysis
- **Unified Schema**: Consistent 12-field schema with preserved persona identities
- **Robust Pipeline**: End-to-end data collection, processing, and unification
- **Comprehensive Testing**: 36 unit tests with 63%+ coverage for core unification logic
- **Persona Preservation**: Original target group identities preserved as persona tags

## 📁 Project Structure

```
HateSpeechDetection_ver2/
├── data/
│   ├── hatexplain/           # HateXplain raw dataset
│   │   ├── dataset.json
│   │   ├── post_id_divisions.json
│   │   └── classes*.npy
│   ├── toxigen/              # ToxiGen raw dataset  
│   │   ├── train.parquet
│   │   ├── annotated.parquet
│   │   └── annotations.parquet
│   └── processed/            # Processed & unified data
│       ├── hatexplain/       # Processed HateXplain
│       ├── toxigen/          # Processed ToxiGen
│       └── unified/          # Final unified dataset
│           ├── unified_train.json
│           ├── unified_val.json
│           ├── unified_test.json
│           └── unified_dataset_stats.json
├── data_collection/          # Dataset downloaders
│   ├── hatexplain_downloader.py
│   ├── toxigen_downloader.py
│   ├── hatexplain_data_presence_validator.py
│   └── toxigen_data_presence_validator.py
├── data_preparation/         # Data processing pipeline
│   ├── data_preparation_hatexplain.py
│   ├── data_preparation_toxigen.py
│   └── data_unification.py
├── tests/                    # Comprehensive test suite
│   ├── test_data_unification.py    # 36 unit tests for unification logic
│   ├── test_hatexplain_downloader.py
│   ├── test_toxigen_data_preparation.py
│   ├── test_hatexplain_data_presence.py
│   ├── test_toxigen_data_presence.py
│   └── test_toxigen_downloader.py
├── requirements.txt          # Project dependencies
├── pyproject.toml           # Project configuration
├── run_tests.py             # Test runner
├── README.md                # This file
└── UNIFICATION_GUIDE.md     # Detailed unification documentation
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

## 🧪 Testing & Validation

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

**See `requirements.txt` for complete dependency list with versions.**

## 📚 Additional Documentation

- **`UNIFICATION_GUIDE.md`** - Detailed unification process and schema documentation
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
