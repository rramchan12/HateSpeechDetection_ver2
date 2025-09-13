# Hate Speech Detection: HateXplain + ToxiGen Unified Dataset

A comprehensive hate speech detection project that unifies **HateXplain** and **ToxiGen** datasets into a single, consistent schema optimized for training robust hate speech detection models. The project focuses specifically on **LGBTQ**, **Mexican**, and **Middle East** target groups.

## 🎯 Project Overview

This project provides:
- **Dual Dataset Integration**: Combines real social media data (HateXplain) with synthetic data (ToxiGen)
- **Filtered Target Groups**: Focuses on 3 specific demographics for targeted analysis
- **Unified Schema**: Consistent 12-field schema for both datasets
- **Robust Pipeline**: End-to-end data collection, processing, and unification
- **High Test Coverage**: Comprehensive test suite with 70%+ coverage

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

## 📊 Unified Dataset Features

### Target Groups & Mapping

| **Group** | **HateXplain Source** | **ToxiGen Source** | **Final Count** |
|-----------|----------------------|-------------------|-----------------|
| **LGBTQ** | Homosexual, Gay | lgbtq | 22,785 |
| **Mexican** | Hispanic, Latino | mexican | 20,632 |
| **Middle East** | Arab | middle_east | 20,904 |

### Label Distribution

- **Binary Labels**: 46.9% hate, 53.1% normal (balanced for training)
- **Multiclass Labels**: hate, offensive, normal, toxic_implicit, benign
- **Sources**: 95.8% ToxiGen (synthetic), 4.2% HateXplain (real social media)

### Unified Schema (12 Fields)

| **Field** | **Purpose** | **Example Values** |
|-----------|-------------|-------------------|
| `text` | Input text | "This is offensive content..." |
| `label_binary` | Binary classification | "hate", "normal" |
| `label_multiclass` | Multi-class labels | "hatespeech", "toxic_implicit", "benign" |
| `target_group_norm` | Normalized group | "lgbtq", "mexican", "middle_east" |
| `persona_tag` | Persona identifier | "LGBTQ", "MEXICAN", "MIDDLE_EAST" |
| `source_dataset` | Data provenance | "hatexplain", "toxigen" |
| `rationale_text` | Label explanation | Human rationale (HateXplain only) |
| `is_synthetic` | Generated flag | true (ToxiGen), false (HateXplain) |
| `fine_tuning_embedding` | Model features | Computed embedding vector |
| `original_id` | Source identifier | Original dataset ID |
| `split` | Data split | "train", "val", "test" |

## 🧪 Testing & Validation

This project includes comprehensive testing for all components with 70%+ coverage.

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

- Schema consistency validation
- Target group filtering
- Label distribution verification  
- Split ratio validation

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
