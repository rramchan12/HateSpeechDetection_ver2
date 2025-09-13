# ToxiGen Hate Speech Detection

A machine learning project for hate speech detection using the ToxiGen dataset.

## Project Structure

```
HateSpeechDetection_ver2/
├── data/
│   └── toxigen/           # Downloaded dataset files
│       ├── train.parquet
│       ├── annotated.parquet
│       └── annotations.parquet
├── data_collection/
│   └── toxigen_downloader.py  # Dataset download utility
├── tests/                 # Test suite
│   ├── test_toxigen_data_presence.py
│   └── test_toxigen_downloader.py
├── requirements.txt       # Project dependencies
├── pyproject.toml        # Project configuration & pytest settings
└── run_tests.py          # Test runner convenience script
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download the ToxiGen dataset:**
```bash
python -m data_collection.toxigen_downloader
```

Or with specific options:
```bash
python -m data_collection.toxigen_downloader --format parquet --overwrite
```

## Testing

This project uses pytest with comprehensive test coverage and multiple test categories.

### Quick Start

```bash
# Run all tests
python run_tests.py all

# Run just unit tests (fast)
python run_tests.py unit

# Run data validation tests with output
python run_tests.py data

# Run without coverage (fastest)
python run_tests.py fast
```

### Direct pytest Usage

```bash
# All tests with coverage
pytest tests/ -v

# Unit tests only
pytest tests/ -v -m "unit"

# Data tests with output
pytest tests/ -v -m "data" -s

# Integration tests
pytest tests/ -v -m "integration"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests, no external dependencies
- `@pytest.mark.integration` - Integration tests with external systems
- `@pytest.mark.data` - Tests requiring dataset files
- `@pytest.mark.slow` - Long-running tests

### Coverage Reports

Coverage reports are automatically generated and saved to `htmlcov/index.html`.

View coverage in browser:
```bash
python run_tests.py coverage
# Open htmlcov/index.html in your browser
```

## Dataset Information

The ToxiGen dataset contains three splits:

- **train**: ~250k synthetic text examples for training
- **annotated**: ~9k human-annotated examples for evaluation  
- **annotations**: ~27k raw annotation data from human study

### Dataset Schema

**Train split fields:**
- `prompt` - Input prompt used for generation
- `generation` - Generated text sample
- `generation_method` - Method used for generation
- `group` - Target demographic group
- `prompt_label` - Label for the prompt
- `roberta_prediction` - RoBERTa model prediction

**Annotated split fields:**
- `text` - Text sample
- `target_group` - Target demographic
- `factual?` - Factual content indicator
- `ingroup_effect` - Ingroup/outgroup effect
- `lewd` - Lewd content indicator
- `framing` - Framing analysis

## Development

### Adding New Tests

1. Create test files in `tests/` following the pattern `test_*.py`
2. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
3. Follow the existing patterns for mocking external dependencies

### Running Specific Tests

```bash
# Run specific test file
pytest tests/test_toxigen_data_presence.py -v

# Run specific test function
pytest tests/test_toxigen_data_presence.py::test_toxigen_data_presence -v

# Run with specific marker
pytest -m "unit and not slow" -v
```

### Configuration

Test configuration is managed in `pyproject.toml`:

- **Coverage threshold**: 70% minimum
- **Test discovery**: `tests/` directory, `test_*.py` files
- **Coverage reports**: Terminal + HTML
- **Markers**: Custom markers for test categorization

## Dependencies

Core dependencies:
- `datasets` - Hugging Face datasets library
- `pandas` - Data manipulation
- `pyarrow` - Parquet file support

Testing dependencies:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities

See `requirements.txt` for full dependency list and versions.