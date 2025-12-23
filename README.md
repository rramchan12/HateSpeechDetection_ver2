# Hate Speech Detection Using LLMs: Integrating Persona-Based and Policy-Based Techniques

## Master's Thesis Implementation - Liverpool John Moores University

**Research Title:** HATEFUL CONTENT DETECTION IN SOCIAL MEDIA USING LARGE LANGUAGE MODELS (LLMs) INCLUDING PERSONA-BASED AND POLICY-BASED TECHNIQUES

**Author:** Ravi Ramchandran | [LinkedIn](https://www.linkedin.com/in/raviramchandran/)  
**Programme:** MS Machine Learning & Artificial Intelligence (ML&AI)  
**Student ID:** 1180967  
**Thesis Supervisor:** Dr Dattatraya Parle | [LinkedIn](https://www.linkedin.com/in/dr-dattatraya-parle-1bb2635/)  
**Institution:** Liverpool John Moores University  
**Research Proposal:** [View PDF](https://github.com/rramchan12/LJMU_MLAI_RR/blob/main/Research_Proposal_RaviR_FinalD.pdf)

---

## Abstract

The rapid escalation of hate speech on social media has created an urgent need for scalable, fair, and context-sensitive automated moderation systems. Traditional rule-based and machine learning approaches struggle with implicit, coded, or context-dependent hate speech, while no prior work has systematically unified persona-based and policy-based techniques within Large Language Models (LLMs). This thesis presents a novel, empirically validated approach that integrates persona-based and policy-based signals using both open-source (GPT-OSS) and commercial (GPT-5) LLMs, explicitly focusing on three high-prevalence personas: LGBTQ+, Mexican, and Middle East. The methodology combines HateXplain and ToxiGen datasets into a unified, balanced corpus of 5,151 samples, employing a modular framework for Instruction Fine-Tuning (IFT) and Supervised Fine-Tuning (SFT) with LoRA. This research advances hate speech detection by providing a reproducible, data-driven methodology that emphasizes prompt design, model adaptation, and fairness-aware evaluation for ethical AI-driven moderation on social media platforms.

### Research Contributions

This thesis advances the field through three primary contributions to the research community:

#### 1. Unified Benchmark Dataset

We publicly release a curated, cleaned, benchmark-ready unified dataset combining HateXplain and ToxiGen sources. The dataset provides 5,151 stratified samples with 47/53 class balance, 36.9% rationale coverage, and explicit demographic group annotations (LGBTQ+, Mexican, Middle East) designed for fairness-aware evaluation. **[Dataset Documentation](data/processed/unified/README.md)**

#### 2. Reusable Prompt Suite for Instruction Fine-Tuning

The thesis provides a reusable prompt suite covering baseline, persona-based, and policy-constrained variants, allowing researchers to benchmark LLM behavior consistently. We introduce general-purpose instruction templates that can be reused across LLM families (GPT-OSS, Qwen, LLaMA) with systematic evaluation protocols. **[Prompt Validation Framework](prompt_engineering/README.md)**

#### 3. Modular Supervised Fine-Tuning Framework

We contribute a parameter-efficient fine-tuning framework implementing LoRA/QLoRA with 4-bit quantization, designed for reproducible experimentation across model architectures. The framework provides multi-GPU training orchestration, explicit rationale supervision, and comprehensive fairness-aware evaluation metrics. **[SFT Framework](finetuning/pipeline/SFT_IMPLEMENTATION_FRAMEWORK_ARCHITECTURE.md)**

## Project Architecture

### `data_collection/`
**Automated dataset acquisition infrastructure for HateXplain and ToxiGen sources.**

Downloads HateXplain from GitHub (4 core files) and ToxiGen from HuggingFace Hub (3 splits). Includes integrity validators for format verification and presence checking.

**Documentation**: [`data_collection/data_collection_README.md`](data_collection/data_collection_README.md)

### `data_preparation/`
**3-stage preprocessing and unification pipeline producing balanced, unified dataset.**

- **Stage 1**: Source-specific preprocessing (HateXplain/ToxiGen normalization)
- **Stage 2**: Schema unification with Approach 3 stratified balancing (47/53 source balance, 36.9% rationale coverage)
- **Stage 3**: Export and validation (5,151 entries, 92% size reduction with quality improvement)

**Documentation**: [`data_preparation/data_preparation_README.md`](data_preparation/data_preparation_README.md)  
**Methodology**: [`data_preparation/UNIFICATION_APPROACH.md`](data_preparation/UNIFICATION_APPROACH.md)

### `finetuning/pipeline/`
**LoRA training and evaluation infrastructure with QLoRA 4-bit quantization.**

- **Phase 1**: LoRA Training (train.py with HuggingFace Accelerate, produces safetensors)
- **Phase 2**: Baseline Evaluation (runner.py loads safetensors, computes metrics)

Supports multi-GPU training (4x A100 80GB), parameter-efficient fine-tuning (r=32, α=32), and comprehensive evaluation metrics (accuracy, precision, recall, F1).

**Documentation**: [`finetuning/pipeline/SFT_IMPLEMENTATION_FRAMEWORK_ARCHITECTURE.md`](finetuning/pipeline/SFT_IMPLEMENTATION_FRAMEWORK_ARCHITECTURE.md)

### `prompt_engineering/`
**Validation framework for testing prompt strategies with Azure AI models.**

Multi-threaded execution framework (5 workers, batch size 10) with 5 prompt strategies (Baseline, Policy, Persona, Combined, Enhanced Combined). Features intelligent rate limiting, exponential backoff retry logic, and runID-based incremental storage.

**Documentation**: [`prompt_engineering/pipeline/PROMPT_VALIDATION_FRAMEWORK_ARCHITECTURE.md`](prompt_engineering/pipeline/PROMPT_VALIDATION_FRAMEWORK_ARCHITECTURE.md)  
**User Guide**: [`prompt_engineering/README.md`](prompt_engineering/README.md)  
**Debugging**: [`prompt_engineering/DEBUG.md`](prompt_engineering/DEBUG.md)

### `data/`
**Raw and processed datasets with unified schema output.**

- `hatexplain/` - Real social media posts (2,427 samples, 100% rationale coverage)
- `toxigen/` - Synthetic generated text (2,724 samples)
- `processed/unified/` - Balanced unified dataset (5,151 entries, 12-field schema)

### `eda/`
**Exploratory data analysis notebooks and visualization outputs.**

Jupyter notebooks for HateXplain, ToxiGen, and unified dataset analysis. Includes distribution analysis, quality metrics, and balance verification.

### `tests/`
**Unit and integration tests with 63% coverage of core unification logic.**

36 tests covering target group normalization, persona tag extraction, label mapping, fine-tuning embeddings, and dataset loading. Run with `python run_tests.py all`.

## Project Structure

```
HateSpeechDetection_ver2/
├── data/                        # Raw and processed datasets
├── data_collection/             # Dataset downloaders and validators
├── data_preparation/            # Preprocessing and unification pipeline
├── eda/                         # Exploratory data analysis notebooks
├── finetuning/                  # LoRA training and evaluation infrastructure
├── prompt_engineering/          # Prompt validation framework
├── tests/                       # Unit and integration tests
├── htmlcov/                     # Test coverage reports
├── requirements.txt             # Project dependencies
├── pyproject.toml              # Project configuration
├── run_tests.py                # Test runner script
└── README.md                   # This file
```

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Configure Azure AI credentials
```

### 2. Data Pipeline
```bash
# Download datasets
python data_collection/hatexplain_downloader.py
python data_collection/toxigen_downloader.py

# Unify datasets
python data_preparation/data_unification.py
```

### 3. Prompt Engineering Validation
```bash
cd prompt_engineering
python prompt_runner.py --test-connection
python prompt_runner.py --data-source unified --sample-size 50 --strategies all
```

### 4. LoRA Fine-Tuning
```bash
cd finetuning/pipeline/lora
accelerate launch train.py --config configs/training_config.yaml

cd ../baseline
python runner.py --model-path ../lora/outputs/checkpoint-final
```

## Documentation Index

### Core Framework Documentation
- **[`data_collection/data_collection_README.md`](data_collection/data_collection_README.md)** - Dataset acquisition and validation
- **[`data_preparation/data_preparation_README.md`](data_preparation/data_preparation_README.md)** - Preprocessing and unification pipeline
- **[`data_preparation/UNIFICATION_APPROACH.md`](data_preparation/UNIFICATION_APPROACH.md)** - Comprehensive unification methodology analysis
- **[`finetuning/pipeline/SFT_IMPLEMENTATION_FRAMEWORK_ARCHITECTURE.md`](finetuning/pipeline/SFT_IMPLEMENTATION_FRAMEWORK_ARCHITECTURE.md)** - LoRA training and evaluation framework
- **[`prompt_engineering/pipeline/PROMPT_VALIDATION_FRAMEWORK_ARCHITECTURE.md`](prompt_engineering/pipeline/PROMPT_VALIDATION_FRAMEWORK_ARCHITECTURE.md)** - Prompt validation framework architecture
- **[`prompt_engineering/README.md`](prompt_engineering/README.md)** - Prompt engineering user guide
- **[`prompt_engineering/DEBUG.md`](prompt_engineering/DEBUG.md)** - Debugging and troubleshooting guide

### Implementation Guides
- **[`prompt_engineering/pipeline/IFT_APPROACH_IMPLEMENTATION.md`](prompt_engineering/pipeline/IFT_APPROACH_IMPLEMENTATION.md)** - In-context fine-tuning implementation
- **[`TRAINING_EVALUATION_DATASET_NOTE.md`](TRAINING_EVALUATION_DATASET_NOTE.md)** - Dataset usage guidelines
- **[`finetuning/A100_SSH_TRAINING_GUIDE.md`](finetuning/A100_SSH_TRAINING_GUIDE.md)** - Remote A100 training setup
- **[`finetuning/VALIDATION_GUIDE.md`](finetuning/VALIDATION_GUIDE.md)** - Model validation procedures

## Unified Dataset Features

### Target Groups & Distribution
| **Group** | **Count** | **Percentage** | **Persona Tags** |
|-----------|-----------|----------------|------------------|
| **LGBTQ** | 2,515 | 48.8% | homosexual, gay, lgbtq |
| **Middle East** | 1,471 | 28.5% | arab, middle_east |
| **Mexican** | 1,165 | 22.6% | hispanic, latino, mexican |

### Quality Metrics
- **Total Entries**: 5,151 (stratified balanced from 64,321 - 92% reduction for quality)
- **Label Balance**: 47.1% hate, 52.9% normal (near-perfect class balance)
- **Source Balance**: 47.1% HateXplain, 52.9% ToxiGen (perfect source balance)
- **Rationale Coverage**: 36.9% (11.5x improvement from original 3.2%)
- **Data Split**: 70.4% train, 10.0% val, 19.6% test

### Unified Schema (12 Fields)
| **Field** | **Description** |
|-----------|----------------|
| `text` | Input text content |
| `label_binary` | Binary classification (hate/normal) |
| `label_multiclass` | Multi-class labels (hatespeech/toxic_implicit/benign_implicit) |
| `target_group_norm` | Normalized group (lgbtq/mexican/middle_east) |
| `persona_tag` | Original identity preserved (homosexual/arab/hispanic) |
| `source_dataset` | Data provenance (hatexplain/toxigen) |
| `rationale_text` | Human explanation (HateXplain only) |
| `is_synthetic` | Generated flag (true for ToxiGen) |
| `fine_tuning_embedding` | Formatted training input |
| `original_id` | Source dataset identifier |
| `split` | Data split (train/val/test) |
| `fine_tuning_label` | Formatted training target |

## Testing & Validation

### Run Tests
```bash
# Run all tests with coverage
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run data validation tests
python run_tests.py data

# Run without coverage (fastest)
python run_tests.py fast

# View coverage report
python run_tests.py coverage
# Open htmlcov/index.html in browser
```

### Test Categories
- **Data Collection Tests**: HateXplain/ToxiGen downloader validation, integrity checks
- **Data Preparation Tests**: Processing pipeline, label mapping, feature extraction
- **Unification Tests**: Schema consistency, target group filtering, persona tag preservation, label distribution, fine-tuning embeddings

## Dependencies

**Core Libraries:**
- `datasets` - HuggingFace datasets for loading
- `pandas` - Data manipulation and analysis
- `pyarrow` - Efficient Parquet file processing
- `numpy` - Numerical computing support

**Prompt Engineering:**
- `azure-ai-inference` - Azure AI model integration with rate limiting
- `python-dotenv` - Environment variable management
- `PyYAML` - YAML configuration file parsing
- `scikit-learn` - Machine learning metrics

**Fine-Tuning:**
- `transformers` - HuggingFace model library
- `peft` - Parameter-efficient fine-tuning (LoRA/QLoRA)
- `accelerate` - Multi-GPU training support
- `bitsandbytes` - 4-bit quantization

**Testing:**
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting

**See `requirements.txt` for complete dependency list with versions.**

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure test coverage remains above 60%
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License & Citation

This project is released as **open-source software** for research and educational purposes.

### Citation Requirement

**Citation is mandatory** when using this framework, methodology, or unified dataset in your work. Please cite:

```bibtex
@mastersthesis{ramchandran2025hatespeechtdetection,
  author = {Ramchandran, Ravi},
  title = {Hateful Content Detection in Social Media Using Large Language Models (LLMs) Including Persona-Based and Policy-Based Techniques},
  school = {Liverpool John Moores University},
  year = {2025},
  type = {Master's Thesis},
  program = {MS Machine Learning \& Artificial Intelligence}
}
```

### Source Dataset Licenses

This project processes publicly available datasets under the following licenses:
- **HateXplain**: MIT License - Cite [Mathew et al., 2021](https://arxiv.org/abs/2012.10289)
- **ToxiGen**: Apache License 2.0 - Cite [Hartvigsen et al., 2022](https://arxiv.org/abs/2203.09509)

Please cite the original papers when using the unified dataset or methodology described in this framework.
