# Unified Hate Speech Detection Dataset

## Overview

This directory contains the **unified, balanced dataset** representing the core data contribution of this thesis. The dataset combines HateXplain (real social media posts) and ToxiGen (synthetic generated text) into a single, high-quality corpus optimized for hate speech detection research with fairness-aware evaluation.

## Dataset Composition

### Files
- **`unified_train.json`** (2,686 samples) - Training set with stratified demographic and label balance
- **`unified_val.json`** (514 samples) - Validation set for hyperparameter tuning
- **`unified_test.json`** (1,009 samples) - Held-out test set for final evaluation
- **`unified_dataset_stats.json`** - Comprehensive dataset statistics and quality metrics

### Total Dataset Statistics
- **Total Samples**: 5,151
- **Label Balance**: 47.1% hate, 52.9% normal
- **Source Balance**: 47.1% HateXplain, 52.9% ToxiGen
- **Rationale Coverage**: 36.9% (1,899 samples with human explanations)
- **Data Split**: 70.4% train, 10.0% val, 19.6% test

### Target Group Distribution
| **Group** | **Count** | **Percentage** | **Persona Tags** |
|-----------|-----------|----------------|------------------|
| **LGBTQ+** | 2,515 | 48.8% | homosexual, gay, lgbtq |
| **Middle East** | 1,471 | 28.5% | arab, middle_east |
| **Mexican** | 1,165 | 22.6% | hispanic, latino, mexican |

## Unified Schema (12 Fields)

Each sample contains the following fields:

| **Field** | **Type** | **Description** |
|-----------|----------|-----------------|
| `text` | string | Input text content for classification |
| `label_binary` | string | Binary classification label (hate/normal) |
| `label_multiclass` | string | Multi-class label (hatespeech/toxic_implicit/benign_implicit) |
| `target_group_norm` | string | Normalized target group (lgbtq/mexican/middle_east) |
| `persona_tag` | string | Original persona identifier (homosexual/arab/hispanic/etc.) |
| `source_dataset` | string | Data provenance (hatexplain/toxigen) |
| `rationale_text` | string/null | Human explanation (HateXplain only, null for ToxiGen) |
| `is_synthetic` | boolean | Generated flag (true for ToxiGen, false for HateXplain) |
| `fine_tuning_embedding` | string | Formatted instruction-following input for SFT |
| `original_id` | string | Source dataset identifier for traceability |
| `split` | string | Data split assignment (train/val/test) |
| `fine_tuning_label` | string | Formatted target label for SFT |

## Data Quality Features

### Stratified Balancing
The dataset achieves near-perfect balance across multiple dimensions:
- **Source balance**: 47/53 split ensures equal representation from real and synthetic data
- **Label balance**: 47/53 hate/normal ratio prevents class bias
- **Demographic balance**: Stratified sampling maintains proportional representation across LGBTQ+, Mexican, and Middle East personas

### Quality Improvements Over Source Datasets
- **92% size reduction**: From 64,321 raw samples to 5,151 curated entries
- **11.5x rationale improvement**: From 3.2% to 36.9% explanation coverage
- **Zero label noise**: Rigorous validation and consistency checks
- **Preserved diversity**: Maintains demographic proportions while improving quality

## Unification Methodology

This dataset was created using **Approach 3: Stratified Sampling + Rationale Preservation** from the comprehensive unification analysis.

**Key Design Decisions:**
1. **Target group selection**: Focus on three high-prevalence personas (LGBTQ+, Mexican, Middle East)
2. **Source balancing**: Equal representation from HateXplain and ToxiGen
3. **Rationale maximization**: Preserve all HateXplain samples with human explanations
4. **Quality-first filtering**: Undersample to eliminate duplicates and low-quality samples

**Full Methodology Documentation:**  
[`../../data_preparation/UNIFICATION_APPROACH.md`](../../data_preparation/UNIFICATION_APPROACH.md)

## Exploratory Data Analysis

Comprehensive EDA notebooks and visualizations analyzing the unified dataset:

### Analysis Notebooks
- **`../../../eda/unified_dataset_eda.ipynb`** - Complete statistical analysis, distribution plots, and quality metrics
- **`../../../eda/outputs/`** - Generated visualizations and summary statistics

### Key Findings
- **Demographic fairness**: No group has <20% or >50% representation
- **Label distribution**: Near-perfect 50/50 balance prevents model bias
- **Text complexity**: Diverse content lengths (10-280 characters) ensure robustness
- **Rationale quality**: Human explanations provide explainability signals for SFT

**EDA Results:**  
[`../../../eda/unified_dataset_eda.ipynb`](../../../eda/unified_dataset_eda.ipynb)

## Usage Examples

### Loading the Dataset (Python)

```python
import json
import pandas as pd

# Load training data
with open('unified_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Convert to DataFrame for analysis
df_train = pd.DataFrame(train_data)

# Check demographic distribution
print(df_train['target_group_norm'].value_counts())

# Check label balance
print(df_train['label_binary'].value_counts())

# Access samples with rationales
rationale_samples = df_train[df_train['rationale_text'].notna()]
print(f"Samples with rationales: {len(rationale_samples)}")
```

### Using for Fine-Tuning

```python
# Access pre-formatted instruction-following inputs
for sample in train_data:
    instruction_input = sample['fine_tuning_embedding']
    target_label = sample['fine_tuning_label']
    
    # Use for SFT training
    # model.train(instruction_input, target_label)
```

## Citation

When using this unified dataset, please cite:

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

**Source Dataset Citations:**
- **HateXplain**: Mathew et al., 2021 - [https://arxiv.org/abs/2012.10289](https://arxiv.org/abs/2012.10289)
- **ToxiGen**: Hartvigsen et al., 2022 - [https://arxiv.org/abs/2203.09509](https://arxiv.org/abs/2203.09509)

## License

This unified dataset inherits the licenses of its source datasets:
- **HateXplain**: MIT License
- **ToxiGen**: Apache License 2.0

The unification methodology and processing pipeline are released as open-source software for research and educational purposes.

## Contact

For questions about the dataset, unification methodology, or usage:
- **Author**: Ravi Ramchandran
- **Institution**: Liverpool John Moores University
- **Supervisor**: Dr Dattatraya Parle
