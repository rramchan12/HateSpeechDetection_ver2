# Data Unification Guide - Filtered Dataset (LGBTQ, Mexican, Middle East)

This document explains how to use the data unification pipeline to merge HateXplain and ToxiGen datasets into a single, consistent schema for hate speech detection. The current implementation is **filtered to include only 3 specific target groups**: LGBTQ, Mexican, and Middle East.

## 🎯 **Target Group Selection & Filtering**

The unified dataset is **filtered to include only 3 specific target groups** out of the 13+ available in the original datasets:

| **Target Group** | **HateXplain Source** | **ToxiGen Source** | **Normalized** | **Persona Tag** | **Final Count** |
|------------------|----------------------|-------------------|----------------|-----------------|-----------------|
| **LGBTQ** | Homosexual, Gay | lgbtq | lgbtq | LGBTQ | 22,785 (35.4%) |
| **Mexican** | Hispanic, Latino | mexican | mexican | MEXICAN | 20,632 (32.1%) |
| **Middle East** | Arab | middle_east | middle_east | MIDDLE_EAST | 20,904 (32.5%) |

**Filtering Impact:**
- **HateXplain**: 2,726 entries retained (14.1% of original ~19K)
- **ToxiGen**: 61,595 entries retained (24.5% of original ~251K)
- **Total Unified**: 64,321 entries (filtered from ~270K original)

All other target groups (Asian, Black, Jewish, Native American, etc.) are **automatically filtered out** during the unification process.

## 📋 **Unified Schema Overview**

The unified dataset follows this comprehensive 12-field schema for the 3 selected target groups:

| **Field** | **Purpose** | **HateXplain Source** | **ToxiGen Source** |
|-----------|-------------|------------------------|-------------------|
| `text` | Core input text | `post_text` | `generation` (mapped from `text`) |
| `label_binary` | Binary target (hate vs normal) | `majority_label` → hate→hate, offensive/normal→normal | `label_binary` → toxic→hate, benign→normal |
| `label_multiclass` | Multi-class labels | `majority_label` (hate/offensive/normal) | `label_binary` → toxic→toxic_implicit, benign→benign |
| `target_group_norm` | Normalized target group | `target` → normalized | `target_group` → normalized |
| `persona_tag` | Persona identifier | Derived from normalized `target` | Derived from normalized `target_group` |
| `source_dataset` | Data provenance | Constant: "hatexplain" | Constant: "toxigen" |
| `rationale_text` | Human explanation | `rationale_text` | null (not available) |
| `is_synthetic` | Generated data flag | false (real social media) | true (machine-generated) |
| `fine_tuning_embedding` | Model fine-tuning features | Augmented with persona/policy tags | Augmented with persona/policy tags |
| `original_id` | Source dataset ID | `post_id` | Generated `toxigen_{idx}` |
| `split` | Train/val/test designation | Inherited from source | Inherited from source |

## 🎯 **Current Dataset Statistics**

After filtering and unification, the dataset contains:

### Overall Metrics
- **Total Entries**: 64,321
- **HateXplain Entries**: 2,726 (4.2%)
- **ToxiGen Entries**: 61,595 (95.8%)
- **Synthetic Ratio**: 95.8%
- **Average Text Length**: 90.1 characters
- **Rationale Coverage**: 3.2% (HateXplain only)

### Label Distribution
**Binary Labels:**
- **hate**: 30,164 (46.9%) 
- **normal**: 34,157 (53.1%)

**Multiclass Labels:**
- **toxic_implicit**: 30,164 (46.9% - from ToxiGen)
- **benign**: 31,431 (48.9% - from ToxiGen)  
- **hatespeech**: 1,064 (1.7% - from HateXplain)
- **offensive**: 1,022 (1.6% - from HateXplain)
- **normal**: 640 (1.0% - from HateXplain)

### Target Group Distribution
- **lgbtq**: 22,785 entries (35.4%)
- **middle_east**: 20,904 entries (32.5%)
- **mexican**: 20,632 entries (32.1%)

## 🚀 **Quick Start**

### Basic Usage

```python
from data_preparation.data_unification import DatasetUnifier

# Initialize unifier
unifier = DatasetUnifier(
    hatexplain_dir="data/processed/hatexplain",
    toxigen_dir="data/processed/toxigen"
)

# Load, unify, and export
unifier.load_datasets()
unifier.unify_datasets()
unifier.print_dataset_summary()
unifier.export_unified_dataset(format='json')
```

### Advanced Usage

```python
# Custom output directory
unifier = DatasetUnifier(
    hatexplain_dir="data/processed/hatexplain",
    toxigen_dir="data/processed/toxigen",
    output_dir="data/processed/custom_unified"
)

# Load and process
unifier.load_datasets()
unifier.unify_datasets()

# Get detailed statistics
stats = unifier.analyze_unified_dataset()
print(f"Total entries: {stats.total_entries}")
print(f"Synthetic ratio: {stats.synthetic_ratio:.1%}")
print(f"Rationale coverage: {stats.rationale_coverage:.1%}")

# Export in multiple formats
unifier.export_unified_dataset(format='json')
unifier.export_unified_dataset(format='csv')     # Requires pandas
unifier.export_unified_dataset(format='parquet') # Requires pandas
```

## 🔄 **Field Mapping Details**

### Label Mapping

**HateXplain Labels:**
- `majority_label: "hate"` → `label_binary: "hate"`, `label_multiclass: "hate"`
- `majority_label: "offensive"` → `label_binary: "normal"`, `label_multiclass: "offensive"`
- `majority_label: "normal"` → `label_binary: "normal"`, `label_multiclass: "normal"`

**ToxiGen Labels:**
- `label_binary: "toxic"` → `label_binary: "hate"`, `label_multiclass: "toxic_implicit"`
- `label_binary: "normal"` → `label_binary: "normal"`, `label_multiclass: "benign"`

### Target Group Normalization

**Filtered Target Groups** (only these 3 are included in the final dataset):

```python
# Current implementation filters to only these mappings:
TARGET_GROUP_NORMALIZATION = {
    # HateXplain → Normalized (filtered)
    'Homosexual': 'lgbtq',
    'Gay': 'lgbtq', 
    'Hispanic': 'mexican',
    'Latino': 'mexican',
    'Arab': 'middle_east',
    
    # ToxiGen → Normalized (filtered)
    'lgbtq': 'lgbtq',
    'mexican': 'mexican', 
    'middle_east': 'middle_east'
}

# All other groups are filtered out during processing
VALID_TARGET_GROUPS = ['lgbtq', 'mexican', 'middle_east']
```

**Note**: The original datasets contain 13+ target groups (Women, African, Islam, Jewish, Black, Asian, etc.), but the current implementation specifically filters to only the 3 groups above. All other entries are excluded from the final unified dataset.

### Persona Tags

**Filtered Persona Tags** (only these 3 are used in the current dataset):

- `LGBTQ`, `MEXICAN`, `MIDDLE_EAST`

**Note**: The original implementation supported 9+ persona tags (`lgbtq`, `mexican`, `middle_east`, `black`, `asian`, `muslim`, `jewish`, `women`, `latino`), but the current filtered dataset only uses the 3 listed above.

## 🎨 **Fine-Tuning Embeddings**

The unified dataset includes fine-tuning embeddings with structured placeholders for the 3 filtered target groups:

**Example for HateXplain with rationale:**

```text
[PERSONA:LGBTQ] this is hate speech text [RATIONALE:contains stereotypes about lgbtq community] [POLICY:HATE_SPEECH_DETECTION]
```

**Example for ToxiGen:**

```text
[PERSONA:MEXICAN] this is generated text [POLICY:HATE_SPEECH_DETECTION]
```

**Example without specific persona:**

```text
this is text without specific target group [POLICY:HATE_SPEECH_DETECTION]
```

**Available Persona Tags in Filtered Dataset:**

- `[PERSONA:LGBTQ]` - For LGBTQ-targeted content
- `[PERSONA:MEXICAN]` - For Mexican-targeted content  
- `[PERSONA:MIDDLE_EAST]` - For Middle East-targeted content

## 📊 **Output Statistics**

The unification process generates comprehensive statistics for the filtered dataset:

```json
{
  "total_entries": 64321,
  "hatexplain_entries": 2726,
  "toxigen_entries": 61595,
  "label_binary_distribution": {
    "normal": 34157,
    "hate": 30164
  },
  "label_multiclass_distribution": {
    "offensive": 1022,
    "hatespeech": 1064,
    "normal": 640,
    "toxic_implicit": 30164,
    "benign": 31431
  },
  "target_group_distribution": {
    "mexican": 20632,
    "lgbtq": 22785,
    "middle_east": 20904
  },
  "persona_tag_distribution": {
    "mexican": 20632,
    "lgbtq": 22785,
    "middle_east": 20904
  },
  "source_distribution": {
    "hatexplain": 2726,
    "toxigen": 61595
  },
  "synthetic_ratio": 0.958,
  "avg_text_length": 90.1,
  "rationale_coverage": 0.032
}
```

## 📁 **Output Structure**

After running the unification, you'll get:

```
data/processed/unified/
├── unified_train.json
├── unified_val.json  
├── unified_test.json
└── unified_dataset_stats.json
```

## 🛠 **Customization**

### Adding New Target Group Mappings

```python
# Extend target group normalization
unifier.TARGET_GROUP_NORMALIZATION.update({
    'CustomGroup': 'custom_normalized_name'
})
```

### Modifying Persona Tags

```python
# Add new top persona tags
unifier.TOP_PERSONA_TAGS.add('new_persona')
```

### Custom Fine-Tuning Templates

```python
def custom_embedding(self, text, target_group_norm, persona_tag, rationale_text, source):
    # Custom template logic
    return f"[CUSTOM:{persona_tag}] {text} [CUSTOM_POLICY]"

# Override method
unifier.create_fine_tuning_embedding = custom_embedding
```

## ✅ **Validation**

The scaffolding includes validation features:

1. **Schema Consistency**: All entries follow the unified schema
2. **Label Mapping**: Correct conversion from source to unified labels
3. **Target Group Normalization**: Consistent target group names
4. **Statistics Generation**: Comprehensive dataset analysis
5. **Format Support**: JSON, CSV, and Parquet export options

## 🎯 **Expected Results (Current Implementation)**

With the **filtered dataset** focusing on 3 target groups, the current results are:

### **Filtering Statistics**
- **Original Dataset Size**: ~270,000 entries (19K HateXplain + 251K ToxiGen)
- **Filtered Dataset Size**: 64,321 entries (2.7K HateXplain + 61.6K ToxiGen)
- **Filtering Ratio**: 23.8% of original data retained

### **Final Dataset Characteristics**
- **Total entries**: 64,321 (filtered from ~270K)
- **Binary label distribution**: 53.1% normal, 46.9% hate (well-balanced)
- **Synthetic ratio**: 95.8% (primarily ToxiGen generated content)
- **Rationale coverage**: 3.2% (only HateXplain provides human rationales)
- **Target groups**: 3 filtered groups (LGBTQ, Mexican, Middle East)
- **Persona tags**: 3 focused personas for targeted fine-tuning

### **Data Quality Indicators**
- **Label Balance**: Near 50/50 split between hate and normal (ideal for training)
- **Group Balance**: Relatively even distribution across 3 target groups (32-35% each)
- **Source Diversity**: Combines real social media (HateXplain) with synthetic data (ToxiGen)
- **Explanation Coverage**: 3.2% of entries include human rationales for model interpretability

This **filtered and unified dataset** provides a focused, high-quality foundation for training hate speech detection models specifically targeting LGBTQ, Mexican, and Middle East demographics, with balanced labels and diverse source content.
