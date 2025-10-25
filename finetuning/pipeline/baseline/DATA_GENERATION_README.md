# Fine-Tuning Data Generation

This document describes the fine-tuning data generation process using `finetuning_data_generator.py`.

## Overview

The data generator converts the unified dataset into instruction format (JSONL) suitable for fine-tuning language models. It creates two versions:

1. **Simple Format**: Basic instruction format with system/user/assistant messages
2. **Optimized Format**: Uses the `combined_optimized` strategy for sophisticated prompting

## Data Source

The generator uses the **original train/val/test splits** from the unified dataset:
- `data/processed/unified/unified_train.json`
- `data/processed/unified/unified_val.json`
- `data/processed/unified/unified_test.json`

Each file contains records with a `split` field indicating the original split assignment.

## Output Formats

### Simple Format

Basic instruction format for fine-tuning:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert hate speech detection system..."
    },
    {
      "role": "user",
      "content": "Analyze this text for hate speech:\n\nPersona: LGBTQ\nText: \"example text\""
    },
    {
      "role": "assistant",
      "content": "{\"classification\": \"hate_speech\", \"confidence\": \"high\", \"reasoning\": \"...\", \"protected_group\": \"LGBTQ\"}"
    }
  ]
}
```

**Output Files:**
- `train.jsonl` - Training data (simple format)
- `validation.jsonl` - Validation data (simple format)
- `test.jsonl` - Test data (simple format, optional)

### Optimized Format

Uses the `combined_optimized` strategy with sophisticated system and user prompts:

- Detailed policy explanations
- Community-specific context
- Examples of hate vs. policy discussion
- Nuanced evaluation criteria

**Output Files:**
- `train_optimized.jsonl` - Training data (optimized format)
- `validation_optimized.jsonl` - Validation data (optimized format)
- `test_optimized.jsonl` - Test data (optimized format, optional)

## Usage

### Basic Usage (Default Settings)

Generate train and validation files in both formats:

```bash
python -m finetuning.pipeline.baseline.runner --generate_data
```

**Default Settings:**
- Unified dir: `./data/processed/unified`
- Output dir: `./finetuning/data/prepared`
- Template: `combined/combined_gptoss_v1.json`
- Strategy: `combined_optimized`
- Include test: `False`

### Include Test Split

```bash
python -m finetuning.pipeline.baseline.runner --generate_data --include_test
```

### Generate Only Simple Format

Skip the optimized version:

```bash
python -m finetuning.pipeline.baseline.runner --generate_data --simple_only
```

### Custom Directories

```bash
python -m finetuning.pipeline.baseline.runner --generate_data \
  --unified_dir ./data/processed/unified \
  --data_output_dir ./custom/output/dir
```

### Custom Template and Strategy

```bash
python -m finetuning.pipeline.baseline.runner --generate_data \
  --prompt_template combined/combined_gptoss_v1.json \
  --strategy combined_optimized
```

## Split Filtering

The generator filters data based on the `split` field in the unified dataset:

- For `unified_train.json`: Keeps only records where `split == "train"`
- For `unified_val.json`: Keeps only records where `split == "val"`
- For `unified_test.json`: Keeps only records where `split == "test"`

This ensures that the generated fine-tuning data uses the **original dataset splits**, not the filename-based organization.

## Expected Output

```
============================================================
FINE-TUNING DATA GENERATION
============================================================
Unified data directory: ./data/processed/unified
Output directory: ./finetuning/data/prepared
Include test split: False
Generate optimized: True
Template: combined/combined_gptoss_v1.json
Strategy: combined_optimized
============================================================

Generating fine-tuning data files...

============================================================
GENERATION COMPLETE
============================================================

SIMPLE format:
  ✓ finetuning/data/prepared/train.jsonl (2686 samples)
  ✓ finetuning/data/prepared/validation.jsonl (55 samples)

OPTIMIZED format:
  ✓ finetuning/data/prepared/train_optimized.jsonl (2686 samples)
  ✓ finetuning/data/prepared/validation_optimized.jsonl (55 samples)

============================================================
[SUCCESS] Fine-tuning data generation complete
Output directory: ./finetuning/data/prepared
============================================================
```

## Data Counts

Based on the unified dataset with split filtering:

| Split | Records (split=train) | Records (split=val) | Records (split=test) |
|-------|----------------------|---------------------|---------------------|
| Train | 2,686 | - | - |
| Val | - | 55 | - |
| Test | - | - | 578 |

**Note:** The unified files contain mixed splits. The generator filters to use only the appropriate split field values.

## File Format Details

### Simple Format Structure

- **System Prompt**: Basic hate speech detection instructions
- **User Prompt**: "Analyze this text for hate speech:" + Persona + Text
- **Assistant Response**: JSON with classification, confidence, reasoning, protected_group

### Optimized Format Structure

- **System Prompt**: Comprehensive policy explanation with community context
- **User Prompt**: Detailed evaluation criteria with examples and community-specific guidance
- **Assistant Response**: Same JSON format as simple version

## Integration with Fine-Tuning

The generated files are ready for use with:

1. **OpenAI Fine-Tuning API**: Upload JSONL files directly
2. **Hugging Face Transformers**: Load with `datasets.load_dataset('json', data_files=...)`
3. **Custom Training Scripts**: Read line-by-line JSONL format

## Comparison with Baseline

| Aspect | Baseline (Pre-Fine-Tuning) | Fine-Tuning Data |
|--------|---------------------------|------------------|
| **Format** | Sophisticated prompting with `combined_optimized` | Simple instruction format |
| **Purpose** | Establish best pre-trained performance | Teach model the task |
| **Data Source** | validation.jsonl (545 samples from custom re-split) | Original splits (train: 2,686, val: 55) |
| **Prompting** | Complex multi-shot with examples | Simple system/user/assistant |

**Key Insight:** The baseline uses sophisticated prompting to get the best performance from the pre-trained model. After fine-tuning with simple instruction format, the model should internalize the task and achieve similar F1 scores with simpler prompts.

## Troubleshooting

### No samples found with split=X

If you see this warning, the generator will use all samples from the file instead:

```
WARNING: No samples found with split=train in unified_train.json
INFO: Using all 3628 samples from file instead
```

This happens when the `split` field doesn't match the filename. The generator is designed to handle this gracefully.

### Strategy not found

If the template doesn't contain the specified strategy:

```
WARNING: Strategy combined_optimized not found in template. Using simple format only.
```

The generator will only create simple format files.

## See Also

- [VALIDATION_GUIDE.md](../../../finetuning/VALIDATION_GUIDE.md) - Baseline calculation methodology
- [runner.py](runner.py) - Main pipeline runner with all modes
- [finetuning_data_generator.py](finetuning_data_generator.py) - Data generation implementation
