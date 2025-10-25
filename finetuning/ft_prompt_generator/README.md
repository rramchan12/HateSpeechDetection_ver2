# Fine-Tuning Prompt Generator

This package converts the unified dataset into instruction format (JSONL) suitable for fine-tuning language models.

## Overview

The generator creates two versions of fine-tuning data:

1. **Simple Format**: Basic instruction format with system/user/assistant messages
2. **Optimized Format**: Uses the `combined_optimized` strategy for sophisticated prompting

## Data Source

The generator uses the **original train/val/test splits** from the unified dataset:
- `data/processed/unified/unified_train.json`
- `data/processed/unified/unified_val.json`
- `data/processed/unified/unified_test.json`

Each file contains records with a `split` field indicating the original split assignment.

## Output Directory

All generated fine-tuning prompts are saved to:
```
finetuning/data/ft_prompts/
```

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
python -m finetuning.ft_prompt_generator.cli
```

**Default Settings:**
- Unified dir: `./data/processed/unified`
- Output dir: `./finetuning/data/ft_prompts`
- Template: `combined/combined_gptoss_v1.json`
- Strategy: `combined_optimized`
- Include test: `False`

### Include Test Split

```bash
python -m finetuning.ft_prompt_generator.cli --include_test
```

### Generate Only Simple Format

Skip the optimized version:

```bash
python -m finetuning.ft_prompt_generator.cli --simple_only
```

### Custom Directories

```bash
python -m finetuning.ft_prompt_generator.cli \
  --unified_dir ./data/processed/unified \
  --output_dir ./custom/output/dir
```

### Custom Template and Strategy

```bash
python -m finetuning.ft_prompt_generator.cli \
  --template combined/combined_gpt5_v1.json \
  --strategy combined_optimized
```

### Debug Mode

```bash
python -m finetuning.ft_prompt_generator.cli --debug
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
FINE-TUNING PROMPT GENERATION
============================================================
Unified data directory: ./data/processed/unified
Output directory: ./data/ft_prompts
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
  ✓ finetuning/data/ft_prompts/train.jsonl (2686 samples)
  ✓ finetuning/data/ft_prompts/validation.jsonl (55 samples)

OPTIMIZED format:
  ✓ finetuning/data/ft_prompts/train_optimized.jsonl (2686 samples)
  ✓ finetuning/data/ft_prompts/validation_optimized.jsonl (55 samples)

============================================================
[SUCCESS] Fine-tuning data generation complete
Output directory: ./finetuning/data/ft_prompts
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

## Package Structure

```
ft_prompt_generator/
├── __init__.py          # Package initialization
├── generator.py         # Core FineTuningDataGenerator class
├── cli.py              # Command-line interface
└── README.md           # This file
```

## Programmatic Usage

You can also use the generator programmatically:

```python
from finetuning.ft_prompt_generator import FineTuningDataGenerator

# Create generator
generator = FineTuningDataGenerator(
    unified_dir="./data/processed/unified",
    output_dir="./finetuning/data/ft_prompts",
    template_path="combined/combined_gptoss_v1.json",
    strategy_name="combined_optimized"
)

# Generate all files
results = generator.generate_all(
    include_test=False,
    generate_optimized=True
)

# Check results
for format_type, files in results.items():
    print(f"{format_type}: {len(files)} files generated")
```

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

- [VALIDATION_GUIDE.md](../finetuning/VALIDATION_GUIDE.md) - Baseline calculation methodology
- [prompt_engineering/](../prompt_engineering/) - Prompt engineering tools and templates
- [data_preparation/](../data_preparation/) - Data unification pipeline
