# LoRA Fine-Tuning Approach for Hate Speech Detection

## Document Overview

**Purpose**: This document provides a comprehensive overview of the LoRA (Low-Rank Adaptation) fine-tuning approach applied to hate speech detection, detailing the experimental configurations and methodologies employed in this research.

**Context**: This work extends the validation methodology for instruction fine-tuning (IFT) of large language models (20B+ parameters) on binary hate speech classification with bias fairness constraints. The approach synthesizes established LoRA research (Hu et al., 2021) and QLoRA efficiency innovations (Dettmers et al., 2023) with empirical experimentation.

**Scope**: Documentation of prompt engineering strategies, training data preparation, and hyperparameter configurations tested during the LoRA fine-tuning experiments on the GPT-OSS-20B model.

**Version**: 2.0  
**Date**: October 26, 2025  
**Author**: HateSpeechDetection Research Team

---

## Table of Contents

1. [Prompt Selection](#1-prompt-selection)
2. [Training Data Configuration](#2-training-data-configuration)
3. [Hyperparameter Configurations](#3-hyperparameter-configurations)
4. [Experimental Results](#4-experimental-results)
5. [References](#5-references)

---

## 1. Prompt Selection

### 1.1 Prompt Engineering Strategy

The prompt template used for instruction fine-tuning follows a chat-based conversational format designed to align the model with hate speech detection objectives through explicit role definition, contextual framing, and structured output formatting.

#### Prompt Template Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert hate speech detection system. Analyze the given text and determine if it contains hate speech targeting protected groups (LGBTQ+, Mexican, Middle Eastern communities)."
    },
    {
      "role": "user", 
      "content": "Analyze this text for hate speech:\n\nPersona: {target_group}\nText: \"{text}\""
    },
    {
      "role": "assistant",
      "content": "{\"classification\": \"{hate_speech|not_hate}\", \"confidence\": \"{high|medium|low}\", \"reasoning\": \"{explanation}\", \"protected_group\": \"{MEXICAN|LGBTQ+|MIDDLE_EAST}\"}"
    }
  ]
}
```

**Design Rationale**:
- **System Role**: Establishes expert persona and defines protected demographic groups explicitly
- **User Role**: Provides demographic context (persona) alongside text for bias-aware classification
- **Assistant Role**: Structured JSON output with classification (`hate_speech` or `not_hate`), confidence level, reasoning, and protected group (for hate speech cases)
- **Clarity**: Three-turn conversation format mimics natural instruction-following pattern
- **Parsability**: JSON response enables reliable extraction of predictions and confidence scores
- **Bias Awareness**: Persona labels in user message and protected_group in assistant response enable demographic-stratified evaluation

### 1.2 Baseline Prompt Template

The chat-based prompt template was designed to provide comprehensive context for hate speech detection with explicit protected group awareness:

**Template Characteristics**:
- **Multi-turn conversation format**: System → User → Assistant pattern
- **Expert persona definition**: System message establishes role as hate speech detection expert
- **Protected group context**: Explicitly mentions LGBTQ+, Mexican, and Middle Eastern communities
- **Demographic labeling**: Each sample includes persona annotation (HISPANIC, MIDDLE_EAST, LGBTQ+) in user message
- **Structured JSON output**: 
  - All samples: `classification`, `confidence`, `reasoning`
  - Hate speech samples additionally include: `protected_group` field
- **Token overhead**: ~150-200 tokens per sample (system message + persona context + text)
- **Data quirk**: Some samples contain double curly braces `{{...}}` in assistant content (escaping artifact)

**Validation Context**:
The chat-based template with explicit protected group context was designed to improve the model's awareness of bias-sensitive demographics during training. The structured output format distinguishes between hate speech (with `protected_group` field identifying the target) and non-hate speech samples. Post-fine-tuning validation showed the model successfully internalized task requirements, achieving F1=0.6692 on production data (1,009 samples). The `protected_group` field in hate speech responses enables precise demographic-stratified bias analysis.

### 1.3 Prompt Format for Training Data

Training samples follow the chat-based conversational format with system, user, and assistant roles:

**For hate speech samples**:
```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert hate speech detection system. Analyze the given text and determine if it contains hate speech targeting protected groups (LGBTQ+, Mexican, Middle Eastern communities)."
    },
    {
      "role": "user",
      "content": "Analyze this text for hate speech:\n\nPersona: HISPANIC\nText: \"so thats where beaners come from\""
    },
    {
      "role": "assistant",
      "content": "{\"classification\": \"hate_speech\", \"confidence\": \"high\", \"reasoning\": \"beaners\", \"protected_group\": \"MEXICAN\"}"
    }
  ]
}
```

**For non-hate speech samples**:
```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert hate speech detection system. Analyze the given text and determine if it contains hate speech targeting protected groups (LGBTQ+, Mexican, Middle Eastern communities)."
    },
    {
      "role": "user",
      "content": "Analyze this text for hate speech:\n\nPersona: MIDDLE_EAST\nText: \"we must learn to live with others even if we do not like them\""
    },
    {
      "role": "assistant",
      "content": "{\"classification\": \"not_hate\", \"confidence\": \"high\", \"reasoning\": \"This text does not contain hate speech.\"}"
    }
  ]
}
```

**Key Properties**:
- **System Message**: Defines the model's role as a hate speech detection expert with context about protected groups
- **User Message**: Contains the text to analyze along with demographic persona annotation (HISPANIC, MIDDLE_EAST, LGBTQ+)
- **Assistant Message**: Structured JSON response with:
  - `classification`: Either `"hate_speech"` or `"not_hate"`
  - `confidence`: Confidence level (typically `"high"`)
  - `reasoning`: Explanation or key offensive terms identified
  - `protected_group`: Target demographic (only for hate_speech samples) - values: `"MEXICAN"`, `"LGBTQ+"`, `"MIDDLE_EAST"`
- **Format**: JSONL (one sample per line) with chat-based messages array for instruction fine-tuning
- **Note**: Some samples have double curly braces `{{...}}` which are escaped during processing

---

## 2. Training Data Configuration

### 2.1 Dataset Composition

**Source**: Unified dataset combining HateXplain and ToxiGen corpora (filtered for LoRA training)  
**Total LoRA Training Samples**: 2,686 (filtered from unified training set)  
**LoRA Validation Split**: 100 samples (size-varied ablation set)  
**Test/Production Validation**: 1,009 samples (unified test set, used for post-training evaluation)

**Rationale for Training Subset (74% of Available Data)**:  
LoRA fine-tuning updates only 0.038%-0.114% of model parameters (7.96M-23.9M trainable params from 20.9B total), making it highly parameter-efficient. Using a smaller, curated subset (2,686 samples) prevents the risk of memorizing training patterns while preserving generalization capability. The 74% sampling strategy balances sufficient data volume for convergence with protection against overfitting to dataset artifacts.

**Unified Dataset Context**:
- Original unified corpus: 5,151 total samples
- Unified training set: 3,628 samples (70.4%)
- Unified validation set: 514 samples (10.0%)
- Unified test set: 1,009 samples (19.6%)

**LoRA Training Data Selection**:
The LoRA training files (`train.jsonl` with 2,686 samples and `validation.jsonl` with 55 samples) represent a filtered subset of the unified dataset. The filtering is performed by the `ft_prompt_generator` module, which selects samples based on split field values from the unified dataset. The test set (1,009 samples) is used exclusively for post-training production validation and is never seen during training.

**Class Distribution** (LoRA training data):
- Hate speech samples: ~45%
- Normal (non-hate) samples: ~55%
- Balanced representation maintained across splits

### 2.2 Data Preprocessing

**Text Normalization**:
- Lowercasing: No (preserved original casing for context)
- Special characters: Retained (important for hate speech patterns)
- URLs/Mentions: Anonymized where present
- Maximum sequence length: 512 tokens (BERT-style tokenization)

**Demographic Annotations**:
The dataset includes target group annotations for bias analysis:
- **LGBTQ+**: 494 samples (49.0% of test set)
- **Mexican**: 209 samples (20.7% of test set)
- **Middle East**: 306 samples (30.3% of test set)

### 2.3 Data Format

**Training File**: `finetuning/data/ft_prompts/train.jsonl`  
**Validation File**: `finetuning/data/ft_prompts/validation.jsonl`

**Sample Structure** (Chat-based format):

*Example 1 - Hate Speech:*
```jsonl
{"messages": [{"role": "system", "content": "You are an expert hate speech detection system. Analyze the given text and determine if it contains hate speech targeting protected groups (LGBTQ+, Mexican, Middle Eastern communities)."}, {"role": "user", "content": "Analyze this text for hate speech:\n\nPersona: HISPANIC\nText: \"so thats where beaners come from\""}, {"role": "assistant", "content": "{\"classification\": \"hate_speech\", \"confidence\": \"high\", \"reasoning\": \"beaners\", \"protected_group\": \"MEXICAN\"}"}]}
```

*Example 2 - Non-Hate Speech:*
```jsonl
{"messages": [{"role": "system", "content": "You are an expert hate speech detection system. Analyze the given text and determine if it contains hate speech targeting protected groups (LGBTQ+, Mexican, Middle Eastern communities)."}, {"role": "user", "content": "Analyze this text for hate speech:\n\nPersona: MIDDLE_EAST\nText: \"we must learn to live with others even if we do not like them\""}, {"role": "assistant", "content": "{\"classification\": \"not_hate\", \"confidence\": \"high\", \"reasoning\": \"This text does not contain hate speech.\"}"}]}
```

**Format Details**:
- **Messages Array**: Each sample contains a conversation with system, user, and assistant turns
- **System Role**: Establishes context and task definition (hate speech detection with protected group focus)
- **User Role**: Presents text with demographic persona label (HISPANIC, MIDDLE_EAST, LGBTQ+) for bias-aware evaluation
- **Assistant Role**: Returns structured JSON with:
  - `classification`: `"hate_speech"` or `"not_hate"`
  - `confidence`: Confidence level (typically `"high"`)
  - `reasoning`: Explanation text or identified offensive terms
  - `protected_group`: Target demographic - only present in hate_speech samples (values: `"MEXICAN"`, `"LGBTQ+"`, `"MIDDLE_EAST"`)
- **Token Limit**: Maximum 512 tokens per conversation (tokenized using model's chat template)
- **Data Note**: Approximately 55% of samples have double curly braces `{{...}}` in assistant content (escaping artifact from data preparation)

### 2.4 Data Augmentation

**Strategy**: No artificial augmentation applied  
**Rationale**: Preserve authentic linguistic patterns and demographic context from original annotated datasets

---

## 3. Hyperparameter Configurations

Three distinct LoRA configurations were evaluated during the experimental phase, each designed to test specific hypotheses about model capacity and module selection.

**For complete experimental results, see**: [LoRA Fine-Tuning Experiment Summary](lora_ft_summary_README.md)

---

### 3.1 Phase 1: Default Configuration

**Config File**: `default.json` | **Hypothesis**: Baseline LoRA with standard rank (r=32) and 2 attention modules

#### Configuration Summary

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base Model | `openai/gpt-oss-20b` (20.9B params) |
| **LoRA** | Rank (r) | 32 |
| | Alpha | 32 |
| | Dropout | 0.05 |
| | Target Modules | `q_proj`, `v_proj` |
| | Trainable Params | 7,962,624 (0.038% of total) |
| **Training** | Learning Rate | 2e-4 |
| | Epochs | 3 |
| | Batch Size (per device) | 4 |
| | Gradient Accumulation | 4 |
| | Effective Batch Size | 64 (4×4 GPUs×4 accum) |
| | Warmup Steps | 100 |
| | Weight Decay | 0.01 |
| | LR Scheduler | Cosine |
| | Total Steps | 126 (42 per epoch) |
| **Quantization** | Load in 4-bit | Yes (NF4) |
| | Compute Dtype | bfloat16 |

---

### 3.2 Phase 2: High Capacity Configuration  

**Config File**: `high_capacity.json` | **Hypothesis**: Doubling LoRA rank to r=64 improves model capacity

#### Configuration Summary

| Category | Parameter | Value | Change from Phase 1 |
|----------|-----------|-------|---------------------|
| **LoRA** | Rank (r) | 64 | 2× increase |
| | Alpha | 64 | 2× increase |
| | Target Modules | `q_proj`, `v_proj` | Same |
| | Trainable Params | 15,925,248 (0.076%) | 2× increase |
| **Training** | Epochs | 5 | +2 epochs |
| | Total Steps | 210 (42 per epoch) | +67% |

**All other parameters**: Same as Phase 1

---

### 3.3 Phase 3: K-Projection with Full Capacity

**Config File**: `k_proj_full_capacity.json` | **Hypothesis**: Adding k_proj module (r=64) improves attention modeling

#### Configuration Summary

| Category | Parameter | Value | Change from Phase 2 |
|----------|-----------|-------|---------------------|
| **LoRA** | Rank (r) | 64 | Same |
| | Target Modules | `q_proj`, `v_proj`, `k_proj` | +k_proj |
| | Trainable Params | 23,887,872 (0.114%) | +50% |
| **Training** | Epochs | 4 | -1 epoch |
| | Total Steps | 168 (42 per epoch) | -20% |

**All other parameters**: Same as Phase 1 and Phase 2

---

### 3.4 Configuration Comparison Matrix

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| **LoRA Rank** | 32 | 64 | 64 |
| **Modules** | q_proj, v_proj | q_proj, v_proj | q_proj, v_proj, k_proj |
| **Trainable Params** | 7.96M | 15.9M | 23.9M |
| **Param % of Total** | 0.038% | 0.076% | 0.114% |
| **Training Epochs** | 3 | 5 | 4 |
| **Total Steps** | 126 | 210 | 168 |
| **Learning Rate** | 2e-4 | 2e-4 | 2e-4 |
| **Batch Size** | 64 | 64 | 64 |
| **4-bit Quant** | NF4 | NF4 | NF4 |

---

### 3.5 Shared Configuration Parameters

The following settings remained constant across all experimental phases:

#### Hardware & Distribution
- **GPUs**: 4× NVIDIA A100 80GB
- **Framework**: PyTorch + DeepSpeed
- **Strategy**: Distributed Data Parallel (DDP)
- **Precision**: bfloat16 mixed precision

#### Training Settings
- **Learning Rate**: 2e-4
- **Batch Size (per device)**: 4
- **Gradient Accumulation**: 4 steps
- **Effective Batch Size**: 64
- **Warmup Steps**: 100
- **Weight Decay**: 0.01
- **Max Gradient Norm**: 1.0
- **LR Scheduler**: Cosine with warmup

#### LoRA Settings
- **Dropout**: 0.05
- **Bias**: none
- **Task Type**: CAUSAL_LM

#### Quantization
- **4-bit**: Enabled (NF4)
- **Compute Dtype**: bfloat16

#### Monitoring
- **Eval Strategy**: Per epoch (42 steps)
- **Logging**: Every 10 steps
- **Early Stopping**: 2 epochs patience, 0.01 threshold

---

## 4. Experimental Results

### 4.1 Complete Results Documentation

Comprehensive experimental results are documented in the **Experiment Summary**:

📊 **[LoRA Fine-Tuning Experiment Summary](lora_ft_summary_README.md)**

**Includes**:
- Training progressions (loss curves, checkpoints)
- Validation metrics (F1, precision, recall, accuracy)
- Bias analysis (FPR/FNR by demographic group)
- Configuration comparisons
- Production validation (1,009 samples)
- Deployment recommendations

### 4.2 Quick Results Overview

| Phase | Config | F1 Score | Best For |
|-------|--------|----------|----------|
| Phase 1 | default (r=32) | 0.6308 | Baseline comparison |
| Phase 2 | high_capacity (r=64) | **0.6852** | **Production deployment** ✅ |
| Phase 3 | k_proj_full (r=64+k) | 0.6491 | Module selection analysis |

**Production F1**: 0.6692 (Phase 2 model on 1,009 samples)

**Navigate to**:
- [Phase 1 Results →](lora_ft_summary_README.md#phase-1-train-with-default-configuration---completed)
- [Phase 2 Results →](lora_ft_summary_README.md#phase-2-train-with-high-capacity-configuration---completed)
- [Phase 3 Results →](lora_ft_summary_README.md#phase-3-k-projection-with-full-capacity---completed)
- [Comparison →](lora_ft_summary_README.md#comparison-summary)
- [Production Validation →](lora_ft_summary_README.md#production-validation-run---best-model)

---

## 5. References

### Primary Research

**LoRA: Low-Rank Adaptation**  
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.  
https://arxiv.org/abs/2106.09685

**QLoRA: Efficient Fine-tuning**  
Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.  
https://arxiv.org/abs/2305.14314

### Datasets

**HateXplain**  
Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021.

**ToxiGen**  
Hartvigsen, T., Gabriel, S., Paullada, A., Sap, M., Guillory, A., & Diaz, M. (2022). *ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection*. ACL 2022.

### Base Model

**GPT-OSS-20B**  
OpenAI. (2024). *GPT-OSS-20B: Open-source GPT model (20.9B parameters)*.  
https://huggingface.co/openai/gpt-oss-20b

---

## Appendix: Project Structure

### Configuration Files
```
finetuning/pipeline/lora/configs/
├── default.json                  # Phase 1 config
├── high_capacity.json            # Phase 2 config (BEST)
├── k_proj_full_capacity.json     # Phase 3 config
└── lora_ft_summary_README.md     # Results documentation
```

### Training & Validation
```
finetuning/
├── data/ft_prompts/
│   ├── train.jsonl              # 2,686 samples
│   └── validation.jsonl         # 55 samples
├── models/
│   ├── lora_default/            # Phase 1 model
│   ├── high_capacity/           # Phase 2 model (PRODUCTION)
│   └── k_proj_full_capacity/    # Phase 3 model
└── outputs/gptoss/post_finetune/
    ├── lora_default/            # Phase 1 results
    ├── high_capacity/           # Phase 2 results + production validation
    └── k_proj_full_capacity/    # Phase 3 results
```

---

**Document Version**: 2.0  
**Last Updated**: October 26, 2025  
**Status**: Complete - 3 phases documented  
**Results**: See [Experiment Summary](lora_ft_summary_README.md)
