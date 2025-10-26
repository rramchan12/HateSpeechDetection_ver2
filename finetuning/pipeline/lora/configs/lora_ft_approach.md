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

The prompt template used for instruction fine-tuning follows a structured format designed to align the model with hate speech detection objectives through clear instruction, context, and expected output formatting.

#### Prompt Template Structure

```json
{
  "instruction": "Classify the following text as 'hate' or 'normal'.",
  "input": "{text}",
  "output_format": "Classification: {label}"
}
```

**Design Rationale**:
- **Simplicity**: Direct classification task without complex reasoning chains
- **Clarity**: Explicit binary choice reduces ambiguity in model responses
- **Consistency**: Standardized format across all training samples
- **Parsability**: Structured output enables reliable extraction of predictions

### 1.2 Baseline Prompt Template

The baseline prompt template (`baseline_v1.json`) was selected based on pre-fine-tuning validation experiments:

**Template Characteristics**:
- Zero-shot instruction format
- No persona-based context or policy guidelines
- Direct text-to-label mapping
- Minimal token overhead (~20-30 tokens per sample)

**Validation Context**:
The baseline template was validated against sophisticated multi-strategy prompts (policy-based, persona-based, combined) in pre-fine-tuning experiments. Post-fine-tuning validation showed the model successfully internalized task requirements, achieving F1=0.6692 on production data (1,009 samples) with simple prompts, comparable to pre-fine-tuning F1=0.615 achieved with complex prompt engineering.

### 1.3 Prompt Format for Training Data

Training samples follow the instruction-tuning format compatible with conversational models:

```jsonl
{
  "instruction": "Classify the following text as 'hate' or 'normal'.",
  "input": "Example text here...",
  "output": "Classification: hate"
}
```

**Key Properties**:
- **Instruction Field**: Consistent task description across all samples
- **Input Field**: Raw text to classify (variable length, max 512 tokens)
- **Output Field**: Expected classification with standard prefix
- **Format**: JSONL (one sample per line) for efficient streaming

---

## 2. Training Data Configuration

### 2.1 Dataset Composition

**Source**: Unified dataset combining HateXplain and ToxiGen corpora  
**Total Samples**: 13,048  
**Training Split**: 2,686 samples (~20.6%)  
**Validation Split**: 55 samples  
**Test Split**: 10,307 samples (production validation)

**Class Distribution**:
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

**Sample Structure**:
```jsonl
{"instruction": "Classify the following text as 'hate' or 'normal'.", "input": "text sample", "output": "Classification: hate"}
{"instruction": "Classify the following text as 'hate' or 'normal'.", "input": "text sample", "output": "Classification: normal"}
```

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
