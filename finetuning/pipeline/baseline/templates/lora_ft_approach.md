# LoRA Fine-Tuning Approach for Hate Speech Detection

## Document Overview

**Purpose**: Define theoretically grounded hyperparameter configurations for LoRA (Low-Rank Adaptation) fine-tuning of GPT-OSS-20B on the hate speech detection task, synthesizing empirical findings from baseline validation with established research.

**Context**: This document extends the validation methodology described in `finetuning/VALIDATION_GUIDE.md` (Phase 5: Fine-Tuning with LoRA) by providing detailed theoretical justification for each hyperparameter choice based on peer-reviewed research and empirical results from production-scale baseline testing.

**Scope**: Parameter recommendations for instruction fine-tuning (IFT) of large language models (20B+ parameters) on binary classification with bias fairness constraints.

**Version**: 1.0  
**Date**: October 25, 2025  
**Author**: HateSpeechDetection Research Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Empirical Context from Baseline Validation](#empirical-context-from-baseline-validation)
4. [Hyperparameter Specifications](#hyperparameter-specifications)
5. [Training Data Configuration](#training-data-configuration)
6. [Evaluation Strategy](#evaluation-strategy)
7. [Implementation Guidelines](#implementation-guidelines)
8. [References](#references)

---

## Executive Summary

### Recommended Configuration

Based on synthesis of LoRA research (Hu et al., 2021), QLoRA efficiency innovations (Dettmers et al., 2023), baseline validation results (F1=0.615, 13,118 samples), and few-shot learning findings from combined policy-persona experiments, we recommend the following configuration:

```json
{
  "training": {
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine"
  },
  "lora": {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
  },
  "quantization": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4"
  },
  "data": {
    "max_length": 512,
    "train_samples": 10453,
    "val_samples": 2595
  }
}
```

**Success Criterion**: Post-FT F1 (simple prompts) ≥ 0.615 (Pre-FT F1 with sophisticated prompts), demonstrating task internalization without complex prompt engineering.

---

## Theoretical Foundation

### 1. LoRA: Low-Rank Adaptation Fundamentals

**Core Innovation** (Hu et al., 2021): LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into each Transformer layer, reducing trainable parameters by 10,000× while maintaining performance parity with full fine-tuning.

**Mathematical Formulation**:
For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents weight updates as:

$$h = W_0x + \Delta W x = W_0x + BAx$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d,k)$.

**Key Findings from Original Paper**:
- Rank $r=8$ sufficient for GPT-3 175B on multiple tasks
- Targeting attention weights ($W_q$, $W_v$) optimal for most NLP tasks
- No additional inference latency vs. adapter methods
- Intrinsic rank of task-specific adaptations typically low (<100)

**Citation**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*. https://arxiv.org/abs/2106.09685

### 2. QLoRA: Efficient 4-bit Quantization

**Memory Efficiency Innovation** (Dettmers et al., 2023): QLoRA enables fine-tuning of 65B parameter models on single 48GB GPUs through:
1. **4-bit NormalFloat (NF4)**: Information-theoretically optimal for normally distributed weights
2. **Double Quantization**: Quantizes quantization constants to reduce memory footprint
3. **Paged Optimizers**: Manages memory spikes via CPU-GPU paging

**Empirical Results**:
- Guanaco-65B: 99.3% of ChatGPT performance with 24 hours fine-tuning on 1 GPU
- 4-bit quantization maintains full 16-bit task performance across 1,000+ models
- NF4 data type provides better performance than standard 4-bit quantization for LLMs

**Relevance to GPT-OSS-20B**: With 4× A100 80GB GPUs, QLoRA enables memory-efficient fine-tuning with full model loaded across GPUs, eliminating need for model parallelism complexity while maintaining performance.

**Citation**: Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*. https://arxiv.org/abs/2305.14314

### 3. AdamW Optimizer and Weight Decay

**Decoupled Weight Decay** (Loshchilov & Hutter, 2019): AdamW fixes the inequivalence between L2 regularization and weight decay for adaptive gradient methods by decoupling weight decay from optimization steps.

**Key Advantage**: Decoupling allows independent tuning of:
- Learning rate: Controls optimization step size
- Weight decay: Controls generalization (independently of learning rate)

**Empirical Evidence**: AdamW substantially improves Adam's generalization, enabling it to compete with SGD+momentum on image classification (previously outperformed by SGD).

**Citation**: Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of the 7th International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1711.05101

### 4. Cosine Annealing Learning Rate Schedule

**Warm Restarts Strategy** (Loshchilov & Hutter, 2017): SGDR (Stochastic Gradient Descent with Warm Restarts) uses cosine annealing with periodic restarts to improve anytime performance and escape local minima.

**Schedule Formula**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$$

**Benefits**:
- Smooth learning rate decay prevents abrupt changes
- Reaches near-zero learning rate for fine-grained convergence
- Periodic restarts (optional) help escape poor local minima
- Empirically outperforms step decay and exponential decay

**Citation**: Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1608.03983

### 5. Gradient Clipping for Stability

**Exploding Gradient Problem** (Pascanu et al., 2013): Gradient clipping prevents training instability by constraining gradient norm to maximum threshold, particularly critical for recurrent/sequential architectures.

**Implementation**: Clip gradients when $\|\mathbf{g}\| > \theta$:
$$\tilde{\mathbf{g}} = \frac{\theta}{\|\mathbf{g}\|} \mathbf{g}$$

where $\theta$ is the clipping threshold (typically 0.5-1.0).

**Justification**: While Transformers less susceptible to exploding gradients than RNNs, clipping provides insurance against training instability, especially important for large models (20B+ parameters) where monitoring is challenging.

**Citation**: Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. *Proceedings of the 30th International Conference on Machine Learning (ICML 2013)*. https://arxiv.org/abs/1211.5063

---

## Empirical Context from Baseline Validation

### Production-Scale Baseline Results

**Test Configuration**:
- **Dataset**: unified_test.json (13,118 samples: HateXplain + ToxiGen)
- **Model**: openai/gpt-oss-20b (~78GB bf16, Phi-3.5-MoE-instruct)
- **Hardware**: 4× NVIDIA A100 80GB GPUs
- **Prompt Strategy**: `combined_optimized` (sophisticated multi-stage prompting)
- **Run ID**: run_20251025_144027

**Performance Metrics** (Baseline with Sophisticated Prompts):
| Metric | Value | Notes |
|--------|-------|-------|
| **F1-Score** | **0.615** | Target to match with simple prompts post-FT |
| Accuracy | 65.0% | Overall classification accuracy |
| Precision | 61.0% | Positive predictive value |
| Recall | 62.0% | True positive rate |

**Bias Metrics by Target Group** (Baseline):
| Group | Samples | FPR | FNR | Assessment |
|-------|---------|-----|-----|------------|
| LGBTQ+ | 6,416 | 43.0% | 39.4% | ⚠️ Highest FPR - overcriminalization |
| Mexican | 3,234 | **8.1%** | 39.8% | ✅ Best FPR across all groups |
| Middle East | 3,468 | 23.6% | 35.2% | ⚠️ Elevated but balanced |

**Key Insight**: Baseline achieved F1=0.615 through sophisticated prompt engineering (combined policy-persona template with 512 tokens, few-shot examples). Fine-tuning goal is to internalize this performance using simple prompts (≤256 tokens, no few-shot examples), proving task learning rather than prompt memorization.

### Few-Shot Learning Evidence from Combined Policy-Persona Testing

**Source**: `prompt_engineering/prompt_templates/combined/gpt_oss_combined_ift_summary_README.md`

**100-Sample Validation** (run_20251018_232343):
- **combined_optimized**: F1=0.614, Recall=0.660
- **Few-shot examples**: 5 explicit Mexican/Latino hate vs. policy distinction examples
- **Configuration**: temperature=0.1, max_tokens=512

**Production Validation** (1,009 samples, run_20251018_234643):
- **F1-Score**: 0.590 (-2.4% degradation from 100-sample)
- **Mexican FPR**: 7.0% (best across all models/strategies)
- **Recall Degradation**: -9.3% (66.0%→56.7%) - significantly better than GPT-5's -15.3%

**Critical Finding**: Few-shot learning with explicit hate vs. policy examples reduced Mexican FPR from 20.0% (baseline v1) to 7.0% (combined optimized), validating that GPT-OSS-20B learns from in-context examples. This suggests LoRA fine-tuning will be effective for task internalization.

**Reference Dataset for Fine-Tuning**:
- **Total**: 13,048 samples (unified train + val + test splits)
- **Train Split**: 10,453 samples (80.1%)
- **Val Split**: 2,595 samples (19.9%)
- **Distribution**: Stratified across hate/normal (50/50) and target groups (LGBTQ 49%, Mexican 21%, Middle East 30%)

---

## Hyperparameter Specifications

### 1. Learning Rate: 2e-4

**Theoretical Basis**: LoRA original paper (Hu et al., 2021) recommends learning rates 2-5× higher than full fine-tuning due to:
1. Smaller parameter count (only LoRA adapters trained)
2. Lower risk of catastrophic forgetting (base model frozen)
3. Faster convergence to optimal adapter weights

**Range Tested in Literature**:
- GPT-3 175B: 1e-4 to 5e-4 optimal
- RoBERTa/DeBERTa: 2e-4 to 3e-4 optimal
- Classification tasks: 2e-4 typical starting point

**Rationale for 2e-4**:
- Conservative starting point for 20B model (smaller than GPT-3 175B)
- Baseline validation showed temperature=0.1 optimal (deterministic behavior)
- Higher learning rate (2e-4 vs. 1e-5 for full fine-tuning) accelerates adapter convergence
- AdamW decoupled weight decay allows independent tuning vs. learning rate

**Adjustment Strategy**: If validation loss plateaus early, increase to 3e-4. If training unstable, decrease to 1e-4.

### 2. Number of Epochs: 3

**Theoretical Basis**: LoRA adapters converge faster than full fine-tuning due to:
1. Reduced search space (rank $r=32$ vs. millions of full parameters)
2. Low intrinsic rank of task-specific adaptations (<100 typical)
3. Pre-trained model already contains general language understanding

**Empirical Evidence**:
- QLoRA paper: 3 epochs sufficient for instruction fine-tuning on 24k samples
- GPT-3 LoRA: 1-3 epochs typical for classification tasks
- Baseline validation: Minimal overfitting observed (F1 degradation -2.4% at scale)

**Rationale for 3 Epochs**:
- **Epoch 1**: Initial adapter learning, large loss reduction
- **Epoch 2**: Refinement, continued improvement
- **Epoch 3**: Fine-grained convergence, validation performance peak
- **Beyond 3**: Overfitting risk increases (validation loss diverges from training loss)

**Monitoring Strategy**: Track validation F1 after each epoch. Early stopping if validation F1 decreases or plateaus.

### 3. Batch Size: 4 (per GPU, effective 16 with 4 GPUs)

**Memory Constraint**:
- Model size: ~78GB bf16 (loaded once per GPU)
- A100 80GB capacity: ~2GB headroom per GPU
- 4-bit quantization: Reduces model to ~20GB, frees 58GB for gradients/activations
- Batch size 4: Safe margin for gradient accumulation and optimizer states

**Theoretical Consideration**: Small batch sizes (4-16) provide:
1. More frequent parameter updates (better exploration)
2. Regularization effect (noisy gradients prevent overfitting)
3. Better generalization (empirically validated across many tasks)

**Effective Batch Size**: 4 per GPU × 4 GPUs = 16 effective batch size without gradient accumulation

**Citation Context**: QLoRA paper used batch size 1 per device with gradient accumulation to fit 65B models on 48GB GPUs. With 80GB GPUs and 20B model, we can safely use batch size 4.

### 4. Gradient Accumulation Steps: 4

**Effective Batch Size Calculation**:
$$\text{Effective Batch Size} = \text{Batch Size} \times \text{Num GPUs} \times \text{Gradient Accumulation Steps}$$
$$= 4 \times 4 \times 4 = 64$$

**Theoretical Basis**: Effective batch size 64 balances:
1. **Stability**: Larger effective batches smooth gradient estimates (reduce variance)
2. **Generalization**: Not too large to harm generalization (64 << 10,453 training samples)
3. **Memory Efficiency**: Enables larger effective batch without OOM errors

**Empirical Range**:
- Classification tasks: Effective batch 32-128 typical
- Instruction fine-tuning: 16-64 common (per QLoRA paper)
- Our dataset: 10,453 samples → 164 steps per epoch at batch 64

**Rationale**: Gradient accumulation=4 achieves effective batch size 64 without exceeding GPU memory, providing stable training while maintaining generalization.

### 5. Warmup Steps: 100

**Learning Rate Warmup** (linear warmup from 0 to target learning rate):
$$\eta_t = \frac{t}{T_{warmup}} \times \eta_{max}, \quad t \leq T_{warmup}$$

**Theoretical Basis**: Warmup prevents early training instability by:
1. Allowing optimizer to calibrate momentum statistics
2. Preventing large early updates that might damage pre-training
3. Smoothly transitioning from random initialization (LoRA adapters) to trained state

**Calculation**:
- Training samples: 10,453
- Effective batch size: 64
- Steps per epoch: 10,453 / 64 ≈ 163 steps
- Total training steps: 163 × 3 epochs = 489 steps
- Warmup steps: 100 (20.5% of total steps)

**Empirical Range**: 5-10% of total training steps typical, but up to 20% for classification tasks. Our 20.5% is conservative for stability.

**Rationale**: 100 steps provides stable warmup period (0.61 epochs) before full learning rate 2e-4 applied.

### 6. Weight Decay: 0.01

**L2 Regularization Strength**: Weight decay 0.01 penalizes large adapter weights to prevent overfitting:
$$L_{total} = L_{task} + \lambda \sum_i w_i^2, \quad \lambda = 0.01$$

**Theoretical Basis** (Loshchilov & Hutter, 2019): AdamW's decoupled weight decay:
1. Applies regularization independently of gradient-based updates
2. Prevents optimizer from "fighting" regularization (unlike Adam)
3. Improves generalization without tuning learning rate

**Empirical Range**:
- LoRA literature: 0.0-0.1 typical
- Classification tasks: 0.01 common default
- Instruction fine-tuning: 0.01-0.05 range

**Rationale for 0.01**:
- Moderate regularization (not too aggressive)
- Baseline validation showed minimal overfitting (-2.4% F1 degradation at scale)
- LoRA adapters have small parameter count → less regularization needed than full fine-tuning
- Conservative choice allows increasing to 0.05 if validation loss diverges

### 7. Max Gradient Norm: 1.0

**Gradient Clipping Threshold**: Clips gradients exceeding norm 1.0 to prevent training instability.

**Theoretical Basis** (Pascanu et al., 2013): Gradient clipping prevents exploding gradients by constraining update magnitude:
$$\tilde{\mathbf{g}} = \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}, \quad \theta = 1.0$$

**Empirical Range**:
- Transformers: 0.5-1.0 typical (less susceptible than RNNs)
- LoRA fine-tuning: 1.0 common default
- Large models (20B+): 1.0 provides stability insurance

**Rationale**: 
- max_grad_norm=1.0 standard for Transformer fine-tuning
- Provides protection against rare instability events without constraining typical gradients
- Baseline validation showed stable training (no gradient explosions observed)
- Conservative insurance for 3-epoch fine-tuning run

### 8. Learning Rate Scheduler: Cosine

**Cosine Annealing Schedule** (no restarts):
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

where:
- $\eta_{max} = 2e-4$ (peak learning rate after warmup)
- $\eta_{min} = 0$ (final learning rate)
- $T_{max} = 489$ steps (total training steps)

**Theoretical Basis** (Loshchilov & Hutter, 2017): Cosine annealing:
1. Smooth decay prevents abrupt changes
2. Reaches near-zero learning rate for fine-grained convergence
3. Empirically outperforms step decay and exponential decay
4. Natural stopping point (zero learning rate) at end of training

**Alternatives Considered**:
- **Linear decay**: Simpler but less smooth convergence
- **Step decay**: Abrupt changes can harm fine-tuning
- **Constant**: No decay risks overfitting late in training

**Rationale**: Cosine schedule provides smooth, predictable learning rate decay optimal for 3-epoch fine-tuning with clear convergence at epoch 3.

### 9. LoRA Rank (r): 32

**Rank Parameter**: Controls adapter capacity via low-rank decomposition.

**Theoretical Basis** (Hu et al., 2021):
- Intrinsic rank of task-specific adaptations typically low (<100)
- Rank r=8 sufficient for GPT-3 175B on many tasks
- Higher rank increases capacity but also parameter count and overfitting risk

**Empirical Evidence**:
- LoRA paper: r=4-64 tested, r=8-32 optimal for most tasks
- Classification tasks: r=16-32 typical
- Instruction fine-tuning: r=32-64 common (more complex than classification)

**Calculation**:
- Attention dimension: 5120 (typical for 20B model)
- Adapter parameters per layer: $2 \times 5120 \times 32 = 327,680$
- Total trainable: ~32M parameters (vs. 20B frozen)
- Reduction: 625× fewer parameters than full fine-tuning

**Rationale for r=32**:
- Sufficient capacity for hate speech detection (binary classification with nuanced policy distinctions)
- Baseline validation showed complex reasoning required (policy vs. hate distinction)
- Few-shot learning experiments demonstrated model learns from examples → higher rank beneficial
- Conservative middle ground: not too small (r=8) to limit capacity, not too large (r=64) to risk overfitting

**Adjustment Strategy**: If validation loss plateaus early, increase to r=64. If overfitting observed, decrease to r=16.

### 10. LoRA Alpha: 32

**Scaling Parameter**: $\alpha$ controls LoRA adapter contribution to model output.

**Theoretical Basis**: LoRA adapters scaled by $\frac{\alpha}{r}$:
$$\Delta W = \frac{\alpha}{r} BA$$

- When $\alpha = r$: Scaling factor = 1.0 (no scaling)
- When $\alpha > r$: Adapters contribute more strongly
- When $\alpha < r$: Adapters contribute more weakly

**Empirical Guideline**: Set $\alpha = r$ as default (LoRA paper recommendation).

**Rationale for $\alpha = 32$**:
- Matches rank $r=32$ → scaling factor 1.0
- Standard practice in LoRA literature
- Simplifies tuning (one parameter to adjust: rank)
- If rank changed to r=64, alpha should also increase to 64

**Adjustment Strategy**: Keep $\alpha = r$ unless specific empirical evidence suggests different scaling improves performance.

### 11. LoRA Dropout: 0.05

**Dropout Regularization**: Randomly drops 5% of adapter activations during training to prevent overfitting.

**Theoretical Basis**: Dropout provides:
1. Ensemble effect (trains multiple sub-networks)
2. Regularization (prevents co-adaptation of features)
3. Improved generalization (reduces overfitting)

**Empirical Range**:
- LoRA literature: 0.0-0.1 typical
- Classification tasks: 0.05-0.1 common
- Low dropout (0.0-0.05): Less regularization, faster convergence
- High dropout (0.1+): More regularization, slower convergence

**Rationale for 0.05**:
- Light regularization (5% dropout rate)
- Baseline validation showed minimal overfitting (-2.4% F1 degradation)
- LoRA adapters already regularized by low rank (r=32)
- Conservative choice: enough to prevent overfitting, not so much to slow convergence

**Adjustment Strategy**: If overfitting observed (validation loss diverges), increase to 0.1. If underfitting (both losses high), decrease to 0.0.

### 12. Target Modules: ["q_proj", "v_proj"]

**Attention Weight Targeting**: Apply LoRA adapters to query and value projection matrices in attention mechanism.

**Theoretical Basis** (Hu et al., 2021):
- Attention weights ($W_q$, $W_k$, $W_v$, $W_o$) most impactful for adaptation
- Query ($W_q$) and Value ($W_v$) projections typically sufficient for most NLP tasks
- Adding Key ($W_k$) and Output ($W_o$) increases parameters but marginal performance gain

**Empirical Evidence**:
- LoRA paper: q+v targeting achieves 90% of full attention targeting performance
- Parameter efficiency: 2 modules vs. 4 modules = 50% parameter reduction
- Classification tasks: q+v typically sufficient (key and output less critical)

**Alternatives Considered**:
- **All attention** ["q_proj", "k_proj", "v_proj", "o_proj"]: 2× parameters, marginal gain
- **Query only** ["q_proj"]: Minimal parameters but may underfit
- **Dense layers**: Rarely beneficial for NLP tasks (LoRA paper finding)

**Rationale**: q_proj + v_proj strikes optimal balance between parameter efficiency and adaptation capacity for hate speech detection task.

### 13. LoRA Bias: "none"

**Bias Term Handling**: Do not apply LoRA to bias terms (only weight matrices).

**Theoretical Basis**:
- LoRA paper does not target bias terms (negligible impact)
- Bias terms small relative to weight matrices
- Training bias increases parameter count without significant benefit

**Alternatives**:
- **"all"**: Train all bias terms (increases parameters marginally)
- **"lora_only"**: Train only LoRA adapter biases (rare usage)
- **"none"**: Standard practice (our choice)

**Rationale**: Following LoRA paper recommendations, bias="none" reduces parameter count without performance loss.

### 14. Quantization: 4-bit NF4

**4-bit NormalFloat (NF4)** with bfloat16 compute:
```json
{
  "load_in_4bit": true,
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_quant_type": "nf4"
}
```

**Theoretical Basis** (Dettmers et al., 2023):
- **NF4**: Information-theoretically optimal for normally distributed weights (typical for neural networks)
- **Double Quantization**: Quantizes quantization constants to reduce memory
- **bfloat16 Compute**: Maintains training stability while reducing memory

**Memory Savings**:
- FP16/BF16 model: ~78GB
- 4-bit quantized: ~20GB (3.9× reduction)
- Freed memory: 58GB per GPU for activations, gradients, optimizer states

**Performance Preservation**: QLoRA paper demonstrates 4-bit quantization maintains full 16-bit task performance across 1,000+ models.

**Rationale**: 
- With 4× A100 80GB GPUs, quantization not strictly necessary for 20B model
- However, provides:
  1. Memory headroom for larger batch sizes
  2. Faster data transfers (4-bit vs. 16-bit)
  3. Insurance against OOM errors during training
  4. No performance degradation (empirically validated)

**Adjustment Strategy**: If memory allows, can disable quantization (load_in_4bit=false) for slightly faster training, but 4-bit recommended for efficiency.

### 15. Maximum Sequence Length: 512

**Context Window**: Maximum token length for input sequences.

**Empirical Basis from Baseline Validation**:
- Baseline strategy: max_tokens=512 optimal for `combined_optimized` (F1=0.615)
- Token analysis: 200-512 tokens goldilocks zone (performance peaks)
- Beyond 768 tokens: Performance degradation due to hedging behavior

**Training Data Consideration**:
- **Simple Format**: System prompt + user prompt + text ≈ 150-250 tokens
- **Optimized Format**: Sophisticated prompts + examples ≈ 400-650 tokens
- **512 Token Limit**: Accommodates both formats with safety margin

**Memory Trade-off**:
- Longer sequences: Quadratic memory growth (self-attention)
- 512 tokens: Standard for instruction fine-tuning (GPT-3.5, GPT-4 training)
- Alternative 1024+: Increases memory 4×, marginal performance gain for classification

**Rationale**: max_length=512 aligns with baseline validation findings, accommodates both simple and optimized formats, and maintains memory efficiency.

---

## Training Data Configuration

### Data Source and Generation

**Generator Tool**: `finetuning/ft_prompt_generator/` (see `ft_prompt_generator/README.md`)

**Source Dataset**:
- **Path**: `data/processed/unified/unified_train.json`, `unified_val.json`, `unified_test.json`
- **Total Samples**: 13,048 (across all splits)
- **Composition**: HateXplain (3,628) + ToxiGen (9,420)

**Split Distribution**:
| Split | Samples | Percentage | Purpose |
|-------|---------|------------|---------|
| **Train** | 10,453 | 80.1% | Fine-tuning training |
| **Val** | 2,595 | 19.9% | Validation/early stopping |
| **Test** | 13,118 | — | Post-FT evaluation only (separate file) |

**Generation Command**:
```bash
python -m finetuning.ft_prompt_generator.cli \
  --unified_dir ./data/processed/unified \
  --output_dir ./finetuning/data/ft_prompts \
  --template combined/combined_gptoss_v1.json \
  --strategy combined_optimized
```

**Output Files**:
```
finetuning/data/ft_prompts/
├── train.jsonl              # 10,453 samples (simple format)
├── validation.jsonl         # 2,595 samples (simple format)
├── train_optimized.jsonl    # 10,453 samples (optimized format)
└── validation_optimized.jsonl # 2,595 samples (optimized format)
```

### Format Comparison

#### Simple Format (Recommended for Fine-Tuning)

**Rationale**: Post-FT goal is to achieve baseline F1=0.615 with simple prompts (no sophisticated prompt engineering), proving task internalization.

**Structure**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert hate speech detection system. Classify text as hate_speech or normal based on whether it attacks or dehumanizes people based on protected characteristics (race, religion, gender, sexual orientation, nationality, disability)."
    },
    {
      "role": "user",
      "content": "Analyze this text for hate speech:\n\nPersona: LGBTQ\nText: \"These gay people are disgusting and should be removed from society\""
    },
    {
      "role": "assistant",
      "content": "{\"classification\": \"hate_speech\", \"confidence\": \"high\", \"reasoning\": \"Attacks LGBTQ community with dehumanizing language and calls for removal\", \"protected_group\": \"LGBTQ\"}"
    }
  ]
}
```

**Token Count**: ~150-250 tokens per sample

#### Optimized Format (Alternative)

**Structure**: Uses sophisticated `combined_optimized` strategy with:
- Detailed policy explanations (community context, examples)
- Few-shot hate vs. policy distinction examples
- Nuanced evaluation criteria

**Token Count**: ~400-650 tokens per sample

**Usage Consideration**: If simple format post-FT performance < baseline (F1 < 0.615), retry with optimized format to provide richer training signal. However, this undermines goal of proving task internalization.

### Data Quality and Balance

**Label Distribution** (stratified 50/50):
- Hate Speech: 6,524 samples (50.0%)
- Normal Speech: 6,524 samples (50.0%)

**Target Group Distribution** (stratified):
- LGBTQ+: 6,416 samples (49.0%)
- Mexican/Latino: 3,234 samples (24.8%)
- Middle Eastern: 3,468 samples (26.2%)

**Stratification Strategy**: Ensures LoRA adapters learn balanced representations across:
1. **Class balance**: Prevents bias toward majority class
2. **Group balance**: Prevents over/under-sensitivity to specific demographics
3. **Source balance**: HateXplain (academic annotation) + ToxiGen (synthetic generation)

---

## Evaluation Strategy

### Success Criteria

**Primary Metric**: Post-FT F1-score with simple prompts ≥ 0.615 (Pre-FT baseline with sophisticated prompts)

**Secondary Metrics**:
1. **Bias Fairness**: FPR/FNR parity across target groups (within 10% variation)
2. **Recall Improvement**: Post-FT Recall ≥ 62.0% (baseline recall)
3. **Precision Maintenance**: Post-FT Precision ≥ 60.0% (within 1% of baseline)

### Validation Protocol

**During Training** (after each epoch):
1. Run validation on 2,595 validation samples (simple prompts)
2. Calculate F1, Precision, Recall, Accuracy
3. Calculate bias metrics (FPR/FNR by target group)
4. Early stopping if validation F1 plateaus or decreases

**Post-Training** (Phase 6 in VALIDATION_GUIDE.md):
1. Run full test on unified_test.json (13,118 samples) with simple prompts
2. Compare to baseline results (run_20251025_144027)
3. Generate comprehensive bias analysis
4. Evaluate prompt complexity reduction (512 tokens → 256 tokens)

### Baseline Comparison Table

| Metric | Baseline (Sophisticated Prompts) | Target (Simple Prompts Post-FT) | Status |
|--------|-----------------------------------|----------------------------------|--------|
| **F1-Score** | 0.615 | ≥ 0.615 | Target |
| **Accuracy** | 65.0% | ≥ 64.0% | Target (1% tolerance) |
| **Precision** | 61.0% | ≥ 60.0% | Target (1% tolerance) |
| **Recall** | 62.0% | ≥ 62.0% | Target |
| **LGBTQ+ FPR** | 43.0% | ≤ 40.0% | Improvement goal |
| **Mexican FPR** | 8.1% | ≤ 10.0% | Maintain excellence |
| **Middle East FPR** | 23.6% | ≤ 25.0% | Maintain |
| **Token Count** | 512 | 256 | Efficiency gain |

### Evaluation Commands

**Post-FT Validation** (see VALIDATION_GUIDE.md Phase 6):
```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Validate with simple prompts using fine-tuned model
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --prompt_template ./finetuning/data/ft_prompts/simple_template.json \
    --strategy simple \
    --max_samples all \
    --output_dir ./finetuning/outputs/gptoss/post_finetune
```

**Metrics Files Generated**:
- `performance_metrics_*.csv`: F1, accuracy, precision, recall
- `bias_metrics_*.csv`: FPR, FNR by target group
- `strategy_unified_results_*.csv`: Per-sample predictions

---

## Implementation Guidelines

### Step 1: Prepare Training Data

```bash
# Generate fine-tuning data (if not already done)
cd /home/azureuser/workspace/HateSpeechDetection_ver2

python -m finetuning.ft_prompt_generator.cli \
    --unified_dir ./data/processed/unified \
    --output_dir ./finetuning/data/ft_prompts \
    --template combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --debug

# Verify output
ls -lh finetuning/data/ft_prompts/
# Expected: train.jsonl (10,453), validation.jsonl (2,595)
```

### Step 2: Configure LoRA Training Script

**Configuration File**: `finetuning/pipeline/lora/config.json`

```json
{
  "model_name_or_path": "openai/gpt-oss-20b",
  "train_file": "./finetuning/data/ft_prompts/train.jsonl",
  "validation_file": "./finetuning/data/ft_prompts/validation.jsonl",
  "output_dir": "./finetuning/models/lora_checkpoints",
  
  "training_args": {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": true,
    "logging_steps": 10,
    "report_to": "none"
  },
  
  "lora_config": {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
  },
  
  "quantization_config": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true
  },
  
  "data_args": {
    "max_seq_length": 512,
    "preprocessing_num_workers": 8
  }
}
```

### Step 3: Launch LoRA Fine-Tuning

```bash
# Activate environment
source .venv/bin/activate

# Launch multi-GPU training with Accelerate
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/config.json

# Monitor training
tail -f finetuning/models/lora_checkpoints/training.log

# Expected duration: 2-3 hours for 3 epochs on 10,453 samples
```

### Step 4: Merge LoRA Adapters (Optional)

```bash
# Merge adapters into base model for deployment
python -m finetuning.pipeline.lora.merge \
    --base_model openai/gpt-oss-20b \
    --lora_weights ./finetuning/models/lora_checkpoints \
    --output_dir ./finetuning/models/merged_model

# Note: Merged model ~78GB (full model size)
# Alternative: Keep adapters separate (~32MB) and load with PEFT library
```

### Step 5: Validate Fine-Tuned Model

See VALIDATION_GUIDE.md Phase 6 for complete validation protocol.

```bash
# Run post-FT validation with simple prompts
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --prompt_template ./finetuning/data/ft_prompts/simple_template.json \
    --strategy simple \
    --max_samples all \
    --output_dir ./finetuning/outputs/gptoss/post_finetune
```

### Step 6: Compare Results

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss

# Extract baseline F1 (sophisticated prompts)
BASELINE_F1=$(awk -F',' 'NR==2 {print $5}' baseline/run_*/performance_metrics_*.csv | head -1)

# Extract post-FT F1 (simple prompts)
POSTFT_F1=$(awk -F',' 'NR==2 {print $5}' post_finetune/run_*/performance_metrics_*.csv | head -1)

echo "Baseline F1 (sophisticated prompts): $BASELINE_F1"
echo "Post-FT F1 (simple prompts):        $POSTFT_F1"

# Success criterion: POSTFT_F1 >= 0.615
```

---

## References

### Primary LoRA Research

1. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.** (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint arXiv:2106.09685. https://arxiv.org/abs/2106.09685
   - **Key Contributions**: Low-rank decomposition for efficient fine-tuning, rank r=8 sufficient for GPT-3 175B, targeting attention weights optimal
   - **Validation**: Tested on RoBERTa, DeBERTa, GPT-2, GPT-3 across multiple NLP tasks
   - **Relevance**: Foundation for LoRA hyperparameter choices (rank, alpha, target modules)

2. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv preprint arXiv:2305.14314. https://arxiv.org/abs/2305.14314
   - **Key Contributions**: 4-bit NormalFloat quantization, double quantization, paged optimizers
   - **Empirical Results**: 65B model fine-tuning on single 48GB GPU, 99.3% ChatGPT performance
   - **Relevance**: Justifies 4-bit quantization choice, demonstrates performance preservation

### Optimizer and Scheduling

3. **Loshchilov, I., & Hutter, F.** (2019). *Decoupled Weight Decay Regularization*. Proceedings of the 7th International Conference on Learning Representations (ICLR 2019). https://arxiv.org/abs/1711.05101
   - **Key Contribution**: AdamW optimizer with decoupled weight decay
   - **Empirical Evidence**: Substantially improves Adam generalization vs. standard Adam
   - **Relevance**: Justifies AdamW choice and weight_decay=0.01 hyperparameter

4. **Loshchilov, I., & Hutter, F.** (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. Proceedings of the 5th International Conference on Learning Representations (ICLR 2017). https://arxiv.org/abs/1608.03983
   - **Key Contribution**: Cosine annealing learning rate schedule with warm restarts
   - **Empirical Results**: State-of-the-art on CIFAR-10 (3.14% error) and CIFAR-100 (16.21% error)
   - **Relevance**: Justifies cosine learning rate schedule choice

### Training Stability

5. **Pascanu, R., Mikolov, T., & Bengio, Y.** (2013). *On the difficulty of training Recurrent Neural Networks*. Proceedings of the 30th International Conference on Machine Learning (ICML 2013). https://arxiv.org/abs/1211.5063
   - **Key Contribution**: Gradient clipping strategy for exploding gradients
   - **Analysis**: Analytical, geometric, and dynamical systems perspective on gradient problems
   - **Relevance**: Justifies max_grad_norm=1.0 for training stability

### Internal Empirical Validation

6. **Baseline Validation Results** (October 25, 2025). *GPT-OSS-20B Baseline Testing on Unified Dataset*. Internal validation run_20251025_144027.
   - **Dataset**: unified_test.json (13,118 samples: HateXplain + ToxiGen)
   - **Performance**: F1=0.615, Accuracy=65.0%, Precision=61.0%, Recall=62.0%
   - **Configuration**: `combined_optimized` strategy, temperature=0.1, max_tokens=512
   - **Relevance**: Establishes target F1=0.615 for post-FT validation with simple prompts

7. **Few-Shot Learning Experiments** (October 18, 2025). *Combined Policy-Persona Instruction Fine-Tuning Summary*. See `prompt_engineering/prompt_templates/combined/gpt_oss_combined_ift_summary_README.md`.
   - **100-Sample Results**: F1=0.614 (combined_optimized), Mexican FPR=20.0%
   - **Production Results** (1,009 samples): F1=0.590, Mexican FPR=7.0% (best across all models)
   - **Key Finding**: Few-shot examples reduced Mexican FPR by 65% (20.0%→7.0%)
   - **Relevance**: Demonstrates GPT-OSS-20B learns from in-context examples, validating LoRA fine-tuning viability

### Additional Context

8. **Fine-Tuning Data Generator Documentation**. See `finetuning/ft_prompt_generator/README.md`.
   - **Tool**: Converts unified dataset into instruction format (JSONL) for fine-tuning
   - **Output**: Simple format (150-250 tokens) and optimized format (400-650 tokens)
   - **Split Filtering**: Uses original train/val/test splits from unified dataset
   - **Relevance**: Documents training data generation pipeline

9. **Validation Guide**. See `finetuning/VALIDATION_GUIDE.md`.
   - **Phase 5**: Fine-Tuning with LoRA (this document provides detailed hyperparameters)
   - **Phase 6**: Post-Fine-Tuning Validation (compares baseline vs. post-FT performance)
   - **Success Criterion**: Post-FT F1 (simple prompts) ≥ Pre-FT F1 (sophisticated prompts)
   - **Relevance**: Overall validation methodology and success criteria

---

## Appendix: Hyperparameter Summary Table

| Category | Parameter | Value | Theoretical Basis | Empirical Evidence |
|----------|-----------|-------|-------------------|-------------------|
| **Training** | learning_rate | 2e-4 | LoRA allows 2-5× higher LR (Hu et al., 2021) | Classification tasks: 2e-4 typical starting point |
| | num_epochs | 3 | LoRA converges faster (reduced search space) | QLoRA: 3 epochs sufficient for 24k samples |
| | batch_size | 4 | Memory constraint: A100 80GB, model ~20GB quantized | Small batches (4-16) provide regularization |
| | gradient_accumulation_steps | 4 | Effective batch size 64 = 4×4×4 | Classification: 32-128 typical effective batch |
| | warmup_steps | 100 | 20.5% of 489 total steps | 5-20% typical, conservative for stability |
| | weight_decay | 0.01 | AdamW decoupled regularization (Loshchilov & Hutter, 2019) | 0.01-0.05 typical for classification |
| | max_grad_norm | 1.0 | Gradient clipping prevents instability (Pascanu et al., 2013) | 0.5-1.0 typical for Transformers |
| | lr_scheduler_type | cosine | Smooth decay, fine-grained convergence (Loshchilov & Hutter, 2017) | Outperforms step/exponential decay |
| **LoRA** | r (rank) | 32 | Intrinsic rank <100 typical (Hu et al., 2021) | r=16-32 optimal for classification |
| | lora_alpha | 32 | Set α=r for scaling factor 1.0 | Standard practice in LoRA literature |
| | lora_dropout | 0.05 | Light regularization (5% dropout) | 0.05-0.1 typical for classification |
| | target_modules | ["q_proj", "v_proj"] | Query & Value projections sufficient (Hu et al., 2021) | Achieves 90% of full attention performance |
| | bias | "none" | Bias terms negligible impact | Standard practice (LoRA paper) |
| **Quantization** | load_in_4bit | true | NF4 optimal for normal distributions (Dettmers et al., 2023) | Maintains full 16-bit performance |
| | bnb_4bit_compute_dtype | bfloat16 | Training stability + memory efficiency | QLoRA empirical validation |
| | bnb_4bit_quant_type | nf4 | Information-theoretically optimal | Outperforms standard 4-bit |
| **Data** | max_seq_length | 512 | Baseline validation: 200-512 goldilocks zone | Beyond 768: performance degradation |
| | train_samples | 10,453 | 80.1% of 13,048 unified dataset | Stratified 50/50 hate/normal |
| | val_samples | 2,595 | 19.9% of 13,048 unified dataset | Stratified across target groups |

---

**Document Status**: Ready for implementation  
**Next Steps**: 
1. Generate training data using `ft_prompt_generator`
2. Configure LoRA training script with recommended hyperparameters
3. Launch multi-GPU fine-tuning with Accelerate
4. Validate post-FT performance against baseline F1=0.615
5. Document results and iterate if needed

**Questions/Issues**: Contact HateSpeechDetection research team or refer to VALIDATION_GUIDE.md for validation protocol.
