# Supervised Fine-Tuning Implementation Framework: Architecture and Design

## Framework Overview

The Supervised Fine-Tuning (SFT) Framework implements a dual-pipeline architecture for establishing baseline performance and parameter-efficient fine-tuning of large language models (20B+ parameters) for hate speech detection. The framework supports both inference-only baseline validation and LoRA-based fine-tuning with automatic multi-GPU distribution via HuggingFace Accelerate. The design emphasizes reproducibility through configuration-driven workflows, scalability through distributed computing, and efficiency through quantization (QLoRA 4-bit) and low-rank adaptation.

## Architectural Block Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│         SFT FRAMEWORK: Training → Evaluation Pipeline        │
└───────────────────────────┬──────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
       PHASE 1                         PHASE 2
┌──────────────────┐  safetensors  ┌─────────────────┐
│ LoRA Training    │───────────────>│ Evaluation      │
│ (train.py)       │  adapter_config│ (runner.py)     │
│ • 4-bit QLoRA    │                │ • Load adapters │
│ • Accelerate     │                │ • Inference     │
│ • 4x GPU         │                │ • Metrics       │
└──────────────────┘                └─────────────────┘
         │                                   │
         ▼                                   ▼
    Outputs:                            Outputs:
    • .safetensors                      • metrics.csv
    • checkpoints                       • report.txt
```

## Execution Workflow Diagram

```text
PHASE 1: LoRA Training
═══════════════════════════════════════════════════════════
┌──────────────────┐     ┌─────────────┐     ┌──────────┐
│ Prepare Data     │────>│ Configure   │────>│ Train    │
│ (ft_prompt_gen)  │     │ Accelerate  │     │ (4x GPU) │
│ • train.jsonl    │     │ • 4 GPUs    │     │ 3 epochs │
│ • validation.json│     │ • bf16      │     │ r=32     │
└──────────────────┘     └─────────────┘     └────┬─────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │ Outputs:         │
                                         │ • .safetensors   │
                                         │ • adapter_config │
                                         │ • checkpoints    │
                                         └───────┬─────────-┘
                                                 │
═════════════════════════════════════════════════╪══════════
PHASE 2: Evaluation                              │
═════════════════════════════════════════════════╪══════════
                                                 ▼
┌──────────────────┐     ┌─────────────┐     ┌──────────┐
│ Load Model       │────>│ Run         │────>│ Calculate│
│ + Adapters       │     │ Inference   │     │ Metrics  │
│ (.safetensors) ◄─┘     │ (1009 test) │     │ F1/FPR   │
└──────────────────┘     └─────────────┘     └────┬─────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │ Outputs:         │
                                         │ • metrics.csv    │
                                         │ • bias.csv       │
                                         │ • report.txt     │
                                         └──────────────────┘
```

## Component Specifications

### 1. LoRA Fine-Tuning Pipeline (`lora/`) - PHASE 1

**Purpose**: Parameter-efficient fine-tuning through Low-Rank Adaptation (LoRA) with 4-bit quantization (QLoRA). This is the **first step** in the workflow, producing trained model adapters.

**Core Components**:

- **Training Script** (`train.py`): Accelerate-based distributed training implementing QLoRA configuration (4-bit NormalFloat4 quantization, bf16 compute), LoRA parameter optimization (rank, alpha, dropout, target modules), and gradient-checkpointing for memory efficiency. Supports config file or command-line parameterization with training/validation data in JSONL format (chat-based messages with system/user/assistant roles).

- **Configuration System** (`configs/*.json`): JSON-based hyperparameter presets including `default.json` (balanced: r=32, α=32, lr=2e-4), `high_capacity.json` (r=64, α=64, lr=1e-4), `memory_efficient.json` (r=16, α=16, lr=3e-4), and `quick_test.json` (r=8, 1 epoch). Supports environment variable substitution and command-line overrides.

- **Merge Utility** (`merge.py`): LoRA adapter fusion with base model weights, converting PEFT checkpoint to standard HuggingFace format for deployment. Preserves model configuration and generates merged model ready for inference.

- **Convergence Checker** (`check_convergence.py`): Training progress analysis tool monitoring loss trends, gradient norms, and early stopping triggers. Visualizes training dynamics through matplotlib plots.

**Training Configuration**:

- **Quantization**: 4-bit NormalFloat4 (nf4) with bf16 compute dtype
- **LoRA Parameters**: Rank 32, alpha 32, dropout 0.05, targets q_proj/v_proj
- **Optimization**: AdamW (lr=2e-4, weight_decay=0.01), cosine learning rate schedule with warmup
- **Training**: 3 epochs, batch size 4 per GPU, gradient accumulation 4 steps (effective batch=64 on 4 GPUs)
- **Validation**: Every 100 steps with optional early stopping (patience=2-3 epochs)
- **Hardware**: Optimized for 4x A100 80GB GPUs (2-3 hours for 10,453 training samples)

**Output Artifacts**:

- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `checkpoint-*/` - Training checkpoints with optimizer states
- `training.log` - Detailed training logs
- TensorBoard events for visualization

### 2. Baseline Validation Pipeline (`baseline/`) - PHASE 2

**Purpose**: Post-training evaluation of fine-tuned models on production test datasets. This is the **second step**, loading trained model adapters (safetensors) for performance assessment.

**Core Components**:

- **Runner** (`runner.py`): CLI orchestrator for inference and evaluation. Loads either pretrained models (for baseline comparison) or fine-tuned models with LoRA adapters (safetensors). Supports single-GPU (default) and multi-GPU (via `--use_accelerate`) execution with concurrent batch processing and comprehensive result persistence. Accepts flexible data sources: unified test dataset (1,009 samples), canned datasets (50/100 samples), or custom JSONL files.

- **Model Loader** (`model_loader/loader.py`): HuggingFace Transformers wrapper loading AutoModelForCausalLM and AutoTokenizer with automatic device mapping. Supports PEFT adapter loading for fine-tuned models, authentication for private models (HF_TOKEN), configurable cache directories, and metadata persistence (model parameters, memory footprint, dtype).

- **Accelerate Connector** (`connector/accelerate_connector.py`): Optional multi-GPU abstraction implementing automatic model distribution across devices with mixed precision (bf16/fp16), gradient accumulation support, and process coordination. Enables distributed inference for faster evaluation on large test sets.

- **Metrics Calculator** (`metrics/calculator.py`): Classification metrics computation reusing prompt_engineering evaluation code. Calculates accuracy, precision, recall, F1-score, and demographic bias metrics (FPR/FNR per target group: LGBTQ+, Middle Eastern, Mexican/Latino).

**Execution Modes**:

- **Connection Test**: `--test_connection` validates model and adapter loading
- **Single-GPU Inference**: Standard execution with automatic device placement
- **Multi-GPU Inference**: `--use_accelerate` with `accelerate launch --num_processes N` for distributed inference
- **Prompt Template Integration**: `--prompt_template` and `--strategy` support evaluation with specific prompting strategies

**Output Artifacts**:

- `validation_results_*.csv` - Per-sample predictions and ground truth
- `performance_metrics_*.csv` - Aggregated classification metrics
- `bias_metrics_*.csv` - Demographic fairness analysis
- `evaluation_report_*.txt` - Human-readable summary
- `validation_log_*.log` - Detailed execution logs

### 3. Shared Infrastructure

**Model Loader** (`baseline/model_loader/loader.py`):
Unified model loading interface supporting both baseline inference and LoRA training. Implements automatic HF_TOKEN authentication, configurable caching (`data/models/`), device map coordination, and metadata persistence. Returns tuple of (model, tokenizer) compatible with both inference and training workflows.

**Accelerate Integration**:
Optional multi-GPU support through HuggingFace Accelerate library. Baseline pipeline enables via `--use_accelerate` flag with `accelerate launch`, while LoRA pipeline requires Accelerate for distributed training. Implements automatic model/optimizer/dataloader distribution with gradient synchronization and mixed precision training.

**Data Management**:
Reuses prompt_engineering data loaders for unified test dataset (1,009 samples), canned datasets (stratified 50/100-sample subsets), and custom JSONL files. Training data generated via `ft_prompt_generator` CLI producing chat-based message format (system/user/assistant roles) with demographic annotations.

**Metrics and Persistence**:
Classification metrics computed via sklearn (accuracy, precision, recall, F1) with demographic bias analysis (per-group FPR/FNR). Baseline outputs: validation_results.csv, test_samples.csv, performance_metrics.csv, evaluation_report.txt. Training outputs: checkpoints, training logs, TensorBoard events, merged models.

## Execution Workflows

### Complete Training-to-Evaluation Workflow

**PHASE 1: LoRA Fine-Tuning** (Must be completed first)

**Step 1 - Environment Setup**:
Configure Accelerate (`accelerate config` - set num_processes=4, mixed_precision=bf16) → Prepare training data via ft_prompt_generator (train.jsonl, validation.jsonl in chat format)

**Step 2 - Model Preparation**:
Load base model with 4-bit quantization → Apply LoRA configuration (rank, alpha, target modules) → Prepare for k-bit training (gradient checkpointing enabled)

**Step 3 - Training Execution**:
Initialize Accelerate Trainer → Load training/validation datasets → Train with gradient accumulation and mixed precision → Validate every N steps → Apply early stopping (optional) → Save checkpoints with safetensors

**Step 4 - Output**:
Generate `adapter_model.safetensors` (trained LoRA adapters) → Save `adapter_config.json` (LoRA hyperparameters) → Create checkpoints with optimizer states → Log training metrics

**Example Commands**:
```bash
# 1. Generate training data (chat-based format)
python -m finetuning.ft_prompt_generator.cli \
    --unified_dir ./data/processed/unified \
    --output_dir ./finetuning/data/ft_prompts \
    --template combined/combined_gptoss_v1.json \
    --strategy combined_optimized

# 2. Train with Accelerate (4 GPUs, 2-3 hours)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

# 3. Monitor training progress
tail -f finetuning/models/lora_checkpoints/training.log
tensorboard --logdir finetuning/models/lora_checkpoints/logs
```

**Output Location**: `finetuning/models/lora_checkpoints/`
- `adapter_model.safetensors` ← **Used in Phase 2**
- `adapter_config.json`
- `checkpoint-*/` directories

---

**PHASE 2: Post-Training Evaluation** (Requires Phase 1 completion)

**Step 1 - Load Trained Model**:
Specify trained model path (`--model_name ./finetuning/models/lora_checkpoints`) → Baseline runner automatically loads safetensors and adapter config → Initialize model with PEFT adapters

**Step 2 - Data Preparation**:
Load production test dataset (unified: 1,009 samples) or canned subsets → Apply sampling limit if needed → Load prompt template (optional)

**Step 3 - Inference Execution**:
Batch samples → Format prompts → Tokenize inputs → Generate predictions with trained model (single-GPU or multi-GPU via Accelerate) → Parse JSON responses → Standardize labels (hate/normal)

**Step 4 - Metrics and Reporting**:
Calculate classification metrics (accuracy, precision, recall, F1) → Compute demographic bias (FPR/FNR per group) → Save validation results CSV → Generate evaluation report → Compare with baseline

**Example Commands**:
```bash
# 1. Evaluate trained model on production data (single GPU)
python -m finetuning.pipeline.baseline.runner \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --max_samples 1009 \
    --output_dir ./finetuning/outputs/post_training

# 2. Multi-GPU evaluation for faster inference (4 GPUs)
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file unified \
    --max_samples 1009

# 3. Quick test with canned dataset (50 samples)
python -m finetuning.pipeline.baseline.runner \
    --model_name ./finetuning/models/lora_checkpoints \
    --data_file canned_50_quick \
    --max_samples 50

# 4. Compare with baseline (pretrained model, no adapters)
python -m finetuning.pipeline.baseline.runner \
    --model_name openai/gpt-oss-20b \
    --data_file unified \
    --max_samples 100 \
    --output_dir ./finetuning/outputs/baseline_comparison
```

**Output Location**: `finetuning/outputs/baseline/run_YYYYMMDD_HHMMSS/`
- `validation_results_*.csv` - Per-sample predictions
- `performance_metrics_*.csv` - F1, accuracy, precision, recall
- `bias_metrics_*.csv` - FPR/FNR by demographic group
- `evaluation_report_*.txt` - Human-readable summary

## Design Principles

**Sequential Training-Evaluation Architecture**: Two-phase workflow enforces clear separation: Phase 1 (LoRA training) produces safetensor adapters, Phase 2 (evaluation) loads adapters for production testing. This prevents training-evaluation leakage and enables independent hyperparameter optimization per phase.

**Configuration-Driven Workflows**: JSON-based hyperparameter presets (LoRA configs) and command-line arguments (baseline runner) eliminate code modifications, accelerating experimental iteration and ensuring reproducibility. All training parameters documented in config files enable peer review validation.

**Distributed Computing via Accelerate**: Native HuggingFace Accelerate integration provides automatic multi-GPU coordination for both training (Phase 1) and inference (Phase 2). Single codebase supports 1-4+ GPUs without modification, abstracting device management complexity.

**Parameter Efficiency Through QLoRA**: 4-bit NormalFloat4 quantization reduces memory footprint by 75% (20B model: ~40GB → ~10GB) while LoRA restricts trainable parameters to <1% of model size (~100M trainable vs 20B total), enabling fine-tuning on consumer-grade GPUs.

**Adapter-Based Deployment**: Safetensor format enables lightweight model distribution (adapters ~200MB vs full model ~40GB). Baseline runner loads base model + adapters at inference time, supporting rapid iteration and A/B testing without full model retraining.

**Reproducibility Through Artifacts**: Phase 1 outputs (safetensors, adapter_config.json, training.log, TensorBoard events) preserve complete training provenance. Phase 2 outputs (validation CSV, metrics, reports) enable independent replication and peer review validation.

**Modular Reusability**: Shared infrastructure (model loader with PEFT support, metrics calculator, data loaders) eliminates duplication. Phase 1 and Phase 2 can evolve independently while maintaining interface compatibility through standardized safetensor/adapter formats.

This framework serves as production infrastructure for supervised fine-tuning experiments, establishing rigorous training-evaluation separation while enabling parameter-efficient adaptation for deployment-ready hate speech detection models. The sequential architecture ensures that evaluation metrics reflect true generalization on held-out production data, not training artifacts.
