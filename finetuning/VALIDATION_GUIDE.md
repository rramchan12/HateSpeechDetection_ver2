# Validation Guide: GPT-OSS-20B Baseline Testing with Multi-GPU Inference

## Overview

This guide provides the **approach and methodology** for validating GPT-OSS-20B through baseline testing using multi-GPU inference with Accelerate on A100 GPUs. This document outlines the validation pipeline setup, execution steps, and connects to detailed validation results.

**Purpose**: Document the validation approach and methodology
- **Audience**: ML Engineers, Researchers
- **Scope**: Baseline validation → Fine-tuning → Post-FT validation
- **Results**: See individual run directories for detailed metrics (performance_metrics_*.csv, bias_metrics_*.csv)

**Current Status**: ✅ Multi-GPU baseline validation pipeline operational
- **Hardware**: 4x NVIDIA A100 80GB GPUs
- **Model**: openai/gpt-oss-20b (~78GB, bf16 precision)
- **Validation Dataset**: unified_val.json (514 samples, stratified)
- **Batch Processing**: 4 samples per GPU batch with intermediate saves
- **Accelerate**: Automatic multi-GPU distribution with file-based result gathering

**Validation Workflow**:
```
Phase 1: Environment Setup (15-20 min)
    ↓
Phase 2: Quick Testing (5-15 min, 30-100 samples)
    ↓
Phase 3: Full Baseline Validation (45-90 min, 514 samples)
    ↓
Phase 4: Results Analysis (metrics already computed)
    ↓
Phase 5: Fine-Tuning with LoRA (2-3 hours)
    ↓
Phase 6: Post-FT Validation & Comparison
```

**Metrics Computed Automatically**:
- ✅ **Performance Metrics**: Accuracy, Precision, Recall, F1-Score (saved to `performance_metrics_*.csv`)
- ✅ **Bias Metrics**: FPR, FNR per target group (saved to `bias_metrics_*.csv`)
- ✅ **Per-Sample Results**: Predictions, rationales (saved to `strategy_unified_results_*.csv`)

**Example Results**:
- Baseline v1 metrics: `prompt_engineering/outputs/baseline_v1/gptoss/run_20251011_085450/`
- Current run metrics: `finetuning/outputs/gptoss/run_20251025_141958/performance_metrics_20251025_141958.csv`

---

## Prerequisites

### 1. Connected to A100 VM

```powershell
# From local machine - VS Code Remote SSH
# Open VS Code → Remote Explorer → Connect to a100vm
```

### 2. Validation Data Available

The pipeline uses the **unified validation dataset** located at:
```
data/processed/unified/unified_val.json
```

**Dataset Details** (as of Oct 25, 2025):
- **Total Samples**: 514
- **Format**: JSON array with `text`, `label_binary`, `target_group_norm`, `source_dataset` fields
- **Distribution**: Stratified across hate/normal and target groups (LGBTQ, Mexican, Middle Eastern)

To verify data:
```bash
# Check file exists and size
ls -lh /home/azureuser/workspace/HateSpeechDetection_ver2/data/processed/unified/unified_val.json

# Count records (should be 514)
python3 -c "import json; print(len(json.load(open('data/processed/unified/unified_val.json'))))"
```

### 3. VS Code Remote SSH Connected

1. Install "Remote - SSH" extension in VS Code (if not already)
2. Press `F1` → "Remote-SSH: Connect to Host" → `a100vm`
3. Open folder: `/home/azureuser/workspace/HateSpeechDetection_ver2`
4. All subsequent steps run from VS Code integrated terminal

### 4. Multi-GPU Setup Configured

The pipeline uses **Accelerate** for automatic multi-GPU distribution:

```bash
# Verify Accelerate configuration
accelerate config --config_file default_config.yaml

# Should show:
# - compute_environment: LOCAL_MACHINE
# - num_machines: 1
# - num_processes: 4 (one per GPU)
# - mixed_precision: bf16
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Environment Setup](#phase-1-environment-setup)
3. [Phase 2: Quick Testing with Canned Data](#phase-2-quick-testing-with-canned-data)
4. [Phase 3: Full Baseline Validation](#phase-3-full-baseline-validation)
5. [Phase 4: Results Analysis](#phase-4-results-analysis)
6. [Phase 5: Fine-Tuning with LoRA](#phase-5-fine-tuning-with-lora)
7. [Phase 6: Post-Fine-Tuning Validation](#phase-6-post-fine-tuning-validation)
8. [Troubleshooting](#troubleshooting)

---

## Phase 1: Environment Setup

**Duration**: 15-20 minutes

### Step 1: Navigate to Project Directory

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2
```

### Step 2: Activate Virtual Environment

The project uses a unified virtual environment:

```bash
# Activate existing .venv
source .venv/bin/activate

# Verify activation
which python
# Expected: /home/azureuser/workspace/HateSpeechDetection_ver2/.venv/bin/python
```

### Step 3: Verify Dependencies

All dependencies are managed through the unified `requirements.txt`:

```bash
# Verify key packages installed
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
```

**Expected Output**:
```
PyTorch: 2.9.0+cu128
CUDA: True
Transformers: 4.48.0.dev0
Accelerate: 1.2.1
```

If packages are missing:
```bash
pip install -r requirements.txt
```

### Step 4: Verify GPUs

```bash
nvidia-smi
```

**Expected**: 4x NVIDIA A100 80GB GPUs with CUDA 12.8

### Step 5: Configure Accelerate (if not already done)

```bash
accelerate config

# Answer prompts:
# - Compute environment: This machine
# - Number of machines: 1
# - Number of processes: 4
# - GPU ids to use: 0,1,2,3
# - Mixed precision: bf16
# - Dynamo backend: no
```

Verify configuration:
```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

---

## Phase 2: Quick Testing with Canned Data

**Duration**: 5-15 minutes  
**Objective**: Verify pipeline works with small stratified samples before running full validation

### Understanding the Baseline Pipeline

The baseline validation pipeline is documented in detail at:
👉 **[finetuning/pipeline/baseline/README.md](pipeline/baseline/README.md)**

Key features:
- **Multi-GPU support** via Accelerate (automatic distribution)
- **Batch processing** (4 samples per GPU batch)
- **Intermediate saves** (JSONL file written after each batch)
- **Sophisticated prompting** using `combined_optimized` strategy
- **Comprehensive metrics** (accuracy, F1, precision, recall, bias metrics)

### Step 1: Quick Test with 30 Samples

Start with a small test to verify the pipeline:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Quick test with 30 samples using Accelerate
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file canned_100_stratified \
    --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --max_samples 30 \
    --output_dir ./finetuning/outputs/gptoss/
```

**What this does**:
- Launches 4 parallel processes (one per GPU)
- Loads model once per GPU (~78GB bf16)
- Processes 30 samples: 30÷4 = 7-8 samples per GPU
- Uses batch_size=4 for efficient processing
- Saves intermediate results after each batch

**Expected Duration**: 5-10 minutes

**Expected Output**:
```
======================================================================
Run ID: run_20251025_HHMMSS
======================================================================

============================================================
Accelerate Connector Initialized
============================================================
Number of GPUs: 4
Mixed Precision: bf16
Batch Size: 1
============================================================

Loading checkpoint shards: 100%|████████| 3/3 [00:10<00:00, 3.4s/it]
[OK] Model loaded on 4 GPU(s)

GPU 0: 100%|████████████████| 2/2 [00:27<00:00, 13.7s/it]

Results saved to: finetuning/outputs/gptoss/run_20251025_HHMMSS

============================================================
EVALUATION METRICS
============================================================
Total Samples: 30
Strategy: combined_optimized

Overall Performance:
  Accuracy:  0.733
  Precision: 0.700
  Recall:    0.750
  F1-Score:  0.724

Bias Metrics by Target Group:
  LGBTQ:
    Sample Count: 13
    FPR: 0.333, FNR: 0.750
  
  MEXICAN:
    Sample Count: 10  
    FPR: 0.143, FNR: 0.500
  
  MIDDLE_EAST:
    Sample Count: 7
    FPR: 0.000, FNR: 0.667
============================================================
```

### Step 2: Examine Output Files

```bash
# Navigate to the run directory
cd finetuning/outputs/gptoss
ls -lt run_*/

# View files created:
# - validation_log_TIMESTAMP.log           # Full request/response logs
# - evaluation_report_TIMESTAMP.txt        # Human-readable summary
# - performance_metrics_TIMESTAMP.csv      # Overall metrics
# - bias_metrics_TIMESTAMP.csv            # Per-group bias metrics
# - strategy_unified_results_TIMESTAMP.csv # Per-sample predictions
# - intermediate_results_TIMESTAMP.jsonl   # Progressive saves (batch by batch)
```

**Key files to check**:

1. **evaluation_report.txt** - Human-readable summary:
```bash
cat run_*/evaluation_report_*.txt
```

2. **bias_metrics.csv** - Per-group metrics:
```bash
cat run_*/bias_metrics_*.csv
```

3. **intermediate_results.jsonl** - Batch-by-batch saves:
```bash
wc -l run_*/intermediate_results_*.jsonl
# Should show 30 lines (one per sample)
```

### Available Canned Datasets

Located in `prompt_engineering/data_samples/`:

| Dataset | Samples | Purpose |
|---------|---------|---------|
| `canned_100_stratified` | 100 | Stratified by target group and label |
| `canned_100_size_varied` | 100 | Varied text lengths |
| `canned_50_quick` | 50 | Quick validation |

All canned datasets are subsets of the unified validation set with proper stratification.

---

## Phase 3: Full Baseline Validation

**Duration**: 45-90 minutes  
**Objective**: Run baseline validation on full 514-sample validation dataset

### Understanding Output Metrics

All validation runs automatically generate three metrics files in the run directory (`finetuning/outputs/gptoss/run_TIMESTAMP/`):

1. **performance_metrics_TIMESTAMP.csv** - Overall performance
   - Columns: `strategy`, `accuracy`, `precision`, `recall`, `f1_score`, `true_positive`, `true_negative`, `false_positive`, `false_negative`
   - Example: `accuracy=0.652, f1_score=0.621`

2. **bias_metrics_TIMESTAMP.csv** - Per-target-group bias analysis
   - Columns: `strategy`, `persona_tag`, `sample_count`, `false_positive_rate`, `false_negative_rate`, `true_positive`, `true_negative`, `false_positive`, `false_negative`
   - Shows FPR/FNR breakdown for LGBTQ, Mexican, Middle Eastern groups
   - Example: `LGBTQ: FPR=0.150, FNR=0.328`

3. **strategy_unified_results_TIMESTAMP.csv** - Individual sample predictions
   - Columns: `strategy`, `sample_id`, `input_text`, `true_label`, `predicted_label`, `persona_tag`, `rationale`
   - Complete record of every prediction with reasoning

### Step 1: Run Full Validation

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Full validation with all 514 samples
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file unified \
    --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --max_samples all \
    --output_dir ./finetuning/outputs/gptoss/baseline
```

**What this does**:
- Loads **unified_val.json** (514 validation samples)
- Distributes across 4 GPUs: ~129 samples per GPU
- Each GPU processes in batches of 4: ~33 batches per GPU
- Writes intermediate saves after each batch (crash recovery)
- Uses sophisticated `combined_optimized` prompting strategy

**Expected Duration**: 45-90 minutes

**Progress Monitoring**:
```bash
# Monitor in real-time (separate terminal)
watch -n 5 nvidia-smi

# Check intermediate file being written
ls -lh finetuning/outputs/gptoss/baseline/run_*/intermediate_results_*.jsonl

# Count completed samples
wc -l finetuning/outputs/gptoss/baseline/run_*/intermediate_results_*.jsonl
```

### Step 2: Review Generated Metrics Files

After completion, examine the auto-generated metrics:

```bash
cd finetuning/outputs/gptoss/baseline

# Find most recent run
LATEST_RUN=$(ls -td run_* | head -1)
cd $LATEST_RUN

# View overall performance metrics
cat performance_metrics_*.csv

# View per-group bias metrics
cat bias_metrics_*.csv

# Sample individual predictions
head -20 strategy_unified_results_*.csv
```

**Expected Performance Metrics** (example from recent run):
```csv
strategy,accuracy,precision,recall,f1_score,true_positive,true_negative,false_positive,false_negative
combined_optimized,0.652,0.615,0.628,0.621,127,208,41,94
```

**Expected Bias Metrics** (example from recent run):
```csv
strategy,persona_tag,sample_count,false_positive_rate,false_negative_rate,true_positive,true_negative,false_positive,false_negative
combined_optimized,lgbtq,167,0.150,0.328,45,85,15,22
combined_optimized,mexican,167,0.118,0.354,42,90,12,23
combined_optimized,middle_east,166,0.137,0.375,40,88,14,24
```

**Interpreting Bias Metrics**:
- **FPR (False Positive Rate)**: Over-prediction of hate → High FPR means model flags too much as hate for this group
- **FNR (False Negative Rate)**: Under-detection of hate → High FNR means model misses actual hate for this group
- **Balanced Fairness**: Similar FPR/FNR across all groups indicates fair treatment

### Step 3: Document Baseline Metrics

Create a baseline reference file for comparison with future fine-tuned models:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss/baseline

# Get most recent run
LATEST_RUN=$(ls -td run_* | head -1)

# Extract F1 from performance metrics
F1=$(awk -F',' 'NR==2 {print $5}' ${LATEST_RUN}/performance_metrics_*.csv)

# Create baseline reference file
cat > BASELINE_METRICS.txt << EOF
=============================================================
PRE-FINE-TUNING BASELINE METRICS
=============================================================
Date: $(date)
Model: openai/gpt-oss-20b
Dataset: unified_val.json (514 samples)
Prompt Strategy: combined_optimized (sophisticated prompting)
Hardware: 4x NVIDIA A100 80GB GPUs
Batch Size: 4 samples per GPU batch

PERFORMANCE METRICS:
  F1-Score: $F1
  Full metrics: ${LATEST_RUN}/performance_metrics_*.csv

BIAS METRICS:
  Per-group FPR/FNR: ${LATEST_RUN}/bias_metrics_*.csv

INDIVIDUAL PREDICTIONS:
  Sample-level results: ${LATEST_RUN}/strategy_unified_results_*.csv

=============================================================
FINE-TUNING TARGET:
- Goal: Post-FT F1 (with simple prompts) ≥ $F1 (with complex prompts)
- This proves fine-tuning internalized the task without needing
  sophisticated prompt engineering
=============================================================
EOF

cat BASELINE_METRICS.txt
```

---

## Phase 4: Results Analysis

**Duration**: 15-30 minutes  
**Objective**: Analyze baseline performance metrics and identify improvement areas

### Step 1: Examine Generated Metrics Files

All metrics are automatically computed and saved during validation. Navigate to your run directory:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss/baseline

# Find most recent run
LATEST_RUN=$(ls -td run_* | head -1)
cd $LATEST_RUN

# List all metrics files
ls -lh *.csv *.txt
```

**Expected files**:
- `performance_metrics_TIMESTAMP.csv` - Overall accuracy, F1, precision, recall
- `bias_metrics_TIMESTAMP.csv` - Per-group FPR, FNR breakdown
- `strategy_unified_results_TIMESTAMP.csv` - Individual sample predictions with rationales
- `evaluation_report_TIMESTAMP.txt` - Human-readable summary report

### Step 2: Validate Bias Metrics (Optional)

The bias metrics are automatically calculated and saved to `bias_metrics_*.csv`. To manually verify the calculations:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

python3 << 'EOF'
import pandas as pd
import glob

# Find most recent results
result_files = glob.glob('finetuning/outputs/gptoss/baseline/run_*/strategy_unified_results_*.csv')
latest_file = sorted(result_files)[-1]

# Load results
df = pd.read_csv(latest_file)

print(f"Loaded {len(df)} samples from: {latest_file}\n")
print("="*60)
print("BIAS METRICS VALIDATION")
print("="*60)

for persona in sorted(df['persona_tag'].unique()):
    persona_df = df[df['persona_tag'] == persona]
    valid_df = persona_df[persona_df['predicted_label'].isin(['hate', 'normal'])]
    
    tp = len(valid_df[(valid_df['predicted_label']=='hate') & (valid_df['true_label']=='hate')])
    tn = len(valid_df[(valid_df['predicted_label']=='normal') & (valid_df['true_label']=='normal')])
    fp = len(valid_df[(valid_df['predicted_label']=='hate') & (valid_df['true_label']=='normal')])
    fn = len(valid_df[(valid_df['predicted_label']=='normal') & (valid_df['true_label']=='hate')])
    
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    fnr = fn/(fn+tp) if (fn+tp)>0 else 0.0
    
    print(f"\n{persona} (n={len(valid_df)}):")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  FPR={fpr:.4f}, FNR={fnr:.4f}")

print("\n" + "="*60)
print("Compare these values with bias_metrics_*.csv to verify accuracy")
EOF
```

### Step 3: Analyze Error Patterns

Identify common misclassifications using the individual predictions file:

```bash
python3 << 'EOF'
import pandas as pd
import glob

# Load most recent results
result_files = glob.glob('finetuning/outputs/gptoss/baseline/run_*/strategy_unified_results_*.csv')
df = pd.read_csv(sorted(result_files)[-1])

# Find false positives (predicted hate, actual normal)
false_positives = df[(df['predicted_label']=='hate') & (df['true_label']=='normal')]

print("="*60)
print(f"FALSE POSITIVES: {len(false_positives)} samples")
print("="*60)
for idx, row in false_positives.head(5).iterrows():
    print(f"\nText: {row['input_text'][:80]}...")
    print(f"Target Group: {row['persona_tag']}")
    print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}")

# Find false negatives (predicted normal, actual hate)
false_negatives = df[(df['predicted_label']=='normal') & (df['true_label']=='hate')]

print("\n" + "="*60)
print(f"FALSE NEGATIVES: {len(false_negatives)} samples")
print("="*60)
for idx, row in false_negatives.head(5).iterrows():
    print(f"\nText: {row['input_text'][:80]}...")
    print(f"Target Group: {row['persona_tag']}")
    print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}")
EOF
```

### Step 4: Compare with Previous Runs (if applicable)

If you have multiple validation runs:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss/baseline

# Compare F1 scores across runs
echo "PERFORMANCE COMPARISON ACROSS RUNS"
echo "="*60
for run in run_*/; do
    echo "=== $run ==="
    awk -F',' 'NR==2 {printf "F1: %.3f, Acc: %.3f, Prec: %.3f, Rec: %.3f\n", $5, $2, $3, $4}' \
        ${run}performance_metrics_*.csv
done
```

**Key Insights to Document**:
1. **Overall Performance**: F1 score to beat with fine-tuning
2. **Bias Patterns**: Which groups have higher FPR/FNR?
3. **Error Patterns**: Common characteristics of false positives/negatives
4. **Improvement Areas**: Target these patterns in fine-tuning data generation

---

## Phase 5: Fine-Tuning with LoRA

**Duration**: 2-3 hours  
**Objective**: Fine-tune GPT-OSS-20B with LoRA adapters to internalize hate speech detection task

### Overview

After establishing baseline metrics with sophisticated prompts, the next step is fine-tuning the model to internalize the task. The goal is for the fine-tuned model to achieve similar or better F1 scores using **simple prompts** compared to the baseline's **sophisticated prompts**.

**Success Criteria**: Post-FT F1 (simple prompts) ≥ Pre-FT F1 (sophisticated prompts)

### Step 1: Generate Fine-Tuning Data

Use the FT Prompt Generator to create training data in multiple formats:

👉 **Detailed documentation**: [finetuning/ft_prompt_generator/README.md](ft_prompt_generator/README.md)

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Generate training data in simple and optimized formats
python -m finetuning.ft_prompt_generator.cli \
    --unified_dir ./data/processed/unified \
    --output_dir ./finetuning/data/ft_prompts \
    --template combined/combined_gptoss_v1.json \
    --strategy combined_optimized
```

**Generated Files** (in `finetuning/data/ft_prompts/`):
- `train.jsonl` - Simple format training data (~80% of unified dataset)
- `validation.jsonl` - Simple format validation data (~20% of unified dataset)
- `train_optimized.jsonl` - Optimized format with sophisticated prompts
- `validation_optimized.jsonl` - Optimized format validation data

**Format Explanation**:
- **Simple Format**: Basic prompt → completion pairs
  ```json
  {
    "messages": [
      {"role": "system", "content": "You are a hate speech classifier."},
      {"role": "user", "content": "Is this hate speech: '...'"},
      {"role": "assistant", "content": "hate"}
    ]
  }
  ```
- **Optimized Format**: Includes detailed reasoning and context from sophisticated prompts
  ```json
  {
    "messages": [
      {"role": "system", "content": "Detailed system prompt with guidelines..."},
      {"role": "user", "content": "Analyze: '...' Target group: lgbtq"},
      {"role": "assistant", "content": "LABEL: hate\nRATIONALE: ..."}
    ]
  }
  ```

### Step 2: Configure LoRA Training

LoRA (Low-Rank Adaptation) allows efficient fine-tuning by training small adapter layers instead of the full model.

**Recommended LoRA Configuration**:
- `lora_rank`: 32-64 (higher = more capacity, but slower)
- `lora_alpha`: 32-64 (scaling factor)
- `lora_dropout`: 0.05-0.1
- `target_modules`: `["q_proj", "v_proj"]` (attention layers)

**Training Hyperparameters**:
- `learning_rate`: 2e-4 to 5e-4
- `epochs`: 2-3
- `batch_size`: 4 per GPU
- `gradient_accumulation_steps`: 2-4
- `warmup_ratio`: 0.1

### Step 3: Run LoRA Fine-Tuning

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Fine-tune with LoRA using optimized format
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.lora.train \
    --model_name openai/gpt-oss-20b \
    --train_file ./finetuning/data/ft_prompts/train_optimized.jsonl \
    --val_file ./finetuning/data/ft_prompts/validation_optimized.jsonl \
    --output_dir ./finetuning/models/lora_checkpoints \
    --lora_rank 32 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1
```

**Training Monitoring**:
```bash
# Monitor GPU usage
watch -n 5 nvidia-smi

# Monitor training logs
tail -f finetuning/models/lora_checkpoints/training.log

# Check validation loss trends
grep "eval_loss" finetuning/models/lora_checkpoints/training.log
```

**Expected Duration**: 2-3 hours for 3 epochs

**Output**: LoRA adapter weights saved to `finetuning/models/lora_checkpoints/`

### Step 4: Merge LoRA Adapters (Optional)

For deployment, you can merge LoRA adapters into the base model:

```bash
python -m finetuning.pipeline.lora.merge \
    --base_model openai/gpt-oss-20b \
    --lora_weights ./finetuning/models/lora_checkpoints \
    --output_dir ./finetuning/models/merged_model
```

**Note**: Merged model will be ~78GB (full model size). Keep LoRA adapters separate for flexibility.

---

## Phase 6: Post-Fine-Tuning Validation

**Duration**: 45-90 minutes  
**Objective**: Validate fine-tuned model with simple prompts and compare to baseline

### Step 1: Run Validation with Fine-Tuned Model

Use **simple prompts** to test if the model internalized the task:

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Validate with simple prompts (no sophisticated prompt engineering)
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

**Key Differences from Baseline Run**:
- `--model_name`: Points to fine-tuned LoRA model instead of base model
- `--prompt_template`: Uses simple template (no sophisticated prompt engineering)
- `--strategy simple`: Basic classification without detailed reasoning
- `--output_dir`: Separate directory for post-FT results

### Step 2: Examine Post-FT Metrics

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss/post_finetune

# Find most recent run
LATEST_RUN=$(ls -td run_* | head -1)
cd $LATEST_RUN

# View performance metrics
cat performance_metrics_*.csv

# View bias metrics
cat bias_metrics_*.csv
```

**Metrics files generated** (same format as baseline):
- `performance_metrics_*.csv` - Accuracy, F1, Precision, Recall
- `bias_metrics_*.csv` - FPR, FNR per target group
- `strategy_unified_results_*.csv` - Individual predictions

### Step 3: Compare Baseline vs Fine-Tuned Performance

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss

# Extract baseline F1 (sophisticated prompts)
BASELINE_F1=$(awk -F',' 'NR==2 {print $5}' baseline/run_*/performance_metrics_*.csv | head -1)

# Extract post-FT F1 (simple prompts)
POSTFT_F1=$(awk -F',' 'NR==2 {print $5}' post_finetune/run_*/performance_metrics_*.csv | head -1)

echo "="*60
echo "PERFORMANCE COMPARISON"
echo "="*60
echo "BASELINE (sophisticated prompts):  F1 = $BASELINE_F1"
echo "FINE-TUNED (simple prompts):       F1 = $POSTFT_F1"
echo ""

# Calculate improvement
python3 << EOF
baseline = float("$BASELINE_F1")
postft = float("$POSTFT_F1")
improvement = ((postft - baseline) / baseline) * 100
if postft >= baseline:
    print(f"✅ SUCCESS: Fine-tuning improved F1 by {improvement:.1f}%")
    print(f"   Model internalized task - no longer needs sophisticated prompts")
else:
    print(f"⚠️  NEEDS WORK: F1 decreased by {abs(improvement):.1f}%")
    print(f"   Consider: more training epochs, different LoRA config, or more training data")
EOF
```

### Step 4: Compare Bias Metrics

```bash
# Compare bias metrics across runs
echo "="*60
echo "BIAS COMPARISON: BASELINE vs FINE-TUNED"
echo "="*60

echo "BASELINE (sophisticated prompts):"
cat baseline/run_*/bias_metrics_*.csv | head -1 | tail -1
cat baseline/run_*/bias_metrics_*.csv | tail -n +2

echo ""
echo "FINE-TUNED (simple prompts):"
cat post_finetune/run_*/bias_metrics_*.csv | head -1 | tail -1
cat post_finetune/run_*/bias_metrics_*.csv | tail -n +2
```

**Key Questions to Analyze**:
1. Did FPR/FNR improve for any target groups?
2. Did fairness balance improve (more similar FPR/FNR across groups)?
3. Are there specific groups that need targeted data augmentation?

### Step 5: Document Fine-Tuning Results

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2/finetuning/outputs/gptoss/post_finetune

# Get most recent run
LATEST_RUN=$(ls -td run_* | head -1)

# Create results summary
cat > FINETUNING_RESULTS.txt << EOF
=============================================================
POST-FINE-TUNING VALIDATION RESULTS
=============================================================
Date: $(date)
Model: openai/gpt-oss-20b + LoRA adapters
Dataset: unified_val.json (514 samples)
Prompt Strategy: simple (no sophisticated prompt engineering)
Hardware: 4x NVIDIA A100 80GB GPUs

BASELINE PERFORMANCE (sophisticated prompts):
  F1-Score: $(awk -F',' 'NR==2 {print $5}' ../../baseline/run_*/performance_metrics_*.csv | head -1)
  Metrics: ../../baseline/run_*/performance_metrics_*.csv

POST-FT PERFORMANCE (simple prompts):
  F1-Score: $(awk -F',' 'NR==2 {print $5}' ${LATEST_RUN}/performance_metrics_*.csv)
  Metrics: ${LATEST_RUN}/performance_metrics_*.csv

BIAS METRICS:
  Baseline: ../../baseline/run_*/bias_metrics_*.csv
  Post-FT:  ${LATEST_RUN}/bias_metrics_*.csv

=============================================================
CONCLUSION:
- Compare F1 scores to determine if fine-tuning was successful
- Analyze bias metrics for fairness improvements
- If successful: deploy fine-tuned model
- If not successful: iterate on training data, hyperparameters, or epochs
=============================================================
EOF

cat FINETUNING_RESULTS.txt
```

---

## Troubleshooting

### Issue: Accelerate Command Not Found

**Error**: `accelerate: command not found`

**Solution**:
```bash
# Activate virtual environment
cd /home/azureuser/workspace/HateSpeechDetection_ver2
source .venv/bin/activate

# Verify accelerate installed
which accelerate
pip show accelerate

# If not installed:
pip install accelerate
```

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Root Cause**: Model is ~78GB bf16. On A100 80GB GPUs, this should fit with headroom, but OOM can occur with large batch sizes or gradient accumulation.

**Solutions**:

1. **Reduce Batch Size**:
```python
# In finetuning/pipeline/baseline/accelerate_connector.py
# Change batch_size from 4 to 2 or 1
self.batch_size = 2
```

2. **Check GPU Memory Before Running**:
```bash
nvidia-smi
# Ensure all 4 GPUs have ~75GB free before starting
```

3. **Clear CUDA Cache**:
```python
# Add to runner.py before model loading
import torch
torch.cuda.empty_cache()
```

### Issue: Model Loading Crashes

**Error**: Model fails to load or crashes during checkpoint loading

**Solutions**:

1. **Clear Transformers Cache**:
```bash
rm -rf ~/.cache/huggingface/transformers/*
rm -rf ~/.cache/huggingface/hub/*
```

2. **Re-download Model**:
```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('openai/gpt-oss-20b', cache_dir='~/.cache/huggingface')"
```

3. **Check Disk Space**:
```bash
df -h ~
# Model requires ~150GB for cached downloads + loaded model
```

### Issue: gather_results() Fails

**Error**: Results not being gathered from all GPUs, or pickle errors during gathering

**Root Cause**: The pipeline uses file-based gathering with pickle + temp files.

**Solutions**:

1. **Check Temp Files**:
```bash
ls -lh /tmp/accelerate_results_*
# Should see files from all 4 GPUs
```

2. **Check Permissions**:
```bash
ls -ld /tmp
# Ensure write permissions
```

3. **Review Logs**:
```bash
grep "gather_results" finetuning/outputs/gptoss/baseline/run_*/validation_log_*.log
grep -i "error" finetuning/outputs/gptoss/baseline/run_*/validation_log_*.log
```

4. **Verify All Processes Completed**:
```bash
# Check intermediate file has results from all GPUs
wc -l finetuning/outputs/gptoss/baseline/run_*/intermediate_results_*.jsonl
# Should match total samples processed
```

### Issue: Intermediate Saves Not Written

**Error**: `intermediate_results.jsonl` file empty or missing

**Solutions**:

1. **Check Write Permissions**:
```bash
ls -ld finetuning/outputs/gptoss/baseline/run_*
# Ensure directory is writable
```

2. **Verify Batch Processing Code**:
```bash
grep -A 10 "with open(intermediate_save_file" finetuning/pipeline/baseline/runner.py
# Ensure file is being written after each batch
```

3. **Check Disk Space**:
```bash
df -h .
# Ensure sufficient space for JSONL output (~100MB for 514 samples)
```

### Issue: Wrong Dataset Loaded

**Error**: Loading `unified_test.json` (13,118 samples) instead of `unified_val.json` (514 samples)

**Symptom**: Terminal shows "GPU 0: 23/64" or other large batch numbers

**Solution**:
```bash
# Verify dataset loader is using validation set
grep "unified_path" prompt_engineering/loaders/unified_dataset_loader.py
# Should show: self.unified_path = ... / "unified_val.json"

# If showing unified_test.json, edit the file:
# Change line 47 to: self.unified_path = self.base_path / "unified_val.json"
```

### Issue: Metrics Mismatch

**Error**: Computed metrics don't match expectations or previous runs

**Solution**:
```bash
# Manually validate bias metrics
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('finetuning/outputs/gptoss/baseline/run_*/strategy_unified_results_*.csv')

# Calculate metrics for one group
persona_df = df[df['persona_tag'] == 'lgbtq']
tp = len(persona_df[(persona_df['predicted_label']=='hate') & (persona_df['true_label']=='hate')])
tn = len(persona_df[(persona_df['predicted_label']=='normal') & (persona_df['true_label']=='normal')])
fp = len(persona_df[(persona_df['predicted_label']=='hate') & (persona_df['true_label']=='normal')])
fn = len(persona_df[(persona_df['predicted_label']=='normal') & (persona_df['true_label']=='hate')])

fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
fnr = fn/(fn+tp) if (fn+tp)>0 else 0.0

print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"FPR={fpr:.4f}, FNR={fnr:.4f}")
EOF

# Compare with bias_metrics_*.csv
cat finetuning/outputs/gptoss/baseline/run_*/bias_metrics_*.csv
```

### Issue: LoRA Fine-Tuning OOM

**Error**: Out of memory during fine-tuning even with LoRA

**Solutions**:

1. **Reduce LoRA Rank**:
```bash
# Use lower rank (less capacity, but less memory)
--lora_rank 16  # instead of 32
```

2. **Reduce Batch Size**:
```bash
--per_device_train_batch_size 2  # instead of 4
--gradient_accumulation_steps 8   # compensate with more accumulation
```

3. **Enable Gradient Checkpointing**:
```bash
--gradient_checkpointing true
```

### Issue: Poor Post-FT Performance

**Error**: Fine-tuned model F1 < baseline F1

**Root Causes & Solutions**:

1. **Insufficient Training**:
   - Try more epochs: `--epochs 5` instead of `--epochs 3`
   - Monitor validation loss to detect early stopping point

2. **Learning Rate Issues**:
   - Too high: model doesn't converge → try `--learning_rate 1e-4`
   - Too low: model doesn't learn → try `--learning_rate 5e-4`

3. **Data Quality**:
   - Use `train_optimized.jsonl` instead of `train.jsonl`
   - Ensure training data is balanced across target groups
   - Check for label noise or mislabeled samples

4. **LoRA Configuration**:
   - Increase rank for more capacity: `--lora_rank 64`
   - Adjust alpha: `--lora_alpha 64`
   - Target more modules: `--target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]'`

---

## Key Takeaways

✅ **Multi-GPU Pipeline Operational**: 4x A100 GPUs with Accelerate  
✅ **Batch Processing**: 4 samples per GPU batch for efficiency  
✅ **Intermediate Saves**: JSONL written after each batch (crash recovery)  
✅ **Comprehensive Metrics**: Auto-generated performance_metrics_*.csv, bias_metrics_*.csv, strategy_unified_results_*.csv  
✅ **Validation Dataset**: unified_val.json (514 samples) loaded by default  
✅ **Canned Datasets**: Quick testing with 30-100 stratified samples  
✅ **Sophisticated Prompting**: combined_optimized strategy for baseline  
✅ **Fine-Tuning Pipeline**: LoRA adapters for efficient fine-tuning  
✅ **Simple Prompts Post-FT**: Test if model internalized task without complex prompting

**Complete Workflow**:
1. ✅ Environment Setup (15-20 min)
2. ✅ Quick Testing with Canned Data (5-15 min)
3. ✅ Full Baseline Validation (45-90 min, 514 samples)
4. ✅ Results Analysis (examine auto-generated metrics files)
5. ✅ Fine-Tuning with LoRA (2-3 hours)
6. ✅ Post-FT Validation with Simple Prompts (45-90 min)
7. ✅ Performance Comparison (baseline vs fine-tuned)

**Metrics References**:
- **Performance**: `finetuning/outputs/gptoss/*/run_*/performance_metrics_*.csv`
- **Bias**: `finetuning/outputs/gptoss/*/run_*/bias_metrics_*.csv`
- **Individual Predictions**: `finetuning/outputs/gptoss/*/run_*/strategy_unified_results_*.csv`

**Next Steps After Validation**:
- Deploy fine-tuned model if F1 improvement achieved
- Iterate on training data/hyperparameters if improvement needed
- Scale to full test set (unified_test.json, 13,118 samples) for final evaluation

---

**Last Updated**: October 25, 2025  
**Pipeline Version**: v2.0 (Multi-GPU with Accelerate + LoRA Fine-Tuning)  
**Model**: openai/gpt-oss-20b (~78GB bf16)
