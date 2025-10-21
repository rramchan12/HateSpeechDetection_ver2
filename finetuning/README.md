# Fine-Tuning for Hate Speech Detection

## Overview

This directory contains guides and data for fine-tuning large language models on A100 GPUs via **remote SSH training**. All training code will be developed directly on the remote A100 VM using VS Code Remote SSH.

**Training Approach**: Remote development and execution on A100 VMs
- Local machine: Documentation and training data
- Remote A100 VM: All code, dependencies, and model training

---

## Directory Structure

```
finetuning/
├── README.md                                  # This file - overview
├── FINE_TUNING_MODEL_SELECTION_README.md     # Model selection guide (GPT-5, GPT-OSS-120B, GPT-OSS-20B)
├── A100_SSH_TRAINING_GUIDE.md                # Complete SSH training guide for A100 VMs
├── VALIDATION_GUIDE.md                        # Training and inference validation for GPT-OSS-120B
└── data/
    └── prepared/
        ├── train.jsonl                        # Training data (3,083 samples)
        └── validation.jsonl                   # Validation data (545 samples)
```

---

## Quick Start

### 1. Read the Model Selection Guide

**File**: `FINE_TUNING_MODEL_SELECTION_README.md`

Understand which model to fine-tune based on:
- Dataset size constraints (3,628 samples)
- Parameter-to-sample ratios
- Cost-benefit analysis
- Expected performance

**Recommendation**: GPT-OSS-20B (optimal sample efficiency, cost-effective)

---

### 2. Connect to Your A100 VM

**Prerequisite**: SSH config already set up (`ssh a100vm` works)

```powershell
# Connect via SSH
ssh a100vm
```

See `A100_SSH_TRAINING_GUIDE.md` for detailed connection instructions.

---

### 3. Transfer Training Data

```powershell
# From your local machine (in finetuning directory)
scp data/prepared/train.jsonl a100vm:~/finetuning/data/
scp data/prepared/validation.jsonl a100vm:~/finetuning/data/
```

---

### 4. Set Up VS Code Remote SSH

**Install Extension**: Remote - SSH (ms-vscode-remote.remote-ssh)

1. Open VS Code
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Select `a100vm`
4. Open folder: `/home/azureuser/finetuning`

Now you can develop directly on the A100 VM!

---

### 5. Follow Validation Guide

**File**: `VALIDATION_GUIDE.md`

Step-by-step guide for:
- Environment setup on remote VM
- Creating training scripts
- Running baseline validation
- Fine-tuning GPT-OSS-120B
- Testing inference
- Evaluating performance

---

## Training Data

### Dataset Summary

**Source**: Unified HateXplain + ToxiGen dataset  
**Total Samples**: 3,628 (prepared from 47,166 original)

**Splits**:
- Training: 3,083 samples (85%) → `data/prepared/train.jsonl`
- Validation: 545 samples (15%) → `data/prepared/validation.jsonl`

**Protected Groups**:
- LGBTQ+: ~49% (~1,779 samples)
- Mexican/Latino: ~21% (~762 samples)
- Middle Eastern: ~30% (~1,087 samples)

**Format**: JSONL with instruction-formatted prompts
```json
{
  "text": "### Instruction:\nClassify the following post...\n### Classification:",
  "label": "hate"
}
```

---

## Baseline Performance (Before Fine-Tuning)

### GPT-OSS-120B Baseline
- **F1-Score**: 0.615 (best overall)
- **Accuracy**: 65.0%
- **Precision**: 61.0%
- **Recall**: 62.0%

**Bias Metrics**:
| Group | FPR | FNR |
|-------|-----|-----|
| LGBTQ+ | 43.0% ⚠️ | 39.4% |
| Mexican | 8.1% ✅ | 39.8% |
| Middle East | 23.6% | 35.2% |

**Target**: Fine-tuning should achieve F1 ≥ 0.620 and reduce LGBTQ+ FPR to <35%

---

## Recommended Workflow

### Phase 1: Environment Setup (30 minutes)
1. Connect to A100 VM via SSH: `ssh a100vm`
2. Create virtual environment and install dependencies
3. Transfer training data from local machine
4. Verify GPU availability (`nvidia-smi`)

### Phase 2: Baseline Validation (1-2 hours)
1. Load GPT-OSS-120B base model
2. Run inference on validation set (545 samples)
3. Calculate baseline F1, precision, recall
4. Measure FPR/FNR by protected group
5. Document baseline performance

### Phase 3: LoRA Fine-Tuning (2-3 hours)
1. Configure LoRA (rank=32, alpha=64, dropout=0.1 for 20B)
2. Train for 3 epochs with early stopping
3. Monitor training loss and validation metrics
4. Save best checkpoint

### Phase 4: Fine-Tuned Validation (1 hour)
1. Load fine-tuned model + LoRA adapter
2. Run inference on validation set
3. Calculate post-fine-tuning metrics
4. Compare against baseline
5. Analyze bias improvements

### Phase 5: Production Deployment (optional)
1. Merge LoRA adapter with base model
2. Export for deployment
3. Set up monitoring pipeline

**Total Time**: 5-7 hours end-to-end

---

## Key Documents

### 1. Model Selection Guide
**File**: `ft_model_selection_README.md` (30+ pages)

**Contents**:
- Comparison matrix: GPT-5, GPT-OSS-120B, GPT-OSS-20B
- Baseline performance analysis from IFT experiments
- Dataset constraints and sample efficiency calculations
- LoRA configuration recommendations
- Cost-benefit analysis (training + inference + TCO)
- Implementation roadmap (5-week plan)

**Key Recommendation**: GPT-OSS-20B
- 6× better sample efficiency than 120B
- $3-5 per training run (vs $12-18 for 120B)
- Expected F1: 0.610-0.630 after fine-tuning

---

### 2. SSH Training Guide
**File**: `A100_SSH_TRAINING_GUIDE.md` (50+ pages)

**Contents**:
- SSH connection with private key
- SSH config setup for easy access
- Data transfer (SCP/SFTP)
- Environment configuration on VM
- Training script templates
- tmux for long-running jobs
- GPU monitoring
- Model retrieval
- Troubleshooting

**Quick Commands**:
```bash
# Connect
ssh a100vm

# Transfer data
scp file.txt a100vm:~/

# Run training in tmux
tmux new -s train
python train.py
# Ctrl+B, D to detach
```

---

### 3. Validation Guide
**File**: `VALIDATION_GUIDE.md`

**Contents**:
- Step-by-step baseline validation
- Training script creation
- LoRA configuration setup
- Fine-tuning execution
- Inference testing
- Performance evaluation
- Comparison framework

**Success Criteria**:
- ✅ Fine-tuned F1 ≥ 0.620 (improve +0.005 from baseline)
- ✅ LGBTQ+ FPR < 35% (reduce from 43%)
- ✅ Mexican FPR ≤ 10% (maintain low rate)

---

## Remote Development with VS Code

### Benefits
- Edit code directly on A100 VM
- No file transfer needed
- Use local VS Code UI
- Access remote terminal
- Real-time code execution

### Setup
1. Install "Remote - SSH" extension
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Select `a100vm`
4. Open `/home/azureuser/finetuning`
5. Install Python extension on remote
6. Select Python interpreter from venv

### Workflow
1. Write code in VS Code (running on VM)
2. Open integrated terminal (connected to VM)
3. Run scripts directly: `python train.py`
4. Monitor with `nvidia-smi`
5. View logs in VS Code

**No SCP needed!** All files are already on the VM.

---

## Support and References

### Baseline Performance Analysis
- **Overall Summary**: `../prompt_engineering/prompt_templates/overall_summary_ift_README.md`
- **GPT-OSS Results**: `../prompt_engineering/prompt_templates/gptoss_ift_summary_README.md`
- **GPT-5 Results**: `../prompt_engineering/prompt_templates/gpt5_ift_summary_README.md`

### Training Data Preparation
- **Unification**: `../data_preparation/data_unification.py`
- **HateXplain**: `../data_preparation/data_preparation_hatexplain.py`
- **ToxiGen**: `../data_preparation/data_preparation_toxigen.py`

### Model Information
- **GPT-OSS-120B**: Phi-3.5-MoE-instruct (120B parameters)
- **GPT-OSS-20B**: Phi-3.5-MoE-instruct (20B parameters)
- **Hugging Face**: Check Azure deployment for exact model IDs

---

## Troubleshooting

### SSH Connection Issues
See `A100_SSH_TRAINING_GUIDE.md` Section "Troubleshooting SSH Connection"

### CUDA Out of Memory
- Reduce batch size: `per_device_train_batch_size=2`
- Increase gradient accumulation: `gradient_accumulation_steps=4`
- Reduce sequence length: `max_seq_length=384`

### Slow Training
- Check GPU utilization: `nvidia-smi` (should be 95-100%)
- Increase data workers: `dataloader_num_workers=8`
- Use gradient checkpointing (enabled by default)

### Model Loading Errors
- Verify model ID: Check Azure deployment or Hugging Face
- Use explicit model path: `--model_id microsoft/Phi-3.5-MoE-instruct`

---

## Next Steps

1. **Read Model Selection Guide** → Understand GPT-OSS-20B recommendation
2. **Connect to A100 VM** → `ssh a100vm`
3. **Set Up VS Code Remote** → Edit code on VM directly
4. **Follow Validation Guide** → Step-by-step training
5. **Document Results** → Compare against baseline (F1=0.615)

---

**Last Updated**: October 21, 2025  
**Training Approach**: Remote SSH on A100 VMs  
**Recommended Model**: GPT-OSS-20B (sample efficiency + cost-effective)  
**Expected Training Time**: 2-3 hours  
**Target Performance**: F1 ≥ 0.620, LGBTQ+ FPR < 35%
