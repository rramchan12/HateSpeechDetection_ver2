# Direct A100 VM Training Guide for GPT-OSS-20B

## Overview

This guide provides a streamlined approach to fine-tuning **GPT-OSS-20B** using **direct SSH access** to your Azure ML A100 VM. This method is more efficient than Azure ML job submission since you have direct control over the training environment.

**Advantages of Direct SSH Training**:
- ✅ Real-time monitoring and debugging
- ✅ Interactive experimentation with immediate feedback
- ✅ Full control over environment and dependencies
- ✅ No Azure ML job submission overhead
- ✅ Easy checkpoint management and model retrieval
- ✅ Can run multiple experiments in parallel (if memory allows)

**Hardware**: NVIDIA A100 GPU (40GB or 80GB VRAM)  
**Expected Training Time**: 2-3 hours for GPT-OSS-20B with LoRA  
**Model**: gpt-oss-20b (20B parameters, Phi-3.5-MoE-instruct)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Data Transfer](#data-transfer)
4. [Environment Configuration](#environment-configuration)
5. [Training Script Setup](#training-script-setup)
6. [Running Training](#running-training)
7. [Monitoring Training](#monitoring-training)
8. [Model Retrieval](#model-retrieval)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. SSH Access to A100 VM

Ensure you have:
- SSH credentials (username, private key file or password)
- VM hostname or IP address
- Port number (typically 22 or custom)

#### Option A: Connect with Private Key (Recommended)

**Step 1: Locate Your Private Key File**

Your private key file is typically:
- Downloaded from Azure Portal when creating the VM
- Named something like `myvm_key.pem`, `id_rsa`, or `azure_key`
- Should have restricted permissions (only you can read it)

**Step 2: Set Correct Permissions (Important!)**

```powershell
# In PowerShell, navigate to where your key is stored
cd C:\Users\YourUsername\.ssh

# If you get permission errors on Windows, you may need to set file permissions:
# Right-click the key file → Properties → Security → Advanced
# - Remove inheritance
# - Remove all users except your account
# - Ensure you have "Read" permission only

# Or use icacls command:
icacls myvm_key.pem /inheritance:r
icacls myvm_key.pem /grant:r "$($env:USERNAME):(R)"
```

**Step 3: Connect Using Private Key**

```powershell
# Basic connection
ssh -i C:\path\to\your\private_key.pem <username>@<vm-hostname-or-ip>

# Example with common key locations:
ssh -i C:\Users\YourUsername\.ssh\azure_key.pem azureuser@my-a100-vm.eastus.cloudapp.azure.com

# Or if your key is in current directory:
ssh -i .\myvm_key.pem azureuser@20.123.45.67

# With custom port (if not using default 22):
ssh -i .\myvm_key.pem -p 2222 azureuser@my-a100-vm.eastus.cloudapp.azure.com
```

**Step 4: Create SSH Config for Easy Access (Optional but Recommended)**

Create/edit `C:\Users\YourUsername\.ssh\config`:

```powershell
# Open config file in notepad
notepad C:\Users\YourUsername\.ssh\config
```

Add the following content:

```
Host a100vm
    HostName my-a100-vm.eastus.cloudapp.azure.com
    User azureuser
    IdentityFile C:\Users\YourUsername\.ssh\azure_key.pem
    Port 22
```

Now you can connect simply with:
```powershell
ssh a100vm
```

#### Option B: Connect with Password

```powershell
# From your local Windows machine (PowerShell)
ssh <username>@<vm-hostname-or-ip>

# Example:
ssh azureuser@my-a100-vm.eastus.cloudapp.azure.com

# You'll be prompted for password
```

#### Finding Your VM Connection Details

If you don't know your VM hostname/IP, find it in Azure Portal:

1. Go to **Azure Portal** → **Virtual Machines**
2. Click on your A100 VM
3. Look for:
   - **Public IP address** (e.g., `20.123.45.67`)
   - **DNS name** (e.g., `my-a100-vm.eastus.cloudapp.azure.com`)
   - **Username** (typically `azureuser` or what you set during creation)

Or use Azure CLI:
```powershell
# Login to Azure
az login

# List VMs and get connection info
az vm list-ip-addresses --name <vm-name> --resource-group <resource-group> --output table

# Get SSH connection string
az vm show --name <vm-name> --resource-group <resource-group> --show-details --query "publicIps" -o tsv
```

#### Troubleshooting SSH Connection

**Issue: Permission denied (publickey)**
```powershell
# Check that you're using the correct key file
# Check that the key has correct permissions (see Step 2 above)
# Verify the username is correct (usually 'azureuser')
```

**Issue: Connection timeout**
```powershell
# Check that the VM is running:
az vm get-instance-view --name <vm-name> --resource-group <resource-group> --query "instanceView.statuses[1].displayStatus"

# Check Network Security Group (NSG) allows SSH on port 22:
az network nsg rule list --nsg-name <nsg-name> --resource-group <resource-group> --output table
```

**Issue: Host key verification failed**
```powershell
# Remove old host key if VM was recreated:
ssh-keygen -R <vm-hostname-or-ip>

# Then try connecting again
ssh -i .\myvm_key.pem azureuser@<vm-hostname>
```

**Test Connection**:
```powershell
# Test with verbose output to debug issues
ssh -v -i .\myvm_key.pem azureuser@<vm-hostname>
```

### 2. Local Files to Transfer

From `HateSpeechDetection_ver2/finetuning/`:
- `data/prepared/train.jsonl` (3,083 samples)
- `data/prepared/validation.jsonl` (545 samples)
- `scripts/train_lora.py` (training script)

---

## Initial Setup

### Step 1: Connect to VM

```powershell
# From your local PowerShell
ssh <username>@<vm-hostname>
```

Example:
```powershell
ssh -i C:\Users\YourUsername\.ssh\azure_key.pem azureuser@my-a100-vm.eastus.cloudapp.azure.com
```

### Step 2: Create Working Directory

```bash
# On the VM
mkdir -p ~/finetuning/data
mkdir -p ~/finetuning/outputs
mkdir -p ~/finetuning/models
cd ~/finetuning
```

### Step 3: Check GPU Availability

```bash
# Verify A100 GPU is available
nvidia-smi

# Expected output should show:
# - GPU: NVIDIA A100 (40GB or 80GB)
# - CUDA Version: 11.7+ or 12.0+
```

**Expected Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    55W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## Data Transfer

### Option 1: SCP (Recommended for Small Files)

```powershell
# From your local PowerShell (in HateSpeechDetection_ver2/finetuning/)

# If using private key authentication:
# Transfer training data
scp -i C:\path\to\your\private_key.pem data/prepared/train.jsonl <username>@<vm-hostname>:~/finetuning/data/

# Transfer validation data
scp -i C:\path\to\your\private_key.pem data/prepared/validation.jsonl <username>@<vm-hostname>:~/finetuning/data/

# Transfer training script
scp -i C:\path\to\your\private_key.pem scripts/train_lora.py <username>@<vm-hostname>:~/finetuning/
```

**Example with Private Key**:
```powershell
scp -i C:\Users\YourUsername\.ssh\azure_key.pem data/prepared/train.jsonl azureuser@my-a100-vm.eastus.cloudapp.azure.com:~/finetuning/data/
scp -i C:\Users\YourUsername\.ssh\azure_key.pem data/prepared/validation.jsonl azureuser@my-a100-vm.eastus.cloudapp.azure.com:~/finetuning/data/
scp -i C:\Users\YourUsername\.ssh\azure_key.pem scripts/train_lora.py azureuser@my-a100-vm.eastus.cloudapp.azure.com:~/finetuning/
```

**If using SSH config (no key needed in command)**:
```powershell
scp data/prepared/train.jsonl a100vm:~/finetuning/data/
scp data/prepared/validation.jsonl a100vm:~/finetuning/data/
scp scripts/train_lora.py a100vm:~/finetuning/
```

### Option 2: SFTP (Alternative)

```powershell
# Start SFTP session with private key
sftp -i C:\path\to\your\private_key.pem <username>@<vm-hostname>

# In SFTP session:
sftp> cd finetuning/data
sftp> put data/prepared/train.jsonl
sftp> put data/prepared/validation.jsonl
sftp> cd ..
sftp> put scripts/train_lora.py
sftp> quit
```

**If using SSH config**:
```powershell
sftp a100vm

# Then use same commands as above
```

### Option 3: Azure Storage (For Large Files)

```bash
# On the VM, if data is in Azure Blob Storage
az login
az storage blob download --account-name <storage_account> \
    --container-name <container> \
    --name train.jsonl \
    --file ~/finetuning/data/train.jsonl \
    --auth-mode login
```

### Verify Data Transfer

```bash
# On the VM
cd ~/finetuning
ls -lh data/
# Should show train.jsonl (~1-2 MB) and validation.jsonl (~300 KB)

ls -lh train_lora.py
# Should show the training script

# Check file contents
head -n 2 data/train.jsonl
```

---

## Environment Configuration

### Step 1: Check Python Version

```bash
python --version
# Should be Python 3.8+ (3.9 or 3.10 recommended)

# If not available, use:
python3 --version
```

### Step 2: Create Virtual Environment (Recommended)

```bash
cd ~/finetuning

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
# Should show: ~/finetuning/venv/bin/python
```

### Step 3: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install core dependencies
pip install transformers==4.36.0
pip install peft==0.7.0
pip install datasets==2.16.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.0
pip install scipy

# Install PyTorch with CUDA support (if not pre-installed)
# Check first if PyTorch is available:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# If torch not installed or CUDA not available:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Expected Output**:
```
Successfully installed transformers-4.36.0
Successfully installed peft-0.7.0
Successfully installed datasets-2.16.0
Successfully installed accelerate-0.25.0
Successfully installed bitsandbytes-0.41.0
```

### Step 4: Verify CUDA and PyTorch

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected Output**:
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 1
```

---

## Training Script Setup

### Step 1: Update Training Script for GPT-OSS-20B

Create an optimized training script for direct execution:

```bash
cd ~/finetuning
nano train_gptoss_20b.py
```

**Paste the following optimized script**:

```python
#!/usr/bin/env python3
"""
Direct LoRA Fine-Tuning for GPT-OSS-20B on A100 VM
Optimized for SSH training with real-time monitoring
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for GPT-OSS-20B")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="gpt-oss-20b",
                        help="Model name or Hugging Face model ID")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Override with specific Hugging Face model ID if needed")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="./data/train.jsonl",
                        help="Path to training data (JSONL format)")
    parser.add_argument("--validation_file", type=str, default="./data/validation.jsonl",
                        help="Path to validation data (JSONL format)")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA arguments (optimized for 20B)
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank (32 recommended for 20B with 3,628 samples)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (2× rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,o_proj,k_proj",
                        help="Comma-separated list of target modules")
    
    # Training arguments (optimized for A100)
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = 4×2=8)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    
    return parser.parse_args()


def load_and_prepare_data(train_file, validation_file, tokenizer, max_seq_length):
    """Load JSONL data and prepare for training"""
    print(f"\nLoading data from:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {validation_file}")
    
    # Load datasets
    dataset = load_dataset('json', data_files={
        'train': train_file,
        'validation': validation_file
    })
    
    print(f"Loaded {len(dataset['train'])} training samples")
    print(f"Loaded {len(dataset['validation'])} validation samples")
    
    # Tokenization function
    def tokenize_function(examples):
        # Assuming JSONL has 'text' field with instruction-formatted content
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_seq_length,
            padding='max_length'
        )
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    print("Tokenization complete.")
    return tokenized_dataset


def setup_lora_model(model_name, model_id, lora_config_args):
    """Load model and apply LoRA configuration"""
    print(f"\nLoading model: {model_id or model_name}")
    print("This may take a few minutes...")
    
    # Determine actual model ID
    actual_model_id = model_id if model_id else model_name
    
    # Check if it's a local path or Hugging Face ID
    if not Path(actual_model_id).exists():
        print(f"  Attempting to load from Hugging Face: {actual_model_id}")
        print(f"  Note: If this fails, you may need to specify the correct model ID")
        print(f"        For gpt-oss-20b, check your Azure deployment or use:")
        print(f"        --model_id <huggingface_org>/<model_name>")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_id,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with A100 optimization
    print("\nLoading model with optimizations for A100...")
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # FP16 for A100
        device_map="auto",           # Automatic GPU placement
        use_cache=False              # Disable KV cache for training
    )
    
    print(f"Model loaded successfully!")
    print(f"  Model parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"  Memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    target_modules = lora_config_args['target_modules'].split(',')
    
    lora_config = LoraConfig(
        r=lora_config_args['lora_r'],
        lora_alpha=lora_config_args['lora_alpha'],
        lora_dropout=lora_config_args['lora_dropout'],
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    print(f"  LoRA rank: {lora_config_args['lora_r']}")
    print(f"  LoRA alpha: {lora_config_args['lora_alpha']}")
    print(f"  LoRA dropout: {lora_config_args['lora_dropout']}")
    print(f"  Target modules: {', '.join(target_modules)}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    print(f"\nTrainable parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Percentage: {trainable_percent:.4f}%")
    
    return model, tokenizer


def main():
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("GPT-OSS-20B LoRA Fine-Tuning")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Setup model and tokenizer
    lora_config_args = {
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'target_modules': args.target_modules
    }
    
    model, tokenizer = setup_lora_model(args.model_name, args.model_id, lora_config_args)
    
    # Load and prepare data
    tokenized_dataset = load_and_prepare_data(
        args.train_file,
        args.validation_file,
        tokenizer,
        args.max_seq_length
    )
    
    # Setup training arguments
    output_dir = Path(args.output_dir) / f"gptoss_20b_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        report_to="none",  # Disable wandb/tensorboard for simple logging
        remove_unused_columns=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True  # Enable for memory efficiency
    )
    
    print(f"\nOutput directory: {output_dir}")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Setup trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print(f"  Total training samples: {len(tokenized_dataset['train'])}")
    print(f"  Total validation samples: {len(tokenized_dataset['validation'])}")
    print(f"  Batch size per device: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Total optimization steps: ~{len(tokenized_dataset['train']) * args.num_train_epochs // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
    print("=" * 80)
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training complete! Saving final model...")
    print("=" * 80)
    
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save training state
    trainer.save_state()
    
    print(f"\nModel saved to: {final_model_path}")
    print(f"Training metrics saved to: {output_dir}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {train_result.metrics.get('train_runtime', 0) / 3600:.2f} hours")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

### Step 2: Make Script Executable

```bash
chmod +x train_gptoss_20b.py
```

---

## Running Training

### Option 1: Interactive Training (Simple)

**For quick testing or short runs**:

```bash
cd ~/finetuning

# Activate virtual environment
source venv/bin/activate

# Run training
python train_gptoss_20b.py \
    --model_name gpt-oss-20b \
    --train_file ./data/train.jsonl \
    --validation_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200
```

**Important**: If `gpt-oss-20b` is not directly available on Hugging Face, you'll need to specify the actual model ID:

```bash
# Example if the model has a different Hugging Face ID:
python train_gptoss_20b.py \
    --model_id microsoft/Phi-3.5-MoE-instruct \
    --train_file ./data/train.jsonl \
    --validation_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --learning_rate 2e-4
```

### Option 2: Background Training with tmux (Recommended)

**For long-running training (2-3 hours)**:

```bash
# Install tmux if not available
sudo apt-get update && sudo apt-get install -y tmux

# Start tmux session
tmux new -s finetuning

# Inside tmux session:
cd ~/finetuning
source venv/bin/activate

# Run training
python train_gptoss_20b.py \
    --model_name gpt-oss-20b \
    --train_file ./data/train.jsonl \
    --validation_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 2>&1 | tee training.log
```

**Detach from tmux session**: Press `Ctrl+B`, then `D`

**Reattach to tmux session**:
```bash
tmux attach -t finetuning
```

**Kill tmux session** (after training completes):
```bash
tmux kill-session -t finetuning
```

### Option 3: Background Training with nohup (Alternative)

```bash
cd ~/finetuning
source venv/bin/activate

nohup python train_gptoss_20b.py \
    --model_name gpt-oss-20b \
    --train_file ./data/train.jsonl \
    --validation_file ./data/validation.jsonl \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 > training.log 2>&1 &

# Get process ID
echo $!

# Disconnect from SSH (training continues in background)
exit
```

**Monitor training**:
```bash
# Reconnect to VM
ssh <username>@<vm-hostname>

# Check training log
tail -f ~/finetuning/training.log

# Check GPU usage
watch -n 5 nvidia-smi
```

---

## Monitoring Training

### Real-Time Monitoring

#### 1. Training Logs

```bash
# Watch training progress
tail -f ~/finetuning/training.log

# Or if using tmux:
tmux attach -t finetuning
```

**Key Metrics to Watch**:
- **Training Loss**: Should decrease consistently (target: <1.0 by end)
- **Evaluation Loss**: Should decrease without diverging from training loss
- **Learning Rate**: Should follow warmup schedule
- **GPU Memory**: Should be ~30-35GB for 20B model with batch=4

**Sample Output**:
```
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
{'eval_loss': 1.156, 'eval_runtime': 45.2, 'epoch': 1.0}
{'loss': 0.987, 'learning_rate': 0.00018, 'epoch': 1.5}
{'eval_loss': 0.923, 'eval_runtime': 44.8, 'epoch': 2.0}
{'loss': 0.845, 'learning_rate': 0.00016, 'epoch': 2.5}
{'eval_loss': 0.812, 'eval_runtime': 45.0, 'epoch': 3.0}
```

#### 2. GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 5 nvidia-smi

# Or for detailed monitoring:
nvidia-smi dmon -s pucvmet -d 5
```

**Healthy Training Indicators**:
- GPU Utilization: 95-100%
- GPU Memory: 30-35GB / 40GB (or 35-45GB / 80GB for A100-80GB)
- GPU Temperature: 60-80°C
- Power Usage: 300-400W (near TDP)

#### 3. Checkpoint Monitoring

```bash
# Check saved checkpoints
ls -lh ~/finetuning/outputs/gptoss_20b_lora_*/

# Expected structure:
# checkpoint-200/
# checkpoint-400/
# checkpoint-600/
# final_model/
# trainer_state.json
# training_args.bin
```

### Early Stopping Indicators

**Good Progress**:
- ✅ Training loss decreasing consistently
- ✅ Eval loss tracking training loss (within 0.1-0.2)
- ✅ GPU utilization near 100%
- ✅ No CUDA out-of-memory errors

**Warning Signs**:
- ⚠️ Eval loss diverging from training loss (overfitting)
- ⚠️ Training loss not decreasing after epoch 1
- ⚠️ GPU utilization < 50% (data loading bottleneck)
- ⚠️ Frequent CUDA errors

**Stop Training If**:
- ❌ Eval loss increasing for 2+ consecutive evaluations
- ❌ Training loss NaN or infinity
- ❌ Out-of-memory errors (reduce batch size)
- ❌ GPU temperature > 85°C sustained

---

## Model Retrieval

### Step 1: Locate Fine-Tuned Model

```bash
cd ~/finetuning/outputs

# Find latest training run
ls -lt

# Navigate to final model
cd gptoss_20b_lora_<timestamp>/final_model

# Verify model files
ls -lh
```

**Expected Files**:
- `adapter_config.json` (LoRA configuration)
- `adapter_model.bin` or `adapter_model.safetensors` (LoRA weights, ~100-200 MB)
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer.json`

### Step 2: Download Model to Local Machine

#### Option A: SCP (Recommended)

```powershell
# From your local PowerShell
cd Q:\workspace\HateSpeechDetection_ver2\finetuning

# Create local model directory
mkdir -p models\gptoss_20b_finetuned

# Download LoRA adapter with private key
scp -i C:\path\to\your\private_key.pem -r <username>@<vm-hostname>:~/finetuning/outputs/gptoss_20b_lora_*/final_model/* models\gptoss_20b_finetuned\

# Or download entire checkpoint directory:
scp -i C:\path\to\your\private_key.pem -r <username>@<vm-hostname>:~/finetuning/outputs/gptoss_20b_lora_*/final_model models\gptoss_20b_finetuned
```

**Example with Private Key**:
```powershell
scp -i C:\Users\YourUsername\.ssh\azure_key.pem -r azureuser@my-a100-vm.eastus.cloudapp.azure.com:~/finetuning/outputs/gptoss_20b_lora_20251021_143022/final_model/* Q:\workspace\HateSpeechDetection_ver2\finetuning\models\gptoss_20b_finetuned\
```

**If using SSH config**:
```powershell
scp -r a100vm:~/finetuning/outputs/gptoss_20b_lora_*/final_model/* models\gptoss_20b_finetuned\
```

#### Option B: Azure Storage Upload (For Sharing)

```bash
# On the VM
cd ~/finetuning/outputs/gptoss_20b_lora_*/final_model

# Upload to Azure Blob Storage
az storage blob upload-batch \
    --account-name <storage_account> \
    --destination <container>/models/gptoss_20b_finetuned \
    --source . \
    --auth-mode login
```

Then download from Azure Storage on your local machine.

### Step 3: Test Fine-Tuned Model Locally

Create a test script on your local machine:

```python
# test_finetuned_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model_name = "gpt-oss-20b"  # Or actual Hugging Face ID
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    "./models/gptoss_20b_finetuned"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

# Test inference
test_input = """Classify the following post as 'hate' or 'not hate':
Post: "You people are ruining this country!"
Classification:"""

inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```

Run:
```powershell
python test_finetuned_model.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors

**Issue**: `OSError: gpt-oss-20b does not appear to be a valid Hugging Face model identifier`

**Solution**: Specify the actual Hugging Face model ID:
```bash
python train_gptoss_20b.py \
    --model_id microsoft/Phi-3.5-MoE-instruct \
    # ... other args
```

Or check your Azure deployment for the correct model path.

#### 2. CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solution 1**: Reduce batch size:
```bash
python train_gptoss_20b.py \
    --per_device_train_batch_size 2 \  # Reduced from 4
    --gradient_accumulation_steps 4 \  # Increased from 2 (keeps effective batch=8)
    # ... other args
```

**Solution 2**: Enable gradient checkpointing (already enabled in script):
```python
gradient_checkpointing=True
```

**Solution 3**: Reduce sequence length:
```bash
python train_gptoss_20b.py \
    --max_seq_length 384 \  # Reduced from 512
    # ... other args
```

#### 3. Slow Data Loading

**Issue**: GPU utilization < 50%, training very slow

**Solution**: Increase data loader workers:
```python
# In TrainingArguments:
dataloader_num_workers=8  # Increase from 4
```

#### 4. SSH Connection Drops

**Issue**: Training stops when SSH disconnects

**Solution**: Use tmux or nohup (see [Running Training](#running-training))

```bash
tmux new -s finetuning
# Run training inside tmux
# Press Ctrl+B, then D to detach
```

Reconnect:
```bash
tmux attach -t finetuning
```

#### 5. Permission Errors

**Issue**: `Permission denied: ~/finetuning/outputs`

**Solution**:
```bash
chmod -R 755 ~/finetuning
mkdir -p ~/finetuning/outputs
```

#### 6. Evaluation Taking Too Long

**Issue**: Evaluation steps take 5+ minutes

**Solution**: Reduce evaluation frequency:
```bash
python train_gptoss_20b.py \
    --eval_steps 400 \  # Increased from 200
    --save_steps 400 \  # Match eval_steps
    # ... other args
```

---

## Quick Reference Commands

### SSH and File Transfer

```powershell
# Connect to VM with private key
ssh -i C:\path\to\your\private_key.pem <username>@<vm-hostname>

# Or if you set up SSH config:
ssh a100vm

# Transfer files to VM
scp -i C:\path\to\your\private_key.pem local_file.txt <username>@<vm-hostname>:~/finetuning/

# Download files from VM
scp -i C:\path\to\your\private_key.pem <username>@<vm-hostname>:~/finetuning/outputs/model/* ./local_dir/

# With SSH config (simpler):
scp local_file.txt a100vm:~/finetuning/
scp a100vm:~/finetuning/outputs/model/* ./local_dir/
```

### Training Management

```bash
# Start training in tmux
tmux new -s finetuning
source ~/finetuning/venv/bin/activate
python train_gptoss_20b.py [args]

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t finetuning

# Monitor GPU
nvidia-smi

# Monitor logs
tail -f training.log
```

### Checkpoints and Models

```bash
# List checkpoints
ls -lh ~/finetuning/outputs/gptoss_20b_lora_*/

# Find latest model
ls -lt ~/finetuning/outputs/ | head -n 5

# Check model size
du -sh ~/finetuning/outputs/gptoss_20b_lora_*/final_model
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Setup** | 10-15 min | SSH connection, environment setup, data transfer |
| **Model Loading** | 5-10 min | Download and load GPT-OSS-20B base model |
| **Training Epoch 1** | 45-60 min | Initial training with warmup |
| **Training Epoch 2** | 45-60 min | Continued training |
| **Training Epoch 3** | 45-60 min | Final training epoch |
| **Evaluation** | 5-10 min | Final validation and checkpoint saving |
| **Model Retrieval** | 5-10 min | Download fine-tuned adapter to local machine |
| **Total** | **2.5-3.5 hours** | End-to-end fine-tuning |

---

## Next Steps After Training

1. **Evaluate Performance**: Run production validation (1,009 samples) to measure F1-score
2. **Compare Against Baseline**: Compare with GPT-OSS-20B baseline (target: F1 ≥ 0.610)
3. **Analyze Bias Metrics**: Calculate FPR/FNR by protected group (LGBTQ+, Mexican, Middle East)
4. **Deploy to Production**: Merge LoRA adapter with base model for inference
5. **Document Results**: Update model card with fine-tuned performance metrics

---

## Support and Documentation

- **Model Selection Guide**: `FINE_TUNING_MODEL_SELECTION_README.md`
- **Azure ML Job Submission**: `AZURE_SUBMISSION_GUIDE.md` (alternative approach)
- **LoRA Configuration**: `README.md` (general fine-tuning documentation)
- **Baseline Performance**: `../prompt_engineering/prompt_templates/overall_summary_ift_README.md`

---

**Last Updated**: October 21, 2025  
**Recommended for**: Direct A100 VM access with SSH  
**Alternative**: Azure ML job submission (see `AZURE_SUBMISSION_GUIDE.md`)
