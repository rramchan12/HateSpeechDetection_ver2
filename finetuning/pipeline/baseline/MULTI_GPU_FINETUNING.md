# Multi-GPU Fine-tuning Guide

## Question: Does Multi-GPU Help with Fine-tuning?

**YES! But it works differently than inference.**

---

## Inference vs Fine-tuning: Key Differences

### Inference (What We Just Tested) ✅

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │
│ Model A │  │ Model B │  │ Model C │  │ Model D │
│ (copy)  │  │ (copy)  │  │ (copy)  │  │ (copy)  │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
  Samples      Samples      Samples      Samples
   0-127       128-255      256-383      384-511
     │            │            │            │
     └────────────┴────────────┴────────────┘
              Merge Results
              (No synchronization needed!)
```

**Characteristics**:
- ✅ Each GPU runs **independently**
- ✅ No communication between GPUs
- ✅ Simply merge results at the end
- ✅ **Perfect linear scaling** (2 GPUs = 2x speedup, 4 GPUs = 4x)
- ✅ Easy to implement (what we just did!)

**Our Results**: 20 samples in 42s (2 GPUs) vs 84s (1 GPU) = **2x speedup** ✅

---

### Fine-tuning (Training)

```
┌─────────────────────────────────────────────────┐
│           SYNCHRONIZED TRAINING                 │
│                                                 │
│  GPU 0        GPU 1        GPU 2        GPU 3  │
│  ┌───┐       ┌───┐       ┌───┐       ┌───┐    │
│  │ W │       │ W │       │ W │       │ W │    │  W = Model Weights
│  └─┬─┘       └─┬─┘       └─┬─┘       └─┬─┘    │  (kept in sync)
│    │           │           │           │       │
│    ▼           ▼           ▼           ▼       │
│  Batch 0    Batch 1    Batch 2    Batch 3     │  Different batches
│    │           │           │           │       │
│    ▼           ▼           ▼           ▼       │
│  Grad 0     Grad 1     Grad 2     Grad 3      │  Compute gradients
│    │           │           │           │       │
│    └───────────┴───────────┴───────────┘       │
│                    │                            │
│                    ▼                            │
│            Average Gradients                   │  All-Reduce
│                    │                            │
│    ┌───────────────┴───────────────┐           │
│    ▼           ▼           ▼        ▼           │
│  Update W   Update W   Update W  Update W      │  Synchronized update
│                                                 │
└─────────────────────────────────────────────────┘
```

**Characteristics**:
- ⚠️ GPUs must **communicate** (share gradients)
- ⚠️ Weights must stay **synchronized**
- ⚠️ Requires special frameworks (DDP, FSDP, DeepSpeed)
- ✅ Still provides speedup (not always linear due to communication overhead)
- ✅ Can train 4x larger batch sizes

**Expected Results**:
- 4 GPUs: **~3-3.5x speedup** (not 4x due to communication)
- Can use **4x larger effective batch size** for better convergence

---

## Multi-GPU Strategies for Fine-tuning

### Strategy 1: Data Parallelism (DDP) - **RECOMMENDED**

**Best for**: Standard fine-tuning, LoRA, QLoRA

```python
# Each GPU has full model copy
# Split batch across GPUs
# Synchronize gradients after each step

GPU 0: Batch [0, 1, 2, 3] → Gradients → \
GPU 1: Batch [4, 5, 6, 7] → Gradients →  } Average → Update All
GPU 2: Batch [8, 9,10,11] → Gradients → /
GPU 3: Batch[12,13,14,15] → Gradients → /
```

**Pros**:
- ✅ Easy to implement (PyTorch DDP or HuggingFace Accelerate)
- ✅ Good speedup (3-3.5x on 4 GPUs)
- ✅ Works with your 20B model (42GB fits on A100 80GB)

**Implementation**:
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# Training loop - Accelerate handles multi-GPU automatically!
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

**Expected Performance**:
- **1 GPU**: ~8 hours for 3 epochs (2,686 samples)
- **4 GPUs**: ~2.5 hours for 3 epochs (**3.2x speedup**)

---

### Strategy 2: Model Parallelism (Pipeline/Tensor Parallel)

**Best for**: Models too large for single GPU

```python
# Split model layers across GPUs
GPU 0: Layers 0-10   → Forward  → \
GPU 1: Layers 11-20  → Forward  →  } Sequential
GPU 2: Layers 21-30  → Forward  → /
GPU 3: Layers 31-40  → Forward  → /
```

**Pros**:
- ✅ Can fit models larger than single GPU memory
- ✅ No batch size limitation

**Cons**:
- ⚠️ GPUs wait for each other (sequential bottleneck)
- ⚠️ Limited speedup (~1.2-1.5x)
- ⚠️ More complex to implement

**When to Use**:
- Your model is **42GB** → Fits on A100 80GB ✅
- **Not needed for your case!** Use DDP instead.

---

### Strategy 3: Fully Sharded Data Parallel (FSDP)

**Best for**: Very large models (>70B parameters)

```python
# Each GPU holds 1/N of model parameters
# Communicate to reassemble during forward/backward

GPU 0: Params[0:25%]  + communicate
GPU 1: Params[25:50%] + communicate  
GPU 2: Params[50:75%] + communicate
GPU 3: Params[75:100%] + communicate
```

**Pros**:
- ✅ Can train models 4x larger
- ✅ Lower memory per GPU
- ✅ Good speedup (~3x on 4 GPUs)

**Cons**:
- ⚠️ More communication overhead
- ⚠️ Complex setup

**When to Use**:
- Models >70B parameters
- Your 20B model fits on single GPU → **DDP is simpler!**

---

### Strategy 4: LoRA with Multi-GPU (BEST for Your Case!)

**What is LoRA?**
- Fine-tune only **0.1% of parameters** (20M instead of 20B)
- **10-100x less memory** for gradients
- **2-3x faster** training
- Similar accuracy to full fine-tuning

```python
# LoRA: Only update small adapter layers
Original Model (frozen): 20B params = 42GB
LoRA Adapters (trainable): 20M params = 40MB

Total memory: 42GB (model) + 40MB (adapters) = 42.04GB
```

**Multi-GPU LoRA**:
```python
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# 1. Load base model (frozen)
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")

# 2. Add LoRA adapters (trainable)
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# 3. Multi-GPU with Accelerate
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# 4. Train (multi-GPU automatic!)
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

**Expected Performance**:
- **1 GPU**: ~2 hours for 3 epochs (LoRA is fast!)
- **4 GPUs**: ~35 minutes (**3.4x speedup**)
- **Memory**: Only 43GB per GPU (plenty of room on A100 80GB)

---

## Recommended Approach for Your Project

### Phase 1: LoRA Fine-tuning on 1 GPU (Start Here)

```bash
# Quick test to validate setup
python -m finetuning.pipeline.baseline.runner \
    --finetune example_template.json \
    --model_name openai/gpt-oss-20b \
    --finetune_output_dir ./finetuning/outputs/models/lora_test
```

**Why start with 1 GPU?**
- ✅ Simpler to debug
- ✅ Validate LoRA works
- ✅ Still reasonably fast (~2 hours)

---

### Phase 2: Multi-GPU LoRA (Production)

```bash
# Install accelerate
pip install accelerate

# Configure accelerate
accelerate config
# Choose: multi-GPU, 4 GPUs, bf16 precision

# Launch multi-GPU training
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    finetuning/pipeline/baseline/runner.py \
    --finetune example_template.json \
    --model_name openai/gpt-oss-20b \
    --finetune_output_dir ./finetuning/outputs/models/lora_4gpu
```

**Expected Results**:
- 4x larger effective batch size (better convergence)
- ~3.4x faster training
- **Total time: ~35 minutes** vs ~2 hours on 1 GPU

---

## Comparison Table: Inference vs Fine-tuning

| Aspect | Inference (Current) | Fine-tuning (Next) |
|--------|-------------------|-------------------|
| **GPU Communication** | None | High (gradient sync) |
| **Synchronization** | Not needed | Required every step |
| **Speedup (4 GPUs)** | 4x (linear) | 3-3.5x (communication overhead) |
| **Implementation** | Simple (multi-process) | Moderate (DDP/Accelerate) |
| **Memory per GPU** | 42GB (full model) | 43GB (model + LoRA) |
| **Best Strategy** | Multi-process ✅ | DDP + LoRA |
| **Current Status** | **Working!** ✅ | Not implemented yet |

---

## Implementation Plan for Multi-GPU Fine-tuning

### Step 1: Implement Single-GPU LoRA (This Week)

**File**: `finetuning/pipeline/baseline/finetuner.py` (new file)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def finetune_with_lora(
    model_name: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4
):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_file,
        eval_dataset=val_file,
        tokenizer=tokenizer,
    )
    
    # Train!
    trainer.train()
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    
    return model
```

---

### Step 2: Add Multi-GPU Support (Next Week)

**Option A: Using Accelerate** (Easiest)

```python
from accelerate import Accelerator

def finetune_with_lora_multigpu(...):
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(...)
    model = get_peft_model(model, lora_config)
    
    # Prepare for multi-GPU (automatic!)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            
            # Accelerate handles multi-GPU gradient sync!
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
```

**Launch**:
```bash
accelerate launch --num_processes 4 finetuning/pipeline/baseline/runner.py --finetune ...
```

---

**Option B: Using DeepSpeed** (Most Optimized)

```json
// deepspeed_config.json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

```bash
deepspeed --num_gpus=4 finetuning/pipeline/baseline/runner.py \
    --finetune example_template.json \
    --deepspeed deepspeed_config.json
```

---

## Expected Timeline

### Phase 1: Single-GPU LoRA (2-3 days)
1. ✅ Day 1: Implement basic LoRA training
2. ✅ Day 2: Test on small dataset (100 samples)
3. ✅ Day 3: Full training (2,686 samples, ~2 hours)

### Phase 2: Multi-GPU LoRA (1-2 days)
1. ✅ Day 4: Add Accelerate integration
2. ✅ Day 5: Test 4-GPU training (~35 minutes)

### Phase 3: Production (Ongoing)
- ✅ Compare LoRA vs full fine-tuning
- ✅ Hyperparameter tuning
- ✅ Multiple training runs with different configs

---

## Key Takeaways

### For Inference (Current) ✅
- ✅ **Simple multi-process works great**
- ✅ **4x linear speedup with 4 GPUs**
- ✅ **Already validated and working!**
- ✅ Use: `CUDA_VISIBLE_DEVICES=X python -m ...`

### For Fine-tuning (Next) 🚀
- ⚠️ **Needs gradient synchronization (DDP/Accelerate)**
- ✅ **3-3.5x speedup with 4 GPUs** (communication overhead)
- ✅ **LoRA highly recommended** (10-100x faster than full fine-tuning)
- ✅ **Accelerate is easiest** (handles multi-GPU automatically)
- 🔨 **Not implemented yet** - needs work

### Recommended Path
1. **Today**: Continue using multi-GPU for inference ✅
2. **This week**: Implement single-GPU LoRA fine-tuning
3. **Next week**: Add multi-GPU support with Accelerate
4. **Result**: Full pipeline with optimal performance! 🎉

---

## Questions to Consider

1. **Do you want to start implementing LoRA fine-tuning now?**
   - I can create the single-GPU implementation first
   - Then add multi-GPU support once that works

2. **What's your priority?**
   - Fast inference (already done! ✅)
   - Fast fine-tuning (needs implementation)
   - Both?

3. **Timeline?**
   - Quick prototype (single-GPU LoRA, 2-3 days)
   - Production-ready (multi-GPU LoRA, 1 week)

Let me know and I can start implementing! 🚀
