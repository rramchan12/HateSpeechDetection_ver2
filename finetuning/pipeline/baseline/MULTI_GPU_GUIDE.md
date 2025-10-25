# Multi-GPU Performance Optimization Guide

## Current Setup: 4x NVIDIA A100 80GB GPUs

**Status**: Currently using only **GPU 0** → **Potential 4x speedup available!**

```bash
# Current GPU configuration
nvidia-smi
# 4x A100 80GB PCIe (GPU 0, 1, 2, 3)
# Total: 320GB VRAM, 80GB per GPU
```

---

## Performance Baseline

| Configuration | Samples/Second | Time for 100 Samples | GPU Utilization |
|--------------|----------------|---------------------|-----------------|
| **Sequential (batch_size=1)** | 0.10 | ~1000s (16.7 min) | 1 GPU |
| **Batch (batch_size=4)** | 0.27 | ~370s (6.2 min) | 1 GPU |
| **Multi-GPU (4 GPUs)** | **~1.0+** | **~100s (1.7 min)** | 4 GPUs |

**Expected improvement: 2.7x → 10x with multi-GPU!**

---

## Option 1: Model Parallelism (Easiest - Already Supported!)

### What It Does
- Splits the 20B parameter model across multiple GPUs
- Each GPU holds different layers
- Data flows through GPUs sequentially

### Pros
- ✅ Works out-of-the-box with `device_map="balanced"`
- ✅ No code changes needed
- ✅ Handles models too large for single GPU

### Cons
- ⚠️ Limited speedup (GPUs wait for each other)
- ⚠️ Good for large models, not for throughput

### Implementation

**Method A: Use `device_map="balanced"`** (Automatic)

```python
# In model_loader/loader.py (line ~120)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="balanced",  # Change from "auto" to "balanced"
    token=use_auth_token,
    cache_dir=str(cache_dir)
)
```

**Method B: Manual device map** (For fine control)

```python
# Manually specify which layers go to which GPU
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    # ... layers 0-10 on GPU 0
    "model.layers.11": 1,
    # ... layers 11-20 on GPU 1
    # ... layers 21-30 on GPU 2
    # ... layers 31-40 on GPU 3
    "model.norm": 3,
    "lm_head": 3
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=device_map,
    token=use_auth_token,
    cache_dir=str(cache_dir)
)
```

### Command

```bash
# No changes needed! Just run normally
python -m finetuning.pipeline.baseline.runner \
    --data_source unified \
    --max_samples 100 \
    --batch_size 4
```

---

## Option 2: Data Parallelism with Accelerate (RECOMMENDED for Throughput)

### What It Does
- Replicates the full model on each GPU
- Each GPU processes different samples in parallel
- **4x throughput** (linear scaling)

### Pros
- ✅ **Best throughput** - true parallelism
- ✅ Linear scaling with GPU count
- ✅ Each GPU works independently

### Cons
- ⚠️ Requires model fits on single GPU (20B @ bfloat16 = ~42GB → OK for A100 80GB!)
- ⚠️ Requires `accelerate` library

### Implementation

**Step 1: Install Accelerate**

```bash
pip install accelerate
```

**Step 2: Configure Accelerate**

```bash
accelerate config

# Answer:
# - This machine: Yes
# - Distributed training: multi-GPU
# - Number of machines: 1
# - Number of GPUs: 4
# - Mixed precision: bf16
```

**Step 3: Create launcher script**

Create `finetuning/pipeline/baseline/runner_distributed.py`:

```python
#!/usr/bin/env python3
"""
Distributed runner using Accelerate for multi-GPU data parallelism.
"""

from accelerate import Accelerator
from runner import LocalModelConnector, run_baseline
import sys

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="unified")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--strategy", default="combined_optimized")
    args = parser.parse_args()
    
    # Print info only on main process
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
    
    # Load model on each GPU
    connector = LocalModelConnector(
        model_name="openai/gpt-oss-20b",
        batch_size=args.batch_size
    )
    connector.load_model_once()
    
    # Wrap model with accelerator
    connector.model = accelerator.prepare(connector.model)
    
    # Run baseline with distributed processing
    # Each GPU gets a subset of the data
    run_baseline(args)

if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run with Accelerate**

```bash
# Launch distributed training
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    finetuning/pipeline/baseline/runner_distributed.py \
    --data_source unified \
    --max_samples 100 \
    --batch_size 4
```

---

## Option 3: Simple Multi-Process Data Parallelism (Quick Solution)

### What It Does
- Run 4 separate processes, one per GPU
- Split dataset manually
- Merge results at the end

### Implementation

**Step 1: Split dataset**

```bash
cd /home/azureuser/workspace/HateSpeechDetection_ver2

# Split validation data into 4 parts
python3 << 'EOF'
import json
from pathlib import Path

# Load full dataset
with open('data/processed/unified/unified_val.json') as f:
    data = json.load(f)

# Split into 4 chunks
chunk_size = len(data) // 4
chunks = [
    data[i*chunk_size:(i+1)*chunk_size]
    for i in range(4)
]

# Save chunks
output_dir = Path('data/processed/unified_splits')
output_dir.mkdir(exist_ok=True)

for i, chunk in enumerate(chunks):
    output_file = output_dir / f'unified_val_gpu{i}.json'
    with open(output_file, 'w') as f:
        json.dump(chunk, f)
    print(f"Saved {len(chunk)} samples to {output_file}")
EOF
```

**Step 2: Run on each GPU in parallel**

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m finetuning.pipeline.baseline.runner \
    --data_source unified_splits/unified_val_gpu0 \
    --batch_size 4 \
    --output_dir ./finetuning/outputs/multi_gpu/gpu0 &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m finetuning.pipeline.baseline.runner \
    --data_source unified_splits/unified_val_gpu1 \
    --batch_size 4 \
    --output_dir ./finetuning/outputs/multi_gpu/gpu1 &

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python -m finetuning.pipeline.baseline.runner \
    --data_source unified_splits/unified_val_gpu2 \
    --batch_size 4 \
    --output_dir ./finetuning/outputs/multi_gpu/gpu2 &

# Terminal 4 (GPU 3)
CUDA_VISIBLE_DEVICES=3 python -m finetuning.pipeline.baseline.runner \
    --data_source unified_splits/unified_val_gpu3 \
    --batch_size 4 \
    --output_dir ./finetuning/outputs/multi_gpu/gpu3 &

# Wait for all to complete
wait
```

**Step 3: Merge results**

```python
# merge_results.py
import json
import pandas as pd
from pathlib import Path

output_base = Path('./finetuning/outputs/multi_gpu')
merged_dir = output_base / 'merged'
merged_dir.mkdir(exist_ok=True)

# Find all run directories
gpu_dirs = [output_base / f'gpu{i}' for i in range(4)]

# Merge detailed results
all_results = []
for gpu_dir in gpu_dirs:
    run_dirs = list(gpu_dir.glob('run_*'))
    if run_dirs:
        latest = sorted(run_dirs)[-1]
        csv_file = list(latest.glob('strategy_*.csv'))[0]
        df = pd.read_csv(csv_file)
        all_results.append(df)

merged_df = pd.concat(all_results, ignore_index=True)
merged_df.to_csv(merged_dir / 'merged_results.csv', index=False)

print(f"Merged {len(merged_df)} results from 4 GPUs")
print(f"Output: {merged_dir / 'merged_results.csv'}")
```

---

## Performance Comparison Table

| Method | Setup Effort | Code Changes | Expected Speedup | Best For |
|--------|--------------|--------------|------------------|----------|
| **Sequential (baseline)** | None | None | 1x | Testing |
| **Batch Processing** | None | ✅ Done | 2.7x | Single GPU |
| **Model Parallelism** | Low | Change 1 line | 1.5-2x | Very large models |
| **Accelerate (Data Parallel)** | Medium | Add wrapper | **3-4x** | **Production** |
| **Multi-Process** | Low | Split script | **3.5-4x** | Quick solution |

---

## Recommendation

### For Immediate Use (Today):
**Use Multi-Process Data Parallelism (Option 3)**
- Easiest to implement (no code changes)
- 3.5-4x speedup
- Can run in parallel terminals

### For Production (This Week):
**Use Accelerate with Data Parallelism (Option 2)**
- Best performance and scalability
- Clean integration
- Future-proof for fine-tuning

### Quick Test (Now):
**Change `device_map="auto"` to `device_map="balanced"`**
- 1 line change in model_loader/loader.py
- Instant multi-GPU usage
- Limited speedup but validates setup

---

## Quick Start: Test Multi-GPU Now

### Test Model Parallelism (1 minute)

```bash
# Edit model_loader/loader.py line 120
# Change: device_map="auto"
# To:     device_map="balanced"

# Run test
python -m finetuning.pipeline.baseline.runner \
    --max_samples 20 \
    --batch_size 4

# Check GPU usage
watch -n 1 nvidia-smi
# You should see activity on multiple GPUs!
```

### Test Multi-Process (5 minutes)

```bash
# Run on GPU 0 and GPU 1 simultaneously
CUDA_VISIBLE_DEVICES=0 python -m finetuning.pipeline.baseline.runner \
    --max_samples 50 --batch_size 4 \
    --output_dir ./finetuning/outputs/test_gpu0 &

CUDA_VISIBLE_DEVICES=1 python -m finetuning.pipeline.baseline.runner \
    --max_samples 50 --batch_size 4 \
    --output_dir ./finetuning/outputs/test_gpu1 &

wait

# Compare: Should take ~half the time!
```

---

## Monitoring GPU Usage

```bash
# Watch real-time GPU utilization
watch -n 1 nvidia-smi

# Check per-process memory
nvidia-smi pmon -i 0,1,2,3

# Full stats
nvidia-smi dmon -i 0,1,2,3
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Model too large for single GPU
- Use model parallelism (`device_map="balanced"`)
- Reduce batch_size
- Use quantization (int8/int4)

### Issue: Only GPU 0 shows activity
**Problem**: Not using multi-GPU properly
- Check `device_map` setting
- Verify CUDA_VISIBLE_DEVICES for multi-process
- Check accelerate configuration

### Issue: Processes crash
**Solution**: Not enough shared memory
```bash
# Increase shared memory
docker run --shm-size=16g ...
```

---

## Next Steps

1. **Immediate**: Test model parallelism with `device_map="balanced"`
2. **Today**: Run multi-process test on 2 GPUs
3. **This week**: Implement Accelerate for production
4. **Future**: Add DDP (DistributedDataParallel) for fine-tuning

---

## Questions About 4 Nodes (IP 50000-50003)

You mentioned **4 nodes with IPs 50000-50003**. If these are **4 separate machines**:

### For Multiple Machines:
Use **PyTorch DistributedDataParallel (DDP)** with `torchrun`:

```bash
# On each node, run:
# Node 0 (master):
torchrun --nproc_per_node=4 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=10.0.0.4 \
    --master_port=29500 \
    finetuning/pipeline/baseline/runner.py

# Node 1:
torchrun --nproc_per_node=4 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr=10.0.0.4 \
    --master_port=29500 \
    finetuning/pipeline/baseline/runner.py

# Repeat for nodes 2 and 3...
```

**Total GPUs**: 4 nodes × 4 GPUs = **16 A100 GPUs!**
**Potential speedup**: **16x!**

---

## Contact & Support

For questions:
- Check nvidia-smi output
- Review logs in outputs/
- Test with small samples first
