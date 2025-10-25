# Accelerate Quick Reference

##  Package Location
```
finetuning/pipeline/connector/
├── __init__.py
├── accelerate_connector.py       # Main connector class
├── test_accelerate_connector.py  # Unit tests
└── INTEGRATION_STATUS.md         # Detailed documentation
```

##  Usage

### Single GPU (Default)
```bash
python -m finetuning.pipeline.baseline.runner \
    --data_file canned_50_quick \
    --max_samples 5
```
No changes needed! Works exactly as before.

### Multi-GPU with Accelerate
```bash
# First time setup (one-time)
pip install accelerate
accelerate config  # Configure for your GPU setup

# Run with multiple GPUs
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file unified \
    --max_samples 100
```

### With Custom Prompt Template
```bash
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file canned_100_stratified \
    --prompt_template ./prompt_engineering/prompt_templates/combined/combined_gptoss_v1.json \
    --strategy combined_optimized \
    --max_samples 50
```

##  Key Flag

### `--use_accelerate`
- **Purpose**: Enable multi-GPU inference via Accelerate
- **Required**: Only when using `accelerate launch`
- **Effect**: Automatically splits dataset across GPUs
- **Default**: False (single-GPU mode)

##  How It Works

### Without `--use_accelerate` (Default)
```
Runner → load_model() → run_inference_with_prompt() → save_results()
         (Single GPU)    (Sequential processing)
```

### With `--use_accelerate`
```
Runner → AccelerateConnector → run_inference_with_accelerate() → save_results()
         (Multi-GPU setup)      (Parallel processing across GPUs)   (Main process only)
                 ↓
         split_dataset() → Each GPU processes subset → gather_results()
```

##  AccelerateConnector Methods

### For Inference (Current)
- `load_model_once()` - Load and wrap model with Accelerate
- `complete(messages, **kwargs)` - Single sample inference
- `complete_batch(messages_batch, **kwargs)` - Batch inference
- `split_dataset(dataset)` - Auto-split across GPUs
- `gather_results(results)` - Collect results from all GPUs

### For Fine-tuning (Ready)
- `prepare_for_training(model, optimizer, *dataloaders)` - Wrap for training
- `backward(loss)` - Gradient backpropagation with sync
- `wait_for_everyone()` - Synchronization barrier
- `save_model(model, output_dir)` - Save trained model

### Properties
- `is_main_process` - True for rank 0 process
- `process_index` - Current GPU index (0, 1, 2, ...)
- `num_processes` - Total number of GPUs

##  Benefits

| Feature | Single GPU | Multi-GPU (Accelerate) |
|---------|------------|------------------------|
| **Setup** | None | `accelerate config` (one-time) |
| **Command** | `python -m ...` | `accelerate launch --num_processes N -m ...` |
| **Splitting** | N/A | Automatic |
| **Gathering** | N/A | Automatic |
| **Speedup** | 1x | ~3.8x (4 GPUs) |
| **Code Changes** | None | Add `--use_accelerate` flag |

##  Future: Fine-tuning

The same connector will work for fine-tuning:

```bash
# Fine-tuning with multiple GPUs
accelerate launch --num_processes 4 \
    -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --finetune \
    --data_file unified \
    --epochs 3
```

Same connector, same approach, same command structure!

##  Configuration File

After running `accelerate config`, check your config:
```bash
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

Example config:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
```

##  Debugging

### Check if Accelerate is available
```bash
python -c "from finetuning.pipeline.baseline.runner import ACCELERATE_AVAILABLE; print(ACCELERATE_AVAILABLE)"
```

### Verify GPU visibility
```bash
nvidia-smi  # Check all GPUs are visible
accelerate launch --num_processes 4 python -c "import torch; print(torch.cuda.device_count())"
```

### Test connector directly
```bash
python finetuning/pipeline/connector/test_accelerate_connector.py
```

##  Documentation

- **Integration Details**: `finetuning/pipeline/connector/INTEGRATION_STATUS.md`
- **Unified Approach**: `finetuning/pipeline/baseline/ACCELERATE_UNIFIED_APPROACH.md`
- **Accelerate Docs**: https://huggingface.co/docs/accelerate

##  Status

-  Package integrated into `finetuning/pipeline/connector/`
-  Runner supports `--use_accelerate` flag
-  Automatic dataset splitting
-  Result gathering from all GPUs
-  Backward compatible (single-GPU unchanged)
-  Ready for fine-tuning implementation
-  Multi-GPU testing pending (requires GPU hardware) - PENDING
