# Accelerate Connector Integration Status

## ✅ COMPLETED

### 1. Package Location
- **Location**: `finetuning/pipeline/connector/`
- **Files**:
  - `__init__.py`
  - `accelerate_connector.py` (main connector class)
  - `test_accelerate_connector.py` (unit tests)

### 2. Runner Integration
- **File**: `finetuning/pipeline/baseline/runner.py`
- **Changes**:
  - Added optional import of `AccelerateConnector`
  - Added `run_inference_with_accelerate()` function
  - Modified `main()` to support `--use_accelerate` flag
  - Updated documentation with multi-GPU examples

### 3. Command-Line Interface
- **New Flag**: `--use_accelerate`
- **Usage**: `accelerate launch --num_processes N -m finetuning.pipeline.baseline.runner --use_accelerate`

---

## 🎯 How It Works

### Single-GPU Mode (Default)
```bash
python -m finetuning.pipeline.baseline.runner \
    --data_file canned_50_quick \
    --max_samples 5
```
- Uses standard `load_model()` from model_loader
- Runs `run_inference_with_prompt()`
- **No changes to existing workflow**

### Multi-GPU Mode (with Accelerate)
```bash
accelerate launch --num_processes 4 -m finetuning.pipeline.baseline.runner \
    --use_accelerate \
    --data_file unified \
    --max_samples 100
```
- Initializes `AccelerateConnector`
- Automatically splits dataset across GPUs
- Each GPU processes its subset
- Results gathered and merged
- Only main process saves results

---

## 📊 Code Flow

### Without Accelerate (Default Path)
```
main() 
  → load_model()
  → load_validation_data()
  → run_inference_with_prompt(model, tokenizer, dataset)
  → save_results()
```

### With Accelerate (New Path)
```
main() 
  → AccelerateConnector(model_name)
  → connector.load_model_once()  # Uses model_loader internally
  → load_validation_data()
  → run_inference_with_accelerate(connector, dataset)
      → connector.split_dataset()  # Auto-split across GPUs
      → connector.complete() for each sample
      → connector.gather_results()  # Merge from all GPUs
  → save_results() (main process only)
```

---

## 🔧 Key Features

### AccelerateConnector Class
```python
class AccelerateConnector:
    def __init__(model_name, cache_dir, batch_size=1, mixed_precision='bf16')
    def load_model_once()  # Load model with Accelerate wrapper
    def complete(messages, **kwargs)  # Single inference
    def complete_batch(messages_batch, **kwargs)  # Batch inference
    def split_dataset(dataset)  # Auto-split across GPUs
    def gather_results(results)  # Gather from all GPUs
    def prepare_for_training(model, optimizer, *dataloaders)  # For future fine-tuning
    def backward(loss)  # For future fine-tuning
```

### Integration Points
1. **Model Loading**: Uses existing `model_loader.load_model()`
2. **Dataset Loading**: Uses existing `load_validation_data()`
3. **Metrics**: Uses existing `save_results()` and metrics classes
4. **Logging**: Integrated with existing logging system

---

## ✅ Benefits

1. **Unified Approach**: Same connector for inference AND future fine-tuning
2. **Automatic Distribution**: No manual dataset splitting required
3. **Scalable**: Works with 1, 2, 4, or N GPUs without code changes
4. **Optional**: Existing single-GPU workflow unchanged
5. **Future-Proof**: Ready for distributed training across multiple nodes

---

## 🚀 Next Steps for Fine-tuning

The same `AccelerateConnector` can be used for fine-tuning:

```python
# In future fine-tuning code:
connector = AccelerateConnector(model_name, ...)
connector.load_model_once()

# Prepare for training
model, optimizer, train_loader = connector.prepare_for_training(
    model, optimizer, train_loader
)

# Training loop
for batch in train_loader:
    loss = model(**batch)
    connector.backward(loss)  # Automatic gradient sync
    optimizer.step()
    optimizer.zero_grad()

# Save model
connector.save_model(model, output_dir)
```

---

## 📝 Configuration

### First-time Setup
```bash
pip install accelerate
accelerate config
```

Interactive prompts:
- This machine
- No distributed training (multi-GPU on single machine)
- Number of GPUs: 4 (or your GPU count)
- Mixed precision: bf16
- Accept defaults for other options

This creates: `~/.cache/huggingface/accelerate/default_config.yaml`

---

## ✅ Testing Status

- ✅ Package moved to correct location
- ✅ Imports updated and working
- ✅ Runner integration complete
- ✅ Command-line flag added
- ✅ Documentation updated
- ⏳ Multi-GPU testing pending (requires GPU setup)
- ⏳ Fine-tuning implementation pending

---

## 📚 References

- **Accelerate Docs**: https://huggingface.co/docs/accelerate
- **Multi-GPU Inference**: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- **Unified Approach**: `finetuning/pipeline/baseline/ACCELERATE_UNIFIED_APPROACH.md`
