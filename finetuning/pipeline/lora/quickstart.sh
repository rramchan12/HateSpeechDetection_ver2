#!/bin/bash
# Quick Start Script for LoRA Fine-Tuning
# Usage: bash finetuning/pipeline/lora/quickstart.sh

set -e  # Exit on error

echo "============================================================"
echo "LoRA Fine-Tuning Quick Start"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check if Accelerate is configured
if ! accelerate env &>/dev/null; then
    echo "⚠️  Accelerate not configured"
    echo "Run: accelerate config"
    exit 1
fi

echo "✓ Accelerate configured"
echo ""

# Check if training data exists
TRAIN_FILE="finetuning/data/ft_prompts/train.jsonl"
VAL_FILE="finetuning/data/ft_prompts/validation.jsonl"

if [[ ! -f "$TRAIN_FILE" ]] || [[ ! -f "$VAL_FILE" ]]; then
    echo "⚠️  Training data not found"
    echo ""
    echo "Generate data with:"
    echo "  python -m finetuning.ft_prompt_generator.cli \\"
    echo "      --unified_dir ./data/processed/unified \\"
    echo "      --output_dir ./finetuning/data/ft_prompts \\"
    echo "      --template combined/combined_gptoss_v1.json \\"
    echo "      --strategy combined_optimized"
    exit 1
fi

echo "✓ Training data found"
echo "  Train: $TRAIN_FILE"
echo "  Val:   $VAL_FILE"
echo ""

# Check number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✓ Detected $NUM_GPUS GPU(s)"
echo ""

# Ask for confirmation
echo "============================================================"
echo "Ready to start LoRA fine-tuning"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: openai/gpt-oss-20b"
echo "  GPUs: $NUM_GPUS"
echo "  Epochs: 3"
echo "  Learning rate: 2e-4"
echo "  LoRA rank: 32"
echo "  Expected duration: 2-3 hours"
echo ""
read -p "Start training? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Launch training
echo ""
echo "============================================================"
echo "Launching LoRA fine-tuning..."
echo "============================================================"
echo ""

accelerate launch --num_processes "$NUM_GPUS" \
    -m finetuning.pipeline.lora.train \
    --config_file ./finetuning/pipeline/lora/configs/default.json

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. View results: ls -lh finetuning/models/lora_checkpoints/"
echo "  2. Validate model:"
echo "     accelerate launch --num_processes $NUM_GPUS \\"
echo "         -m finetuning.pipeline.baseline.runner \\"
echo "         --use_accelerate \\"
echo "         --model_name ./finetuning/models/lora_checkpoints \\"
echo "         --data_file unified \\"
echo "         --max_samples 100"
echo ""
echo "  3. Merge adapters (optional):"
echo "     python -m finetuning.pipeline.lora.merge \\"
echo "         --base_model openai/gpt-oss-20b \\"
echo "         --adapter_path ./finetuning/models/lora_checkpoints \\"
echo "         --output_dir ./finetuning/models/merged_model"
echo ""
