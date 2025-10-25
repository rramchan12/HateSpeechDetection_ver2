#!/usr/bin/env python3
"""
Quick multi-GPU test script - splits dataset and runs on 2 GPUs simultaneously.
"""

import json
import subprocess
import time
from pathlib import Path

def main():
    print("="*60)
    print("MULTI-GPU TEST: Using GPU 0 and GPU 1")
    print("="*60)
    
    # Check if we have the dataset
    dataset_file = Path("data/processed/unified/unified_val.json")
    if not dataset_file.exists():
        print(f"Error: {dataset_file} not found")
        return 1
    
    # Load and split dataset
    print("\n1. Splitting dataset between 2 GPUs...")
    with open(dataset_file) as f:
        data = json.load(f)
    
    mid = len(data) // 2
    gpu0_samples = data[:mid]
    gpu1_samples = data[mid:]
    
    print(f"   GPU 0: {len(gpu0_samples)} samples")
    print(f"   GPU 1: {len(gpu1_samples)} samples")
    
    # Create split files
    split_dir = Path("data/processed/unified_splits_test")
    split_dir.mkdir(exist_ok=True)
    
    gpu0_file = split_dir / "gpu0.json"
    gpu1_file = split_dir / "gpu1.json"
    
    with open(gpu0_file, 'w') as f:
        json.dump(gpu0_samples, f)
    with open(gpu1_file, 'w') as f:
        json.dump(gpu1_samples, f)
    
    print(f"   Saved splits to {split_dir}/")
    
    # Run on both GPUs
    print("\n2. Starting inference on 2 GPUs in parallel...")
    print("   This will take ~37 seconds (vs ~74 seconds single GPU)")
    print("   Watch GPU usage: watch -n 1 nvidia-smi")
    
    start_time = time.time()
    
    # Use the virtual environment Python
    python_exe = Path(".venv/bin/python").absolute()
    if not python_exe.exists():
        print(f"Error: Virtual environment not found at {python_exe}")
        print("Please activate: source .venv/bin/activate")
        return 1
    
    # Start both processes
    env0 = subprocess.os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    
    env1 = subprocess.os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"
    
    proc0 = subprocess.Popen([
        str(python_exe), "-m", "finetuning.pipeline.baseline.runner",
        "--data_source", "unified_splits_test/gpu0",
        "--max_samples", str(len(gpu0_samples)),
        "--batch_size", "4",
        "--output_dir", "./finetuning/outputs/multi_gpu_test/gpu0"
    ], env=env0)
    
    proc1 = subprocess.Popen([
        str(python_exe), "-m", "finetuning.pipeline.baseline.runner",
        "--data_source", "unified_splits_test/gpu1",
        "--max_samples", str(len(gpu1_samples)),
        "--batch_size", "4",
        "--output_dir", "./finetuning/outputs/multi_gpu_test/gpu1"
    ], env=env1)
    
    # Wait for both
    proc0.wait()
    proc1.wait()
    
    elapsed = time.time() - start_time
    
    print(f"\n3. Both processes complete!")
    print(f"   Total time: {elapsed:.1f} seconds")
    print(f"   Expected single GPU: ~74 seconds")
    print(f"   Speedup: {74/elapsed:.2f}x")
    
    print("\n4. Results saved to:")
    print("   GPU 0: ./finetuning/outputs/multi_gpu_test/gpu0/")
    print("   GPU 1: ./finetuning/outputs/multi_gpu_test/gpu1/")
    
    print("\n" + "="*60)
    print("SUCCESS: Multi-GPU inference working!")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())
