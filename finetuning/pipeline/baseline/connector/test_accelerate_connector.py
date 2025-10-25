#!/usr/bin/env python3
"""
Test script for AccelerateConnector - validates multi-GPU inference works.

Usage:
    # Single GPU (automatic)
    python test_accelerate_connector.py
    
    # Multi-GPU with accelerate launch
    accelerate launch --num_processes 4 test_accelerate_connector.py
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from finetuning.connector import AccelerateConnector

def main():
    print("\n" + "="*60)
    print("Testing AccelerateConnector")
    print("="*60)
    
    # Initialize connector
    connector = AccelerateConnector(
        model_name="openai/gpt-oss-20b",
        batch_size=2,
        mixed_precision='bf16'
    )
    
    # Load model
    print("\nLoading model...")
    connector.load_model_once()
    
    if connector.is_main_process:
        print(f"[OK] Model loaded on {connector.num_processes} GPU(s)\n")
    
    # Test single completion
    if connector.is_main_process:
        print("Testing single completion...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."}
    ]
    
    response = connector.complete(messages, max_tokens=10)
    
    if connector.is_main_process:
        print(f"Response: {response.choices[0].message.content}")
        print("[OK] Single completion works!\n")
    
    # Test batch completion
    if connector.is_main_process:
        print("Testing batch completion...")
    
    messages_batch = [
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say 'test1'."}
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say 'test2'."}
        ]
    ]
    
    responses = connector.complete_batch(messages_batch, max_tokens=10)
    
    if connector.is_main_process:
        for i, resp in enumerate(responses):
            print(f"Response {i+1}: {resp.choices[0].message.content}")
        print("[OK] Batch completion works!\n")
    
    if connector.is_main_process:
        print("="*60)
        print("[SUCCESS] AccelerateConnector test passed!")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
