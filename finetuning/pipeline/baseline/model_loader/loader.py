"""
Model loading utilities for baseline validation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


def load_model(model_name: str = "gpt-oss-20b") -> Tuple:
    """
    Load GPT-OSS model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        OSError: If model cannot be loaded
    """
    print(f"Loading model: {model_name}")
    print("This may take 5-10 minutes...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"[OK] Model loaded successfully!")
        print(f"  Parameters: {model.num_parameters() / 1e9:.1f}B")
        print(f"  Memory: {model.get_memory_footprint() / 1e9:.2f} GB")
        
        return model, tokenizer
        
    except OSError as e:
        print(f"[FAILED] Failed to load model: {e}")
        raise
