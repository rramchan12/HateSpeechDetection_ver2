"""
Model loading utilities for baseline validation
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional


def load_model(
    model_name: str = "openai/gpt-oss-20b",
    use_auth_token: Optional[str] = None
) -> Tuple:
    """
    Load GPT-OSS model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        use_auth_token: HuggingFace token (reads from HF_TOKEN env if None)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        OSError: If model cannot be loaded
        
    Environment Variables:
        HF_TOKEN: HuggingFace API token for private models
    """
    # Get token from environment if not provided
    if use_auth_token is None:
        use_auth_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if use_auth_token:
            print(f"[OK] Using HuggingFace token from environment")
    
    print(f"Loading model: {model_name}")
    print("This may take 5-10 minutes...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=use_auth_token
        )
        
        # Auto-detect best dtype (bfloat16 if available, else float16)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print(f"  Using dtype: bfloat16 (GPU supports it)")
        else:
            dtype = torch.float16
            print(f"  Using dtype: float16")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            token=use_auth_token
        )
        
        print(f"[OK] Model loaded successfully!")
        print(f"  Parameters: {model.num_parameters() / 1e9:.1f}B")
        print(f"  Memory: {model.get_memory_footprint() / 1e9:.2f} GB")
        
        return model, tokenizer
        
    except OSError as e:
        error_msg = str(e)
        print(f"[FAILED] Failed to load model: {error_msg}")
        
        # Provide helpful suggestions
        if "is not a local folder and is not a valid model identifier" in error_msg:
            print(f"\n[HELP] The model '{model_name}' was not found on HuggingFace Hub.")
            print(f"  Suggestions:")
            print(f"  1. Check if the model name is correct")
            print(f"  2. Verify model exists: python -m finetuning.model_download.hf_model_downloader --verify {model_name}")
            print(f"  3. See suggested models: python -m finetuning.model_download.hf_model_downloader --suggest")
            print(f"\n  Popular alternatives:")
            print(f"    --model_name microsoft/phi-2")
            print(f"    --model_name microsoft/Phi-3-mini-4k-instruct")
            print(f"    --model_name google/flan-t5-large")
        elif "401" in error_msg or "403" in error_msg or "token" in error_msg.lower():
            print(f"\n[HELP] This appears to be a private model requiring authentication.")
            print(f"  Set your HuggingFace token:")
            print(f"    export HF_TOKEN=hf_xxxxxxxxxxxxx")
            print(f"  Or login with: huggingface-cli login")
        
        raise
