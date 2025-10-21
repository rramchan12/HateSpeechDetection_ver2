"""
Baseline inference execution

Reuses data loading patterns from prompt_engineering when applicable.
"""

import json
import torch
import time
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def load_validation_data(data_file: str, max_samples: int = None) -> List[Dict]:
    """
    Load validation data from JSONL file.
    
    Args:
        data_file: Path to JSONL file
        max_samples: Maximum samples to load (None for all)
        
    Returns:
        List of data samples
    """
    print(f"\nLoading data from: {data_file}")
    
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Total samples: {len(data)}")
    return data


def run_inference(
    model,
    tokenizer,
    data_file: str,
    output_file: str,
    max_samples: int = None,
    max_length: int = 512,
    max_new_tokens: int = 10,
    temperature: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Run baseline inference on validation data
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded tokenizer
        data_file: Path to JSONL data file
        output_file: Path to save results
        max_samples: Maximum samples to process (None for all)
        max_length: Max input token length
        max_new_tokens: Max generated tokens
        temperature: Sampling temperature
        
    Returns:
        List of inference results
    """
    # Load data using helper function
    data = load_validation_data(data_file, max_samples)
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(tqdm(data, desc="Inference")):
        try:
            # Extract prompt (remove label from text if present)
            prompt = item['text'].split('### Classification:')[0] + '### Classification:'
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract prediction
            if '### Classification:' in response:
                prediction = response.split('### Classification:')[-1].strip().lower()
            else:
                prediction = response.strip().lower()
            
            # Normalize prediction
            if 'hate' in prediction and 'not' not in prediction:
                pred_label = 'hate'
            elif 'not' in prediction or 'normal' in prediction:
                pred_label = 'normal'
            else:
                pred_label = 'unknown'
            
            # Store result
            results.append({
                'sample_id': i,
                'prompt': prompt,
                'true_label': item.get('label', 'unknown'),
                'prediction': pred_label,
                'raw_response': response
            })
            
            # Progress update every 50 samples
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(data) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{len(data)} | {rate:.1f} samples/sec | ETA: {remaining/60:.1f} min")
        
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            results.append({
                'sample_id': i,
                'prompt': item.get('text', '')[:100],
                'true_label': item.get('label', 'unknown'),
                'prediction': 'error',
                'error': str(e)
            })
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"[OK] Inference complete! Total time: {total_time/60:.1f} minutes")
    print(f"  Average: {len(data)/total_time:.2f} samples/second")
    
    return results
