"""
Accelerate Connector for unified inference and fine-tuning.

This connector uses HuggingFace Accelerate to provide automatic multi-GPU support
for both inference and fine-tuning operations. It wraps the model loading from
model_loader and provides a unified interface compatible with the existing runner.
"""

import torch
from accelerate import Accelerator
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add project root for imports
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from finetuning.pipeline.baseline.model_loader import load_model


class AccelerateConnector:
    """
    Unified connector for inference and fine-tuning using Accelerate.
    
    This connector provides automatic multi-GPU support using HuggingFace Accelerate.
    It maintains compatibility with the existing LocalModelConnector interface while
    adding multi-GPU capabilities for both inference and fine-tuning.
    
    Key Features:
    - Automatic multi-GPU distribution
    - Unified interface for inference and training
    - Compatible with existing runner code
    - Supports batch processing
    - Handles gradient synchronization for training
    
    Attributes:
        model_name: HuggingFace model identifier
        cache_dir: Optional cache directory for models
        batch_size: Batch size for inference
        accelerator: Accelerate instance for multi-GPU coordination
        model: Loaded model (wrapped by Accelerate)
        tokenizer: Model tokenizer
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 1,
        mixed_precision: str = 'bf16',
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize Accelerate connector.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'openai/gpt-oss-20b')
            cache_dir: Optional cache directory for model files
            batch_size: Batch size for inference (default: 1)
            mixed_precision: Precision mode ('no', 'fp16', 'bf16') (default: 'bf16')
            gradient_accumulation_steps: Steps for gradient accumulation (default: 1)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        
        # Print info only on main process
        if self.accelerator.is_main_process:
            num_gpus = self.accelerator.num_processes
            print(f"\n{'='*60}")
            print(f"Accelerate Connector Initialized")
            print(f"{'='*60}")
            print(f"Number of GPUs: {num_gpus}")
            print(f"Mixed Precision: {mixed_precision}")
            print(f"Gradient Accumulation: {gradient_accumulation_steps} steps")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*60}\n")
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.accelerator.is_main_process
    
    @property
    def process_index(self) -> int:
        """Get the current process index."""
        return self.accelerator.process_index
    
    @property
    def num_processes(self) -> int:
        """Get the total number of processes."""
        return self.accelerator.num_processes
    
    def load_model_once(self):
        """
        Load model and tokenizer if not already loaded.
        
        Uses the model_loader to load the model, then wraps it with Accelerate
        for multi-GPU distribution. Only loads once per connector instance.
        """
        if self.model is None:
            # Load model using existing model_loader
            model, tokenizer = load_model(self.model_name, cache_dir=self.cache_dir)
            
            # Move model to accelerator's device (handles multi-GPU automatically)
            self.model = self.accelerator.prepare(model)
            self.tokenizer = tokenizer
            
            if self.accelerator.is_main_process:
                print(f"[OK] Model prepared on {self.accelerator.num_processes} GPU(s)")
    
    def complete(self, messages: List[Dict], **kwargs) -> Any:
        """
        Single sample completion (maintains compatibility with existing code).
        
        Processes a single inference request. Compatible with the existing
        LocalModelConnector interface.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Generation parameters (max_tokens, temperature, top_p)
            
        Returns:
            Response object with choices attribute containing the model output
        """
        self.load_model_once()
        
        # Format prompt using tokenizer's chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.95),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Return in compatible format
        class Choice:
            def __init__(self, text):
                self.message = type('Message', (), {'content': text})()
        
        class Response:
            def __init__(self, text):
                self.choices = [Choice(text)]
        
        return Response(response_text)
    
    def complete_batch(self, messages_batch: List[List[Dict]], **kwargs) -> List[Any]:
        """
        Batch completion for improved throughput on GPU.
        
        Processes multiple samples in parallel on the GPU for better performance.
        Uses dynamic padding to handle variable-length inputs efficiently.
        
        Args:
            messages_batch: List of message lists, one per sample
            **kwargs: Generation parameters
            
        Returns:
            List of Response objects, one per input
        """
        self.load_model_once()
        
        # Convert all messages to prompt texts
        prompt_texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        
        # Tokenize with padding for batch processing
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True  # Dynamic padding for batch
        )
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        
        # Generate for entire batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.95),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode each response
        responses = []
        for i, output in enumerate(outputs):
            # Skip the input tokens
            response_text = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Return in compatible format
            class Choice:
                def __init__(self, text):
                    self.message = type('Message', (), {'content': text})()
            
            class Response:
                def __init__(self, text):
                    self.choices = [Choice(text)]
            
            responses.append(Response(response_text))
        
        return responses
    
    def split_dataset(self, dataset: List[Any]) -> List[Any]:
        """
        Split dataset across processes automatically.
        
        Uses Accelerate's split_between_processes to automatically distribute
        the dataset across available GPUs. Each process gets a subset.
        
        Args:
            dataset: Full dataset to split
            
        Returns:
            Subset of dataset for this process
        """
        # Accelerate handles splitting automatically
        with self.accelerator.split_between_processes(dataset) as process_data:
            return list(process_data)
    
    def gather_results(self, results: List[Any]) -> List[Any]:
        """
        Gather results from all processes.
        
        Collects results from all GPUs and returns the combined list.
        Only the main process receives all results; other processes get None.
        
        Args:
            results: Results from this process
            
        Returns:
            All results combined (main process only) or None (other processes)
        """
        # Gather results from all processes
        all_results = self.accelerator.gather_for_metrics(results)
        
        # Only main process has all results
        if self.accelerator.is_main_process:
            return all_results
        else:
            return None
    
    def prepare_for_training(self, model, optimizer, *dataloaders):
        """
        Prepare components for distributed training.
        
        Wraps model, optimizer, and dataloaders with Accelerate for
        multi-GPU training with automatic gradient synchronization.
        
        Args:
            model: Model to train
            optimizer: Optimizer instance
            *dataloaders: DataLoader instances
            
        Returns:
            Tuple of prepared (model, optimizer, *dataloaders)
        """
        return self.accelerator.prepare(model, optimizer, *dataloaders)
    
    def backward(self, loss):
        """
        Backward pass with automatic gradient synchronization.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        self.accelerator.backward(loss)
    
    def wait_for_everyone(self):
        """Wait for all processes to reach this point."""
        self.accelerator.wait_for_everyone()
    
    def save_model(self, model, output_dir: str):
        """
        Save model (only main process saves).
        
        Args:
            model: Model to save
            output_dir: Directory to save to
        """
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
            print(f"[OK] Model saved to {output_dir}")
