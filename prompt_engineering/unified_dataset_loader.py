"""
Unified Dataset Loader for Prompt Strategy Validation

This module provides a unified interface for loading test datasets from either
canned samples or the unified test dataset. It supports filtering by number of
samples and provides a consistent data structure for both data sources.

Classes:
    DatasetType: Enum for dataset type selection
    UnifiedDatasetLoader: Main loader class with filtering capabilities

Functions:
    load_dataset(): Factory function for loading datasets
"""

import json
import logging
import random
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class DatasetType(Enum):
    """Enumeration for dataset type selection"""
    CANNED = "canned"
    UNIFIED = "unified"


class UnifiedDatasetLoader:
    """
    Unified dataset loader for prompt strategy validation.
    
    This class provides a consistent interface for loading test samples from
    either canned samples (small curated set) or unified test dataset (large
    comprehensive set). Supports filtering by sample count and maintains
    consistent data structure across both sources.
    """
    
    def __init__(self):
        """Initialize the dataset loader with logging"""
        self.logger = logging.getLogger(__name__)
        
        # Define dataset paths
        self.base_path = Path(__file__).parent
        self.canned_path = self.base_path / "data_samples" / "canned_basic_all.json"
        self.unified_path = self.base_path.parent / "data" / "processed" / "unified" / "unified_test.json"
        
        # Cache for loaded datasets
        self._canned_cache = None
        self._unified_cache = None
    
    def _load_canned_samples(self) -> List[Dict[str, Any]]:
        """
        Load canned samples from JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of canned sample dictionaries
            
        Raises:
            FileNotFoundError: If canned samples file is not found
            json.JSONDecodeError: If JSON file is malformed
        """
        if self._canned_cache is not None:
            return self._canned_cache
            
        if not self.canned_path.exists():
            raise FileNotFoundError(f"Canned samples file not found: {self.canned_path}")
        
        try:
            with open(self.canned_path, 'r', encoding='utf-8') as f:
                # Handle both formats: {"samples": [...]} and [...]
                data = json.load(f)
                if isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                elif isinstance(data, list):
                    samples = data
                else:
                    raise ValueError("Invalid canned samples format")
                
                self._canned_cache = samples
                self.logger.info(f"Loaded {len(samples)} canned samples")
                return samples
                
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in canned samples file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading canned samples: {e}")
    
    def _load_unified_samples(self) -> List[Dict[str, Any]]:
        """
        Load unified test samples from JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of unified test sample dictionaries
            
        Raises:
            FileNotFoundError: If unified test file is not found
            json.JSONDecodeError: If JSON file is malformed
        """
        if self._unified_cache is not None:
            return self._unified_cache
            
        if not self.unified_path.exists():
            raise FileNotFoundError(f"Unified test file not found: {self.unified_path}")
        
        try:
            with open(self.unified_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                
                if not isinstance(samples, list):
                    raise ValueError("Invalid unified test format: expected list")
                
                self._unified_cache = samples
                self.logger.info(f"Loaded {len(samples)} unified test samples")
                return samples
                
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in unified test file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading unified test samples: {e}")
    
    def load_samples(self, dataset_type: Union[str, DatasetType], 
                    num_samples: Union[int, str] = "all", 
                    random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load samples from the specified dataset with optional filtering.
        
        Args:
            dataset_type (Union[str, DatasetType]): Type of dataset to load ("canned" or "unified")
            num_samples (Union[int, str]): Number of samples to return ("all" or integer)
            random_seed (Optional[int]): Random seed for reproducible sampling
            
        Returns:
            List[Dict[str, Any]]: List of sample dictionaries
            
        Raises:
            ValueError: If dataset_type or num_samples is invalid
            FileNotFoundError: If dataset file is not found
        """
        # Normalize dataset type
        if isinstance(dataset_type, str):
            try:
                dataset_type = DatasetType(dataset_type.lower())
            except ValueError:
                raise ValueError(f"Invalid dataset type: {dataset_type}. Must be 'canned' or 'unified'")
        
        # Load the appropriate dataset
        if dataset_type == DatasetType.CANNED:
            samples = self._load_canned_samples()
        elif dataset_type == DatasetType.UNIFIED:
            samples = self._load_unified_samples()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Handle sample filtering
        if isinstance(num_samples, str):
            if num_samples.lower() == "all":
                return samples
            else:
                raise ValueError(f"Invalid num_samples string: {num_samples}. Must be 'all' or an integer")
        
        if isinstance(num_samples, int):
            if num_samples <= 0:
                raise ValueError("num_samples must be positive")
            
            if num_samples >= len(samples):
                self.logger.warning(f"Requested {num_samples} samples but only {len(samples)} available")
                return samples
            
            # Set random seed for reproducible results
            if random_seed is not None:
                random.seed(random_seed)
            
            # Return random sample
            return random.sample(samples, num_samples)
        
        raise ValueError(f"Invalid num_samples type: {type(num_samples)}. Must be int or 'all'")
    
    def get_dataset_info(self, dataset_type: Union[str, DatasetType]) -> Dict[str, Any]:
        """
        Get information about the specified dataset.
        
        Args:
            dataset_type (Union[str, DatasetType]): Type of dataset
            
        Returns:
            Dict[str, Any]: Dataset information including size, path, etc.
        """
        # Normalize dataset type
        if isinstance(dataset_type, str):
            dataset_type = DatasetType(dataset_type.lower())
        
        if dataset_type == DatasetType.CANNED:
            samples = self._load_canned_samples()
            return {
                "type": "canned",
                "path": str(self.canned_path),
                "total_samples": len(samples),
                "description": "Small curated set of test samples"
            }
        elif dataset_type == DatasetType.UNIFIED:
            samples = self._load_unified_samples()
            return {
                "type": "unified",
                "path": str(self.unified_path),
                "total_samples": len(samples),
                "description": "Large comprehensive test dataset"
            }
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def validate_sample_structure(self, samples: List[Dict[str, Any]]) -> bool:
        """
        Validate that samples have the expected structure.
        
        Args:
            samples (List[Dict[str, Any]]): List of samples to validate
            
        Returns:
            bool: True if all samples have valid structure
            
        Raises:
            ValueError: If sample structure is invalid
        """
        required_fields = [
            "text", "label_binary", "label_multiclass", 
            "target_group_norm", "persona_tag", "source_dataset"
        ]
        
        if not samples:
            raise ValueError("Sample list is empty")
        
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {i} is not a dictionary")
            
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Sample {i} missing required field: {field}")
                
            # Validate label_binary values
            if sample.get("label_binary") not in ["hate", "normal"]:
                raise ValueError(f"Sample {i} has invalid label_binary: {sample.get('label_binary')}")
        
        return True
    
    def clear_cache(self):
        """Clear cached datasets to force reload on next access"""
        self._canned_cache = None
        self._unified_cache = None
        self.logger.info("Dataset cache cleared")


def load_dataset(dataset_type: Union[str, DatasetType], 
                num_samples: Union[int, str] = "all",
                random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Factory function to load dataset samples.
    
    Args:
        dataset_type (Union[str, DatasetType]): Type of dataset ("canned" or "unified")
        num_samples (Union[int, str]): Number of samples ("all" or integer)
        random_seed (Optional[int]): Random seed for reproducible sampling
        
    Returns:
        List[Dict[str, Any]]: List of sample dictionaries
        
    Example:
        >>> samples = load_dataset("canned", 5)
        >>> samples = load_dataset("unified", "all")
        >>> samples = load_dataset("unified", 100, random_seed=42)
    """
    loader = UnifiedDatasetLoader()
    return loader.load_samples(dataset_type, num_samples, random_seed)


def get_dataset_info(dataset_type: Union[str, DatasetType]) -> Dict[str, Any]:
    """
    Factory function to get dataset information.
    
    Args:
        dataset_type (Union[str, DatasetType]): Type of dataset
        
    Returns:
        Dict[str, Any]: Dataset information
    """
    loader = UnifiedDatasetLoader()
    return loader.get_dataset_info(dataset_type)