#!/usr/bin/env python3
"""
ToxiGen Dataset Preparation and Cleansing Operations

This module provides comprehensive data preparation operations for the ToxiGen dataset,
including data loading, cleaning, preprocessing, feature extraction, and export functionality.

Author: Data Preparation Pipeline
Date: September 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import re
import sys
from dataclasses import dataclass
import logging

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not available. Some features will be limited.")
    HAS_PANDAS = False

try:
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not available. Label encoding will be basic.")
    HAS_SKLEARN = False
    
    # Simple LabelEncoder replacement
    class LabelEncoder:
        def __init__(self):
            self.classes_ = None
            self._label_to_idx = {}
            
        def fit(self, y):
            self.classes_ = np.unique(y)
            self._label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
            return self
            
        def transform(self, y):
            return [self._label_to_idx[label] for label in y]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Data class to hold dataset statistics for train.parquet fields only"""
    total_entries: int
    text_length_mean: float
    text_length_median: float
    text_length_std: float
    label_distribution: Dict[str, int]  # Based on prompt_label: toxic/benign
    group_distribution: Dict[str, int]  # Target group distribution
    generation_method_distribution: Dict[str, int]  # Generation method distribution
    avg_roberta_prediction: float
    prompt_label_distribution: Dict[str, int]  # Raw prompt_label counts
    missing_values: int
    duplicate_count: int


class ToxiGenDataProcessor:
    """
    Main class for ToxiGen dataset preparation and cleansing operations.
    
    This class handles:
    - Data loading and validation
    - Data cleansing and preprocessing
    - Feature extraction and transformation
    - Statistical analysis and reporting
    - Data export and splitting
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the ToxiGen data processor.
        
        Args:
            data_dir: Path to the data directory. If None, uses default project structure.
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[1]
            self.data_dir = self.project_root / "data" / "toxigen"
        else:
            self.data_dir = Path(data_dir)
        
        self.train_data = None
        self.annotated_data = None
        self.annotations_data = None
        self.processed_data = None
        self.label_encoders = {}
        
        logger.info(f"Initialized ToxiGen processor with data directory: {self.data_dir}")
    
    def load_raw_data(self) -> Dict[str, Any]:
        """
        Load raw ToxiGen dataset from train.parquet file only.
        
        Returns:
            Dictionary containing loaded data and metadata
        """
        logger.info("Loading raw ToxiGen dataset from train.parquet...")
        
        # Load training data from Parquet only (most efficient)
        train_parquet = self.data_dir / "train.parquet"
        
        if not train_parquet.exists():
            raise FileNotFoundError(f"Required file train.parquet not found in {self.data_dir}")
        
        if not HAS_PANDAS:
            raise ImportError("pandas is required for Parquet file processing")
        
        try:
            self.train_data = pd.read_parquet(train_parquet)
            logger.info(f"Loaded training data from Parquet: {len(self.train_data)} entries")
        except Exception as e:
            raise RuntimeError(f"Failed to load train.parquet: {e}")
            
        # Create annotated dataset from training data with proper field mapping
        logger.info("Creating annotated dataset from training data for analysis...")
        self.annotated_data = self.train_data.copy()
        
        # Map train.parquet fields to expected fields for processing
        # Use 'generation' as 'text' for consistency with other processing
        if 'generation' in self.annotated_data.columns:
            self.annotated_data['text'] = self.annotated_data['generation']
        
        # Map 'group' to 'target_group' for consistency
        if 'group' in self.annotated_data.columns:
            self.annotated_data['target_group'] = self.annotated_data['group']
        
        # Use prompt_label directly for labeling (1=toxic, 0=benign)
        if 'prompt_label' in self.annotated_data.columns:
            self.annotated_data['label_binary'] = self.annotated_data['prompt_label'].apply(
                lambda x: 'toxic' if x == 1 else 'benign'
            )
        else:
            logger.warning("prompt_label field not found, setting default labels")
            self.annotated_data['label_binary'] = 'benign'
        
        # Use roberta_prediction as toxicity score
        if 'roberta_prediction' in self.annotated_data.columns:
            self.annotated_data['toxicity_score'] = self.annotated_data['roberta_prediction']
        else:
            self.annotated_data['toxicity_score'] = 0.0
        
        # Map generation_method to actual_method for consistency
        if 'generation_method' in self.annotated_data.columns:
            self.annotated_data['actual_method'] = self.annotated_data['generation_method']
        
        logger.info(f"Created annotated dataset: {len(self.annotated_data)} entries")
        
        # No separate annotations data - using train.parquet only
        self.annotations_data = None
        
        # Setup label encoders
        self._setup_label_encoders()
        
        return {
            'train_data': self.train_data,
            'annotated_data': self.annotated_data,
            'annotations_data': self.annotations_data,
            'label_encoders': self.label_encoders
        }
    
    def _setup_label_encoders(self):
        """Setup label encoders for train.parquet fields only"""
        try:
            if self.annotated_data is not None and HAS_PANDAS:
                # Target group encoder
                if 'target_group' in self.annotated_data.columns:
                    groups = self.annotated_data['target_group'].dropna().unique()
                    encoder_groups = LabelEncoder()
                    encoder_groups.fit(groups)
                    self.label_encoders['target_group'] = encoder_groups
                    logger.info(f"Loaded target group encoder: {len(groups)} groups")
                
                # Label binary encoder (toxic/benign based on prompt_label)
                label_binary = ['benign', 'toxic']
                encoder_label = LabelEncoder()
                encoder_label.fit(label_binary)
                self.label_encoders['label_binary'] = encoder_label
                logger.info("Loaded label binary encoder")
                
                # Generation method encoder
                if 'generation_method' in self.annotated_data.columns:
                    methods = self.annotated_data['generation_method'].dropna().unique()
                    encoder_methods = LabelEncoder()
                    encoder_methods.fit(methods)
                    self.label_encoders['generation_method'] = encoder_methods
                    logger.info(f"Loaded generation method encoder: {len(methods)} methods")
                
        except Exception as e:
            logger.warning(f"Could not setup label encoders: {e}")
    
    def analyze_data_quality(self) -> DatasetStats:
        """
        Perform comprehensive data quality analysis based on train.parquet fields.
        
        Returns:
            DatasetStats object containing analysis results
        """
        logger.info("Analyzing data quality...")
        
        if self.annotated_data is None:
            raise ValueError("Annotated data not loaded. Call load_raw_data() first.")
        
        if not HAS_PANDAS:
            raise ValueError("Pandas is required for data analysis")
        
        # Use annotated data for quality analysis
        df = self.annotated_data
        
        # Basic statistics
        total_entries = len(df)
        text_lengths = df['text'].str.len()
        
        # Label distribution based on prompt_label (1=toxic, 0=benign)
        if 'prompt_label' in df.columns:
            toxic_count = (df['prompt_label'] == 1).sum()
            benign_count = (df['prompt_label'] == 0).sum()
            label_distribution = {
                'toxic': toxic_count,
                'benign': benign_count
            }
            # Raw prompt_label distribution for reference
            prompt_label_distribution = df['prompt_label'].value_counts().to_dict()
        else:
            label_distribution = {'toxic': 0, 'benign': total_entries}
            prompt_label_distribution = {0: total_entries}
        
        # Group distribution
        group_counts = df['target_group'].value_counts().to_dict() if 'target_group' in df.columns else {}
        
        # Generation method distribution
        method_counts = df['generation_method'].value_counts().to_dict() if 'generation_method' in df.columns else {}
        
        # RoBERTa prediction statistics
        avg_roberta_prediction = df['roberta_prediction'].mean() if 'roberta_prediction' in df.columns else 0.0
        
        stats = DatasetStats(
            total_entries=total_entries,
            text_length_mean=float(text_lengths.mean()),
            text_length_median=float(text_lengths.median()),
            text_length_std=float(text_lengths.std()),
            label_distribution=label_distribution,
            group_distribution=group_counts,
            generation_method_distribution=method_counts,
            avg_roberta_prediction=float(avg_roberta_prediction),
            prompt_label_distribution=prompt_label_distribution,
            missing_values=int(df.isnull().sum().sum()),
            duplicate_count=int(df.duplicated().sum())
        )
        
        logger.info("Data quality analysis completed")
        return stats
    
    def print_data_summary(self, stats: Optional[DatasetStats] = None):
        """
        Print a comprehensive summary of the dataset based on train.parquet fields.
        
        Args:
            stats: DatasetStats object. If None, will compute stats.
        """
        if stats is None:
            stats = self.analyze_data_quality()
        
        print("\n" + "="*60)
        print("TOXIGEN DATASET SUMMARY")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total entries: {stats.total_entries:,}")
        print(f"  Average text length: {stats.text_length_mean:.1f} characters")
        print(f"  Text length median: {stats.text_length_median:.1f} characters")
        print(f"  Text length std: {stats.text_length_std:.1f} characters")
        print(f"  Missing values: {stats.missing_values:,}")
        print(f"  Duplicate entries: {stats.duplicate_count:,}")
        print(f"  Average RoBERTa prediction: {stats.avg_roberta_prediction:.3f}")
        
        print(f"\nLabel Distribution (Binary):")
        total_labels = sum(stats.label_distribution.values())
        for label, count in sorted(stats.label_distribution.items()):
            percentage = (count / total_labels) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nPrompt Label Distribution (Raw):")
        for label, count in sorted(stats.prompt_label_distribution.items()):
            percentage = (count / stats.total_entries) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        if stats.group_distribution:
            print(f"\nTarget Group Distribution:")
            total_groups = sum(stats.group_distribution.values())
            # Show top 10 most common groups
            sorted_groups = sorted(stats.group_distribution.items(), key=lambda x: x[1], reverse=True)
            for group, count in sorted_groups[:10]:
                percentage = (count / total_groups) * 100
                print(f"  {group}: {count:,} ({percentage:.1f}%)")
            if len(sorted_groups) > 10:
                other_count = sum(count for _, count in sorted_groups[10:])
                other_percentage = (other_count / total_groups) * 100
                print(f"  Others ({len(sorted_groups) - 10} groups): {other_count:,} ({other_percentage:.1f}%)")
        
        if stats.generation_method_distribution:
            print(f"\nGeneration Method Distribution:")
            for method, count in sorted(stats.generation_method_distribution.items()):
                percentage = (count / stats.total_entries) * 100
                print(f"  {method}: {count:,} ({percentage:.1f}%)")
        
        print("="*60)
    
    def clean_text_data(self) -> Dict[str, Any]:
        """
        Perform text cleaning operations on the dataset.
        
        Returns:
            Dictionary with cleaning statistics
        """
        logger.info("Starting text cleaning operations...")
        
        if self.annotated_data is None:
            raise ValueError("Annotated data not loaded. Call load_raw_data() first.")
        
        if not HAS_PANDAS:
            raise ValueError("Pandas is required for text cleaning")
        
        df = self.annotated_data.copy()
        
        cleaning_stats = {
            'entries_processed': 0,
            'empty_texts_found': 0,
            'texts_with_urls': 0,
            'texts_with_mentions': 0,
            'texts_with_hashtags': 0,
            'texts_with_newlines': 0,
            'invalid_entries_removed': 0
        }
        
        cleaned_texts = []
        
        for idx, row in df.iterrows():
            try:
                text = row.get('text', '')
                
                if not text or not isinstance(text, str):
                    cleaning_stats['empty_texts_found'] += 1
                    continue
                
                # Track text characteristics before cleaning
                if 'http' in text or 'www.' in text:
                    cleaning_stats['texts_with_urls'] += 1
                if '@' in text:
                    cleaning_stats['texts_with_mentions'] += 1
                if '#' in text:
                    cleaning_stats['texts_with_hashtags'] += 1
                if '\n' in text or '\r' in text:
                    cleaning_stats['texts_with_newlines'] += 1
                
                # Clean the text
                cleaned_text = self._clean_text(text)
                
                # Create cleaned row
                cleaned_row = row.copy()
                cleaned_row['text'] = cleaned_text
                cleaned_row['original_text_length'] = len(text)
                cleaned_row['cleaned_text_length'] = len(cleaned_text)
                cleaned_row['text_cleaned'] = True
                
                cleaned_texts.append(cleaned_row)
                cleaning_stats['entries_processed'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing entry {idx}: {e}")
                cleaning_stats['invalid_entries_removed'] += 1
        
        # Update the annotated data with cleaned texts
        if cleaned_texts:
            self.annotated_data = pd.DataFrame(cleaned_texts)
        
        logger.info(f"Text cleaning completed. Processed {cleaning_stats['entries_processed']} entries")
        return cleaning_stats
    
    def _clean_text(self, text: str) -> str:
        """
        Clean individual text entries.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Optional: Normalize URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Optional: Normalize user mentions
        text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)
        
        # Optional: Normalize hashtags
        text = re.sub(r'#[a-zA-Z0-9_]+', '[HASHTAG]', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!?]{3,}', '!!!', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def extract_features(self) -> Union[pd.DataFrame, List[Dict]]:
        """
        Extract features from the cleaned dataset.
        
        Returns:
            DataFrame with extracted features (if pandas available) or list of dicts
        """
        logger.info("Extracting features from dataset...")
        
        if self.annotated_data is None:
            raise ValueError("Annotated data not loaded. Call load_raw_data() first.")
        
        if not HAS_PANDAS:
            raise ValueError("Pandas is required for feature extraction")
        
        df = self.annotated_data.copy()
        features_list = []
        
        for idx, row in df.iterrows():
            try:
                text = row.get('text', '')
                
                # Basic features
                features = {
                    'text_id': f"toxigen_{idx}",
                    'text': text,
                    'text_length': len(text),
                    'word_count': len(text.split()) if text else 0,
                    'target_group': row.get('target_group', 'unknown'),
                    'toxicity_ai': float(row.get('toxicity_ai', 0.0)),
                    'toxicity_human': float(row.get('toxicity_human', 0.0)),
                    'stereotyping': row.get('stereotyping', 'unknown'),
                    'intent': float(row.get('intent', 0.0)),
                    'factual': row.get('factual?', 'unknown'),
                    'framing': row.get('framing', 'unknown'),
                    'ingroup_effect': row.get('ingroup_effect', 'unknown'),
                    'lewd': row.get('lewd', 'unknown')
                }
                
                # Binary label (threshold-based)
                toxicity_threshold = 3.0
                features['label_binary'] = 'hate' if features['toxicity_human'] >= toxicity_threshold else 'normal'
                
                # Text-based features
                text_lower = text.lower() if text else ''
                features['contains_url'] = '[url]' in text_lower or 'http' in text_lower
                features['contains_mention'] = '[user]' in text_lower or '@' in text_lower
                features['contains_hashtag'] = '[hashtag]' in text_lower or '#' in text_lower
                features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0.0
                features['exclamation_count'] = text.count('!')
                features['question_count'] = text.count('?')
                features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
                features['sentence_count'] = len([s for s in text.split('.') if s.strip()]) if text else 0
                
                # Advanced features
                features['toxicity_gap'] = abs(features['toxicity_ai'] - features['toxicity_human'])
                features['high_intent'] = features['intent'] >= 4.0
                features['uses_stereotyping'] = 'stereotyping' in features['stereotyping'].lower() if features['stereotyping'] != 'unknown' else False
                
                # Generation metadata (if available)
                if 'actual_method' in row:
                    features['generation_method'] = row['actual_method']
                elif 'generation_method' in row:
                    features['generation_method'] = row['generation_method']
                else:
                    features['generation_method'] = 'unknown'
                
                features_list.append(features)
                
            except Exception as e:
                logger.warning(f"Error extracting features for entry {idx}: {e}")
        
        if HAS_PANDAS:
            df = pd.DataFrame(features_list)
            logger.info(f"Feature extraction completed. Shape: {df.shape}")
            return df
        else:
            logger.info(f"Feature extraction completed. {len(features_list)} records")
            return features_list
    
    def prepare_training_data(self, target_encoding: str = 'label_binary', test_size: float = 0.2, val_size: float = 0.1) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[List[Dict], List[Dict], List[Dict]]]:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            target_encoding: Type of label encoding ('label_binary', 'target_group', 'stereotyping')
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data) - DataFrames if pandas available, else lists
        """
        logger.info(f"Preparing training data with {target_encoding} encoding...")
        
        # Extract features
        features_data = self.extract_features()
        
        if not HAS_PANDAS:
            raise ValueError("Pandas is required for data preparation")
        
        features_df = features_data
        
        # Encode labels if encoder is available
        if target_encoding in self.label_encoders:
            encoder = self.label_encoders[target_encoding]
            try:
                # Get the target column for encoding
                if target_encoding == 'label_binary':
                    target_column = 'label_binary'
                elif target_encoding == 'target_group':
                    target_column = 'target_group'
                elif target_encoding == 'stereotyping':
                    target_column = 'stereotyping'
                else:
                    target_column = target_encoding
                
                # Create label_encoded column
                label_encoded_values = []
                for _, row in features_df.iterrows():
                    label = row[target_column]
                    if HAS_SKLEARN and hasattr(encoder, 'transform'):
                        if label in encoder.classes_:
                            encoded = encoder.transform([label])[0]
                        else:
                            encoded = -1
                    else:
                        encoded = encoder._label_to_idx.get(label, -1)
                    label_encoded_values.append(encoded)
                
                features_df['label_encoded'] = label_encoded_values
            except Exception as e:
                logger.warning(f"Could not encode all labels: {e}")
                features_df['label_encoded'] = -1
        
        # Stratified split to maintain label distribution
        if 'target_group' in features_df.columns:
            # Split by target group to ensure balanced representation
            train_dfs, val_dfs, test_dfs = [], [], []
            
            for group in features_df['target_group'].unique():
                group_df = features_df[features_df['target_group'] == group].copy()
                n_total = len(group_df)
                
                if n_total < 3:  # Too few samples, put all in train
                    group_df['split'] = 'train'
                    train_dfs.append(group_df)
                    continue
                
                # Calculate split sizes
                n_test = max(1, int(n_total * test_size))
                n_val = max(1, int(n_total * val_size))
                n_train = n_total - n_test - n_val
                
                # Shuffle and split
                group_df = group_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                train_df = group_df.iloc[:n_train].copy()
                val_df = group_df.iloc[n_train:n_train+n_val].copy()
                test_df = group_df.iloc[n_train+n_val:].copy()
                
                train_df['split'] = 'train'
                val_df['split'] = 'val'
                test_df['split'] = 'test'
                
                train_dfs.append(train_df)
                val_dfs.append(val_df)
                test_dfs.append(test_df)
            
            # Combine all splits
            train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
            val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
            test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        else:
            # Simple random split
            features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
            n_total = len(features_df)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)
            
            test_df = features_df.iloc[:n_test].copy()
            val_df = features_df.iloc[n_test:n_test+n_val].copy()
            train_df = features_df.iloc[n_test+n_val:].copy()
            
            train_df['split'] = 'train'
            val_df['split'] = 'val'
            test_df['split'] = 'test'
        
        logger.info(f"Data preparation completed:")
        logger.info(f"  Training: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def export_processed_data(self, output_dir: Optional[str] = None, format: str = 'json'):
        """
        Export processed data in various formats.
        
        Args:
            output_dir: Output directory. If None, uses project structure.
            format: Export format ('csv', 'json', 'parquet') - defaults to 'json' if pandas not available
        """
        if output_dir is None:
            output_dir = self.project_root / "data" / "processed" / "toxigen"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use JSON format if pandas not available
        if not HAS_PANDAS and format in ['csv', 'parquet']:
            format = 'json'
            logger.warning("Pandas not available, using JSON format instead")
        
        logger.info(f"Exporting processed data to {output_dir} in {format} format...")
        
        # Prepare data splits
        train_data, val_data, test_data = self.prepare_training_data()
        
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, data in datasets.items():
            if HAS_PANDAS and hasattr(data, 'to_csv'):
                # DataFrame version
                df = data
                if format == 'csv':
                    output_file = output_dir / f"toxigen_{split_name}.csv"
                    df.to_csv(output_file, index=False)
                elif format == 'json':
                    output_file = output_dir / f"toxigen_{split_name}.json"
                    df.to_json(output_file, orient='records', indent=2)
                elif format == 'parquet':
                    output_file = output_dir / f"toxigen_{split_name}.parquet"
                    df.to_parquet(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            else:
                # List version
                if format == 'json':
                    output_file = output_dir / f"toxigen_{split_name}.json"
                    with output_file.open('w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Format {format} requires pandas. Use 'json' format instead.")
            
            logger.info(f"Exported {split_name} split: {output_file}")
        
        # Export summary statistics
        stats = self.analyze_data_quality()
        stats_file = output_dir / "toxigen_dataset_stats.json"
        
        # Convert all values to native Python types for JSON serialization
        stats_dict = {
            'total_entries': int(stats.total_entries),
            'text_length_mean': float(stats.text_length_mean),
            'text_length_median': float(stats.text_length_median),
            'text_length_std': float(stats.text_length_std),
            'label_distribution': {k: int(v) for k, v in stats.label_distribution.items()},
            'group_distribution': {k: int(v) for k, v in stats.group_distribution.items()},
            'generation_method_distribution': {k: int(v) for k, v in stats.generation_method_distribution.items()},
            'avg_roberta_prediction': float(stats.avg_roberta_prediction),
            'prompt_label_distribution': {k: int(v) for k, v in stats.prompt_label_distribution.items()},
            'missing_values': int(stats.missing_values),
            'duplicate_count': int(stats.duplicate_count)
        }
        
        with stats_file.open('w') as f:
            json.dump(stats_dict, f, indent=2)
        
        logger.info(f"Exported dataset statistics: {stats_file}")
        logger.info(f"Data export completed to {output_dir}")


def main():
    """Main function to demonstrate the data preparation pipeline."""
    try:
        # Initialize processor
        processor = ToxiGenDataProcessor()
        
        # Step 1: Load raw data
        print("Step 1: Loading raw data...")
        raw_data = processor.load_raw_data()
        
        # Step 2: Analyze data quality
        print("\nStep 2: Analyzing data quality...")
        stats = processor.analyze_data_quality()
        processor.print_data_summary(stats)
        
        # Step 3: Clean text data
        print("\nStep 3: Cleaning text data...")
        cleaning_stats = processor.clean_text_data()
        
        print("\nCleaning Results:")
        for key, value in cleaning_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        # Step 4: Extract features and prepare training data
        print("\nStep 4: Extracting features and preparing training data...")
        train_data, val_data, test_data = processor.prepare_training_data()
        
        print(f"\nFeature Extraction Results:")
        print(f"  Training samples: {len(train_data):,}")
        print(f"  Validation samples: {len(val_data):,}")
        print(f"  Test samples: {len(test_data):,}")
        
        # Sample features for display
        if len(train_data) > 0:
            if isinstance(train_data, list):
                sample_features = list(train_data[0].keys()) if train_data else []
            elif HAS_PANDAS and hasattr(train_data, 'columns'):
                sample_features = list(train_data.columns)
            else:
                sample_features = []
                
            if sample_features:
                print(f"  Sample features: {', '.join(sample_features[:10])}")
                if len(sample_features) > 10:
                    print(f"    ... and {len(sample_features) - 10} more features")
        
        # Step 5: Export processed data
        print("\nStep 5: Exporting processed data...")
        processor.export_processed_data(format='json')
        
        print("\nData preparation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
