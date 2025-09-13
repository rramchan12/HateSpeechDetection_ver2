#!/usr/bin/env python3
"""
HateXplain Dataset Preparation and Cleansing Operations

This module provides comprehensive data preparation operations for the HateXplain dataset,
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
    """Data class to hold dataset statistics"""
    total_entries: int
    label_distribution: Dict[str, int]
    persona_distribution: Dict[str, int]  # Add persona/target distribution
    avg_tokens_per_post: float
    min_tokens: int
    max_tokens: int
    missing_rationales: int
    annotation_agreement: float


class HateXplainDataProcessor:
    """
    Main class for HateXplain dataset preparation and cleansing operations.
    
    This class handles:
    - Data loading and validation
    - Data cleansing and preprocessing
    - Feature extraction and transformation
    - Statistical analysis and reporting
    - Data export and splitting
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the HateXplain data processor.
        
        Args:
            data_dir: Path to the data directory. If None, uses default project structure.
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[1]
            self.data_dir = self.project_root / "data" / "hatexplain"
        else:
            self.data_dir = Path(data_dir)
        
        self.dataset = None
        self.divisions = None
        self.processed_data = None
        self.label_encoders = {}
        
        logger.info(f"Initialized HateXplain processor with data directory: {self.data_dir}")
    
    def load_raw_data(self) -> Dict[str, Any]:
        """
        Load raw HateXplain dataset and division files.
        
        Returns:
            Dictionary containing loaded data and metadata
        """
        logger.info("Loading raw HateXplain dataset...")
        
        # Load main dataset
        dataset_file = self.data_dir / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with dataset_file.open('r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # Load post ID divisions
        divisions_file = self.data_dir / "post_id_divisions.json"
        if not divisions_file.exists():
            raise FileNotFoundError(f"Divisions file not found: {divisions_file}")
        
        with divisions_file.open('r', encoding='utf-8') as f:
            self.divisions = json.load(f)
        
        # Load label encoders
        self._load_label_encoders()
        
        logger.info(f"Successfully loaded {len(self.dataset)} entries")
        logger.info(f"Data splits: {', '.join([f'{k}: {len(v)}' for k, v in self.divisions.items()])}")
        
        return {
            'dataset': self.dataset,
            'divisions': self.divisions,
            'label_encoders': self.label_encoders
        }
    
    def _load_label_encoders(self):
        """Load pre-trained label encoders"""
        try:
            # Load 3-class encoder (hatespeech, normal, offensive)
            classes_file = self.data_dir / "classes.npy"
            if classes_file.exists():
                encoder_3class = LabelEncoder()
                encoder_3class.classes_ = np.load(classes_file, allow_pickle=True)
                if HAS_SKLEARN:
                    self.label_encoders['3class'] = encoder_3class
                else:
                    # For basic encoder, set up the mapping manually
                    encoder_3class.fit(encoder_3class.classes_)
                    self.label_encoders['3class'] = encoder_3class
                logger.info(f"Loaded 3-class encoder: {encoder_3class.classes_}")
            
            # Load 2-class encoder (toxic, non-toxic)
            classes_two_file = self.data_dir / "classes_two.npy"
            if classes_two_file.exists():
                encoder_2class = LabelEncoder()
                encoder_2class.classes_ = np.load(classes_two_file, allow_pickle=True)
                if HAS_SKLEARN:
                    self.label_encoders['2class'] = encoder_2class
                else:
                    # For basic encoder, set up the mapping manually
                    encoder_2class.fit(encoder_2class.classes_)
                    self.label_encoders['2class'] = encoder_2class
                logger.info(f"Loaded 2-class encoder: {encoder_2class.classes_}")
                
        except Exception as e:
            logger.warning(f"Could not load label encoders: {e}")
    
    def analyze_data_quality(self) -> DatasetStats:
        """
        Perform comprehensive data quality analysis.
        
        Returns:
            DatasetStats object containing analysis results
        """
        logger.info("Analyzing data quality...")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_raw_data() first.")
        
        # Basic statistics
        total_entries = len(self.dataset)
        token_counts = []
        label_counts = defaultdict(int)
        target_counts = defaultdict(int)  # Add target counts for personas
        missing_rationales = 0
        annotation_agreements = []
        
        for post_id, entry in self.dataset.items():
            # Token count analysis
            token_count = len(entry.get('post_tokens', []))
            token_counts.append(token_count)
            
            # Label distribution analysis
            annotators = entry.get('annotators', [])
            labels = [ann.get('label') for ann in annotators if ann.get('label')]
            
            # Count each label
            for label in labels:
                label_counts[label] += 1
            
            # Target/persona distribution analysis
            targets = [ann.get('target', []) for ann in annotators if ann.get('target')]
            all_targets = []
            for target_list in targets:
                if isinstance(target_list, list):
                    all_targets.extend(target_list)
                else:
                    all_targets.append(target_list)
            
            # Count each target (persona)
            for target in all_targets:
                if target:  # Skip empty targets
                    target_counts[target] += 1
            
            # Check rationales
            if not entry.get('rationales'):
                missing_rationales += 1
            
            # Calculate annotation agreement for this post
            if len(labels) > 1:
                most_common_label = Counter(labels).most_common(1)[0][1]
                agreement = most_common_label / len(labels)
                annotation_agreements.append(agreement)
        
        # Calculate statistics
        avg_tokens = np.mean(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        avg_agreement = np.mean(annotation_agreements) if annotation_agreements else 0
        
        stats = DatasetStats(
            total_entries=total_entries,
            label_distribution=dict(label_counts),
            persona_distribution=dict(target_counts),  # Add persona distribution
            avg_tokens_per_post=avg_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            missing_rationales=missing_rationales,
            annotation_agreement=avg_agreement
        )
        
        logger.info("Data quality analysis completed")
        return stats
    
    def print_data_summary(self, stats: Optional[DatasetStats] = None):
        """
        Print a comprehensive summary of the dataset.
        
        Args:
            stats: DatasetStats object. If None, will compute stats.
        """
        if stats is None:
            stats = self.analyze_data_quality()
        
        print("\n" + "="*60)
        print("HATEXPLAIN DATASET SUMMARY")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total entries: {stats.total_entries:,}")
        print(f"  Average tokens per post: {stats.avg_tokens_per_post:.1f}")
        print(f"  Token range: {stats.min_tokens} - {stats.max_tokens}")
        print(f"  Missing rationales: {stats.missing_rationales:,} ({stats.missing_rationales/stats.total_entries*100:.1f}%)")
        print(f"  Average annotation agreement: {stats.annotation_agreement:.2f}")
        
        print(f"\nLabel Distribution:")
        total_annotations = sum(stats.label_distribution.values())
        for label, count in sorted(stats.label_distribution.items()):
            percentage = (count / total_annotations) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nPersona/Target Distribution:")
        total_personas = sum(stats.persona_distribution.values())
        # Show top 10 most common personas
        sorted_personas = sorted(stats.persona_distribution.items(), key=lambda x: x[1], reverse=True)
        for persona, count in sorted_personas[:10]:
            percentage = (count / total_personas) * 100
            print(f"  {persona}: {count:,} ({percentage:.1f}%)")
        if len(sorted_personas) > 10:
            other_count = sum(count for _, count in sorted_personas[10:])
            other_percentage = (other_count / total_personas) * 100
            print(f"  Others ({len(sorted_personas) - 10} personas): {other_count:,} ({other_percentage:.1f}%)")
        
        if self.divisions:
            print(f"\nData Splits:")
            for split_name, post_ids in self.divisions.items():
                print(f"  {split_name}: {len(post_ids):,} post IDs")
        
        print("="*60)
    
    def clean_text_data(self) -> Dict[str, Any]:
        """
        Perform text cleaning operations on the dataset.
        
        Returns:
            Dictionary with cleaning statistics
        """
        logger.info("Starting text cleaning operations...")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_raw_data() first.")
        
        cleaning_stats = {
            'entries_processed': 0,
            'empty_posts_found': 0,
            'posts_with_urls': 0,
            'posts_with_mentions': 0,
            'posts_with_hashtags': 0,
            'invalid_entries_removed': 0
        }
        
        cleaned_dataset = {}
        
        for post_id, entry in self.dataset.items():
            try:
                # Basic validation
                if not entry.get('post_tokens') or not isinstance(entry['post_tokens'], list):
                    cleaning_stats['empty_posts_found'] += 1
                    continue
                
                # Clean tokens
                original_tokens = entry['post_tokens']
                cleaned_tokens = self._clean_token_list(original_tokens)
                
                # Update statistics
                text_content = ' '.join(original_tokens)
                if 'http' in text_content or 'www.' in text_content:
                    cleaning_stats['posts_with_urls'] += 1
                if '@' in text_content:
                    cleaning_stats['posts_with_mentions'] += 1
                if '#' in text_content:
                    cleaning_stats['posts_with_hashtags'] += 1
                
                # Create cleaned entry
                cleaned_entry = entry.copy()
                cleaned_entry['post_tokens'] = cleaned_tokens
                cleaned_entry['original_token_count'] = len(original_tokens)
                cleaned_entry['cleaned_token_count'] = len(cleaned_tokens)
                
                cleaned_dataset[post_id] = cleaned_entry
                cleaning_stats['entries_processed'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing entry {post_id}: {e}")
                cleaning_stats['invalid_entries_removed'] += 1
        
        self.dataset = cleaned_dataset
        
        logger.info(f"Text cleaning completed. Processed {cleaning_stats['entries_processed']} entries")
        return cleaning_stats
    
    def _clean_token_list(self, tokens: List[str]) -> List[str]:
        """
        Clean individual tokens in a post.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of cleaned tokens
        """
        cleaned_tokens = []
        
        for token in tokens:
            if not isinstance(token, str):
                continue
            
            # Convert to lowercase
            token = token.lower().strip()
            
            # Skip empty tokens
            if not token:
                continue
            
            # Optional: Normalize URLs
            if token.startswith(('http://', 'https://', 'www.')):
                token = '[URL]'
            
            # Optional: Normalize user mentions
            elif token.startswith('@'):
                token = '[USER]'
            
            # Optional: Normalize hashtags
            elif token.startswith('#'):
                token = '[HASHTAG]'
            
            # Remove excessive punctuation
            token = re.sub(r'[!?]{3,}', '!!!', token)
            token = re.sub(r'\.{3,}', '...', token)
            
            cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def extract_features(self) -> Union[pd.DataFrame, List[Dict]]:
        """
        Extract features from the cleaned dataset.
        
        Returns:
            DataFrame with extracted features (if pandas available) or list of dicts
        """
        logger.info("Extracting features from dataset...")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_raw_data() first.")
        
        features_list = []
        
        for post_id, entry in self.dataset.items():
            try:
                # Basic features
                features = {
                    'post_id': post_id,
                    'text': ' '.join(entry.get('post_tokens', [])),
                    'token_count': len(entry.get('post_tokens', [])),
                    'has_rationales': len(entry.get('rationales', [])) > 0,
                    'num_annotators': len(entry.get('annotators', []))
                }
                
                # Extract rationale text
                post_tokens = entry.get('post_tokens', [])
                rationales = entry.get('rationales', [])
                
                if features['has_rationales'] and rationales and post_tokens:
                    # Combine all rationale indices from all annotators
                    combined_rationale = [0] * len(post_tokens)
                    for rationale_list in rationales:
                        if isinstance(rationale_list, list):
                            for i, val in enumerate(rationale_list):
                                if i < len(combined_rationale) and val == 1:
                                    combined_rationale[i] = 1
                    
                    # Extract tokens marked as rationales
                    rationale_tokens = []
                    for i, token in enumerate(post_tokens):
                        if i < len(combined_rationale) and combined_rationale[i] == 1:
                            rationale_tokens.append(token)
                    
                    if rationale_tokens:
                        features['rationale_text'] = ' '.join(rationale_tokens)
                    else:
                        features['rationale_text'] = 'NA'
                else:
                    features['rationale_text'] = 'NA'
                
                # Extract majority label and target information
                annotators = entry.get('annotators', [])
                if annotators:
                    labels = [ann.get('label') for ann in annotators if ann.get('label')]
                    targets = [ann.get('target', []) for ann in annotators if ann.get('target')]
                    
                    if labels:
                        label_counts = Counter(labels)
                        majority_label = label_counts.most_common(1)[0][0]
                        features['majority_label'] = majority_label
                        features['label_agreement'] = label_counts[majority_label] / len(labels)
                    else:
                        features['majority_label'] = 'unknown'
                        features['label_agreement'] = 0.0
                    
                    # Extract target information
                    if targets:
                        # Flatten all targets and find the most common
                        all_targets = []
                        for target_list in targets:
                            if isinstance(target_list, list):
                                all_targets.extend(target_list)
                            else:
                                all_targets.append(target_list)
                        
                        # Remove 'None' targets and get unique targets
                        valid_targets = [t for t in all_targets if t and t != 'None']
                        if valid_targets:
                            target_counts = Counter(valid_targets)
                            features['target'] = target_counts.most_common(1)[0][0]
                            features['target_list'] = list(set(valid_targets))
                        else:
                            features['target'] = 'None'
                            features['target_list'] = ['None']
                    else:
                        features['target'] = 'None'
                        features['target_list'] = ['None']
                else:
                    features['majority_label'] = 'unknown'
                    features['label_agreement'] = 0.0
                    features['target'] = 'None'
                    features['target_list'] = ['None']
                
                # Text-based features
                text = features['text'].lower()
                features['contains_url'] = '[url]' in text or 'http' in text
                features['contains_mention'] = '[user]' in text or '@' in text
                features['contains_hashtag'] = '[hashtag]' in text or '#' in text
                features['avg_word_length'] = np.mean([len(word) for word in entry.get('post_tokens', [])]) if entry.get('post_tokens') else 0
                features['exclamation_count'] = text.count('!')
                features['question_count'] = text.count('?')
                features['caps_ratio'] = sum(1 for c in features['text'] if c.isupper()) / len(features['text']) if features['text'] else 0
                
                # Determine split
                for split_name, post_ids in self.divisions.items():
                    if post_id in post_ids:
                        features['split'] = split_name
                        break
                else:
                    features['split'] = 'unknown'
                
                features_list.append(features)
                
            except Exception as e:
                logger.warning(f"Error extracting features for {post_id}: {e}")
        
        if HAS_PANDAS:
            df = pd.DataFrame(features_list)
            logger.info(f"Feature extraction completed. Shape: {df.shape}")
            return df
        else:
            logger.info(f"Feature extraction completed. {len(features_list)} records")
            return features_list
    
    def prepare_training_data(self, target_encoding: str = '3class') -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[List[Dict], List[Dict], List[Dict]]]:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            target_encoding: Type of label encoding ('3class' or '2class')
            
        Returns:
            Tuple of (train_data, val_data, test_data) - DataFrames if pandas available, else lists
        """
        logger.info(f"Preparing training data with {target_encoding} encoding...")
        
        # Extract features
        features_data = self.extract_features()
        
        if HAS_PANDAS and isinstance(features_data, pd.DataFrame):
            # Pandas version
            features_df = features_data
            
            # Encode labels if encoder is available
            if target_encoding in self.label_encoders:
                encoder = self.label_encoders[target_encoding]
                try:
                    # Create label_encoded column
                    label_encoded_values = []
                    for _, row in features_df.iterrows():
                        label = row['majority_label']
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
            
            # Split data
            train_df = features_df[features_df['split'] == 'train'].copy()
            val_df = features_df[features_df['split'] == 'val'].copy()
            test_df = features_df[features_df['split'] == 'test'].copy()
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  Training: {len(train_df)} samples")
            logger.info(f"  Validation: {len(val_df)} samples")
            logger.info(f"  Test: {len(test_df)} samples")
            
            return train_df, val_df, test_df
        else:
            # List version
            features_list = features_data
            
            # Encode labels if encoder is available
            if target_encoding in self.label_encoders:
                encoder = self.label_encoders[target_encoding]
                for item in features_list:
                    try:
                        if item['majority_label'] in encoder.classes_:
                            item['label_encoded'] = encoder.transform([item['majority_label']])[0]
                        else:
                            item['label_encoded'] = -1
                    except:
                        item['label_encoded'] = -1
            
            # Split data
            train_data = [item for item in features_list if item['split'] == 'train']
            val_data = [item for item in features_list if item['split'] == 'val']
            test_data = [item for item in features_list if item['split'] == 'test']
            
            logger.info(f"Data preparation completed:")
            logger.info(f"  Training: {len(train_data)} samples")
            logger.info(f"  Validation: {len(val_data)} samples")
            logger.info(f"  Test: {len(test_data)} samples")
            
            return train_data, val_data, test_data
    
    def export_processed_data(self, output_dir: Optional[str] = None, format: str = 'json'):
        """
        Export processed data in various formats.
        
        Args:
            output_dir: Output directory. If None, uses project structure.
            format: Export format ('csv', 'json', 'parquet') - defaults to 'json' if pandas not available
        """
        if output_dir is None:
            output_dir = self.project_root / "data" / "processed" / "hatexplain"
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
                    output_file = output_dir / f"hatexplain_{split_name}.csv"
                    df.to_csv(output_file, index=False)
                elif format == 'json':
                    output_file = output_dir / f"hatexplain_{split_name}.json"
                    df.to_json(output_file, orient='records', indent=2)
                elif format == 'parquet':
                    output_file = output_dir / f"hatexplain_{split_name}.parquet"
                    df.to_parquet(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            else:
                # List version
                if format == 'json':
                    output_file = output_dir / f"hatexplain_{split_name}.json"
                    with output_file.open('w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Format {format} requires pandas. Use 'json' format instead.")
            
            logger.info(f"Exported {split_name} split: {output_file}")
        
        # Export summary statistics
        stats = self.analyze_data_quality()
        stats_file = output_dir / "dataset_stats.json"
        with stats_file.open('w') as f:
            json.dump({
                'total_entries': stats.total_entries,
                'label_distribution': stats.label_distribution,
                'persona_distribution': stats.persona_distribution,  # Add persona distribution
                'avg_tokens_per_post': stats.avg_tokens_per_post,
                'min_tokens': stats.min_tokens,
                'max_tokens': stats.max_tokens,
                'missing_rationales': stats.missing_rationales,
                'annotation_agreement': stats.annotation_agreement
            }, f, indent=2)
        
        logger.info(f"Data export completed to {output_dir}")


def main():
    """Main function to demonstrate the data preparation pipeline."""
    try:
        # Initialize processor
        processor = HateXplainDataProcessor()
        
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
