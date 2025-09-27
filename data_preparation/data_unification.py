"""
Data Unification Module for HateXplain and ToxiGen Datasets

This module provides functionality to merge HateXplain and ToxiGen datasets into a unified schema
according to the specified minimal schema requirements. The unified dataset enables consistent
training across both hate speech detection datasets.

Unified Schema:
- text: Core input text
- label_binary: Binary classification (hate vs normal)
- label_multiclass: Optional multi-class labels 
- target_group_norm: Normalized target group names
- persona_tag: Top persona tags (LGBTQ, Mexican, Middle East, etc.)
- source_dataset: Dataset provenance (hatexplain/toxigen)
- rationale_text: Explanation for labels (when available)
- is_synthetic: Flag for generated/augmented data
- fine_tuning_embedding: Scaffold for fine-tuning with persona/policy placeholders
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pandas for efficient processing
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("Pandas not available. Using basic Python processing.")


@dataclass
class UnifiedDatasetStats:
    """Statistics for the unified dataset"""
    total_entries: int
    hatexplain_entries: int
    toxigen_entries: int
    label_binary_distribution: Dict[str, int]
    label_multiclass_distribution: Dict[str, int]
    target_group_distribution: Dict[str, int]
    persona_tag_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    synthetic_ratio: float
    avg_text_length: float
    rationale_coverage: float


class DatasetUnifier:
    """
    Main class for unifying HateXplain and ToxiGen datasets into a consistent schema.
    
    This class handles:
    - Loading processed datasets from both sources
    - Mapping fields according to unified schema
    - Normalizing target groups and persona tags
    - Creating fine-tuning embeddings with placeholders
    - Exporting unified datasets in various formats
    """
    
    # Target group normalization mapping - ONLY for LGBTQ, Mexican, and Middle East
    TARGET_GROUP_NORMALIZATION = {
        # HateXplain target normalization (only selected groups)
        'Homosexual': 'lgbtq',
        'Gay': 'lgbtq', 
        'Hispanic': 'mexican',
        'Latino': 'mexican',
        'Arab': 'middle_east',
        
        # ToxiGen target groups (only selected groups)
        'lgbtq': 'lgbtq',
        'mexican': 'mexican', 
        'middle_east': 'middle_east'
    }
    
    # Valid target groups for filtering
    VALID_TARGET_GROUPS = {'lgbtq', 'mexican', 'middle_east'}
    
    # Top persona tags for fine-tuning (only the 3 selected)
    TOP_PERSONA_TAGS = {'lgbtq', 'mexican', 'middle_east'}
    
    def __init__(self, hatexplain_dir: str, toxigen_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the dataset unifier.
        
        Args:
            hatexplain_dir: Directory containing processed HateXplain data
            toxigen_dir: Directory containing processed ToxiGen data  
            output_dir: Output directory for unified dataset (defaults to data/processed/unified)
        """
        self.hatexplain_dir = Path(hatexplain_dir)
        self.toxigen_dir = Path(toxigen_dir)
        
        if output_dir is None:
            self.output_dir = Path("data/processed/unified")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset containers
        self.hatexplain_data = {}  # train/val/test splits
        self.toxigen_data = {}     # train/val/test splits
        self.unified_data = {}     # unified train/val/test splits
        
        logger.info(f"Initialized DatasetUnifier")
        logger.info(f"  HateXplain dir: {self.hatexplain_dir}")
        logger.info(f"  ToxiGen dir: {self.toxigen_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def load_datasets(self):
        """Load both HateXplain and ToxiGen processed datasets."""
        logger.info("Loading processed datasets...")
        
        # Load HateXplain splits
        for split in ['train', 'val', 'test']:
            hatexplain_file = self.hatexplain_dir / f"hatexplain_{split}.json"
            if hatexplain_file.exists():
                with open(hatexplain_file, 'r', encoding='utf-8') as f:
                    self.hatexplain_data[split] = json.load(f)
                logger.info(f"Loaded HateXplain {split}: {len(self.hatexplain_data[split])} entries")
            else:
                logger.warning(f"HateXplain {split} file not found: {hatexplain_file}")
        
        # Load ToxiGen splits
        for split in ['train', 'val', 'test']:
            toxigen_file = self.toxigen_dir / f"toxigen_{split}.json"
            if toxigen_file.exists():
                with open(toxigen_file, 'r', encoding='utf-8') as f:
                    self.toxigen_data[split] = json.load(f)
                logger.info(f"Loaded ToxiGen {split}: {len(self.toxigen_data[split])} entries")
            else:
                logger.warning(f"ToxiGen {split} file not found: {toxigen_file}")
    
    def normalize_target_group(self, target_group: str, source: str) -> Optional[str]:
        """
        Normalize target group names according to unified schema.
        Only processes LGBTQ, Mexican, and Middle East groups.
        
        Args:
            target_group: Original target group name
            source: Source dataset ('hatexplain' or 'toxigen')
            
        Returns:
            Normalized target group name if valid, None if not in selected groups
        """
        if not target_group or target_group in ['None', 'none', 'null']:
            return None
        
        # Clean the target group string
        target_clean = str(target_group).strip()
        
        # Apply normalization mapping
        normalized = self.TARGET_GROUP_NORMALIZATION.get(target_clean, target_clean.lower())
        
        # Only return if it's one of our valid target groups
        if normalized in self.VALID_TARGET_GROUPS:
            return normalized
        
        return None
    
    def extract_persona_tag(self, original_target_group: str, target_group_norm: Optional[str]) -> Optional[str]:
        """
        Extract persona tag preserving original identity while normalizing target groups.
        
        Args:
            original_target_group: Original target group from source dataset
            target_group_norm: Normalized target group name
            
        Returns:
            Original persona tag (lowercased) if target group is valid, None otherwise
        """
        # Only return persona tag if the normalized target group is valid
        if target_group_norm and target_group_norm in self.VALID_TARGET_GROUPS:
            # Return the original target group (lowercased) as persona tag
            return original_target_group.lower().strip()
        
        return None
    
    def map_hatexplain_labels(self, majority_label: str) -> Tuple[str, str]:
        """
        Map HateXplain labels to unified schema.
        
        Args:
            majority_label: HateXplain majority_label (hate/offensive/normal)
            
        Returns:
            Tuple of (binary_label, multiclass_label)
        """
        # Binary mapping: hate → hate, offensive+normal → normal
        if majority_label == 'hate':
            binary_label = 'hate'
        else:  # offensive or normal
            binary_label = 'normal'
        
        # Multiclass: keep original
        multiclass_label = majority_label
        
        return binary_label, multiclass_label
    
    def map_toxigen_labels(self, label_binary: str) -> Tuple[str, str]:
        """
        Map ToxiGen labels to unified schema.
        
        Args:
            label_binary: ToxiGen label_binary (toxic/benign from prompt_label conversion)
            
        Returns:
            Tuple of (binary_label, multiclass_label)
        """
        # Binary mapping: toxic → hate, benign/normal → normal
        if label_binary == 'toxic':
            binary_label = 'hate'
            multiclass_label = 'toxic_implicit'  # ToxiGen is implicit toxicity
        else:  # benign or normal
            binary_label = 'normal'
            multiclass_label = 'benign_implicit'  # ToxiGen is implicit toxicity
        
        return binary_label, multiclass_label
    
    def create_fine_tuning_embedding(self, text: str, target_group_norm: Optional[str], 
                                   persona_tag: Optional[str], rationale_text: Optional[str],
                                   source: str) -> str:
        """
        Create fine-tuning embedding with persona, rationale, and policy placeholders.
        
        Args:
            text: Original text
            target_group_norm: Normalized target group
            persona_tag: Persona tag (if applicable)
            rationale_text: Rationale explanation (if available)
            source: Source dataset
            
        Returns:
            Fine-tuning embedding string with placeholders
        """
        embedding_parts = []
        
        # Add persona placeholder if applicable
        if persona_tag:
            embedding_parts.append(f"[PERSONA:{persona_tag.upper()}]")
        
        # Add original text
        embedding_parts.append(text)
        
        # Add rationale placeholder if available (HateXplain only)
        if rationale_text and rationale_text not in ['NA', 'None', '']:
            embedding_parts.append(f"[RATIONALE:{rationale_text}]")
        
        # Add policy placeholder
        embedding_parts.append("[POLICY:HATE_SPEECH_DETECTION]")
        
        return " ".join(embedding_parts)
    
    def unify_entry(self, entry: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """
        Convert a single entry to unified schema.
        Only processes entries with LGBTQ, Mexican, or Middle East target groups.
        
        Args:
            entry: Original entry from either dataset
            source: Source dataset ('hatexplain' or 'toxigen')
            
        Returns:
            Unified entry following the minimal schema, or None if target group not selected
        """
        # First check if this entry has a valid target group
        if source == 'hatexplain':
            target_raw = entry.get('target', 'None')
        elif source == 'toxigen':
            target_raw = entry.get('target_group', 'none')
        else:
            return None
        
        target_group_norm = self.normalize_target_group(target_raw, source)
        
        # Skip entries that don't have our selected target groups
        if not target_group_norm:
            return None
        
        unified = {}
        
        if source == 'hatexplain':
            # HateXplain field mapping
            unified['text'] = entry.get('text', '')
            
            # Label mapping
            majority_label = entry.get('majority_label', 'normal')
            binary_label, multiclass_label = self.map_hatexplain_labels(majority_label)
            unified['label_binary'] = binary_label
            unified['label_multiclass'] = multiclass_label
            
            # Target group normalization (already validated above)
            unified['target_group_norm'] = target_group_norm
            unified['persona_tag'] = self.extract_persona_tag(target_raw, target_group_norm)
            
            # Source and synthetic flags
            unified['source_dataset'] = 'hatexplain'
            # Check if this entry was synthetically augmented (from future synthetic generation)
            unified['is_synthetic'] = entry.get('is_synthetic', False)
            
            # Rationale text
            rationale = entry.get('rationale_text', 'NA')
            unified['rationale_text'] = rationale if rationale not in ['NA', 'None', ''] else None
            
        elif source == 'toxigen':
            # ToxiGen field mapping
            unified['text'] = entry.get('text', '')
            
            # Label mapping - ToxiGen uses 'label_binary' field with values 'toxic'/'benign'
            label_binary = entry.get('label_binary', 'benign')
            binary_label, multiclass_label = self.map_toxigen_labels(label_binary)
            unified['label_binary'] = binary_label
            unified['label_multiclass'] = multiclass_label
            
            # Target group normalization (already validated above)
            unified['target_group_norm'] = target_group_norm
            unified['persona_tag'] = self.extract_persona_tag(target_raw, target_group_norm)
            
            # Source and synthetic flags
            unified['source_dataset'] = 'toxigen'
            unified['is_synthetic'] = False  # Real examples from ToxiGen dataset, not synthetically generated in this pipeline
            
            # No rationale available for ToxiGen
            unified['rationale_text'] = None
        
        # Create fine-tuning embedding
        unified['fine_tuning_embedding'] = self.create_fine_tuning_embedding(
            unified['text'],
            unified['target_group_norm'],
            unified['persona_tag'],
            unified['rationale_text'],
            source
        )
        
        # Add metadata
        unified['original_id'] = entry.get('post_id', entry.get('text_id', ''))
        unified['split'] = entry.get('split', 'unknown')
        
        return unified
    
    def unify_datasets(self):
        """Unify both datasets according to the minimal schema, filtering for selected target groups only."""
        logger.info("Unifying datasets for LGBTQ, Mexican, and Middle East target groups only...")
        
        if not self.hatexplain_data and not self.toxigen_data:
            raise ValueError("No datasets loaded. Call load_datasets() first.")
        
        # Process each split
        for split in ['train', 'val', 'test']:
            unified_split = []
            hatexplain_processed = 0
            toxigen_processed = 0
            hatexplain_filtered = 0
            toxigen_filtered = 0
            
            # Process HateXplain entries
            if split in self.hatexplain_data:
                for entry in self.hatexplain_data[split]:
                    hatexplain_processed += 1
                    unified_entry = self.unify_entry(entry, 'hatexplain')
                    if unified_entry is not None:
                        unified_split.append(unified_entry)
                        hatexplain_filtered += 1
            
            # Process ToxiGen entries
            if split in self.toxigen_data:
                for entry in self.toxigen_data[split]:
                    toxigen_processed += 1
                    unified_entry = self.unify_entry(entry, 'toxigen')
                    if unified_entry is not None:
                        unified_split.append(unified_entry)
                        toxigen_filtered += 1
            
            self.unified_data[split] = unified_split
            logger.info(f"Unified {split} split: {len(unified_split)} entries (filtered from {hatexplain_processed + toxigen_processed})")
            logger.info(f"  HateXplain: {hatexplain_filtered}/{hatexplain_processed} entries")
            logger.info(f"  ToxiGen: {toxigen_filtered}/{toxigen_processed} entries")
    
    def analyze_unified_dataset(self) -> UnifiedDatasetStats:
        """Analyze the unified dataset and generate statistics."""
        logger.info("Analyzing unified dataset...")
        
        if not self.unified_data:
            raise ValueError("No unified data available. Call unify_datasets() first.")
        
        # Combine all splits for analysis
        all_entries = []
        for split_data in self.unified_data.values():
            all_entries.extend(split_data)
        
        if not all_entries:
            raise ValueError("No entries found in unified dataset")
        
        # Calculate statistics
        total_entries = len(all_entries)
        hatexplain_entries = sum(1 for e in all_entries if e['source_dataset'] == 'hatexplain')
        toxigen_entries = sum(1 for e in all_entries if e['source_dataset'] == 'toxigen')
        
        # Label distributions
        label_binary_dist = {}
        label_multiclass_dist = {}
        for entry in all_entries:
            # Binary labels
            binary_label = entry['label_binary']
            label_binary_dist[binary_label] = label_binary_dist.get(binary_label, 0) + 1
            
            # Multiclass labels
            multiclass_label = entry['label_multiclass']
            label_multiclass_dist[multiclass_label] = label_multiclass_dist.get(multiclass_label, 0) + 1
        
        # Target group distribution
        target_group_dist = {}
        for entry in all_entries:
            target_group = entry['target_group_norm']
            target_group_dist[target_group] = target_group_dist.get(target_group, 0) + 1
        
        # Persona tag distribution
        persona_tag_dist = {}
        for entry in all_entries:
            persona_tag = entry['persona_tag']
            if persona_tag:
                persona_tag_dist[persona_tag] = persona_tag_dist.get(persona_tag, 0) + 1
        
        # Source distribution
        source_dist = {
            'hatexplain': hatexplain_entries,
            'toxigen': toxigen_entries
        }
        
        # Synthetic ratio
        synthetic_count = sum(1 for e in all_entries if e['is_synthetic'])
        synthetic_ratio = synthetic_count / total_entries if total_entries > 0 else 0.0
        
        # Average text length
        text_lengths = [len(e['text']) for e in all_entries if e['text']]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0.0
        
        # Rationale coverage
        rationale_count = sum(1 for e in all_entries if e['rationale_text'])
        rationale_coverage = rationale_count / total_entries if total_entries > 0 else 0.0
        
        stats = UnifiedDatasetStats(
            total_entries=total_entries,
            hatexplain_entries=hatexplain_entries,
            toxigen_entries=toxigen_entries,
            label_binary_distribution=label_binary_dist,
            label_multiclass_distribution=label_multiclass_dist,
            target_group_distribution=target_group_dist,
            persona_tag_distribution=persona_tag_dist,
            source_distribution=source_dist,
            synthetic_ratio=synthetic_ratio,
            avg_text_length=avg_text_length,
            rationale_coverage=rationale_coverage
        )
        
        logger.info("Dataset analysis completed")
        return stats
    
    def print_dataset_summary(self, stats: Optional[UnifiedDatasetStats] = None):
        """Print comprehensive summary of the unified dataset."""
        if stats is None:
            stats = self.analyze_unified_dataset()
        
        print("\n" + "="*70)
        print("UNIFIED DATASET SUMMARY")
        print("="*70)
        
        print(f"\nDataset Overview:")
        print(f"  Total entries: {stats.total_entries:,}")
        print(f"  HateXplain entries: {stats.hatexplain_entries:,} ({stats.hatexplain_entries/stats.total_entries*100:.1f}%)")
        print(f"  ToxiGen entries: {stats.toxigen_entries:,} ({stats.toxigen_entries/stats.total_entries*100:.1f}%)")
        print(f"  Synthetic ratio: {stats.synthetic_ratio:.1%}")
        print(f"  Average text length: {stats.avg_text_length:.1f} characters")
        print(f"  Rationale coverage: {stats.rationale_coverage:.1%}")
        
        print(f"\nBinary Label Distribution:")
        for label, count in sorted(stats.label_binary_distribution.items()):
            percentage = (count / stats.total_entries) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nMulticlass Label Distribution:")
        for label, count in sorted(stats.label_multiclass_distribution.items()):
            percentage = (count / stats.total_entries) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nTarget Group Distribution (Top 10):")
        sorted_groups = sorted(stats.target_group_distribution.items(), 
                             key=lambda x: x[1], reverse=True)
        for group, count in sorted_groups[:10]:
            percentage = (count / stats.total_entries) * 100
            print(f"  {group}: {count:,} ({percentage:.1f}%)")
        
        if stats.persona_tag_distribution:
            print(f"\nPersona Tag Distribution:")
            for tag, count in sorted(stats.persona_tag_distribution.items(), 
                                   key=lambda x: x[1], reverse=True):
                percentage = (count / stats.total_entries) * 100
                print(f"  {tag}: {count:,} ({percentage:.1f}%)")
        
        print("="*70)
    
    def export_unified_dataset(self, format: str = 'json'):
        """
        Export the unified dataset in the specified format.
        
        Args:
            format: Export format ('json', 'csv', 'parquet')
        """
        if not self.unified_data:
            raise ValueError("No unified data to export. Call unify_datasets() first.")
        
        logger.info(f"Exporting unified dataset in {format} format...")
        
        # Export each split
        for split, data in self.unified_data.items():
            if not data:
                continue
            
            if format == 'json':
                output_file = self.output_dir / f"unified_{split}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv' and HAS_PANDAS:
                output_file = self.output_dir / f"unified_{split}.csv"
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False, encoding='utf-8')
            
            elif format == 'parquet' and HAS_PANDAS:
                output_file = self.output_dir / f"unified_{split}.parquet"
                df = pd.DataFrame(data)
                df.to_parquet(output_file, index=False)
            
            else:
                if not HAS_PANDAS:
                    logger.warning(f"Pandas not available. Using JSON format instead of {format}")
                    format = 'json'
                    output_file = self.output_dir / f"unified_{split}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {split} split: {output_file} ({len(data)} entries)")
        
        # Export dataset statistics
        stats = self.analyze_unified_dataset()
        stats_file = self.output_dir / "unified_dataset_stats.json"
        
        stats_dict = {
            'total_entries': stats.total_entries,
            'hatexplain_entries': stats.hatexplain_entries,
            'toxigen_entries': stats.toxigen_entries,
            'label_binary_distribution': stats.label_binary_distribution,
            'label_multiclass_distribution': stats.label_multiclass_distribution,
            'target_group_distribution': stats.target_group_distribution,
            'persona_tag_distribution': stats.persona_tag_distribution,
            'source_distribution': stats.source_distribution,
            'synthetic_ratio': stats.synthetic_ratio,
            'avg_text_length': stats.avg_text_length,
            'rationale_coverage': stats.rationale_coverage
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported dataset statistics: {stats_file}")
        logger.info(f"Dataset unification completed. Output directory: {self.output_dir}")
    
    def balance_source_distribution(self, toxigen_multiplier: float = 1.0):
        """
        Balance the representation between HateXplain (persona hate) and ToxiGen (implicit hate)
        by equalizing or proportionally adjusting the number of samples from each source dataset.
        
        Args:
            toxigen_multiplier: How many times more ToxiGen samples than HateXplain
                               (1.0 = equal, 2.0 = double, 3.0 = triple, etc.)
        
        Returns:
            UnifiedDatasetStats: Updated statistics after balancing
        """
        logger.info(f"Balancing source distribution with 1:{toxigen_multiplier} HateXplain:ToxiGen ratio...")
        
        if not self.unified_data:
            raise ValueError("No unified data available. Call unify_datasets() first.")
        
        # Combine all unified data
        all_entries = []
        for split_data in self.unified_data.values():
            all_entries.extend(split_data)
        
        if not all_entries:
            raise ValueError("No entries found in unified dataset")
        
        # Separate by source
        hatexplain_entries = [e for e in all_entries if e['source_dataset'] == 'hatexplain']
        toxigen_entries = [e for e in all_entries if e['source_dataset'] == 'toxigen']
        
        logger.info(f"Current source distribution:")
        logger.info(f"  HateXplain (persona hate): {len(hatexplain_entries):,} samples")
        logger.info(f"  ToxiGen (implicit hate): {len(toxigen_entries):,} samples")
        logger.info(f"  Current ratio: 1:{len(toxigen_entries)/len(hatexplain_entries) if hatexplain_entries else 0:.1f}")
        
        # Calculate target sizes
        hatexplain_samples = len(hatexplain_entries)
        toxigen_target = int(hatexplain_samples * toxigen_multiplier)
        total_target = hatexplain_samples + toxigen_target
        
        logger.info(f"\nTarget distribution:")
        logger.info(f"  HateXplain: {hatexplain_samples:,} samples ({100/(1+toxigen_multiplier):.1f}%)")
        logger.info(f"  ToxiGen: {toxigen_target:,} samples ({100*toxigen_multiplier/(1+toxigen_multiplier):.1f}%)")
        logger.info(f"  Total: {total_target:,} samples")
        
        # Keep all HateXplain entries (valuable persona hate with rationales)
        balanced_hatexplain = hatexplain_entries.copy()
        
        # Stratified ToxiGen sampling maintaining target group and class balance
        balanced_toxigen = []
        
        # Get current ToxiGen target group distribution
        toxigen_by_group = {}
        for entry in toxigen_entries:
            group = entry['target_group_norm']
            if group not in toxigen_by_group:
                toxigen_by_group[group] = []
            toxigen_by_group[group].append(entry)
        
        import random
        random.seed(42)  # For reproducible results
        
        logger.info(f"\nStratified ToxiGen sampling:")
        
        for target_group in ['lgbtq', 'mexican', 'middle_east']:
            group_entries = toxigen_by_group.get(target_group, [])
            
            if not group_entries:
                logger.warning(f"  {target_group}: No entries found, skipping")
                continue
            
            # Calculate proportional quota for this group
            group_proportion = len(group_entries) / len(toxigen_entries)
            group_quota = int(toxigen_target * group_proportion)
            
            # Separate by class within group
            group_hate = [e for e in group_entries if e['label_binary'] == 'hate']
            group_normal = [e for e in group_entries if e['label_binary'] == 'normal']
            
            # Maintain class balance within group (50/50 split)
            hate_quota = group_quota // 2
            normal_quota = group_quota - hate_quota
            
            # Sample with replacement if necessary to meet quotas
            if len(group_hate) >= hate_quota:
                sampled_hate = random.sample(group_hate, hate_quota)
            else:
                sampled_hate = group_hate.copy()
                logger.warning(f"  {target_group}: Only {len(group_hate)} hate samples available (needed {hate_quota})")
            
            if len(group_normal) >= normal_quota:
                sampled_normal = random.sample(group_normal, normal_quota)
            else:
                sampled_normal = group_normal.copy()
                logger.warning(f"  {target_group}: Only {len(group_normal)} normal samples available (needed {normal_quota})")
            
            balanced_toxigen.extend(sampled_hate + sampled_normal)
            
            logger.info(f"  {target_group}: {len(sampled_hate + sampled_normal):,} samples "
                       f"({len(sampled_hate)} hate, {len(sampled_normal)} normal)")
        
        # Combine balanced sources
        balanced_entries = balanced_hatexplain + balanced_toxigen
        
        logger.info(f"\nBalanced dataset summary:")
        logger.info(f"  HateXplain: {len(balanced_hatexplain):,} samples ({len(balanced_hatexplain)/len(balanced_entries)*100:.1f}%)")
        logger.info(f"  ToxiGen: {len(balanced_toxigen):,} samples ({len(balanced_toxigen)/len(balanced_entries)*100:.1f}%)")
        logger.info(f"  Total: {len(balanced_entries):,} samples")
        logger.info(f"  Reduction: {len(all_entries) - len(balanced_entries):,} samples removed ({(len(all_entries) - len(balanced_entries))/len(all_entries)*100:.1f}%)")
        
        # Redistribute back to splits maintaining original split ratios
        self._redistribute_to_splits(balanced_entries, all_entries)
        
        # Analyze and return updated statistics
        final_stats = self.analyze_unified_dataset()
        
        logger.info(f"\nFinal source distribution:")
        for source, count in final_stats.source_distribution.items():
            percentage = (count / final_stats.total_entries) * 100
            logger.info(f"  {source}: {count:,} samples ({percentage:.1f}%)")
        
        logger.info(f"Final class distribution:")
        for label, count in final_stats.label_binary_distribution.items():
            percentage = (count / final_stats.total_entries) * 100
            logger.info(f"  {label}: {count:,} samples ({percentage:.1f}%)")
        
        return final_stats
    
    def _redistribute_to_splits(self, balanced_entries: List[Dict[str, Any]], original_entries: List[Dict[str, Any]]):
        """
        Redistribute balanced entries back to train/val/test splits maintaining original ratios.
        
        Args:
            balanced_entries: List of balanced unified entries
            original_entries: List of original unified entries for ratio calculation
        """
        # Calculate original split ratios
        original_split_counts = {}
        for split, data in self.unified_data.items():
            original_split_counts[split] = len(data)
        
        total_original = len(original_entries)
        original_split_ratios = {
            split: count / total_original 
            for split, count in original_split_counts.items()
            if total_original > 0
        }
        
        logger.info(f"Redistributing to splits with original ratios: {original_split_ratios}")
        
        # Shuffle balanced entries for random distribution
        import random
        random.shuffle(balanced_entries)
        
        # Redistribute to splits
        start_idx = 0
        for split, ratio in original_split_ratios.items():
            split_size = int(len(balanced_entries) * ratio)
            end_idx = start_idx + split_size
            
            # Handle final split to include any remaining entries
            if split == list(original_split_ratios.keys())[-1]:
                end_idx = len(balanced_entries)
            
            self.unified_data[split] = balanced_entries[start_idx:end_idx]
            logger.info(f"  {split}: {len(self.unified_data[split]):,} samples ({len(self.unified_data[split])/len(balanced_entries)*100:.1f}%)")
            start_idx = end_idx


def main():
    """Main function to demonstrate dataset unification with source balancing."""
    # Initialize unifier
    unifier = DatasetUnifier(
        hatexplain_dir="data/processed/hatexplain",
        toxigen_dir="data/processed/toxigen"
    )
    
    # Load datasets
    unifier.load_datasets()
    
    # Unify datasets
    unifier.unify_datasets()
    
    # Print original summary
    print("\n" + "="*70)
    print("ORIGINAL UNIFIED DATASET")
    print("="*70)
    unifier.print_dataset_summary()
    
    # Balance source distribution (1:1 ratio by default)
    print("\n" + "="*70)
    print("BALANCING SOURCE DISTRIBUTION (1:1 RATIO)")
    print("="*70)
    balanced_stats = unifier.balance_source_distribution(toxigen_multiplier=1.0)
    
    # Print balanced summary
    print("\n" + "="*70)
    print("BALANCED UNIFIED DATASET")
    print("="*70)
    unifier.print_dataset_summary(balanced_stats)
    
    # Export balanced unified dataset
    unifier.export_unified_dataset(format='json')
    
    print("\n" + "*" * 60)
    print("*** Dataset unification and balancing completed successfully! ***")
    print(f"*** Final dataset: {balanced_stats.total_entries:,} samples ***")
    print(f"*** HateXplain: {balanced_stats.hatexplain_entries:,} samples ({balanced_stats.hatexplain_entries/balanced_stats.total_entries*100:.1f}%) ***")
    print(f"*** ToxiGen: {balanced_stats.toxigen_entries:,} samples ({balanced_stats.toxigen_entries/balanced_stats.total_entries*100:.1f}%) ***")
    print(f"*** Rationale coverage: {balanced_stats.rationale_coverage:.1%} ***")
    print("*" * 60)


if __name__ == "__main__":
    main()
