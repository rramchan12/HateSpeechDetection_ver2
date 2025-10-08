#!/usr/bin/env python3
"""
Dataset Sampler for Hate Speech Detection
==========================================

Creates representative samples from the unified training dataset for systematic
prompt testing. Uses stratified sampling to maintain distribution balance across
target groups, labels, and data sources.

Usage:
    python dataset_sampler.py --size 100 --output canned_100_stratified.json
    python dataset_sampler.py --size 50 --output canned_50_quick.json
    python dataset_sampler.py --size 200 --output canned_200_comprehensive.json
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import sys
import os

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class StratifiedSampler:
    """
    Stratified sampler that maintains distribution balance across multiple dimensions:
    - Target groups (LGBTQ, Middle East, Mexican)  
    - Binary labels (hate/normal)
    - Source datasets (HatEXplain/ToxiGen)
    - Text length variety
    """
    
    def __init__(self, data_path: str):
        """Initialize sampler with training data."""
        self.data_path = Path(data_path)
        self.data = self._load_data()
        self.stats = self._calculate_stats()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load the unified training dataset."""
        print(f"Loading training data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data):,} training samples")
        return data
        
    def _calculate_stats(self) -> Dict[str, Any]:
        """Calculate dataset statistics for informed sampling."""
        stats = {
            'total_size': len(self.data),
            'target_groups': Counter(),
            'labels': Counter(), 
            'sources': Counter(),
            'combined': defaultdict(int)
        }
        
        for sample in self.data:
            target = sample['target_group_norm']
            label = sample['label_binary']
            source = sample['source_dataset']
            
            stats['target_groups'][target] += 1
            stats['labels'][label] += 1
            stats['sources'][source] += 1
            stats['combined'][f"{target}||{label}"] += 1
            
        return stats
    
    def _print_stats(self):
        """Print dataset distribution statistics."""
        print("\nDataset Distribution Analysis:")
        print("=" * 50)
        
        total = self.stats['total_size']
        print(f"Total samples: {total:,}")
        
        print(f"\nTarget Groups:")
        for group, count in self.stats['target_groups'].most_common():
            pct = (count / total) * 100
            print(f"  - {group}: {count:,} ({pct:.1f}%)")
            
        print(f"\nBinary Labels:")
        for label, count in self.stats['labels'].items():
            pct = (count / total) * 100
            print(f"  - {label}: {count:,} ({pct:.1f}%)")
            
        print(f"\nData Sources:")
        for source, count in self.stats['sources'].items():
            pct = (count / total) * 100
            print(f"  - {source}: {count:,} ({pct:.1f}%)")
    
    def create_stratified_sample(self, sample_size: int, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Create a stratified sample maintaining proportional representation.
        
        Args:
            sample_size: Target number of samples
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled data points
        """
        print(f"\nCreating stratified sample of size {sample_size}")
        random.seed(random_seed)
        
        # Group data by target_group + label combination for stratified sampling
        strata = defaultdict(list)
        for sample in self.data:
            key = f"{sample['target_group_norm']}||{sample['label_binary']}"
            strata[key].append(sample)
            
        # Calculate proportional allocation
        total_samples = len(self.data)
        sample_allocation = {}
        allocated_total = 0
        
        for stratum, samples in strata.items():
            proportion = len(samples) / total_samples
            allocated = max(1, round(sample_size * proportion))  # At least 1 sample per stratum
            sample_allocation[stratum] = min(allocated, len(samples))  # Don't exceed available samples
            allocated_total += sample_allocation[stratum]
            
        # Adjust if we're over/under the target due to rounding
        if allocated_total != sample_size:
            # Adjust the largest strata first
            sorted_strata = sorted(sample_allocation.items(), key=lambda x: len(strata[x[0]]), reverse=True)
            diff = sample_size - allocated_total
            
            for stratum, current_alloc in sorted_strata:
                if diff == 0:
                    break
                    
                available = len(strata[stratum]) - current_alloc
                if diff > 0 and available > 0:
                    # Need more samples
                    add = min(diff, available)
                    sample_allocation[stratum] += add
                    diff -= add
                elif diff < 0 and current_alloc > 1:
                    # Need fewer samples  
                    remove = min(-diff, current_alloc - 1)
                    sample_allocation[stratum] -= remove
                    diff += remove
        
        # Sample from each stratum
        final_sample = []
        print(f"\nSampling Strategy:")
        
        for stratum, allocation in sample_allocation.items():
            stratum_samples = random.sample(strata[stratum], allocation)
            final_sample.extend(stratum_samples)
            
            target_group, label = stratum.split('||')
            original_count = len(strata[stratum])
            pct_sampled = (allocation / original_count) * 100
            print(f"  - {target_group} + {label}: {allocation}/{original_count} ({pct_sampled:.1f}%)")
            
        # Final shuffle to mix strata
        random.shuffle(final_sample)
        
        print(f"\nCreated sample with {len(final_sample)} examples")
        return final_sample
    
    def create_size_varied_sample(self, sample_size: int, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Create sample with varied text lengths for comprehensive testing.
        """
        print(f"\nCreating size-varied sample of {sample_size} examples")
        
        # First get stratified sample
        stratified_sample = self.create_stratified_sample(sample_size * 2, random_seed)  # Get more to choose from
        
        # Categorize by text length
        short_texts = []    # < 50 chars
        medium_texts = []   # 50-150 chars  
        long_texts = []     # > 150 chars
        
        for sample in stratified_sample:
            text_len = len(sample['text'])
            if text_len < 50:
                short_texts.append(sample)
            elif text_len <= 150:
                medium_texts.append(sample)
            else:
                long_texts.append(sample)
        
        # Target distribution: 30% short, 50% medium, 20% long
        target_short = int(sample_size * 0.3)
        target_medium = int(sample_size * 0.5)  
        target_long = sample_size - target_short - target_medium
        
        random.seed(random_seed)
        final_sample = []
        
        # Sample from each length category
        if short_texts and target_short > 0:
            final_sample.extend(random.sample(short_texts, min(target_short, len(short_texts))))
        if medium_texts and target_medium > 0:
            final_sample.extend(random.sample(medium_texts, min(target_medium, len(medium_texts))))
        if long_texts and target_long > 0:
            final_sample.extend(random.sample(long_texts, min(target_long, len(long_texts))))
            
        # Fill remaining slots if needed
        remaining = sample_size - len(final_sample)
        if remaining > 0:
            all_remaining = [s for s in stratified_sample if s not in final_sample]
            if all_remaining:
                final_sample.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        
        random.shuffle(final_sample)
        print(f"Created size-varied sample: {len(final_sample)} examples")
        return final_sample[:sample_size]
    
    def validate_sample(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate sample maintains good distribution properties."""
        sample_stats = {
            'size': len(sample),
            'target_groups': Counter(),
            'labels': Counter(),
            'sources': Counter()
        }
        
        for item in sample:
            sample_stats['target_groups'][item['target_group_norm']] += 1
            sample_stats['labels'][item['label_binary']] += 1  
            sample_stats['sources'][item['source_dataset']] += 1
            
        return sample_stats
    
    def print_sample_validation(self, sample: List[Dict[str, Any]]):
        """Print sample validation statistics."""
        stats = self.validate_sample(sample)
        total = stats['size']
        
        print(f"\nSample Validation Report:")
        print("=" * 40)
        print(f"Sample size: {total}")
        
        print(f"\nTarget Group Distribution:")
        for group, count in stats['target_groups'].most_common():
            pct = (count / total) * 100
            orig_pct = (self.stats['target_groups'][group] / self.stats['total_size']) * 100
            diff = pct - orig_pct
            status = "OK" if abs(diff) < 5 else "WARNING"
            print(f"  {status}: {group}: {count} ({pct:.1f}%) [orig: {orig_pct:.1f}%, diff: {diff:+.1f}%]")
            
        print(f"\nLabel Distribution:")
        for label, count in stats['labels'].items():
            pct = (count / total) * 100
            orig_pct = (self.stats['labels'][label] / self.stats['total_size']) * 100
            diff = pct - orig_pct
            status = "OK" if abs(diff) < 10 else "WARNING"
            print(f"  {status}: {label}: {count} ({pct:.1f}%) [orig: {orig_pct:.1f}%, diff: {diff:+.1f}%]")


def main():
    """Main sampling workflow."""
    parser = argparse.ArgumentParser(description="Create representative samples from unified training data")
    parser.add_argument("--size", type=int, default=100, 
                       help="Sample size (default: 100)")
    parser.add_argument("--output", type=str, default="canned_100_stratified.json",
                       help="Output filename (default: canned_100_stratified.json)")
    parser.add_argument("--method", choices=['stratified', 'size_varied'], default='stratified',
                       help="Sampling method (default: stratified)")  
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--stats-only", action='store_true',
                       help="Only print dataset statistics, don't create sample")
    
    args = parser.parse_args()
    
    # Paths
    data_path = Path(__file__).parent.parent / "data" / "processed" / "unified" / "unified_train.json"
    output_dir = Path(__file__).parent / "data_samples"
    output_path = output_dir / args.output
    
    # Validate paths
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        return 1
        
    output_dir.mkdir(exist_ok=True)
    
    # Initialize sampler
    print("Initializing Dataset Sampler")
    print("=" * 50)
    sampler = StratifiedSampler(str(data_path))
    sampler._print_stats()
    
    if args.stats_only:
        print(f"\nStats-only mode complete.")
        return 0
    
    # Create sample
    print(f"\nCreating {args.method} sample...")
    
    if args.method == 'stratified':
        sample = sampler.create_stratified_sample(args.size, args.seed)
    else:  # size_varied
        sample = sampler.create_size_varied_sample(args.size, args.seed)
    
    # Validate sample
    sampler.print_sample_validation(sample)
    
    # Save sample
    print(f"\nSaving sample to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"Sample saved successfully!")
    print(f"\nUsage: python prompt_runner.py --data-source {args.output.replace('.json', '')} --sample-size {args.size}")
    
    return 0


if __name__ == "__main__":
    exit(main())