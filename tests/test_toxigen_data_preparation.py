#!/usr/bin/env python3
"""
Unit tests for ToxiGen data preparation pipeline.

This module contains comprehensive tests for the ToxiGen data preparation,
including data loading, cleaning, feature extraction, and export functionality.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preparation.data_preparation_toxigen import ToxiGenDataProcessor, DatasetStats


class TestToxiGenDataPreparation(unittest.TestCase):
    """Test cases for ToxiGen data preparation pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.processor = ToxiGenDataProcessor()
        
    def test_data_loading(self):
        """Test that data loading works correctly."""
        # Load data
        data = self.processor.load_raw_data()
        
        # Check that data was loaded
        self.assertIsNotNone(data)
        self.assertIn('train_data', data)
        self.assertIn('annotated_data', data)
        self.assertIn('label_encoders', data)
        
        # Check that training data exists
        self.assertIsNotNone(self.processor.train_data)
        self.assertGreater(len(self.processor.train_data), 0)
        
        # Check that annotated data was created
        self.assertIsNotNone(self.processor.annotated_data)
        self.assertGreater(len(self.processor.annotated_data), 0)
        
        print(f" Data loading test passed - {len(self.processor.train_data)} records loaded")
    
    def test_label_encoders(self):
        """Test that label encoders are set up correctly."""
        # Ensure data is loaded
        if self.processor.train_data is None:
            self.processor.load_raw_data()
        
        # Check that encoders exist (based on train.parquet fields)
        self.assertIn('target_group', self.processor.label_encoders)
        self.assertIn('label_binary', self.processor.label_encoders)
        self.assertIn('generation_method', self.processor.label_encoders)
        
        # Check that encoders have classes
        target_encoder = self.processor.label_encoders['target_group']
        self.assertIsNotNone(target_encoder.classes_)
        self.assertGreater(len(target_encoder.classes_), 0)
        
        # Check binary label encoder
        label_encoder = self.processor.label_encoders['label_binary']
        self.assertIsNotNone(label_encoder.classes_)
        self.assertIn('toxic', label_encoder.classes_)
        self.assertIn('benign', label_encoder.classes_)
        
        print(f" Label encoders test passed - {len(target_encoder.classes_)} target groups")
    
    def test_data_quality_analysis(self):
        """Test data quality analysis functionality."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Perform analysis
        stats = self.processor.analyze_data_quality()
        
        # Check that stats are valid
        self.assertIsInstance(stats, DatasetStats)
        self.assertGreater(stats.total_entries, 0)
        self.assertIsInstance(stats.label_distribution, dict)
        self.assertIsInstance(stats.group_distribution, dict)
        self.assertGreater(stats.text_length_mean, 0)
        self.assertGreater(stats.text_length_median, 0)
        self.assertGreater(stats.text_length_std, 0)
        
        # Check that distributions sum correctly
        total_labels = sum(stats.label_distribution.values())
        self.assertEqual(total_labels, stats.total_entries)
        
        total_groups = sum(stats.group_distribution.values())
        self.assertEqual(total_groups, stats.total_entries)
        
        # Check for expected labels (based on prompt_label)
        self.assertIn('toxic', stats.label_distribution)
        self.assertIn('benign', stats.label_distribution)
        
        print(f" Data quality analysis test passed - {stats.total_entries} entries analyzed")
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Get original data size
        try:
            import pandas as pd
            if hasattr(self.processor.annotated_data, 'shape'):
                original_size = len(self.processor.annotated_data)
            else:
                original_size = len(self.processor.annotated_data)
        except:
            original_size = len(self.processor.annotated_data)
        
        # Perform cleaning
        cleaning_stats = self.processor.clean_text_data()
        
        # Check cleaning stats
        self.assertIsInstance(cleaning_stats, dict)
        self.assertIn('entries_processed', cleaning_stats)
        self.assertIn('empty_texts_found', cleaning_stats)
        self.assertGreater(cleaning_stats['entries_processed'], 0)
        
        print(f" Text cleaning test passed - {cleaning_stats['entries_processed']} entries processed")
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        # Ensure data is loaded and cleaned
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
            self.processor.clean_text_data()
        
        # Extract features
        features = self.processor.extract_features()
        
        # Check that features were extracted
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
        
        # Check feature structure
        if hasattr(features, 'columns'):
            # DataFrame
            expected_features = [
                'text_id', 'text', 'text_length', 'word_count', 'target_group',
                'label_binary'  # Updated to match new field name
            ]
            for feature in expected_features:
                self.assertIn(feature, features.columns)
            
            # Check for train.parquet specific features
            self.assertIn('generation_method', features.columns)
            
            print(f" Feature extraction test passed - {features.shape[1]} features extracted")
        else:
            # List of dicts
            expected_features = [
                'text_id', 'text', 'text_length', 'word_count', 'target_group',
                'toxicity_ai', 'toxicity_human', 'stereotyping', 'toxicity_binary'
            ]
            sample = features[0]
            for feature in expected_features:
                self.assertIn(feature, sample)
            
            print(f" Feature extraction test passed - {len(sample)} features extracted")
    
    def test_training_data_preparation(self):
        """Test training data preparation and splitting."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Prepare training data
        train_data, val_data, test_data = self.processor.prepare_training_data()
        
        # Check that splits exist
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(val_data)
        self.assertIsNotNone(test_data)
        
        # Check that splits have data
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertGreater(len(test_data), 0)
        
        # Check that total equals original
        total_split = len(train_data) + len(val_data) + len(test_data)
        original_total = len(self.processor.annotated_data)
        self.assertEqual(total_split, original_total)
        
        print(f" Training data preparation test passed - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def test_export_functionality(self):
        """Test data export functionality."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export data
            self.processor.export_processed_data(output_dir=temp_dir, format='json')
            
            # Check that files were created
            temp_path = Path(temp_dir)
            
            train_file = temp_path / "toxigen_train.json"
            val_file = temp_path / "toxigen_val.json"
            test_file = temp_path / "toxigen_test.json"
            stats_file = temp_path / "toxigen_dataset_stats.json"
            
            self.assertTrue(train_file.exists())
            self.assertTrue(val_file.exists())
            self.assertTrue(test_file.exists())
            self.assertTrue(stats_file.exists())
            
            # Check that files have content
            with open(train_file, 'r') as f:
                train_data = json.load(f)
                self.assertGreater(len(train_data), 0)
            
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
                self.assertIn('total_entries', stats_data)
                self.assertIn('label_distribution', stats_data)
                self.assertIn('group_distribution', stats_data)
            
            print(f" Export functionality test passed - Files created in {temp_dir}")
    
    def test_synthetic_field_creation(self):
        """Test that synthetic fields are created correctly from train.parquet data."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Check synthetic fields (based on train.parquet)
        try:
            import pandas as pd
            if hasattr(self.processor.annotated_data, 'columns'):
                # DataFrame version
                self.assertIn('text', self.processor.annotated_data.columns)
                self.assertIn('target_group', self.processor.annotated_data.columns)
                self.assertIn('label_binary', self.processor.annotated_data.columns)
                self.assertIn('generation_method', self.processor.annotated_data.columns)
                
                # Check that values are reasonable
                sample_row = self.processor.annotated_data.iloc[0]
                self.assertGreater(len(sample_row['text']), 0)
                self.assertIn(sample_row['target_group'], self.processor.label_encoders['target_group'].classes_)
                self.assertIn(sample_row['label_binary'], ['toxic', 'benign'])
                
                print(" Synthetic field creation test passed - Fields created from train.parquet")
            else:
                # List version
                sample = self.processor.annotated_data[0]
                self.assertIn('text', sample)
                self.assertIn('target_group', sample)
                self.assertIn('label_binary', sample)
                
                # Check that values are reasonable
                self.assertGreater(len(sample['text']), 0)
                self.assertGreaterEqual(sample['toxicity_human'], 0.0)
                self.assertLessEqual(sample['toxicity_human'], 5.0)
        except ImportError:
            # No pandas, test list version
            sample = self.processor.annotated_data[0]
            self.assertIn('text', sample)
            self.assertIn('target_group', sample)
            self.assertIn('toxicity_human', sample)
            self.assertIn('stereotyping', sample)
        
        print(f" Synthetic field creation test passed")
    
    def test_text_feature_extraction(self):
        """Test that text-based features are extracted correctly."""
        # Ensure data is loaded
        if self.processor.annotated_data is None:
            self.processor.load_raw_data()
        
        # Extract features
        features = self.processor.extract_features()
        
        # Check text-based features
        if hasattr(features, 'iloc'):
            # DataFrame version
            sample = features.iloc[0]
            self.assertIn('text_length', sample.index)
            self.assertIn('word_count', sample.index)
            self.assertIn('avg_word_length', sample.index)
            self.assertIn('caps_ratio', sample.index)
            self.assertIn('sentence_count', sample.index)
            
            # Check that values are reasonable
            self.assertGreater(sample['text_length'], 0)
            self.assertGreater(sample['word_count'], 0)
            self.assertGreater(sample['avg_word_length'], 0)
            self.assertGreaterEqual(sample['caps_ratio'], 0.0)
            self.assertLessEqual(sample['caps_ratio'], 1.0)
        else:
            # List version
            sample = features[0]
            self.assertIn('text_length', sample)
            self.assertIn('word_count', sample)
            self.assertIn('avg_word_length', sample)
            self.assertIn('caps_ratio', sample)
            self.assertIn('sentence_count', sample)
            
            # Check that values are reasonable
            self.assertGreater(sample['text_length'], 0)
            self.assertGreater(sample['word_count'], 0)
            self.assertGreater(sample['avg_word_length'], 0)
            self.assertGreaterEqual(sample['caps_ratio'], 0.0)
            self.assertLessEqual(sample['caps_ratio'], 1.0)
        
        print(f" Text feature extraction test passed")


class TestToxiGenIntegration(unittest.TestCase):
    """Integration tests for the full ToxiGen pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete data preparation pipeline."""
        print("\n" + "="*50)
        print("RUNNING FULL TOXIGEN PIPELINE INTEGRATION TEST")
        print("="*50)
        
        # Initialize processor
        processor = ToxiGenDataProcessor()
        
        # Step 1: Load data
        print("Step 1: Loading data...")
        data = processor.load_raw_data()
        self.assertIsNotNone(data)
        print(f" Loaded {len(processor.train_data)} training records")
        
        # Step 2: Analyze quality
        print("Step 2: Analyzing data quality...")
        stats = processor.analyze_data_quality()
        self.assertGreater(stats.total_entries, 0)
        print(f" Analyzed {stats.total_entries} entries")
        
        # Step 3: Clean text
        print("Step 3: Cleaning text...")
        cleaning_stats = processor.clean_text_data()
        self.assertGreater(cleaning_stats['entries_processed'], 0)
        print(f" Cleaned {cleaning_stats['entries_processed']} text entries")
        
        # Step 4: Extract features
        print("Step 4: Extracting features...")
        features = processor.extract_features()
        self.assertGreater(len(features), 0)
        print(f" Extracted features for {len(features)} entries")
        
        # Step 5: Prepare training data
        print("Step 5: Preparing training splits...")
        train_data, val_data, test_data = processor.prepare_training_data()
        total_samples = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_samples, stats.total_entries)
        print(f" Prepared splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        print("\n" + "="*50)
        print("FULL PIPELINE INTEGRATION TEST PASSED")
        print("="*50)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add individual tests
    suite.addTest(TestToxiGenDataPreparation('test_data_loading'))
    suite.addTest(TestToxiGenDataPreparation('test_label_encoders'))
    suite.addTest(TestToxiGenDataPreparation('test_synthetic_field_creation'))
    suite.addTest(TestToxiGenDataPreparation('test_data_quality_analysis'))
    suite.addTest(TestToxiGenDataPreparation('test_text_cleaning'))
    suite.addTest(TestToxiGenDataPreparation('test_feature_extraction'))
    suite.addTest(TestToxiGenDataPreparation('test_text_feature_extraction'))
    suite.addTest(TestToxiGenDataPreparation('test_training_data_preparation'))
    suite.addTest(TestToxiGenDataPreparation('test_export_functionality'))
    
    # Add integration test
    suite.addTest(TestToxiGenIntegration('test_full_pipeline'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TOXIGEN DATA PREPARATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL TESTS PASSED! ToxiGen data preparation pipeline is working correctly.")
    else:
        print(f"\n {len(result.failures + result.errors)} test(s) failed.")
    
    print(f"{'='*60}")