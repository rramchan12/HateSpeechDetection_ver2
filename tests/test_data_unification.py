"""
Unit Tests for Data Unification Module

This module contains comprehensive unit tests for the DatasetUnifier class
and its methods, ensuring robust dataset merging functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.data_unification import DatasetUnifier, UnifiedDatasetStats


class TestDatasetUnifier:
    """Test cases for the DatasetUnifier class initialization and basic functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_hatexplain_dir = "test_hatexplain"
        self.test_toxigen_dir = "test_toxigen"
        self.test_output_dir = "test_output"
        
        self.unifier = DatasetUnifier(
            hatexplain_dir=self.test_hatexplain_dir,
            toxigen_dir=self.test_toxigen_dir,
            output_dir=self.test_output_dir
        )
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test DatasetUnifier initialization."""
        assert str(self.unifier.hatexplain_dir) == self.test_hatexplain_dir
        assert str(self.unifier.toxigen_dir) == self.test_toxigen_dir
        assert str(self.unifier.output_dir) == self.test_output_dir
        
        # Test data containers are initialized as empty
        assert self.unifier.hatexplain_data == {}
        assert self.unifier.toxigen_data == {}
        assert self.unifier.unified_data == {}
    
    @pytest.mark.unit
    def test_initialization_default_output_dir(self):
        """Test DatasetUnifier initialization with default output directory."""
        unifier = DatasetUnifier(
            hatexplain_dir=self.test_hatexplain_dir,
            toxigen_dir=self.test_toxigen_dir
        )
        # Use Path to handle Windows/Unix path differences
        assert str(unifier.output_dir) == str(Path("data/processed/unified"))
    
    @pytest.mark.unit
    def test_constants(self):
        """Test that class constants are properly defined."""
        # Test target group normalization mapping
        assert 'Arab' in self.unifier.TARGET_GROUP_NORMALIZATION
        assert self.unifier.TARGET_GROUP_NORMALIZATION['Arab'] == 'middle_east'
        assert self.unifier.TARGET_GROUP_NORMALIZATION['Homosexual'] == 'lgbtq'
        assert self.unifier.TARGET_GROUP_NORMALIZATION['Hispanic'] == 'mexican'
        
        # Test valid target groups
        expected_valid_groups = {'lgbtq', 'mexican', 'middle_east'}
        assert self.unifier.VALID_TARGET_GROUPS == expected_valid_groups
        
        # Test top persona tags
        expected_persona_tags = {'lgbtq', 'mexican', 'middle_east'}
        assert self.unifier.TOP_PERSONA_TAGS == expected_persona_tags


class TestNormalizeTargetGroup:
    """Test cases for the normalize_target_group method."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
    
    @pytest.mark.unit
    def test_normalize_target_group_hatexplain_valid(self):
        """Test target group normalization for valid HateXplain groups."""
        # Test Arab -> middle_east
        result = self.unifier.normalize_target_group("Arab", "hatexplain")
        assert result == "middle_east"
        
        # Test Homosexual -> lgbtq
        result = self.unifier.normalize_target_group("Homosexual", "hatexplain")
        assert result == "lgbtq"
        
        # Test Gay -> lgbtq
        result = self.unifier.normalize_target_group("Gay", "hatexplain")
        assert result == "lgbtq"
        
        # Test Hispanic -> mexican
        result = self.unifier.normalize_target_group("Hispanic", "hatexplain")
        assert result == "mexican"
        
        # Test Latino -> mexican
        result = self.unifier.normalize_target_group("Latino", "hatexplain")
        assert result == "mexican"
    
    @pytest.mark.unit
    def test_normalize_target_group_toxigen_valid(self):
        """Test target group normalization for valid ToxiGen groups."""
        # Test direct mappings
        result = self.unifier.normalize_target_group("lgbtq", "toxigen")
        assert result == "lgbtq"
        
        result = self.unifier.normalize_target_group("mexican", "toxigen")
        assert result == "mexican"
        
        result = self.unifier.normalize_target_group("middle_east", "toxigen")
        assert result == "middle_east"
    
    @pytest.mark.unit
    def test_normalize_target_group_invalid(self):
        """Test target group normalization for invalid groups."""
        # Test groups not in our selected set
        result = self.unifier.normalize_target_group("Women", "hatexplain")
        assert result is None
        
        result = self.unifier.normalize_target_group("African American", "hatexplain")
        assert result is None
        
        result = self.unifier.normalize_target_group("Jewish", "hatexplain")
        assert result is None
        
        # Test ToxiGen invalid groups
        result = self.unifier.normalize_target_group("women", "toxigen")
        assert result is None
        
        result = self.unifier.normalize_target_group("black", "toxigen")
        assert result is None
    
    @pytest.mark.unit
    def test_normalize_target_group_none_empty(self):
        """Test target group normalization for None and empty values."""
        # Test None
        result = self.unifier.normalize_target_group(None, "hatexplain")
        assert result is None
        
        # Test empty string
        result = self.unifier.normalize_target_group("", "hatexplain")
        assert result is None
        
        # Test 'None' string
        result = self.unifier.normalize_target_group("None", "hatexplain")
        assert result is None
        
        # Test 'none' string
        result = self.unifier.normalize_target_group("none", "hatexplain")
        assert result is None
        
        # Test 'null' string
        result = self.unifier.normalize_target_group("null", "hatexplain")
        assert result is None
    
    @pytest.mark.unit
    def test_normalize_target_group_whitespace(self):
        """Test target group normalization with whitespace."""
        # Test with leading/trailing whitespace
        result = self.unifier.normalize_target_group("  Arab  ", "hatexplain")
        assert result == "middle_east"
        
        result = self.unifier.normalize_target_group(" Homosexual ", "hatexplain")
        assert result == "lgbtq"
        
        result = self.unifier.normalize_target_group("  lgbtq  ", "toxigen")
        assert result == "lgbtq"


class TestExtractPersonaTag:
    """Test cases for the extract_persona_tag method."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
    
    @pytest.mark.unit
    def test_extract_persona_tag_valid_lgbtq(self):
        """Test persona tag extraction for valid LGBTQ target group."""
        result = self.unifier.extract_persona_tag("Homosexual", "lgbtq")
        assert result == "homosexual"
        
        result = self.unifier.extract_persona_tag("Gay", "lgbtq")
        assert result == "gay"
        
        result = self.unifier.extract_persona_tag("LGBTQ", "lgbtq")
        assert result == "lgbtq"
    
    @pytest.mark.unit
    def test_extract_persona_tag_valid_mexican(self):
        """Test persona tag extraction for valid Mexican target group."""
        result = self.unifier.extract_persona_tag("Hispanic", "mexican")
        assert result == "hispanic"
        
        result = self.unifier.extract_persona_tag("Latino", "mexican")
        assert result == "latino"
        
        result = self.unifier.extract_persona_tag("Mexican", "mexican")
        assert result == "mexican"
    
    @pytest.mark.unit
    def test_extract_persona_tag_valid_middle_east(self):
        """Test persona tag extraction for valid Middle East target group."""
        result = self.unifier.extract_persona_tag("Arab", "middle_east")
        assert result == "arab"
        
        result = self.unifier.extract_persona_tag("Middle Eastern", "middle_east")
        assert result == "middle eastern"
    
    @pytest.mark.unit
    def test_extract_persona_tag_invalid_target_group_norm(self):
        """Test persona tag extraction with invalid normalized target groups."""
        result = self.unifier.extract_persona_tag("Women", "women")
        assert result is None
        
        result = self.unifier.extract_persona_tag("African American", "african_american")
        assert result is None
    
    @pytest.mark.unit
    def test_extract_persona_tag_none_target_group_norm(self):
        """Test persona tag extraction when target_group_norm is None."""
        result = self.unifier.extract_persona_tag("Homosexual", None)
        assert result is None
        
        result = self.unifier.extract_persona_tag("Arab", None)
        assert result is None
    
    @pytest.mark.unit
    def test_extract_persona_tag_whitespace_handling(self):
        """Test persona tag extraction with whitespace in original target group."""
        result = self.unifier.extract_persona_tag("  Homosexual  ", "lgbtq")
        assert result == "homosexual"
        
        result = self.unifier.extract_persona_tag(" Hispanic ", "mexican")
        assert result == "hispanic"
        
        result = self.unifier.extract_persona_tag("  Arab  ", "middle_east")
        assert result == "arab"
    
    @pytest.mark.unit
    def test_extract_persona_tag_case_handling(self):
        """Test persona tag extraction preserves original casing (after lowercasing)."""
        result = self.unifier.extract_persona_tag("HOMOSEXUAL", "lgbtq")
        assert result == "homosexual"
        
        result = self.unifier.extract_persona_tag("HiSpAnIc", "mexican")
        assert result == "hispanic"
        
        result = self.unifier.extract_persona_tag("aRaB", "middle_east")
        assert result == "arab"


class TestMapLabels:
    """Test cases for label mapping methods."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
    
    @pytest.mark.unit
    def test_map_hatexplain_labels(self):
        """Test HateXplain label mapping."""
        # Test hate label
        binary, multiclass = self.unifier.map_hatexplain_labels("hate")
        assert binary == "hate"
        assert multiclass == "hate"
        
        # Test offensive label
        binary, multiclass = self.unifier.map_hatexplain_labels("offensive")
        assert binary == "normal"
        assert multiclass == "offensive"
        
        # Test normal label
        binary, multiclass = self.unifier.map_hatexplain_labels("normal")
        assert binary == "normal"
        assert multiclass == "normal"
    
    @pytest.mark.unit
    def test_map_toxigen_labels(self):
        """Test ToxiGen label mapping."""
        # Test toxic label
        binary, multiclass = self.unifier.map_toxigen_labels("toxic")
        assert binary == "hate"
        assert multiclass == "toxic_implicit"
        
        # Test benign label
        binary, multiclass = self.unifier.map_toxigen_labels("benign")
        assert binary == "normal"
        assert multiclass == "benign_implicit"
        
        # Test normal label (edge case)
        binary, multiclass = self.unifier.map_toxigen_labels("normal")
        assert binary == "normal"
        assert multiclass == "benign_implicit"


class TestCreateFineTuningEmbedding:
    """Test cases for creating fine-tuning embeddings."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
    
    @pytest.mark.unit
    def test_create_fine_tuning_embedding_with_persona_and_rationale(self):
        """Test fine-tuning embedding creation with persona and rationale."""
        text = "This is test text"
        target_group_norm = "lgbtq"
        persona_tag = "homosexual"
        rationale_text = "This contains hate speech because..."
        source = "hatexplain"
        
        result = self.unifier.create_fine_tuning_embedding(
            text, target_group_norm, persona_tag, rationale_text, source
        )
        
        expected = "[PERSONA:HOMOSEXUAL] This is test text [RATIONALE:This contains hate speech because...] [POLICY:HATE_SPEECH_DETECTION]"
        assert result == expected
    
    @pytest.mark.unit
    def test_create_fine_tuning_embedding_with_persona_no_rationale(self):
        """Test fine-tuning embedding creation with persona but no rationale."""
        text = "This is test text"
        target_group_norm = "mexican"
        persona_tag = "hispanic"
        rationale_text = None
        source = "toxigen"
        
        result = self.unifier.create_fine_tuning_embedding(
            text, target_group_norm, persona_tag, rationale_text, source
        )
        
        expected = "[PERSONA:HISPANIC] This is test text [POLICY:HATE_SPEECH_DETECTION]"
        assert result == expected
    
    @pytest.mark.unit
    def test_create_fine_tuning_embedding_no_persona_no_rationale(self):
        """Test fine-tuning embedding creation without persona or rationale."""
        text = "This is test text"
        target_group_norm = "middle_east"
        persona_tag = None
        rationale_text = None
        source = "toxigen"
        
        result = self.unifier.create_fine_tuning_embedding(
            text, target_group_norm, persona_tag, rationale_text, source
        )
        
        expected = "This is test text [POLICY:HATE_SPEECH_DETECTION]"
        assert result == expected
    
    @pytest.mark.unit
    def test_create_fine_tuning_embedding_empty_rationale(self):
        """Test fine-tuning embedding creation with empty rationale."""
        text = "This is test text"
        target_group_norm = "lgbtq"
        persona_tag = "gay"
        rationale_text = ""  # Empty string should be treated as None
        source = "hatexplain"
        
        result = self.unifier.create_fine_tuning_embedding(
            text, target_group_norm, persona_tag, rationale_text, source
        )
        
        # Empty rationale should not be included
        expected = "[PERSONA:GAY] This is test text [POLICY:HATE_SPEECH_DETECTION]"
        assert result == expected
    
    @pytest.mark.unit
    def test_create_fine_tuning_embedding_na_rationale(self):
        """Test fine-tuning embedding creation with 'NA' rationale."""
        text = "This is test text"
        target_group_norm = "lgbtq"
        persona_tag = "lgbtq"
        rationale_text = "NA"  # Should be treated as None
        source = "hatexplain"
        
        result = self.unifier.create_fine_tuning_embedding(
            text, target_group_norm, persona_tag, rationale_text, source
        )
        
        # 'NA' rationale should not be included
        expected = "[PERSONA:LGBTQ] This is test text [POLICY:HATE_SPEECH_DETECTION]"
        assert result == expected


class TestUnifyEntry:
    """Test cases for the unify_entry method."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
    
    @pytest.mark.unit
    def test_unify_entry_hatexplain_valid(self):
        """Test unifying a valid HateXplain entry."""
        hatexplain_entry = {
            'text': 'This is hate speech against gay people',
            'target': 'Homosexual',
            'majority_label': 'hate',
            'rationale_text': 'Contains slurs and threats',
            'post_id': 'he_12345',
            'split': 'train'
        }
        
        result = self.unifier.unify_entry(hatexplain_entry, 'hatexplain')
        
        assert result is not None
        assert result['text'] == 'This is hate speech against gay people'
        assert result['target_group_norm'] == 'lgbtq'
        assert result['persona_tag'] == 'homosexual'
        assert result['label_binary'] == 'hate'
        assert result['label_multiclass'] == 'hate'
        assert result['source_dataset'] == 'hatexplain'
        assert result['is_synthetic'] == False
        assert result['rationale_text'] == 'Contains slurs and threats'
        assert result['original_id'] == 'he_12345'
        assert result['split'] == 'train'
        assert '[PERSONA:HOMOSEXUAL]' in result['fine_tuning_embedding']
        assert '[RATIONALE:Contains slurs and threats]' in result['fine_tuning_embedding']
    
    @pytest.mark.unit
    def test_unify_entry_hatexplain_arab(self):
        """Test unifying a HateXplain entry with Arab target."""
        hatexplain_entry = {
            'text': 'Anti-Arab hate speech',
            'target': 'Arab',
            'majority_label': 'hate',
            'rationale_text': 'Contains ethnic slurs',
            'post_id': 'he_67890',
            'split': 'train'
        }
        
        result = self.unifier.unify_entry(hatexplain_entry, 'hatexplain')
        
        assert result is not None
        assert result['target_group_norm'] == 'middle_east'
        assert result['persona_tag'] == 'arab'  # Preserves original
        assert '[PERSONA:ARAB]' in result['fine_tuning_embedding']
    
    @pytest.mark.unit
    def test_unify_entry_hatexplain_invalid_target(self):
        """Test unifying a HateXplain entry with invalid target group."""
        hatexplain_entry = {
            'text': 'Some text about women',
            'target': 'Women',  # Not in our selected target groups
            'majority_label': 'hate',
            'rationale_text': 'Some rationale',
            'post_id': 'he_invalid',
            'split': 'train'
        }
        
        result = self.unifier.unify_entry(hatexplain_entry, 'hatexplain')
        assert result is None  # Should be filtered out
    
    @pytest.mark.unit
    def test_unify_entry_toxigen_valid(self):
        """Test unifying a valid ToxiGen entry."""
        toxigen_entry = {
            'text': 'Toxic statement about Mexican people',
            'target_group': 'mexican',
            'label_binary': 'toxic',
            'text_id': 'tg_12345',
            'split': 'train'
        }
        
        result = self.unifier.unify_entry(toxigen_entry, 'toxigen')
        
        assert result is not None
        assert result['text'] == 'Toxic statement about Mexican people'
        assert result['target_group_norm'] == 'mexican'
        assert result['persona_tag'] == 'mexican'
        assert result['label_binary'] == 'hate'
        assert result['label_multiclass'] == 'toxic_implicit'
        assert result['source_dataset'] == 'toxigen'
        assert result['is_synthetic'] == False  # Real examples from ToxiGen, not synthetically generated in this pipeline
        assert result['rationale_text'] is None
        assert result['original_id'] == 'tg_12345'
        assert result['split'] == 'train'
        assert '[PERSONA:MEXICAN]' in result['fine_tuning_embedding']
    
    @pytest.mark.unit
    def test_unify_entry_toxigen_benign(self):
        """Test unifying a benign ToxiGen entry."""
        toxigen_entry = {
            'text': 'Neutral statement about LGBTQ community',
            'target_group': 'lgbtq',
            'label_binary': 'benign',
            'text_id': 'tg_67890',
            'split': 'val'
        }
        
        result = self.unifier.unify_entry(toxigen_entry, 'toxigen')
        
        assert result is not None
        assert result['label_binary'] == 'normal'
        assert result['label_multiclass'] == 'benign_implicit'
    
    @pytest.mark.unit
    def test_unify_entry_toxigen_invalid_target(self):
        """Test unifying a ToxiGen entry with invalid target group."""
        toxigen_entry = {
            'text': 'Some text about women',
            'target_group': 'women',  # Not in our selected target groups
            'label_binary': 'toxic',
            'text_id': 'tg_invalid',
            'split': 'train'
        }
        
        result = self.unifier.unify_entry(toxigen_entry, 'toxigen')
        assert result is None  # Should be filtered out
    
    @pytest.mark.unit
    def test_unify_entry_hatexplain_synthetic_flag(self):
        """Test unifying a HateXplain entry with synthetic flag."""
        hatexplain_entry = {
            'text': 'Synthetically generated hate speech',
            'target': 'Homosexual',
            'majority_label': 'hate',
            'rationale_text': 'Generated rationale',
            'post_id': 'he_synthetic',
            'split': 'train',
            'is_synthetic': True  # Synthetic augmentation
        }
        
        result = self.unifier.unify_entry(hatexplain_entry, 'hatexplain')
        
        assert result is not None
        assert result['is_synthetic'] == True  # Should preserve synthetic flag
    
    @pytest.mark.unit
    def test_unify_entry_invalid_source(self):
        """Test unifying an entry with invalid source."""
        entry = {
            'text': 'Some text',
            'target': 'Homosexual',
            'majority_label': 'hate'
        }
        
        result = self.unifier.unify_entry(entry, 'invalid_source')
        assert result is None


class TestLoadDatasets:
    """Test cases for the load_datasets method."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.hatexplain_dir = Path(self.temp_dir) / "hatexplain"
        self.toxigen_dir = Path(self.temp_dir) / "toxigen"
        self.hatexplain_dir.mkdir()
        self.toxigen_dir.mkdir()
        
        self.unifier = DatasetUnifier(
            hatexplain_dir=str(self.hatexplain_dir),
            toxigen_dir=str(self.toxigen_dir)
        )
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.unit
    def test_load_datasets_success(self):
        """Test successful loading of datasets."""
        # Create test data files
        hatexplain_train_data = [
            {'text': 'hate speech', 'target': 'Homosexual', 'majority_label': 'hate'}
        ]
        toxigen_train_data = [
            {'text': 'toxic content', 'target_group': 'lgbtq', 'label_binary': 'toxic'}
        ]
        
        # Write test files
        with open(self.hatexplain_dir / "hatexplain_train.json", 'w') as f:
            json.dump(hatexplain_train_data, f)
        
        with open(self.toxigen_dir / "toxigen_train.json", 'w') as f:
            json.dump(toxigen_train_data, f)
        
        # Load datasets
        self.unifier.load_datasets()
        
        # Verify data was loaded
        assert 'train' in self.unifier.hatexplain_data
        assert 'train' in self.unifier.toxigen_data
        assert len(self.unifier.hatexplain_data['train']) == 1
        assert len(self.unifier.toxigen_data['train']) == 1
        assert self.unifier.hatexplain_data['train'][0]['text'] == 'hate speech'
        assert self.unifier.toxigen_data['train'][0]['text'] == 'toxic content'
    
    @pytest.mark.unit
    def test_load_datasets_missing_files(self):
        """Test loading datasets when files are missing."""
        # Don't create any files
        self.unifier.load_datasets()
        
        # Verify empty containers
        assert self.unifier.hatexplain_data == {}
        assert self.unifier.toxigen_data == {}
    
    @pytest.mark.unit
    def test_load_datasets_partial_files(self):
        """Test loading datasets when only some files exist."""
        # Create only HateXplain train file
        hatexplain_train_data = [
            {'text': 'hate speech', 'target': 'Arab', 'majority_label': 'hate'}
        ]
        
        with open(self.hatexplain_dir / "hatexplain_train.json", 'w') as f:
            json.dump(hatexplain_train_data, f)
        
        self.unifier.load_datasets()
        
        # Verify only HateXplain train was loaded
        assert 'train' in self.unifier.hatexplain_data
        assert len(self.unifier.hatexplain_data['train']) == 1
        assert self.unifier.toxigen_data == {}


class TestAnalyzeUnifiedDataset:
    """Test cases for the analyze_unified_dataset method."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.unifier = DatasetUnifier(
            hatexplain_dir="test_hatexplain",
            toxigen_dir="test_toxigen"
        )
        
        # Create mock unified data
        self.unifier.unified_data = {
            'train': [
                {
                    'text': 'hate speech',
                    'label_binary': 'hate',
                    'label_multiclass': 'hate',
                    'target_group_norm': 'lgbtq',
                    'persona_tag': 'homosexual',
                    'source_dataset': 'hatexplain',
                    'is_synthetic': False,
                    'rationale_text': 'Contains slurs'
                },
                {
                    'text': 'toxic content',
                    'label_binary': 'hate',
                    'label_multiclass': 'toxic_implicit',
                    'target_group_norm': 'mexican',
                    'persona_tag': 'mexican',
                    'source_dataset': 'toxigen',
                    'is_synthetic': False,  # Real ToxiGen examples
                    'rationale_text': None
                },
                {
                    'text': 'normal content',
                    'label_binary': 'normal',
                    'label_multiclass': 'benign_implicit',
                    'target_group_norm': 'middle_east',
                    'persona_tag': 'arab',
                    'source_dataset': 'toxigen',
                    'is_synthetic': False,  # Real ToxiGen examples
                    'rationale_text': None
                }
            ]
        }
    
    @pytest.mark.unit
    def test_analyze_unified_dataset_success(self):
        """Test successful analysis of unified dataset."""
        stats = self.unifier.analyze_unified_dataset()
        
        assert isinstance(stats, UnifiedDatasetStats)
        assert stats.total_entries == 3
        assert stats.hatexplain_entries == 1
        assert stats.toxigen_entries == 2
        
        # Check label distributions
        assert stats.label_binary_distribution == {'hate': 2, 'normal': 1}
        assert stats.label_multiclass_distribution == {
            'hate': 1, 'toxic_implicit': 1, 'benign_implicit': 1
        }
        
        # Check target group distribution
        assert stats.target_group_distribution == {
            'lgbtq': 1, 'mexican': 1, 'middle_east': 1
        }
        
        # Check persona tag distribution
        assert stats.persona_tag_distribution == {
            'homosexual': 1, 'mexican': 1, 'arab': 1
        }
        
        # Check synthetic ratio
        assert stats.synthetic_ratio == 2/3  # 2 out of 3 are synthetic
        
        # Check rationale coverage
        assert stats.rationale_coverage == 1/3  # 1 out of 3 has rationale
    
    @pytest.mark.unit
    def test_analyze_unified_dataset_empty(self):
        """Test analysis with empty unified dataset."""
        self.unifier.unified_data = {}
        
        with pytest.raises(ValueError, match="No unified data available"):
            self.unifier.analyze_unified_dataset()
    
    @pytest.mark.unit
    def test_analyze_unified_dataset_no_data(self):
        """Test analysis when no unified data is available."""
        self.unifier.unified_data = None
        
        with pytest.raises(ValueError, match="No unified data available"):
            self.unifier.analyze_unified_dataset()


if __name__ == "__main__":
    pytest.main([__file__])