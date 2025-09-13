import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.hatexplain_data_presence_validator import (
    validate_hatexplain_data_presence,
    test_hatexplain_data_presence,
    _HATEXPLAIN_CORE_FILES,
    _HATEXPLAIN_OPTIONAL_FILES,
    _EXPECTED_SPLITS
)


@pytest.mark.unit
def test_hatexplain_constants():
    """Test that required constants are properly defined."""
    assert isinstance(_HATEXPLAIN_CORE_FILES, list)
    assert "dataset.json" in _HATEXPLAIN_CORE_FILES
    assert "post_id_divisions.json" in _HATEXPLAIN_CORE_FILES
    
    assert isinstance(_HATEXPLAIN_OPTIONAL_FILES, list)
    assert "classes.npy" in _HATEXPLAIN_OPTIONAL_FILES
    assert "classes_two.npy" in _HATEXPLAIN_OPTIONAL_FILES
    
    assert isinstance(_EXPECTED_SPLITS, list)
    assert "train" in _EXPECTED_SPLITS
    assert "val" in _EXPECTED_SPLITS
    assert "test" in _EXPECTED_SPLITS


@pytest.mark.unit
def test_hatexplain_missing_directory():
    """Test validation fails when data directory doesn't exist."""
    with patch('pathlib.Path.is_dir', return_value=False):
        with pytest.raises(AssertionError) as excinfo:
            validate_hatexplain_data_presence()
        
        assert "Missing directory 'data/hatexplain'" in str(excinfo.value)
        assert "download the HateXplain dataset" in str(excinfo.value)


@pytest.mark.unit
def test_hatexplain_missing_core_files():
    """Test validation fails when core files are missing."""
    with patch('pathlib.Path.is_dir', return_value=True):
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(AssertionError) as excinfo:
                validate_hatexplain_data_presence()
            
            assert "Missing expected HateXplain core files" in str(excinfo.value)
            assert "dataset.json" in str(excinfo.value)
            assert "post_id_divisions.json" in str(excinfo.value)


@pytest.mark.unit
def test_hatexplain_corrupted_dataset_json():
    """Test validation fails when dataset.json is corrupted."""
    with patch('pathlib.Path.is_dir', return_value=True):
        # Mock files exist but dataset.json has invalid content
        def mock_exists(self):
            return str(self).endswith(('dataset.json', 'post_id_divisions.json'))
        
        def mock_stat(self):
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1000
            return mock_stat_result
        
        # Mock json.load to raise JSONDecodeError
        def mock_json_load(f):
            raise json.JSONDecodeError("Expecting value", "invalid json", 0)
        
        with patch('pathlib.Path.exists', mock_exists):
            with patch('pathlib.Path.stat', mock_stat):
                with patch('pathlib.Path.relative_to', return_value=Path("data/hatexplain/dataset.json")):
                    with patch('builtins.open', mock_open()):
                        with patch('json.load', mock_json_load):
                            with pytest.raises(AssertionError) as excinfo:
                                validate_hatexplain_data_presence()
                            
                            assert "corrupted or not valid JSON" in str(excinfo.value)


@pytest.mark.unit
def test_hatexplain_valid_dataset():
    """Test validation succeeds with valid dataset structure."""
    # Mock dataset.json content
    mock_dataset = {
        "12345_post": {
            "post_id": "12345_post",
            "annotators": [{"label": "normal", "annotator_id": 1}],
            "rationales": [[0, 1, 0, 1]],
            "post_tokens": ["this", "is", "a", "test"]
        },
        "12346_post": {
            "post_id": "12346_post", 
            "annotators": [{"label": "hate", "annotator_id": 2}],
            "rationales": [[1, 1, 0, 0]],
            "post_tokens": ["another", "test", "post", "here"]
        }
    }
    
    # Mock post_id_divisions.json content
    mock_divisions = {
        "train": ["12345_post"],
        "val": ["12346_post"], 
        "test": []
    }
    
    # Mock file system
    def mock_exists(self):
        return str(self).endswith(('dataset.json', 'post_id_divisions.json'))
    
    def mock_stat(self):
        mock_stat_result = Mock()
        mock_stat_result.st_size = 5000 if 'dataset.json' in str(self) else 500
        return mock_stat_result
    
    def mock_open_func(file_path, *args, **kwargs):
        if 'dataset.json' in str(file_path):
            return mock_open(read_data=json.dumps(mock_dataset)).return_value
        elif 'post_id_divisions.json' in str(file_path):
            return mock_open(read_data=json.dumps(mock_divisions)).return_value
        else:
            raise FileNotFoundError()
    
    with patch('pathlib.Path.is_dir', return_value=True):
        with patch('pathlib.Path.exists', mock_exists):
            with patch('pathlib.Path.stat', mock_stat):
                with patch('pathlib.Path.relative_to', return_value=Path("data/hatexplain/test.json")):
                    with patch('builtins.open', mock_open_func):
                        # Capture print output
                        with patch('builtins.print') as mock_print:
                            result = validate_hatexplain_data_presence()
                            
                            # Verify function returns result
                            assert isinstance(result, dict)
                            assert "dataset.json" in result
                            assert "post_id_divisions.json" in result
                            
                            # Verify print was called with summary
                            mock_print.assert_called()
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            summary_printed = any("HateXplain dataset summary" in call for call in print_calls)
                            assert summary_printed


@pytest.mark.unit
def test_hatexplain_test_wrapper():
    """Test the test_hatexplain_data_presence wrapper function."""
    # Mock the main validation function
    mock_result = {"dataset.json": {"total_entries": 100}}
    
    with patch('data_collection.hatexplain_data_presence_validator.validate_hatexplain_data_presence', return_value=mock_result):
        result = test_hatexplain_data_presence(None)  # capsys parameter not used in implementation
        assert result == mock_result


@pytest.mark.integration
def test_hatexplain_validator_import():
    """Integration test: verify the validator module can be imported."""
    from data_collection.hatexplain_data_presence_validator import validate_hatexplain_data_presence, test_hatexplain_data_presence
    
    # Test that functions exist
    assert callable(validate_hatexplain_data_presence)
    assert callable(test_hatexplain_data_presence)
    
    # Test that constants are defined
    from data_collection.hatexplain_data_presence_validator import _HATEXPLAIN_CORE_FILES, _HATEXPLAIN_OPTIONAL_FILES, _EXPECTED_SPLITS
    assert len(_HATEXPLAIN_CORE_FILES) >= 2
    assert len(_HATEXPLAIN_OPTIONAL_FILES) >= 2
    assert len(_EXPECTED_SPLITS) == 3


@pytest.mark.data
def test_hatexplain_real_data_if_present():
    """Data test: if HateXplain data is present, validate it."""
    try:
        result = validate_hatexplain_data_presence()
        # If we get here, data is present and valid
        assert isinstance(result, dict)
        assert "dataset.json" in result
        print("✅ HateXplain data validation passed with real data")
    except AssertionError as e:
        if "Missing directory" in str(e) or "Missing expected" in str(e):
            pytest.skip("HateXplain data not present - skipping data validation test")
        else:
            # Re-raise if it's a different validation error
            raise