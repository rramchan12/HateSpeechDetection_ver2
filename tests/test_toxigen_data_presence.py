import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.toxigen_data_presence_validator import (
    validate_toxigen_data_presence,
    test_toxigen_data_presence,
    _find_split_file,
    _count_records,
    _peek_fields,
    _ALLOWED_EXTS,
    _EXPECTED_SPLITS
)


@pytest.mark.unit
def test_toxigen_constants():
    """Test that required constants are properly defined."""
    assert isinstance(_ALLOWED_EXTS, list)
    assert ".parquet" in _ALLOWED_EXTS
    assert len(_ALLOWED_EXTS) == 1  # Only Parquet files for efficiency
    
    assert isinstance(_EXPECTED_SPLITS, list)
    assert "train" in _EXPECTED_SPLITS
    assert "annotated" in _EXPECTED_SPLITS
    assert "annotations" in _EXPECTED_SPLITS


@pytest.mark.unit
def test_find_split_file():
    """Test the _find_split_file helper function with Parquet files only."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test Parquet files
        train_file = temp_path / "train.parquet"
        train_file.touch()
        
        annotated_file = temp_path / "annotated.parquet"
        annotated_file.touch()
        
        # Test finding existing Parquet files
        result = _find_split_file(temp_path, "train")
        assert result == train_file
        
        result = _find_split_file(temp_path, "annotated")
        assert result == annotated_file
        
        # Test missing file
        result = _find_split_file(temp_path, "missing")
        assert result is None


@pytest.mark.unit
def test_find_split_file_sharded():
    """Test _find_split_file with sharded files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sharded file
        sharded_file = temp_path / "train-00000-of-00001.parquet"
        sharded_file.touch()
        
        result = _find_split_file(temp_path, "train")
        assert result == sharded_file


@pytest.mark.unit
def test_count_records_parquet():
    """Test _count_records with Parquet files."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        parquet_file = temp_path / "test.parquet"
        
        # Create test Parquet content
        import pandas as pd
        test_data = pd.DataFrame({
            "text": ["line 1", "line 2", "line 3"],
            "label": ["positive", "negative", "positive"]
        })
        test_data.to_parquet(parquet_file)
        
        result = _count_records(parquet_file)
        assert result == 3


@pytest.mark.unit
def test_peek_fields_parquet():
    """Test _peek_fields with Parquet files."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        parquet_file = temp_path / "test.parquet"
        
        # Create test Parquet content
        import pandas as pd
        test_data = pd.DataFrame({
            "text": ["sample"],
            "label": ["positive"], 
            "score": [0.9]
        })
        test_data.to_parquet(parquet_file)
        
        result = _peek_fields(parquet_file)
        assert "text" in result
        assert "label" in result
        assert "score" in result


@pytest.mark.unit
def test_toxigen_missing_directory():
    """Test validation fails when data directory doesn't exist."""
    with patch('pathlib.Path.is_dir', return_value=False):
        with pytest.raises(AssertionError) as excinfo:
            validate_toxigen_data_presence()
        
        assert "Missing directory 'data/toxigen'" in str(excinfo.value)
        assert "download the ToxiGen dataset" in str(excinfo.value)


@pytest.mark.unit
def test_toxigen_missing_splits():
    """Test validation fails when split files are missing."""
    with patch('pathlib.Path.is_dir', return_value=True):
        with patch('data_collection.toxigen_data_presence_validator._find_split_file', return_value=None):
            with pytest.raises(AssertionError) as excinfo:
                validate_toxigen_data_presence()
            
            assert "Missing expected toxigen data split files" in str(excinfo.value)
            for split in _EXPECTED_SPLITS:
                assert split in str(excinfo.value)


@pytest.mark.unit
def test_toxigen_valid_dataset():
    """Test validation succeeds with valid dataset structure."""
    # Mock file paths
    def mock_find_split_file(data_dir, split):
        return data_dir / f"{split}.jsonl"
    
    def mock_count_records(path):
        if "train" in str(path):
            return 1000
        elif "annotated" in str(path):
            return 100
        elif "annotations" in str(path):
            return 500
        return 0
    
    def mock_peek_fields(path):
        if "train" in str(path):
            return ["prompt", "generation", "label"]
        elif "annotated" in str(path):
            return ["text", "target_group", "label"]
        elif "annotations" in str(path):
            return ["text", "annotation_id", "label"]
        return []
    
    mock_project_root = Path("/fake/project")
    
    with patch('pathlib.Path.is_dir', return_value=True):
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_resolve.return_value.parents = [mock_project_root, Path("/")]
            
            with patch('data_collection.toxigen_data_presence_validator._find_split_file', side_effect=mock_find_split_file):
                with patch('data_collection.toxigen_data_presence_validator._count_records', side_effect=mock_count_records):
                    with patch('data_collection.toxigen_data_presence_validator._peek_fields', side_effect=mock_peek_fields):
                        with patch('pathlib.Path.relative_to', side_effect=lambda x: Path(f"data/toxigen/{x.name}")):
                            # Capture print output
                            with patch('builtins.print') as mock_print:
                                result = validate_toxigen_data_presence()
                                
                                # Verify function returns result
                                assert isinstance(result, dict)
                                for split in _EXPECTED_SPLITS:
                                    assert split in result
                                
                                # Verify print was called with summary
                                mock_print.assert_called()
                                print_calls = [str(call) for call in mock_print.call_args_list]
                                summary_printed = any("ToxiGen data summary" in call for call in print_calls)
                                assert summary_printed


@pytest.mark.unit
def test_toxigen_empty_train_split():
    """Test validation fails when train split is empty."""
    def mock_find_split_file(data_dir, split):
        return data_dir / f"{split}.jsonl"
    
    def mock_count_records(path):
        if "train" in str(path):
            return 0  # Empty train split
        return 100
    
    def mock_peek_fields(path):
        return ["text", "label"]
    
    with patch('pathlib.Path.is_dir', return_value=True):
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_resolve.return_value.parents = [Path("/fake/project"), Path("/")]
            
            with patch('data_collection.toxigen_data_presence_validator._find_split_file', side_effect=mock_find_split_file):
                with patch('data_collection.toxigen_data_presence_validator._count_records', side_effect=mock_count_records):
                    with patch('data_collection.toxigen_data_presence_validator._peek_fields', side_effect=mock_peek_fields):
                        with patch('pathlib.Path.relative_to', side_effect=lambda x: Path(f"data/toxigen/{x.name}")):
                            with patch('builtins.print'):  # Suppress print output
                                with pytest.raises(AssertionError) as excinfo:
                                    validate_toxigen_data_presence()
                                
                                assert "Train split appears empty" in str(excinfo.value)


@pytest.mark.unit
def test_toxigen_test_wrapper():
    """Test the test_toxigen_data_presence wrapper function."""
    # Mock the main validation function
    mock_result = {"train": {"records": "1000", "file": "data/toxigen/train.jsonl"}}
    
    with patch('data_collection.toxigen_data_presence_validator.validate_toxigen_data_presence', return_value=mock_result):
        result = test_toxigen_data_presence(None)  # capsys parameter not used in implementation
        assert result == mock_result


@pytest.mark.integration
def test_toxigen_validator_import():
    """Integration test: verify the validator module can be imported."""
    from data_collection.toxigen_data_presence_validator import validate_toxigen_data_presence, test_toxigen_data_presence
    
    # Test that functions exist
    assert callable(validate_toxigen_data_presence)
    assert callable(test_toxigen_data_presence)
    
    # Test that helper functions exist
    from data_collection.toxigen_data_presence_validator import _find_split_file, _count_records, _peek_fields
    assert callable(_find_split_file)
    assert callable(_count_records)
    assert callable(_peek_fields)
    
    # Test that constants are defined
    from data_collection.toxigen_data_presence_validator import _ALLOWED_EXTS, _EXPECTED_SPLITS
    assert len(_ALLOWED_EXTS) == 1  # Only Parquet files for efficiency
    assert len(_EXPECTED_SPLITS) == 3


@pytest.mark.data
def test_toxigen_real_data_if_present():
    """Data test: if ToxiGen data is present, validate it."""
    try:
        result = validate_toxigen_data_presence()
        # If we get here, data is present and valid
        assert isinstance(result, dict)
        for split in _EXPECTED_SPLITS:
            assert split in result
        print("✅ ToxiGen data validation passed with real data")
    except AssertionError as e:
        if "Missing directory" in str(e) or "Missing expected" in str(e):
            pytest.skip("ToxiGen data not present - skipping data validation test")
        else:
            # Re-raise if it's a different validation error
            raise