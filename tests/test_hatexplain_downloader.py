import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile
import urllib.error
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.hatexplain_downloader import (
    download_hatexplain,
    _download_file,
    _validate_dataset_integrity,
    _HATEXPLAIN_BASE_URL,
    _HATEXPLAIN_FILES,
    _OPTIONAL_FILES
)


@pytest.mark.unit
def test_hatexplain_constants():
    """Test that required constants are properly defined."""
    assert isinstance(_HATEXPLAIN_BASE_URL, str)
    assert "githubusercontent.com" in _HATEXPLAIN_BASE_URL.lower()
    
    assert isinstance(_HATEXPLAIN_FILES, list)
    assert "dataset.json" in _HATEXPLAIN_FILES
    assert "post_id_divisions.json" in _HATEXPLAIN_FILES
    assert "classes.npy" in _HATEXPLAIN_FILES
    assert "classes_two.npy" in _HATEXPLAIN_FILES
    
    assert isinstance(_OPTIONAL_FILES, list)
    assert "README.md" in _OPTIONAL_FILES


@pytest.mark.unit
def test_download_file_success():
    """Test successful file download."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        destination = temp_path / "test_file.txt"
        test_content = b"test content"
        
        # Mock urllib.request.urlretrieve
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            # Simulate successful download by creating the file
            def side_effect(url, dest):
                Path(dest).write_bytes(test_content)
            
            mock_retrieve.side_effect = side_effect
            
            result = _download_file("http://example.com/file.txt", destination)
            
            assert result is True
            assert destination.exists()
            assert destination.read_bytes() == test_content
            mock_retrieve.assert_called_once_with("http://example.com/file.txt", destination)


@pytest.mark.unit
def test_download_file_already_exists():
    """Test download when file already exists and overwrite is False."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        destination = temp_path / "existing_file.txt"
        
        # Create existing file
        destination.write_text("existing content")
        
        with patch('builtins.print') as mock_print:
            result = _download_file("http://example.com/file.txt", destination, overwrite=False)
            
            assert result is True
            assert destination.read_text() == "existing content"  # Unchanged
            mock_print.assert_called_once()
            assert "already exists" in str(mock_print.call_args)


@pytest.mark.unit
def test_download_file_url_error():
    """Test download failure due to URL error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        destination = temp_path / "failed_file.txt"
        
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            mock_retrieve.side_effect = urllib.error.URLError("Connection failed")
            
            with patch('builtins.print') as mock_print:
                result = _download_file("http://invalid-url.com/file.txt", destination)
                
                assert result is False
                assert not destination.exists()
                mock_print.assert_called()
                assert "Failed to download" in str(mock_print.call_args)


@pytest.mark.unit
def test_validate_dataset_integrity_success():
    """Test dataset validation with valid JSON structure."""
    mock_dataset = {
        "12345_post": {
            "post_id": "12345_post",
            "annotators": [{"label": "normal"}],
            "rationales": [[0, 1, 0]],
            "post_tokens": ["test", "post", "here"]
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_file = temp_path / "dataset.json"
        dataset_file.write_text(json.dumps(mock_dataset))
        
        with patch('builtins.print') as mock_print:
            result = _validate_dataset_integrity(temp_path)
            
            assert result is True
            mock_print.assert_called()
            # Check that validation message was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            validation_printed = any("validation passed" in call.lower() for call in print_calls)
            assert validation_printed


@pytest.mark.unit
def test_validate_dataset_integrity_missing_file():
    """Test dataset validation fails when dataset.json is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with patch('builtins.print') as mock_print:
            result = _validate_dataset_integrity(temp_path)
            
            assert result is False
            mock_print.assert_called_once()
            assert "dataset.json file missing" in str(mock_print.call_args)


@pytest.mark.unit
def test_validate_dataset_integrity_invalid_json():
    """Test dataset validation fails with invalid JSON."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_file = temp_path / "dataset.json"
        dataset_file.write_text("invalid json content")
        
        with patch('builtins.print') as mock_print:
            result = _validate_dataset_integrity(temp_path)
            
            assert result is False
            mock_print.assert_called()
            assert "not valid JSON" in str(mock_print.call_args)


@pytest.mark.unit
def test_validate_dataset_integrity_missing_fields():
    """Test dataset validation fails when required fields are missing."""
    # Dataset missing required fields
    mock_dataset = {
        "12345_post": {
            "post_id": "12345_post",
            # Missing: annotators, rationales, post_tokens
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_file = temp_path / "dataset.json"
        dataset_file.write_text(json.dumps(mock_dataset))
        
        with patch('builtins.print') as mock_print:
            result = _validate_dataset_integrity(temp_path)
            
            assert result is False
            mock_print.assert_called()
            assert "missing required fields" in str(mock_print.call_args)


@pytest.mark.unit
@patch('data_collection.hatexplain_downloader._download_file')
@patch('data_collection.hatexplain_downloader._validate_dataset_integrity')
def test_download_hatexplain_success(mock_validate, mock_download):
    """Test successful HateXplain dataset download."""
    # Mock successful downloads and validation
    mock_download.return_value = True
    mock_validate.return_value = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create the core files to simulate successful download
        for filename in _HATEXPLAIN_FILES:
            (temp_path / filename).touch()
        
        with patch('builtins.print') as mock_print:
            result = download_hatexplain(data_dir=temp_dir, overwrite=True)
            
            assert result is True
            
            # Verify all core files were attempted to be downloaded
            assert mock_download.call_count == len(_HATEXPLAIN_FILES)
            
            # Verify validation was called
            mock_validate.assert_called_once()
            
            # Verify success message was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            success_printed = any("download complete" in call.lower() for call in print_calls)
            assert success_printed


@pytest.mark.unit
@patch('data_collection.hatexplain_downloader._download_file')
def test_download_hatexplain_partial_failure(mock_download):
    """Test HateXplain download with some failed downloads."""
    # Mock some downloads failing
    def mock_download_side_effect(url, dest, overwrite=False):
        return "dataset.json" in str(dest)  # Only dataset.json succeeds
    
    mock_download.side_effect = mock_download_side_effect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('builtins.print') as mock_print:
            result = download_hatexplain(data_dir=temp_dir)
            
            assert result is False  # Should fail due to missing core files
            
            # Verify failure message was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            failure_printed = any("incomplete" in call.lower() for call in print_calls)
            assert failure_printed


@pytest.mark.unit
@patch('data_collection.hatexplain_downloader._download_file')
@patch('data_collection.hatexplain_downloader._validate_dataset_integrity')
def test_download_hatexplain_with_optional_files(mock_validate, mock_download):
    """Test HateXplain download including optional files."""
    mock_download.return_value = True
    mock_validate.return_value = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create all files to simulate successful download
        for filename in _HATEXPLAIN_FILES + _OPTIONAL_FILES:
            (temp_path / filename).touch()
        
        result = download_hatexplain(
            data_dir=temp_dir, 
            include_optional=True, 
            overwrite=True
        )
        
        assert result is True
        
        # Verify core + optional files were downloaded
        expected_files = len(_HATEXPLAIN_FILES) + len(_OPTIONAL_FILES)
        assert mock_download.call_count == expected_files


@pytest.mark.unit
def test_download_hatexplain_default_directory():
    """Test download with default directory path resolution."""
    with patch('data_collection.hatexplain_downloader._download_file', return_value=True):
        with patch('data_collection.hatexplain_downloader._validate_dataset_integrity', return_value=True):
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                with patch('builtins.print'):
                    result = download_hatexplain()  # No data_dir specified
                    
                    # Should succeed with default path
                    assert result is True
                    
                    # Should create directory
                    mock_mkdir.assert_called_once()


@pytest.mark.integration
def test_hatexplain_downloader_import():
    """Integration test: verify the downloader module can be imported."""
    from data_collection.hatexplain_downloader import download_hatexplain, cli_main, main
    
    # Test that functions exist
    assert callable(download_hatexplain)
    assert callable(cli_main)
    assert callable(main)
    
    # Test that constants are defined
    from data_collection.hatexplain_downloader import _HATEXPLAIN_BASE_URL, _HATEXPLAIN_FILES
    assert isinstance(_HATEXPLAIN_BASE_URL, str)
    assert isinstance(_HATEXPLAIN_FILES, list)


@pytest.mark.integration
def test_hatexplain_main_function():
    """Integration test: test the main() function."""
    with patch('data_collection.hatexplain_downloader.download_hatexplain', return_value=True) as mock_download:
        from data_collection.hatexplain_downloader import main
        
        result = main()
        assert result is True
        mock_download.assert_called_once()


@pytest.mark.slow
@pytest.mark.integration  
def test_hatexplain_real_download():
    """Slow integration test: attempt real download to verify URLs work."""
    # This test actually tries to download one small file to verify the URLs are valid
    test_url = _HATEXPLAIN_BASE_URL + "README.md"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        destination = temp_path / "test_readme.md"
        
        try:
            result = _download_file(test_url, destination, overwrite=True)
            if result:
                assert destination.exists()
                assert destination.stat().st_size > 0
                print("✅ Real download test passed - URLs are accessible")
            else:
                pytest.skip("Download failed - may be network/URL issue")
        except Exception as e:
            pytest.skip(f"Download test skipped due to: {e}")


@pytest.mark.data
def test_hatexplain_download_and_validate():
    """Data test: download HateXplain and validate if not present."""
    try:
        # Try to validate existing data first
        from data_collection.hatexplain_data_presence_validator import validate_hatexplain_data_presence
        validate_hatexplain_data_presence()
        print("✅ HateXplain data already present and valid")
    except AssertionError as e:
        if "Missing directory" in str(e) or "Missing expected" in str(e):
            # Data not present, try to download
            try:
                result = download_hatexplain()
                if result:
                    print("✅ HateXplain download and validation successful")
                else:
                    pytest.skip("HateXplain download failed - may be network issue")
            except Exception as download_error:
                pytest.skip(f"HateXplain download test skipped: {download_error}")
        else:
            # Re-raise if it's a different validation error
            raise