#!/usr/bin/env python3
"""
HateXplain Dataset Downloader

Downloads the HateXplain dataset from the official GitHub repository.
The dataset includes hate speech detection data with explainability annotations.

Repository: https://github.com/hate-alert/HateXplain
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Tuple
import json

# Base URL for HateXplain raw files on GitHub
_HATEXPLAIN_BASE_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"

# Core dataset files to download
_HATEXPLAIN_FILES = [
    "dataset.json",           # Main dataset with annotations
    "post_id_divisions.json", # Train/val/test split definitions
    "classes.npy",           # Three-class encoder (hatespeech, normal, offensive)
    "classes_two.npy",       # Two-class encoder (toxic, non-toxic)
]

# Optional files
_OPTIONAL_FILES = [
    "README.md",  # Dataset documentation
]


def _download_file(url: str, destination: Path, overwrite: bool = False) -> bool:
    """Download a single file from URL to destination path.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if downloaded successfully, False otherwise
    """
    if destination.exists() and not overwrite:
        print(f"✓ {destination.name} already exists (use --overwrite to replace)")
        return True
        
    try:
        print(f"📥 Downloading {destination.name}...")
        urllib.request.urlretrieve(url, destination)
        
        # Verify file was downloaded and has content
        if destination.exists() and destination.stat().st_size > 0:
            print(f"✅ Downloaded {destination.name} ({destination.stat().st_size:,} bytes)")
            return True
        else:
            print(f"❌ Failed to download {destination.name} - file is empty or missing")
            return False
            
    except urllib.error.URLError as e:
        print(f"❌ Failed to download {destination.name}: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error downloading {destination.name}: {e}")
        return False


def _validate_dataset_integrity(data_dir: Path) -> bool:
    """Validate that the downloaded dataset has the expected structure.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check main dataset file
    dataset_file = data_dir / "dataset.json"
    if not dataset_file.exists():
        print("❌ Main dataset.json file missing")
        return False
        
    try:
        with dataset_file.open('r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, dict) or len(dataset) == 0:
            print("❌ dataset.json appears to be empty or invalid")
            return False
            
        # Check a sample entry structure
        sample_key = next(iter(dataset.keys()))
        sample_entry = dataset[sample_key]
        
        required_fields = ['post_id', 'annotators', 'rationales', 'post_tokens']
        missing_fields = [field for field in required_fields if field not in sample_entry]
        
        if missing_fields:
            print(f"❌ Sample entry missing required fields: {missing_fields}")
            return False
            
        print(f"✅ Dataset validation passed - {len(dataset):,} entries found")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ dataset.json is not valid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False


def download_hatexplain(
    data_dir: Optional[str] = None,
    overwrite: bool = False,
    include_optional: bool = False
) -> bool:
    """Download the HateXplain dataset from GitHub.
    
    Args:
        data_dir: Directory to save dataset files (default: ./data/hatexplain)
        overwrite: Whether to overwrite existing files
        include_optional: Whether to download optional files like README
        
    Returns:
        True if download was successful, False otherwise
    """
    # Determine data directory
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data" / "hatexplain"
    else:
        data_dir = Path(data_dir)
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Downloading HateXplain dataset to: {data_dir}")
    
    # Determine files to download
    files_to_download = _HATEXPLAIN_FILES.copy()
    if include_optional:
        files_to_download.extend(_OPTIONAL_FILES)
    
    # Download each file
    success_count = 0
    failed_files = []
    
    for filename in files_to_download:
        url = _HATEXPLAIN_BASE_URL + filename
        destination = data_dir / filename
        
        if _download_file(url, destination, overwrite):
            success_count += 1
        else:
            failed_files.append(filename)
    
    # Report results
    total_files = len(files_to_download)
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {success_count}/{total_files}")
    
    if failed_files:
        print(f"   ❌ Failed: {', '.join(failed_files)}")
        
    # Validate dataset if core files downloaded successfully
    core_success = all(
        (data_dir / filename).exists() 
        for filename in _HATEXPLAIN_FILES
    )
    
    if core_success:
        print(f"\n🔍 Validating dataset integrity...")
        if _validate_dataset_integrity(data_dir):
            print(f"\n🎉 HateXplain dataset download complete!")
            print(f"   📂 Location: {data_dir}")
            print(f"   📄 Files: {success_count} downloaded")
            return True
        else:
            print(f"\n⚠️  Dataset downloaded but validation failed")
            return False
    else:
        print(f"\n❌ Download incomplete - missing core dataset files")
        return False


def cli_main():
    """Command-line interface for the HateXplain downloader."""
    parser = argparse.ArgumentParser(
        description="Download the HateXplain dataset from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hatexplain_downloader.py
  python hatexplain_downloader.py --data-dir ./my_data --overwrite
  python hatexplain_downloader.py --include-optional
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory to save dataset files (default: ./data/hatexplain)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download optional files like README.md"
    )
    
    args = parser.parse_args()
    
    # Run the download
    success = download_hatexplain(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        include_optional=args.include_optional
    )
    
    sys.exit(0 if success else 1)


def main():
    """Main entry point when imported as a module."""
    return download_hatexplain()


if __name__ == "__main__":
    cli_main()
