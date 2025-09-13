import os
from pathlib import Path
import json
import sys
from typing import Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover
    pq = None  # type: ignore

# Accepted file extensions for splits
_ALLOWED_EXTS = [".jsonl", ".json", ".parquet", ".csv"]
_EXPECTED_SPLITS = ["train", "val", "test"]

# HateXplain specific files
_HATEXPLAIN_CORE_FILES = ["dataset.json", "post_id_divisions.json"]
_HATEXPLAIN_OPTIONAL_FILES = ["classes.npy", "classes_two.npy"]


def _find_split_file(data_dir: Path, split: str) -> Optional[Path]:
    """Return the first matching file for a given split name in data_dir.

    The function searches for files named like split.* with allowed extensions.
    """
    for ext in _ALLOWED_EXTS:
        candidate = data_dir / f"{split}{ext}"
        if candidate.is_file():
            return candidate
    # Also allow pattern split_*.* (e.g., train-00000-of-00001.parquet) common in sharded outputs
    for path in data_dir.glob(f"{split}*"):
        if path.is_file() and path.suffix in _ALLOWED_EXTS:
            return path
    return None


def _count_records(path: Path) -> Optional[int]:  # pragma: no cover - small IO helper
    ext = path.suffix.lower()
    try:
        if ext == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        if ext == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle object with key 'data' or list directly
            if isinstance(data, list):
                return len(data)
            for key in ("data", "examples", "rows"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return None
        if ext == ".csv":
            if pd is None:
                return None
            df = pd.read_csv(path)
            return len(df)
        if ext == ".parquet":
            # Prefer pyarrow metadata (fast) over full load
            if pq is not None:
                try:
                    pf = pq.ParquetFile(path)
                    return pf.metadata.num_rows  # type: ignore[attr-defined]
                except Exception:
                    pass
            if pd is not None:
                df = pd.read_parquet(path)
                return len(df)
            return None
    except Exception as e:  # pragma: no cover
        print(f"Warning: failed to count records in {path.name}: {e}", file=sys.stderr)
        return None
    return None


def _peek_fields(path: Path, max_fields: int = 6) -> List[str]:  # pragma: no cover - exploratory helper
    ext = path.suffix.lower()
    try:
        if ext in (".jsonl", ".json"):
            with path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if ext == ".jsonl":
                    import json as _json
                    obj = _json.loads(first_line) if first_line else {}
                else:
                    import json as _json
                    obj_raw = _json.load(f) if first_line == '' else _json.loads(first_line)
                    if isinstance(obj_raw, list) and obj_raw:
                        obj = obj_raw[0]
                    elif isinstance(obj_raw, dict):
                        # attempt to descend into first list value
                        for v in obj_raw.values():
                            if isinstance(v, list) and v:
                                obj = v[0]
                                break
                        else:
                            obj = obj_raw
                    else:
                        obj = {}
                if isinstance(obj, dict):
                    return list(obj.keys())[:max_fields]
        if ext in (".csv", ".parquet") and pd is not None:
            if ext == ".csv":
                df = pd.read_csv(path, nrows=1)
            else:
                df = pd.read_parquet(path, columns=None)
                df = df.head(1)
            return list(df.columns)[:max_fields]
    except Exception:
        return []
    return []


def validate_hatexplain_data_presence():
    """Validate that HateXplain data directory & expected files exist and print a summary.

    HateXplain dataset comes as a single dataset.json file with split definitions
    in post_id_divisions.json, rather than separate train/val/test files.

    This function is intentionally tolerant about exact file formats to accommodate
    various pre-processing or export steps. It will raise an AssertionError with a clear
    message if data is missing, guiding how to populate the data.

    Returns:
        Dict containing summary information about detected dataset and splits.
    """
    project_root = Path(__file__).resolve().parents[1]  # Go up to project root
    data_dir = project_root / "data" / "hatexplain"
    
    if not data_dir.is_dir():
        raise AssertionError(
            "Missing directory 'data/hatexplain'. Please download the HateXplain dataset "
            "(e.g., via hatexplain_downloader.py or from GitHub) before running validation."
        )

    # Check for core HateXplain files
    missing_files = []
    summary = {}
    
    for filename in _HATEXPLAIN_CORE_FILES:
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
            continue
            
        # Get file info
        file_size = file_path.stat().st_size
        summary[filename] = {
            "file": str(file_path.relative_to(project_root)),
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2)
        }
    
    if missing_files:
        raise AssertionError(
            f"Missing expected HateXplain core files: {', '.join(missing_files)}. "
            "Please download the dataset using hatexplain_downloader.py or manually "
            "from https://github.com/hate-alert/HateXplain/tree/master/Data"
        )
    
    # Analyze dataset.json content
    dataset_file = data_dir / "dataset.json"
    try:
        with dataset_file.open('r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, dict):
            raise AssertionError("dataset.json does not contain a valid dictionary")
            
        total_entries = len(dataset)
        summary["dataset.json"]["total_entries"] = total_entries
        
        # Sample a few entries to get field information
        if total_entries > 0:
            sample_key = next(iter(dataset.keys()))
            sample_entry = dataset[sample_key]
            
            # Check required fields
            required_fields = ['post_id', 'annotators', 'rationales', 'post_tokens']
            present_fields = [field for field in required_fields if field in sample_entry]
            missing_required = [field for field in required_fields if field not in sample_entry]
            
            summary["dataset.json"]["sample_fields"] = present_fields
            if missing_required:
                summary["dataset.json"]["missing_fields"] = missing_required
                
            # Count tokens in sample
            if 'post_tokens' in sample_entry:
                summary["dataset.json"]["sample_tokens"] = len(sample_entry['post_tokens'])
                
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise AssertionError(f"dataset.json is corrupted or not valid JSON: {e}")
    except Exception as e:
        raise AssertionError(f"Error reading dataset.json: {e}")
    
    # Analyze post_id_divisions.json for split information
    divisions_file = data_dir / "post_id_divisions.json"
    try:
        with divisions_file.open('r', encoding='utf-8') as f:
            divisions = json.load(f)
            
        split_info = {}
        for split_name in _EXPECTED_SPLITS:
            if split_name in divisions:
                split_ids = divisions[split_name]
                split_info[split_name] = len(split_ids) if isinstance(split_ids, list) else "?"
            else:
                split_info[split_name] = "missing"
                
        summary["post_id_divisions.json"]["splits"] = split_info
        
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise AssertionError(f"post_id_divisions.json is corrupted or not valid JSON: {e}")
    except Exception as e:
        raise AssertionError(f"Error reading post_id_divisions.json: {e}")
    
    # Check for optional files
    optional_summary = {}
    for filename in _HATEXPLAIN_OPTIONAL_FILES:
        file_path = data_dir / filename
        if file_path.exists():
            optional_summary[filename] = {
                "file": str(file_path.relative_to(project_root)),
                "size_bytes": file_path.stat().st_size
            }
    
    if optional_summary:
        summary["optional_files"] = optional_summary
    
    # Print a human-readable summary
    print("HateXplain dataset summary:")
    print(f"  📊 Total entries: {summary['dataset.json']['total_entries']:,}")
    
    # Show split information
    splits = summary["post_id_divisions.json"]["splits"]
    print(f"  📂 Data splits:")
    for split_name, count in splits.items():
        if isinstance(count, int):
            print(f"    - {split_name}: {count:,} post IDs")
        else:
            print(f"    - {split_name}: {count}")
    
    # Show sample fields
    if "sample_fields" in summary["dataset.json"]:
        fields = summary["dataset.json"]["sample_fields"]
        print(f"  🔍 Sample fields: {', '.join(fields)}")
    
    # Show files
    print(f"  📁 Core files:")
    for filename in _HATEXPLAIN_CORE_FILES:
        if filename in summary:
            size_mb = summary[filename]["size_mb"]
            print(f"    - {filename}: {size_mb} MB")
    
    if "optional_files" in summary:
        print(f"  📁 Optional files:")
        for filename, info in summary["optional_files"].items():
            size_kb = round(info["size_bytes"] / 1024, 1)
            print(f"    - {filename}: {size_kb} KB")
    
    # Basic sanity check
    total_split_ids = sum(
        count for count in splits.values() 
        if isinstance(count, int)
    )
    
    if total_split_ids > 0 and total_split_ids != summary["dataset.json"]["total_entries"]:
        print(f"  ⚠️  Warning: Split IDs total ({total_split_ids:,}) != dataset entries ({summary['dataset.json']['total_entries']:,})")
    
    return summary


def test_hatexplain_data_presence(capsys):
    """Test that HateXplain data directory & expected splits exist and print a summary.

    This test is intentionally tolerant about exact file formats to accommodate
    various pre-processing or export steps. It will fail fast with a clear
    message guiding how to populate the data if missing.
    """
    return validate_hatexplain_data_presence()


if __name__ == "__main__":
    """Run the validator when executed directly."""
    try:
        summary = validate_hatexplain_data_presence()
        print(f"\n✅ Validation successful! Found {len(summary)} splits.")
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)
