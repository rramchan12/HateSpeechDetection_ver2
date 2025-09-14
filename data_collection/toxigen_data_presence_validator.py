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

# Accepted file extensions for splits - Parquet only for efficiency
_ALLOWED_EXTS = [".parquet"]
_EXPECTED_SPLITS = ["train", "annotated", "annotations"]


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
    """Count records in Parquet files only."""
    ext = path.suffix.lower()
    try:
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
    """Get field names from Parquet files only."""
    ext = path.suffix.lower()
    try:
        if ext == ".parquet" and pd is not None:
            df = pd.read_parquet(path, columns=None)
            df = df.head(1)
            return list(df.columns)[:max_fields]
    except Exception:
        return []
    return []


def validate_toxigen_data_presence():
    """Validate that toxigen data directory & expected Parquet splits exist and print a summary.

    This function expects Parquet files for efficient data processing. It will raise an 
    AssertionError with a clear message if Parquet data is missing, guiding how to download them.

    Returns:
        Dict containing summary information about detected Parquet splits.
    """
    project_root = Path(__file__).resolve().parents[1]  # Go up to project root
    data_dir = project_root / "data" / "toxigen"
    
    if not data_dir.is_dir():
        raise AssertionError(
            "Missing directory 'data/toxigen'. Please download the ToxiGen dataset "
            "in Parquet format (e.g., via toxigen_downloader.py) before running validation."
        )

    summary: Dict[str, Dict[str, Optional[str]]] = {}
    missing_splits: List[str] = []

    for split in _EXPECTED_SPLITS:
        path = _find_split_file(data_dir, split)
        if not path:
            missing_splits.append(split)
            continue
        count = _count_records(path)
        fields = _peek_fields(path)
        summary[split] = {
            "file": str(path.relative_to(project_root)),
            "ext": path.suffix,
            "records": str(count) if count is not None else "?",
            "fields": ", ".join(fields) if fields else "?",
        }

    if missing_splits:
        raise AssertionError(
            "Missing expected toxigen data split files: " + ", ".join(missing_splits) +
            ". Place the Parquet files (train.parquet/annotated.parquet/annotations.parquet) "
            "in 'data/toxigen'. Use toxigen_downloader.py with --format parquet to download them."
        )

    # Print a human-readable summary
    print("ToxiGen data summary:")
    for split in _EXPECTED_SPLITS:
        meta = summary[split]
        print(f"  - {split}: {meta['records']} records | file={meta['file']} | fields={meta['fields']}")

    # Basic sanity: at least some rows in train if count known
    train_meta = summary.get("train")
    if train_meta and train_meta.get("records", "?") != "?":
        try:
            record_count = int(train_meta["records"])
            if record_count <= 0:
                raise AssertionError("Train split appears empty.")
        except ValueError:
            pass  # Skip check if records is not a valid number
    
    return summary


def test_toxigen_data_presence(capsys):
    """Test that toxigen data directory & expected Parquet splits exist and print a summary.

    This test expects Parquet files for efficient data processing. It will fail fast with a clear
    message guiding how to download the Parquet files if missing.
    """
    return validate_toxigen_data_presence()


if __name__ == "__main__":
    """Run the validator when executed directly."""
    try:
        summary = validate_toxigen_data_presence()
        print(f"\n>> Validation successful! Found {len(summary)} splits.")
    except AssertionError as e:
        print(f">> ERROR: Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f">> UNEXPECTED ERROR: {e}")
        sys.exit(1)