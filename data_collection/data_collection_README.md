# Data Collection Implementation

## Overview

The data collection module implements automated acquisition and validation infrastructure for two hate speech detection datasets: HateXplain and ToxiGen. The implementation provides deterministic data retrieval from authoritative sources with comprehensive validation mechanisms ensuring data integrity before downstream processing. The module architecture separates download orchestration from presence validation, enabling independent execution for initial acquisition and subsequent integrity verification.

## Module Components

### HateXplain Downloader (`hatexplain_downloader.py`)

The HateXplain downloader retrieves the explainable hate speech dataset from the official GitHub repository (https://github.com/hate-alert/HateXplain). The implementation downloads four core files: `dataset.json` (main annotations), `post_id_divisions.json` (train/val/test splits), `classes.npy` (three-class encoder: hatespeech/normal/offensive), and `classes_two.npy` (binary toxic/non-toxic encoder). The downloader implements HTTP retrieval via `urllib.request` with overwrite protection, preventing accidental data replacement. The dataset contains social media posts with multi-annotator hate speech labels and token-level attention explanations, supporting both classification and interpretability research.

### ToxiGen Downloader (`toxigen_downloader.py`)

The ToxiGen downloader fetches synthetic hate speech data from HuggingFace Hub (toxigen/toxigen-data) comprising three splits: `train` (~250k synthetic generations), `annotated` (human-evaluated subset), and `annotations` (raw human study data). The implementation leverages the `datasets` library for authenticated Hub access with automatic token detection from HuggingFace CLI credentials. The downloader supports dual output formats: Parquet (optimized for columnar analytics, requires `pyarrow`) and JSON Lines (text-based portability). Format selection through `--format` flag enables workflow-specific optimization, with Parquet preferred for large-scale processing and JSONL for human-readable inspection. The implementation materializes dataset splits to local disk (`data/toxigen/`) for stable offline access, eliminating repeated Hub queries during development and testing.

### Data Presence Validators (`hatexplain_data_presence_validator.py`, `toxigen_data_presence_validator.py`)

The validation modules implement comprehensive integrity checking through file existence verification, format validation, and record count reporting. The HateXplain validator searches for train/val/test splits across multiple formats (`.jsonl`, `.json`, `.parquet`, `.csv`) with flexible file naming patterns accommodating sharded outputs (e.g., `train-00000-of-00001.parquet`). The validator verifies HateXplain-specific core files (`dataset.json`, `post_id_divisions.json`) and optional encoders. Record counting implements format-specific optimizations: JSONL via line enumeration, JSON via deserialization, Parquet via metadata inspection using `pyarrow.ParquetFile.metadata.num_rows` (fast) with Pandas fallback, and CSV via dataframe loading. The ToxiGen validator restricts to Parquet format for consistency, validating the three expected splits (train/annotated/annotations) with identical metadata-based counting for performance.

## Usage

**Download HateXplain dataset:**
```bash
python -m data_collection.hatexplain_downloader --output-dir ./data/hatexplain
```

**Download ToxiGen dataset (Parquet format):**
```bash
python -m data_collection.toxigen_downloader --format parquet --output-dir ./data/toxigen
```

**Validate HateXplain data presence:**
```bash
python -m data_collection.hatexplain_data_presence_validator --data-dir ./data/hatexplain
```

**Validate ToxiGen data presence:**
```bash
python -m data_collection.toxigen_data_presence_validator --data-dir ./data/toxigen
```

## Design Principles

The implementation emphasizes reproducibility through deterministic source references (GitHub commit hashes, HuggingFace Hub dataset IDs), idempotency via overwrite protection preventing accidental data corruption, validation separation enabling independent integrity checks without re-downloading, and format flexibility supporting workflow-specific optimization (Parquet for analytics, JSONL for portability). The modular architecture enables CI/CD integration where downloaders execute during environment setup and validators run as pre-processing checks, ensuring data availability before pipeline execution.
