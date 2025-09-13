"""Utility to download and materialize the ToxiGen dataset locally.

This script fetches the three published splits from the Hugging Face Hub:
  - train (synthetic generation set ~250k rows)
  - annotated (human evaluation processed)
  - annotations (raw human study annotations)

It then writes them into ``data/toxigen`` (configurable) in either Parquet or
JSON Lines format so that tests (e.g. ``tests/test_toxigen_data_presence.py``)
and downstream pipelines can operate on a stable on-disk representation
without re-querying the Hub each run.

Example (CLI):
	python -m data_collection.toxigen_downloader --format parquet --overwrite

Programmatic:
	from data_collection.toxigen_downloader import download_toxigen
	download_toxigen(format="jsonl")

Notes:
  * Requires the ``datasets`` library (``pip install datasets``) and for
	Parquet output also ``pyarrow``. For JSON Lines only the former is needed.
  * ``use_auth_token`` defaults to True so that if your HF CLI is logged in
	the authenticated token is used automatically (some mirrors require it).
  * If a file already exists and ``overwrite`` is False, that split is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional
import os


try:  # Import lazily so the module can still be introspected without deps.
	from datasets import load_dataset, Dataset
except Exception as e:  # pragma: no cover - import error path
	load_dataset = None  # type: ignore
	_IMPORT_ERROR = e
else:  # pragma: no cover - only executed when available (runtime path)
	_IMPORT_ERROR = None


SplitName = Literal["train", "annotated", "annotations"]
DEFAULT_SPLITS: List[SplitName] = ["train", "annotated", "annotations"]
DATASET_ID = "toxigen/toxigen-data"


class DownloadError(RuntimeError):
	"""Raised when a required dependency or condition prevents download."""


def _ensure_datasets_available() -> None:
	if load_dataset is None:
		raise DownloadError(
			"The 'datasets' library is required but not importable. Install with 'pip install datasets'."
		) from _IMPORT_ERROR  # type: ignore[arg-type]


def _materialize_split(
	split: SplitName,
	output_dir: Path,
	file_format: Literal["parquet", "jsonl"],
	overwrite: bool,
	use_auth_token: Optional[bool | str],
	keep_columns: Optional[Iterable[str]],
) -> Path:
	"""Download a single split and write it to disk.

	Returns the path to the written (or pre-existing) file.
	"""
	ext = "parquet" if file_format == "parquet" else "jsonl"
	out_path = output_dir / f"{split}.{ext}"
	if out_path.exists() and not overwrite:
		return out_path

	ds_loaded = load_dataset(  # type: ignore[operator]
		DATASET_ID, name=split, token=use_auth_token
	)
	# Some configs may still return a DatasetDict with a single split key;
	# normalize to a Dataset instance.
	if hasattr(ds_loaded, "keys") and not isinstance(ds_loaded, Dataset):  # DatasetDict-like
		# Try common keys
		for key in ("train", split, "default"):
			if key in ds_loaded:  # type: ignore[operator]
				ds = ds_loaded[key]  # type: ignore[index]
				break
		else:  # fallback to first value
			first_key = list(ds_loaded.keys())[0]  # type: ignore[operator]
			ds = ds_loaded[first_key]  # type: ignore[index]
	else:
		ds = ds_loaded  # type: ignore[assignment]

	if keep_columns:
		# Only keep columns that exist (tolerant to mismatch per split)
		existing = [c for c in keep_columns if c in ds.column_names]
		if existing:
			ds = ds.select_columns(existing)

	if file_format == "parquet":
		# dataset has a native to_parquet method
		ds.to_parquet(str(out_path))  # pragma: no cover - IO heavy
	else:  # jsonl
		# Write as JSON Lines for streaming friendliness
		ds.to_json(str(out_path), orient="records", lines=True)  # pragma: no cover

	return out_path


def download_toxigen(
	output_dir: Path | str = "data/toxigen",
	format: Literal["parquet", "jsonl"] = "parquet",
	overwrite: bool = False,
	use_auth_token: Optional[bool | str] = True,
	splits: Optional[Iterable[SplitName]] = None,
	keep_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
	"""Download and materialize the ToxiGen dataset splits.

	Parameters
	----------
	output_dir : Path | str
		Directory where files will be written (created if needed).
	format : {"parquet", "jsonl"}
		Output file format. Parquet is columnar & space efficient; JSONL is
		human-readable & streaming friendly.
	overwrite : bool
		If False (default), existing files are left untouched and treated as
		successful results.
	use_auth_token : bool | str | None
		Passed through to Hugging Face ``load_dataset``. ``True`` uses the
		cached token from `huggingface-cli login` if available.
	splits : iterable
		Custom subset of splits; defaults to the canonical three.
	keep_columns : iterable
		Optional whitelist of columns to retain (intersection applied per
		split). Useful to shrink on-disk size if you only need text/labels.

	Returns
	-------
	Dict[str, Path]
		Mapping from split name to the path of the written (or reused) file.
	"""
	_ensure_datasets_available()

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	chosen_splits: List[SplitName] = list(splits) if splits else list(DEFAULT_SPLITS)
	results: Dict[str, Path] = {}
	for split in chosen_splits:
		try:
			print(f"[toxigen] Processing split '{split}' (format={format}, overwrite={overwrite})")
			path = _materialize_split(
				split=split,
				output_dir=output_dir,
				file_format=format,
				overwrite=overwrite,
				use_auth_token=use_auth_token,
				keep_columns=keep_columns,
			)
		except Exception as e:
			raise DownloadError(f"Failed to materialize split '{split}': {e}") from e
		results[split] = path
	return results


def _summarize(paths: Dict[str, Path]) -> str:  # pragma: no cover - formatting helper
	lines = ["ToxiGen splits materialized:"]
	for split, path in paths.items():
		try:
			size_mb = path.stat().st_size / 1_048_576
			lines.append(f"  - {split:<11} -> {path} ({size_mb:.2f} MB)")
		except OSError:
			lines.append(f"  - {split:<11} -> {path} (size unknown)")
	return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Download & persist the ToxiGen dataset splits")
	p.add_argument(
		"--output-dir", "-o", default="data/toxigen", help="Directory to place the materialized splits"
	)
	p.add_argument(
		"--format",
		"-f",
		choices=["parquet", "jsonl"],
		default="parquet",
		help="Output file format (parquet recommended)",
	)
	p.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing files if they already exist",
	)
	p.add_argument(
		"--no-auth",
		action="store_true",
		help="Do not use a Hugging Face auth token (may fail if gated)",
	)
	p.add_argument(
		"--token",
		default=None,
		help="Explicit Hugging Face token (overrides HF_TOKEN / default auth). Avoid embedding secrets in scripts.",
	)
	p.add_argument(
		"--splits",
		nargs="*",
		default=None,
		help="Custom subset of splits (default: train annotated annotations)",
	)
	p.add_argument(
		"--keep-columns",
		nargs="*",
		default=None,
		help="Optional list of columns to retain (intersection per split)",
	)
	return p


def cli_main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI wrapper
	# Resolve auth preference order: --no-auth > --token > env > default(True)
	parser = _build_arg_parser()
	args = parser.parse_args(argv)
	resolved_token: Optional[bool | str]
	if args.no_auth:
		resolved_token = None
	elif args.token:
		resolved_token = args.token
	else:
		resolved_token = os.environ.get("HF_TOKEN") or True
	try:
		paths = download_toxigen(
			output_dir=args.output_dir,
			format=args.format,
			overwrite=args.overwrite,
			use_auth_token=resolved_token,
			splits=args.splits,
			keep_columns=args.keep_columns,
		)
	except DownloadError as e:
		print(f"Error: {e}", file=sys.stderr)
		return 2
	except Exception as e:  # Unknown error
		print(f"Unexpected failure: {e}", file=sys.stderr)
		return 1
	print(_summarize(paths))
	return 0


def main() -> Dict[str, Path]:
	"""Convenience programmatic entry point.

	Equivalent to calling ``download_toxigen()`` with all defaults. Returns
	the mapping of split name to written (or existing) file paths. This keeps
	a conventional ``main()`` available for import while the CLI uses
	``cli_main`` to avoid signature clashes.
	"""
	return download_toxigen()


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(cli_main())

