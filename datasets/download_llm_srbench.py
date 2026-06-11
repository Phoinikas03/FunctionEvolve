#!/usr/bin/env python3
"""Reusable downloader for the llm-srbench dataset (HuggingFace: nnheui/llm-srbench).

Downloads the snapshot into ``datasets/llm-srbench`` (next to this script) so it matches
the layout expected by ``src/dataset.py`` (lsr_bench_data.hdf5 + data/*.parquet).

Usage
-----
    python datasets/download_llm_srbench.py            # download if missing
    python datasets/download_llm_srbench.py --force    # re-download even if present
    python datasets/download_llm_srbench.py --dest /some/dir/llm-srbench

The destination matches the directory name used throughout the repo (``llm-srbench``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

HF_REPO_ID = "nnheui/llm-srbench"
HDF5_FILENAME = "lsr_bench_data.hdf5"
DEFAULT_DEST = Path(__file__).resolve().parent / "llm-srbench"


def _has_expected_files(path: Path) -> bool:
    data_dir = path / "data"
    if not (path / HDF5_FILENAME).is_file() or not data_dir.is_dir():
        return False
    return any(data_dir.glob("*.parquet"))


def download(dest: Path, force: bool = False) -> Path:
    if _has_expected_files(dest) and not force:
        print(f"[llm-srbench] Already present at {dest} (use --force to re-download).")
        return dest
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with: pip install huggingface_hub"
        ) from e

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[llm-srbench] Downloading {HF_REPO_ID} -> {dest} ...")
    snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(dest))
    if not _has_expected_files(dest):
        raise SystemExit(
            f"[llm-srbench] Download finished but {dest} is missing expected files "
            f"({HDF5_FILENAME} and data/*.parquet)."
        )
    print(f"[llm-srbench] Done: {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"Destination directory (default: {DEFAULT_DEST}).")
    ap.add_argument("--force", action="store_true", help="Re-download even if already present.")
    args = ap.parse_args()
    download(args.dest, force=args.force)


if __name__ == "__main__":
    main()
