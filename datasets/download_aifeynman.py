#!/usr/bin/env python3
"""Reusable downloader for the AI-Feynman Symbolic Regression Database.

Fetches the large AI-Feynman data archives into ``datasets/aifeynman`` (next to
this script).  The CSV metadata files are intentionally kept as repo-tracked,
locally corrected canonical metadata and are not downloaded by default.

Source: https://space.mit.edu/home/tegmark/aifeynman.html
The host/filenames are configurable via --base-url in case the mirror moves.

Usage
-----
    python datasets/download_aifeynman.py            # download missing archives
    python datasets/download_aifeynman.py --force    # re-download archives
    python datasets/download_aifeynman.py --include-csv  # also fetch upstream CSVs
    python datasets/download_aifeynman.py --base-url https://your.mirror/path
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://space.mit.edu/home/tegmark"
DEFAULT_DEST = Path(__file__).resolve().parent / "aifeynman"

# Large data archives.  Metadata CSVs are tracked in git after local fixes, so
# the default downloader must not overwrite them.
ARCHIVE_FILES = [
    "Feynman_with_units.tar.gz",
    "bonus_with_units.tar.gz",
]

CSV_FILES = [
    "FeynmanEquations.csv",
    "BonusEquations.csv",
    "units.csv",
]


def _download_one(url: str, out: Path) -> bool:
    try:
        print(f"[aifeynman] GET {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(out, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"[aifeynman]   -> {out} ({out.stat().st_size} bytes)")
        return True
    except Exception as e:  # noqa: BLE001 - report and continue with other files
        if out.exists():
            out.unlink(missing_ok=True)
        print(f"[aifeynman]   !! failed: {e}")
        return False


def download(
    dest: Path,
    base_url: str,
    force: bool = False,
    include_csv: bool = False,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    files = ARCHIVE_FILES + (CSV_FILES if include_csv else [])
    for name in files:
        out = dest / name
        if out.exists() and not force:
            print(f"[aifeynman] skip (present): {name}")
            ok.append(name)
            continue
        (ok if _download_one(f"{base_url.rstrip('/')}/{name}", out) else failed).append(name)
    print(f"[aifeynman] Done. ok={len(ok)} failed={len(failed)} dest={dest}")
    if failed:
        print(f"[aifeynman] Could not fetch: {failed}\n"
              f"            Verify the URLs at https://space.mit.edu/home/tegmark/aifeynman.html "
              f"and retry with --base-url.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"Destination directory (default: {DEFAULT_DEST}).")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help=f"Base URL to download from (default: {DEFAULT_BASE_URL}).")
    ap.add_argument("--force", action="store_true", help="Re-download selected files even if present.")
    ap.add_argument(
        "--include-csv",
        action="store_true",
        help="Also download upstream CSV metadata. This can overwrite local canonical fixes when used with --force.",
    )
    args = ap.parse_args()
    download(args.dest, args.base_url, force=args.force, include_csv=args.include_csv)


if __name__ == "__main__":
    main()
