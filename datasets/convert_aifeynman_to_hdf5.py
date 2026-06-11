#!/usr/bin/env python3
"""Convert AI-Feynman text archives into a random-access HDF5 file.

The downloaded AI-Feynman payload stores one whitespace-delimited text file per
equation inside large .tar.gz archives.  Random access into those archives is
slow because gzip streams must be scanned from the beginning.  This script
materializes the data as HDF5:

    datasets/aifeynman/aifeynman_data.hdf5
      /feynman/<Filename>/X
      /feynman/<Filename>/y
      /bonus/<Filename>/X
      /bonus/<Filename>/y

Each input text row is ``x1 x2 ... xN y``.  Metadata from the CSV files is stored
as HDF5 attributes on each equation group.

Usage
-----
    python datasets/convert_aifeynman_to_hdf5.py
    python datasets/convert_aifeynman_to_hdf5.py --force
    python datasets/convert_aifeynman_to_hdf5.py --compression lzf
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "aifeynman"
DEFAULT_OUTPUT = DEFAULT_SOURCE_DIR / "aifeynman_data.hdf5"


ARCHIVES = {
    "feynman": {
        "csv": "FeynmanEquations.csv",
        "tar": "Feynman_with_units.tar.gz",
        "prefix": "Feynman_with_units",
        # The upstream tarball has three member names that do not exactly match
        # FeynmanEquations.csv.  The data columns match these canonical CSV rows.
        "aliases": {
            "I.15.10": "I.15.1",
            "I.48.20": "I.48.2",
            "II.11.7": "II.11.17",
        },
    },
    "bonus": {
        "csv": "BonusEquations.csv",
        "tar": "bonus_with_units.tar.gz",
        "prefix": "bonus_with_units",
        "aliases": {},
    },
}


def _json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_row_dict(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            continue
        out[str(key)] = _json_default(value)
    return out


def _load_equation_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df.dropna(subset=["Filename"]).copy()


def _load_units(units_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(units_path, encoding="utf-8-sig")
    df = df.dropna(subset=["Variable"])
    unit_map: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record = _clean_row_dict(row)
        var = str(record.pop("Variable"))
        record.pop("Unnamed: 7", None)
        unit_map[var] = record
    return unit_map


def _feature_names(row: pd.Series) -> list[str]:
    # A few AI-Feynman CSV rows have an incorrect "# variables" value, while
    # the v*_name columns and text data agree.  Treat the non-empty v*_name
    # columns as authoritative.
    names: list[str] = []
    for i in range(1, 11):
        value = row.get(f"v{i}_name")
        if pd.notna(value) and str(value):
            names.append(str(value))
    return names


def _ranges(row: pd.Series) -> Dict[str, list[float]]:
    ranges: Dict[str, list[float]] = {}
    for i, name in enumerate(_feature_names(row), start=1):
        lo = row.get(f"v{i}_low")
        hi = row.get(f"v{i}_high")
        if pd.notna(lo) and pd.notna(hi):
            ranges[name] = [float(lo), float(hi)]
    return ranges


def _variable_units(
    target: str,
    features: Iterable[str],
    units: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    names = [target, *list(features)]
    return {name: units[name] for name in names if name in units}


def _dataset_kwargs(compression: Optional[str]) -> Dict[str, Any]:
    if compression is None:
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": compression, "shuffle": True}


def _write_attrs(group: h5py.Group, attrs: Dict[str, Any]) -> None:
    for key, value in attrs.items():
        if isinstance(value, (dict, list, tuple)):
            group.attrs[key] = json.dumps(value, ensure_ascii=True, default=_json_default)
        elif value is None:
            continue
        else:
            group.attrs[key] = value


def _convert_split(
    *,
    h5: h5py.File,
    source_dir: Path,
    split: str,
    units: Dict[str, Dict[str, Any]],
    compression: Optional[str],
) -> int:
    spec = ARCHIVES[split]
    table = _load_equation_table(source_dir / spec["csv"])
    rows = {str(row["Filename"]): row for _, row in table.iterrows()}
    split_group = h5.require_group(split)
    kwargs = _dataset_kwargs(compression)
    converted = 0

    tar_path = source_dir / spec["tar"]
    prefix = spec["prefix"]
    aliases = spec.get("aliases", {})
    converted_names: set[str] = set()
    start = time.perf_counter()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            name = aliases.get(member_name, member_name)
            row = rows.get(name)
            if row is None:
                print(f"[{split}] skip unknown member: {member.name}", flush=True)
                continue

            case_start = time.perf_counter()
            fileobj = tar.extractfile(member)
            if fileobj is None:
                print(f"[{split}] skip unreadable member: {member.name}", flush=True)
                continue

            data = np.loadtxt(fileobj, dtype=np.float64)
            features = _feature_names(row)
            n_vars = len(features)
            if data.ndim != 2 or data.shape[1] != n_vars + 1:
                raise ValueError(
                    f"{member.name}: expected {n_vars + 1} columns, got {data.shape}"
                )

            case_group = split_group.require_group(name)
            for ds_name in ("X", "y"):
                if ds_name in case_group:
                    del case_group[ds_name]
            case_group.create_dataset("X", data=data[:, :n_vars], **kwargs)
            case_group.create_dataset("y", data=data[:, n_vars], **kwargs)

            target = str(row["Output"])
            _write_attrs(
                case_group,
                {
                    "source_archive": spec["tar"],
                    "source_member": f"{prefix}/{member_name}",
                    "filename": name,
                    "archive_filename": member_name,
                    "output": target,
                    "formula": str(row["Formula"]),
                    "n_variables": n_vars,
                    "csv_n_variables": int(row["# variables"]),
                    "feature_names": features,
                    "ranges": _ranges(row),
                    "metadata": _clean_row_dict(row),
                    "variable_units": _variable_units(target, features, units),
                },
            )

            converted += 1
            converted_names.add(name)
            elapsed = time.perf_counter() - case_start
            total_elapsed = time.perf_counter() - start
            alias_note = f" archive={member_name}" if member_name != name else ""
            print(
                f"[{split}] {converted:>3d}/{len(rows)} {name}: "
                f"shape={data.shape} write_s={elapsed:.2f} "
                f"total_s={total_elapsed:.1f}{alias_note}",
                flush=True,
            )

    missing = sorted(set(rows) - converted_names)
    if missing:
        print(f"[{split}] missing CSV rows not found in archive: {missing}", flush=True)
    return converted


def convert(
    source_dir: Path,
    output: Path,
    *,
    force: bool = False,
    compression: Optional[str] = None,
) -> None:
    source_dir = source_dir.resolve()
    output = output.resolve()

    for spec in ARCHIVES.values():
        for filename in (spec["csv"], spec["tar"]):
            path = source_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
    units_path = source_dir / "units.csv"
    if not units_path.is_file():
        raise FileNotFoundError(units_path)

    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    units = _load_units(units_path)
    started = time.perf_counter()
    try:
        with h5py.File(tmp_output, "w") as h5:
            h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            h5.attrs["source_dir"] = str(source_dir)
            h5.attrs["format"] = "aifeynman_hdf5_v1"
            h5.attrs["layout"] = "/<split>/<case>/X and /<split>/<case>/y"
            h5.attrs["compression"] = compression or "none"
            h5.attrs["units_json"] = json.dumps(units, ensure_ascii=True, default=_json_default)

            n_feynman = _convert_split(
                h5=h5,
                source_dir=source_dir,
                split="feynman",
                units=units,
                compression=compression,
            )
            n_bonus = _convert_split(
                h5=h5,
                source_dir=source_dir,
                split="bonus",
                units=units,
                compression=compression,
            )
            h5.attrs["n_feynman"] = n_feynman
            h5.attrs["n_bonus"] = n_bonus

        if output.exists():
            output.unlink()
        os.replace(tmp_output, output)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    print(
        f"[done] wrote {output} ({output.stat().st_size / (1024 ** 3):.2f} GiB) "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"AI-Feynman dataset directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HDF5 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="HDF5 dataset compression (default: none; lzf is fast, gzip is smaller but slow).",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    convert(
        source_dir=args.source_dir,
        output=args.output,
        force=args.force,
        compression=compression,
    )


if __name__ == "__main__":
    main()
