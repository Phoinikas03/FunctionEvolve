#!/usr/bin/env python3
"""Generate noisy variants of the llm-srbench dataset.

Creates copies of ``datasets/llm-srbench`` with a *relative* multiplicative perturbation
applied to the dependent variable (y):

    y' = y * (1 + eps * N(0, 1))

By default two variants are produced:
    datasets/llm-srbench-noise1pct   (eps = 0.01)
    datasets/llm-srbench-noise5pct   (eps = 0.05)

Each variant mirrors the source layout (lsr_bench_data.hdf5 + data/*.parquet + gt csv),
so it is a drop-in replacement usable via ``--data-dir`` / ``set_repo_dir(...)``.

Only the y column is changed; X stays identical. The HDF5 sample layout follows
``src/dataset.py``: each equation group holds split datasets whose first column is y
(2-D ``(n_samples, 1 + n_features)`` arrays), or a sub-group with ``X``/``y`` members.

Which splits get noise is controlled by --splits:
    train  (default) : only the training targets are perturbed (test/ood stay clean,
                       so NMSE on held-out data and GT-symbolic checks stay meaningful)
    all              : every split's y is perturbed

Usage
-----
    python datasets/make_noisy_llm_srbench.py                      # 1% and 5%, train only
    python datasets/make_noisy_llm_srbench.py --levels 0.01 0.05 0.1
    python datasets/make_noisy_llm_srbench.py --splits all
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

HDF5_FILENAME = "lsr_bench_data.hdf5"
DEFAULT_SRC = Path(__file__).resolve().parent / "llm-srbench"
TRAIN_SPLIT_NAMES = {"train", "train_data"}


def _pct_tag(eps: float) -> str:
    """0.01 -> '1pct', 0.05 -> '5pct', 0.025 -> '2.5pct'."""
    pct = eps * 100
    s = f"{pct:g}".rstrip("0").rstrip(".") if pct != int(pct) else str(int(pct))
    return f"{s}pct"


def _iter_sample_datasets(h5, splits: str):
    """Yield (perturb_fn, split_name) for each y-bearing array to perturb.

    perturb_fn(rng, eps) reads, perturbs and writes y back in place.
    """
    import h5py

    targets = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        comps = name.split("/")
        leaf = comps[-1]
        if leaf == "y":
            # Group format: .../<split>/y  ->  split is the parent component
            split_name = comps[-2] if len(comps) >= 2 else ""
        elif leaf == "X":
            return  # never perturb features
        else:
            # 2-D format: .../<split>  with column 0 == y
            split_name = leaf
        if splits == "train" and split_name not in TRAIN_SPLIT_NAMES:
            return
        targets.append((name, leaf == "y"))

    h5.visititems(visit)
    return targets


def _perturb_file(hdf5_path: Path, eps: float, splits: str, seed: int) -> int:
    import h5py

    rng = np.random.default_rng(seed)
    n_changed = 0
    with h5py.File(hdf5_path, "r+") as h5:
        targets = _iter_sample_datasets(h5, splits)
        for name, is_group_y in sorted(targets):
            dset = h5[name]
            data = np.asarray(dset[()], dtype=np.float64)
            if is_group_y:
                noise = rng.standard_normal(size=data.shape)
                dset[...] = data * (1.0 + eps * noise)
            else:
                if data.ndim != 2 or data.shape[1] < 1:
                    continue
                y = data[:, 0]
                data[:, 0] = y * (1.0 + eps * rng.standard_normal(size=y.shape))
                dset[...] = data
            n_changed += 1
    return n_changed


def make_variant(src: Path, eps: float, splits: str, seed: int, force: bool) -> Path:
    if not (src / HDF5_FILENAME).is_file():
        raise SystemExit(
            f"Source dataset not found at {src} (expected {HDF5_FILENAME}). "
            f"Run datasets/download_llm_srbench.py first."
        )
    dest = src.parent / f"{src.name}-noise{_pct_tag(eps)}"
    if dest.exists():
        if not force:
            raise SystemExit(f"{dest} already exists (use --force to overwrite).")
        shutil.rmtree(dest)
    print(f"[noisy] Copying {src} -> {dest}")
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".cache", ".git"))
    n = _perturb_file(dest / HDF5_FILENAME, eps=eps, splits=splits, seed=seed)
    print(f"[noisy] eps={eps} ({_pct_tag(eps)}): perturbed y in {n} '{splits}' array(s) -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"Clean source dataset dir (default: {DEFAULT_SRC}).")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.01, 0.05],
                    help="Relative noise levels (default: 0.01 0.05).")
    ap.add_argument("--splits", choices=["train", "all"], default="train",
                    help="Which splits' y to perturb (default: train).")
    ap.add_argument("--seed", type=int, default=20260531, help="Base RNG seed.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing variant dirs.")
    args = ap.parse_args()
    for i, eps in enumerate(args.levels):
        # Distinct, deterministic seed per level.
        make_variant(args.src, eps=eps, splits=args.splits, seed=args.seed + i, force=args.force)


if __name__ == "__main__":
    main()
