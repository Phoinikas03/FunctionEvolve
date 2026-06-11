# Datasets

Large data payloads are **not** tracked by git (see `../.gitignore`); only the helper
scripts (and `llm-srbench/gt_expressions.csv` + `build_gt_csv.py`) are versioned. Use the
scripts below to (re)materialize the datasets locally.

## Layout (after downloading)

```
datasets/
  llm-srbench/                 # HuggingFace nnheui/llm-srbench snapshot
    lsr_bench_data.hdf5        #   sample arrays (col 0 = y, cols 1.. = X)
    data/*.parquet             #   per-split metadata (name, symbols, expression, ...)
    gt_expressions.csv
  aifeynman/                   # AI-Feynman Symbolic Regression Database
    FeynmanEquations.csv  BonusEquations.csv  units.csv
    Feynman_with_units.tar.gz  bonus_with_units.tar.gz
  llm-srbench-noise1pct/       # generated: y *= (1 + 0.01 * N(0,1))  on train split
  llm-srbench-noise5pct/       # generated: y *= (1 + 0.05 * N(0,1))  on train split
```

The default on-disk location resolved by `src/dataset.py` is
`<LOCAL_ROOT>/symregression/datasets/llm-srbench` (override with
`$SYMREG_DATASETS_DIR`), so a download into `datasets/llm-srbench` is picked up
automatically without relying on the HuggingFace `~/.cache` directory.

## Download

```bash
python datasets/download_llm_srbench.py      # -> datasets/llm-srbench (HuggingFace)
python datasets/download_aifeynman.py        # -> datasets/aifeynman   (space.mit.edu)
```

Both are idempotent (skip when already present; `--force` to refresh).
`download_aifeynman.py` downloads only the large archives by default; the
AI-Feynman CSV metadata is tracked in git after local fixes.  Use
`--include-csv` only when intentionally fetching upstream metadata.  It also
takes `--base-url` in case the mirror moves.

For faster random access to AI-Feynman samples, convert the text archives to HDF5:

```bash
python datasets/convert_aifeynman_to_hdf5.py
```

This writes `datasets/aifeynman/aifeynman_data.hdf5` with groups
`/feynman/<case>/X`, `/feynman/<case>/y`, `/bonus/<case>/X`, and
`/bonus/<case>/y`.  The numeric payload stays ignored by git.

## Noisy llm-srbench variants

Relative multiplicative perturbation of the dependent variable `y`:
`y' = y * (1 + eps * N(0, 1))`.

```bash
python datasets/make_noisy_llm_srbench.py            # eps = 0.01 and 0.05, train split only
python datasets/make_noisy_llm_srbench.py --splits all
python datasets/make_noisy_llm_srbench.py --levels 0.01 0.05 0.1 --force
```

By default only the **train** targets are perturbed (test / ood stay clean so held-out
NMSE and GT-symbolic checks remain meaningful). Each variant mirrors the source layout
and is a drop-in replacement via `--data-dir datasets/llm-srbench-noise1pct` /
`set_repo_dir(...)`.
