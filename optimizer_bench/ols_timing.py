#!/usr/bin/env python3
"""Average full _ols_core single-evaluation time over 10 real expressions from logs/full.

Times the entire _ols_core call (design-matrix basis eval + fixed-part eval +
lstsq + y_pred/residual/MSE), not just the lstsq solve. Cache-hit short-circuit
calls are tracked separately so the reported figure is the true cost of one full
VARPRO OLS evaluation.

Run with OLS_PROFILE=1 so structure.py records the per-call timing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OLS_PROFILE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizer import StructureOptimizer
from src.optimizer import structure as _struct
from optimizer_bench.bench_cases import generate_all_cases, _SPLITS

_HDF5 = Path(__file__).resolve().parent.parent / "datasets" / "llm-srbench" / "lsr_bench_data.hdf5"

TARGET_EQS = ["BPG0", "BPG18", "BPG16", "CRK33", "CRK6",
              "CRK28", "CRK17", "PO1", "PO22", "PO40"]


def _load_training_data(split, equation_name):
    import h5py
    prefix = _SPLITS[split]["hdf5_prefix"]
    with h5py.File(_HDF5, "r") as f:
        item = f[f"/{prefix}/{equation_name}"]
        keys = [k for k in item.keys() if "train" in k.lower()] or [list(item.keys())[0]]
        data = np.array(item[keys[0]], dtype=np.float64)
    return data[:, 1:], data[:, 0]


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    assert _struct._OLS_PROFILE_ON, "set OLS_PROFILE=1 before import"

    cases = {c.equation_name: c for c in generate_all_cases(max_per_split=0)
             if c.transform == "T0"}

    rows = []
    for eq in TARGET_EQS:
        case = cases.get(eq)
        if case is None:
            print(f"  [skip] {eq}: no T0 case"); continue
        try:
            X, y = _load_training_data(case.split, case.equation_name)
        except Exception as e:
            print(f"  [skip] {eq}: data load failed {e}"); continue

        _struct._OLS_PROF.update(full_n=0, full_t=0.0, hit_n=0, hit_t=0.0)
        opt = StructureOptimizer(timeout=timeout, max_iter=1000,
                                 bound=100.0, penalty=1e20)
        t0 = time.perf_counter()
        try:
            opt.optimize(case.skeleton, case.param_names,
                         case.feature_names, X, y)
        except Exception as e:
            print(f"  [skip] {eq}: optimize raised {e}"); continue
        wall = time.perf_counter() - t0

        p = dict(_struct._OLS_PROF)
        fn, ft = p["full_n"], p["full_t"]
        hn = p["hit_n"]
        mean_us = (ft / fn * 1e6) if fn else float("nan")
        rows.append((eq, case.n_params, X.shape[0], fn, hn, mean_us, ft, wall))
        print(f"  {eq:7s} np={case.n_params} ns={X.shape[0]:4d} nf={X.shape[1]} | "
              f"full evals={fn:7d} (cache hits={hn:6d}) mean={mean_us:8.2f}us "
              f"total={ft:6.3f}s ({(ft/wall*100 if wall else 0):4.1f}% of {wall:5.2f}s)")

    if not rows:
        print("nothing measured"); return

    tot_full = sum(r[3] for r in rows)
    tot_hit = sum(r[4] for r in rows)
    tot_ft = sum(r[6] for r in rows)
    tot_wall = sum(r[7] for r in rows)
    means = [r[5] for r in rows if r[3]]
    print("\n" + "=" * 74)
    print(f"expressions measured           : {len(rows)}")
    print(f"total FULL _ols_core evals     : {tot_full:,}  (+ {tot_hit:,} cache-hit calls)")
    print(f"total full-eval time           : {tot_ft:.3f} s")
    print(f"GLOBAL mean per full _ols_core : {tot_ft/tot_full*1e6:.2f} us")
    print(f"per-expression mean            : avg={np.mean(means):.2f}us  "
          f"median={np.median(means):.2f}us  "
          f"range=[{min(means):.2f}, {max(means):.2f}]us")
    print(f"_ols_core share of optimize    : {tot_ft/tot_wall*100:.1f}% "
          f"({tot_ft:.2f}s / {tot_wall:.2f}s wall)")


if __name__ == "__main__":
    main()
