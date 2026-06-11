#!/usr/bin/env python3
"""
Optimizer benchmark main script (phased execution + multiprocessing).

Test cases are split into 4 phases by transform type, executed sequentially:
  T0  Original formula (no perturbation)
  T1  Add composite zero terms (T1a + T1b)
  T2  Feature/expression power augmentation (T2a + T2b)
  T4  Add rational zero terms (T4a + T4b)

Each phase evaluates 5 optimizers (L-BFGS-B / DE / CMA-ES / least_squares / Structure)
in parallel, outputs per-phase CSV and summary, then generates a final summary.

Usage:
    python -m optimizer_bench.run_bench [--phase T0,T1] [--timeout 60] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizer import get_optimizer, OPTIMIZER_NAMES
from optimizer_bench.bench_cases import (
    BenchCase, generate_all_cases, _find_snapshot_dir, _SPLITS,
)


# ---------------------------------------------------------------------------
# Phase definitions: transform names for each phase
# ---------------------------------------------------------------------------

PHASES = [
    ("T0", ["T0"]),
    ("T1", ["T1a", "T1b"]),
    ("T2", ["T2a", "T2b"]),
    ("T4", ["T4a", "T4b"]),
]

PHASE_NAMES = [p[0] for p in PHASES]

PHASE_DESCRIPTIONS = {
    "T0": "Original formula (no perturbation)",
    "T1": "Composite zero terms (T1a + T1b)",
    "T2": "Feature/expression power (T2a + T2b)",
    "T4": "Rational zero terms (T4a + T4b)",
}


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _worker(args):
    """
    Execute a single (case, optimizer) combination in a subprocess.
    """
    (case_id, split, equation_name, transform, skeleton,
     param_names, gt_params, feature_names, n_params,
     optimizer_name, X, y, timeout, n_starts) = args

    t0 = time.perf_counter()
    var_y = float(np.var(y))
    timed_out = False

    try:
        opt = get_optimizer(
            optimizer_name,
            timeout=timeout,
            n_restarts=n_starts,
            bound=100.0,
            penalty=1e20,
            max_iter=1000,
        )
        result = opt.optimize(
            skeleton=skeleton,
            param_names=list(param_names),
            feature_names=list(feature_names),
            X_train=X,
            y_train=y,
            parent_params=None,
        )
        best_mse = result.best_mse
        n_feval = result.n_feval
    except TimeoutError:
        timed_out = True
        best_mse = float("inf")
        n_feval = 0
    except Exception:
        best_mse = float("inf")
        n_feval = 0

    elapsed = time.perf_counter() - t0
    if elapsed >= timeout:
        timed_out = True
    nmse = best_mse / var_y if var_y > 0 else float("inf")

    return {
        "case_id": case_id, "split": split,
        "equation_name": equation_name, "transform": transform,
        "n_params": n_params, "optimizer": optimizer_name,
        "time_s": round(elapsed, 3),
        "final_mse": best_mse,
        "final_nmse": nmse,
        "converged": nmse < 1e-10,
        "n_feval": n_feval,
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training_data(split: str, equation_name: str) -> Tuple[np.ndarray, np.ndarray]:
    import h5py
    snap_dir = _find_snapshot_dir()
    hdf5_path = snap_dir / "lsr_bench_data.hdf5"
    hdf5_prefix = _SPLITS[split]["hdf5_prefix"]
    eq_path = f"/{hdf5_prefix}/{equation_name}"
    with h5py.File(hdf5_path, "r") as f:
        item = f[eq_path]
        train_key = [k for k in item.keys() if "train" in k.lower()]
        if not train_key:
            train_key = [list(item.keys())[0]]
        data = np.array(item[train_key[0]], dtype=np.float64)
    return data[:, 1:], data[:, 0]


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "case_id", "split", "equation_name", "transform", "n_params",
    "optimizer", "time_s", "final_mse", "final_nmse",
    "converged", "n_feval", "timed_out",
]


def _write_csv(results: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["case_id"], x["optimizer"])):
            writer.writerow({
                "case_id": r["case_id"],
                "split": r["split"],
                "equation_name": r["equation_name"],
                "transform": r["transform"],
                "n_params": r["n_params"],
                "optimizer": r["optimizer"],
                "time_s": r["time_s"],
                "final_mse": f"{r['final_mse']:.6e}",
                "final_nmse": f"{r['final_nmse']:.6e}",
                "converged": r["converged"],
                "n_feval": r["n_feval"],
                "timed_out": r["timed_out"],
            })


# ---------------------------------------------------------------------------
# Single phase execution
# ---------------------------------------------------------------------------

def _run_phase(
    phase_name: str,
    phase_cases: List[BenchCase],
    data_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    timeout: float,
    n_starts: int,
    n_workers: int,
    output_dir: Path,
) -> List[dict]:
    """Execute all (case x optimizer) tasks for a single phase, return results list."""

    desc = PHASE_DESCRIPTIONS.get(phase_name, "")
    n_cases = len(phase_cases)
    n_tasks = n_cases * len(OPTIMIZER_NAMES)

    print(f"\n{'='*70}")
    print(f"Phase {phase_name}: {desc}")
    print(f"  Cases: {n_cases}, Optimizers: {len(OPTIMIZER_NAMES)}, "
          f"Total tasks: {n_tasks}")
    print(f"{'='*70}")

    tasks = []
    for case in phase_cases:
        if case.equation_name not in data_cache:
            continue
        X, y = data_cache[case.equation_name]
        for opt_name in OPTIMIZER_NAMES:
            tasks.append((
                case.case_id, case.split, case.equation_name, case.transform,
                case.skeleton, case.param_names, case.gt_params,
                case.feature_names, case.n_params,
                opt_name, X, y, timeout, n_starts,
            ))

    if not tasks:
        print("  (no runnable tasks)")
        return []

    hard_timeout = timeout + 15
    results: List[dict] = []
    done = 0
    t_phase_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_worker, t): t for t in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            done += 1
            task = future_to_task[future]
            case_id, split, equation_name, transform = task[:4]
            optimizer_name = task[9]
            n_params = task[8]
            try:
                r = future.result(timeout=hard_timeout)
            except (concurrent.futures.TimeoutError, TimeoutError):
                r = {
                    "case_id": case_id, "split": split,
                    "equation_name": equation_name, "transform": transform,
                    "n_params": n_params, "optimizer": optimizer_name,
                    "time_s": hard_timeout, "final_mse": float("inf"),
                    "final_nmse": float("inf"), "converged": False,
                    "n_feval": 0, "timed_out": True,
                }
            except Exception:
                r = {
                    "case_id": case_id, "split": split,
                    "equation_name": equation_name, "transform": transform,
                    "n_params": n_params, "optimizer": optimizer_name,
                    "time_s": 0.0, "final_mse": float("inf"),
                    "final_nmse": float("inf"), "converged": False,
                    "n_feval": 0, "timed_out": False,
                }

            status = "✓" if r["converged"] else ("⏰" if r["timed_out"] else "✗")
            print(
                f"  [{done:4d}/{len(tasks)}] {status} "
                f"{r['case_id']:25s} {r['optimizer']:14s} "
                f"NMSE={r['final_nmse']:.2e} "
                f"t={r['time_s']:6.1f}s "
                f"feval={r['n_feval']:7d}",
                flush=True,
            )
            results.append(r)

    phase_elapsed = time.time() - t_phase_start

    phase_csv = output_dir / f"results_{phase_name}.csv"
    _write_csv(results, phase_csv)
    print(f"\n  Phase {phase_name} done: {len(results)} results, "
          f"elapsed {phase_elapsed:.1f}s -> {phase_csv}")

    _print_phase_summary(phase_name, results)
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_phase_summary(phase_name: str, results: List[dict]):
    """Print per-optimizer summary for a single phase."""
    if not results:
        return
    desc = PHASE_DESCRIPTIONS.get(phase_name, "")
    print(f"\n  --- {phase_name} ({desc}) summary by optimizer ---")
    print(f"  {'Optimizer':14s} {'Conv%':>6s} {'AvgTime':>8s} {'MedTime':>8s} "
          f"{'AvgNMSE':>10s} {'AvgFeval':>9s} {'Timeout%':>8s}")
    for opt in OPTIMIZER_NAMES:
        rs = [r for r in results if r["optimizer"] == opt]
        if not rs:
            continue
        n = len(rs)
        conv = sum(1 for r in rs if r["converged"]) / n * 100
        times = [r["time_s"] for r in rs]
        avg_t, med_t = np.mean(times), np.median(times)
        nmse_vals = [r["final_nmse"] for r in rs if np.isfinite(r["final_nmse"])]
        avg_nmse = np.mean(nmse_vals) if nmse_vals else float("inf")
        avg_feval = np.mean([r["n_feval"] for r in rs])
        to_pct = sum(1 for r in rs if r["timed_out"]) / n * 100
        print(f"  {opt:14s} {conv:5.1f}% {avg_t:7.1f}s {med_t:7.1f}s "
              f"{avg_nmse:10.2e} {avg_feval:8.0f} {to_pct:7.1f}%")


def _print_full_summary(all_results: List[dict]):
    """Print cross-phase overall summary."""
    if not all_results:
        return

    print("\n" + "=" * 70)
    print("Global Summary")
    print("=" * 70)

    print("\n--- By Optimizer ---")
    print(f"{'Optimizer':14s} {'Conv%':>6s} {'AvgTime':>8s} {'MedTime':>8s} "
          f"{'AvgNMSE':>10s} {'AvgFeval':>9s} {'Timeout%':>8s}")
    for opt in OPTIMIZER_NAMES:
        rs = [r for r in all_results if r["optimizer"] == opt]
        if not rs:
            continue
        n = len(rs)
        conv = sum(1 for r in rs if r["converged"]) / n * 100
        times = [r["time_s"] for r in rs]
        avg_t, med_t = np.mean(times), np.median(times)
        nmse_vals = [r["final_nmse"] for r in rs if np.isfinite(r["final_nmse"])]
        avg_nmse = np.mean(nmse_vals) if nmse_vals else float("inf")
        avg_feval = np.mean([r["n_feval"] for r in rs])
        to_pct = sum(1 for r in rs if r["timed_out"]) / n * 100
        print(f"{opt:14s} {conv:5.1f}% {avg_t:7.1f}s {med_t:7.1f}s "
              f"{avg_nmse:10.2e} {avg_feval:8.0f} {to_pct:7.1f}%")

    phase_transforms = {pn: ts for pn, ts in PHASES}
    active_phases = sorted(
        set(r["transform"][:2] if r["transform"] != "T0" else "T0"
            for r in all_results),
        key=lambda x: PHASE_NAMES.index(x) if x in PHASE_NAMES else 99,
    )

    print("\n--- Convergence Rate Cross Table (Phase x Optimizer) ---")
    header = f"{'Phase':>8s}"
    for opt in OPTIMIZER_NAMES:
        header += f" {opt:>14s}"
    print(header)
    for pn in active_phases:
        transforms = phase_transforms.get(pn, [pn])
        row = f"{pn:>8s}"
        for opt in OPTIMIZER_NAMES:
            rs = [r for r in all_results
                  if r["optimizer"] == opt and r["transform"] in transforms]
            if rs:
                conv = sum(1 for r in rs if r["converged"]) / len(rs) * 100
                row += f" {conv:13.1f}%"
            else:
                row += f" {'—':>14s}"
        print(row)

    print("\n--- By Dataset ---")
    print(f"{'Split':8s} {'Conv%':>6s} {'AvgTime':>8s}")
    for split in sorted(set(r["split"] for r in all_results)):
        rs = [r for r in all_results if r["split"] == split]
        if not rs:
            continue
        n = len(rs)
        conv = sum(1 for r in rs if r["converged"]) / n * 100
        avg_t = np.mean([r["time_s"] for r in rs])
        print(f"{split:8s} {conv:5.1f}% {avg_t:7.1f}s")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_benchmark(
    max_per_split: int = 0,
    timeout: float = 60.0,
    n_starts: int = 5,
    n_workers: int = 8,
    output_dir: str = "optimizer_bench/results",
    phases: Optional[List[str]] = None,
):
    print("=" * 70)
    print("Optimizer Benchmark (Phased Execution)")
    print("=" * 70)

    if phases is None:
        run_phases = PHASES
    else:
        valid = {pn for pn, _ in PHASES}
        for p in phases:
            if p not in valid:
                print(f"[ERR] Unknown phase '{p}', available: {', '.join(PHASE_NAMES)}")
                return []
        run_phases = [(pn, ts) for pn, ts in PHASES if pn in phases]

    print(f"\nPhases to run: {', '.join(pn for pn, _ in run_phases)}")
    print(f"Optimizers: {', '.join(OPTIMIZER_NAMES)}")
    print(f"Timeout: {timeout}s, Restarts: {n_starts}, Workers: {n_workers}")

    all_cases = generate_all_cases(max_per_split=max_per_split)
    transform_to_cases: Dict[str, List[BenchCase]] = {}
    for c in all_cases:
        transform_to_cases.setdefault(c.transform, []).append(c)

    data_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    eq_names = sorted(set(c.equation_name for c in all_cases))
    print(f"\nPreloading {len(eq_names)} datasets...")
    for case in all_cases:
        if case.equation_name not in data_cache:
            try:
                X, y = _load_training_data(case.split, case.equation_name)
                data_cache[case.equation_name] = (X, y)
            except Exception as e:
                print(f"  [ERR] {case.equation_name}: {e}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[dict] = []
    t_total_start = time.time()

    for phase_name, transforms in run_phases:
        phase_cases: List[BenchCase] = []
        for tf in transforms:
            phase_cases.extend(transform_to_cases.get(tf, []))

        if not phase_cases:
            print(f"\n[WARN] Phase {phase_name}: no matching cases, skipping")
            continue

        phase_results = _run_phase(
            phase_name, phase_cases, data_cache,
            timeout, n_starts, n_workers, out_dir,
        )
        all_results.extend(phase_results)

    total_elapsed = time.time() - t_total_start

    if all_results:
        combined_csv = out_dir / "results_all.csv"
        _write_csv(all_results, combined_csv)
        print(f"\nCombined results -> {combined_csv}")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    _print_full_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimizer Benchmark (Phased Execution)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--max-per-split", type=int, default=0,
                        help="Max GTs per split (0=all, default: 0)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout in seconds per (case, optimizer) task (default: 60)")
    parser.add_argument("--n-starts", type=int, default=3,
                        help="Random restarts per optimizer (default: 3)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--output-dir", type=str, default="optimizer_bench/results",
                        help="Output directory for results (default: optimizer_bench/results)")
    parser.add_argument(
        "--phase", type=str, default=None,
        help="Comma-separated list of phases, e.g. 'T0,T1' (default: all)\n"
             "Available: " + ", ".join(
                 f"{pn}={PHASE_DESCRIPTIONS[pn]}"
                 for pn in PHASE_NAMES
             ),
    )
    args = parser.parse_args()

    phases = None
    if args.phase:
        phases = [p.strip() for p in args.phase.split(",") if p.strip()]

    run_benchmark(
        max_per_split=args.max_per_split,
        timeout=args.timeout,
        n_starts=args.n_starts,
        n_workers=args.workers,
        output_dir=args.output_dir,
        phases=phases,
    )


if __name__ == "__main__":
    main()
