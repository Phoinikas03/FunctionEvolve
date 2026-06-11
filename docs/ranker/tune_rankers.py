#!/usr/bin/env python3
"""Grid-search ranker parameters for selecting five candidates from full pools."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ranker_experiment import (  # noqa: E402
    Candidate,
    apply_complexities,
    compute_complexity_map,
    hit,
    iter_effective_logs,
    parse_final_candidates,
    parse_final_status,
    parse_matched_candidate_ids,
    safe_log,
)


SELECT_N = 5
_WORKER_CASES: list[dict[str, object]] = []
BASELINE_HEURISTIC_EXCLUDED_TASKS = {"direct_prompt", "llmsr", "openevolve"}


def complexity_score(cand: Candidate, param_weight: float, special_weight: float, op_weight: float) -> float:
    return (
        cand.tree_size
        + op_weight * cand.op_count
        + param_weight * cand.param_count
        + special_weight * cand.special_count
    )


def pareto_front_weighted(
    candidates: list[Candidate],
    param_weight: float,
    special_weight: float,
    op_weight: float,
) -> list[Candidate]:
    objectives = {
        cand.cid: (
            safe_log(cand.train_nmse),
            complexity_score(cand, param_weight, special_weight, op_weight),
        )
        for cand in candidates
    }
    front = []
    for cand in candidates:
        values = objectives[cand.cid]
        dominated = False
        for other in candidates:
            if other.cid == cand.cid:
                continue
            other_values = objectives[other.cid]
            if all(o <= v for o, v in zip(other_values, values)) and any(
                o < v for o, v in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


def select_pareto_variant(
    candidates: list[Candidate],
    param_weight: float,
    special_weight: float,
    op_weight: float,
    n: int = SELECT_N,
) -> list[int]:
    """Select by non-dominated sorting over train NMSE and complexity.

    The first Pareto front is selected first; if it contains fewer than ``n``
    candidates, it is removed and the next front is selected, until the fixed
    reporting budget is filled. Each front is ordered by training NMSE.
    """
    remaining = list(candidates)
    selected: list[int] = []
    while remaining and len(selected) < n:
        front = pareto_front_weighted(
            remaining,
            param_weight=param_weight,
            special_weight=special_weight,
            op_weight=op_weight,
        )
        if not front:
            break
        front = sorted(
            front,
            key=lambda cand: (
                safe_log(cand.train_nmse),
                complexity_score(cand, param_weight, special_weight, op_weight),
                cand.original_rank,
            ),
        )
        for cand in front:
            if cand.cid not in selected:
                selected.append(cand.cid)
            if len(selected) >= n:
                break
        front_ids = {cand.cid for cand in front}
        remaining = [cand for cand in remaining if cand.cid not in front_ids]
    if len(selected) < n:
        for cand in candidates:
            if cand.cid not in selected:
                selected.append(cand.cid)
            if len(selected) >= n:
                break
    return selected


def select_with_top1(
    candidates: list[Candidate],
    ranked: list[Candidate],
    force_top1: bool,
    n: int = SELECT_N,
) -> list[int]:
    selected: list[int] = []
    if force_top1 and candidates:
        selected.append(candidates[0].cid)
    for cand in ranked:
        if cand.cid not in selected:
            selected.append(cand.cid)
        if len(selected) >= n:
            break
    return selected


def select_occam_variant(
    candidates: list[Candidate],
    log10_delta: float,
    param_weight: float,
    special_weight: float,
    op_weight: float,
    force_top1: bool,
    n: int = SELECT_N,
) -> list[int]:
    if not candidates:
        return []
    finite_train = [cand.train_nmse for cand in candidates if math.isfinite(cand.train_nmse)]
    if not finite_train:
        return [cand.cid for cand in candidates[:n]]
    best_log10 = math.log10(max(min(finite_train), 1e-300))
    near = [
        cand
        for cand in candidates
        if math.log10(max(cand.train_nmse, 1e-300)) - best_log10 <= log10_delta
    ]
    if len(near) < n:
        near = candidates
    # During grid search, use the Occam idea:
    # keep only the strongest NMSE neighborhood, then rank by simplicity.
    front = near
    ranked = sorted(
        front,
        key=lambda cand: (
            complexity_score(cand, param_weight, special_weight, op_weight),
            safe_log(cand.train_nmse),
            cand.original_rank,
        ),
    )
    if len(ranked) < n:
        seen = {cand.cid for cand in ranked}
        ranked.extend(
            cand
            for cand in sorted(
                candidates,
                key=lambda cand: (
                    complexity_score(cand, param_weight, special_weight, op_weight),
                    safe_log(cand.train_nmse),
                    cand.original_rank,
                ),
            )
            if cand.cid not in seen
        )
    return select_with_top1(candidates, ranked, force_top1, n=n)


def mdl_value(cand: Candidate, alpha: float, beta: float, gamma: float, op_beta: float) -> float:
    # ``gamma`` is accepted for compatibility with older tuning CSVs, but
    # the deployed heuristic is train-only and does not use test/validation NMSE.
    return (
        safe_log(cand.train_nmse)
        + alpha * cand.param_count
        + beta * cand.tree_size
        + op_beta * cand.op_count
    )


def select_mdl_variant(
    candidates: list[Candidate],
    alpha: float,
    beta: float,
    gamma: float,
    op_beta: float,
    force_top1: bool,
    n: int = SELECT_N,
) -> list[int]:
    ranked = sorted(
        candidates,
        key=lambda cand: (mdl_value(cand, alpha, beta, gamma, op_beta), cand.original_rank),
    )
    return select_with_top1(candidates, ranked, force_top1, n=n)


def iter_statistics_effective_logs(
    logs_dir: Path,
    datasets: set[str] | None,
) -> list[tuple[str, str, str, str, Path]]:
    latest: dict[tuple[str, str, tuple[str, str], str], Path] = {}
    for dataset_dir in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if datasets is not None and dataset not in datasets:
            continue
        for task_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            task = task_dir.name
            if task in BASELINE_HEURISTIC_EXCLUDED_TASKS:
                continue
            for model_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                model = model_dir.name
                if "legacy" in model.lower():
                    continue
                effective_model = "no-llm" if task.startswith("only_structure") else model
                for date_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                    if date_dir.name == "checkpoint":
                        continue
                    for path in sorted(date_dir.iterdir()):
                        if path.suffix not in {".txt", ".log"}:
                            continue
                        test_case = path.stem.rsplit("_", 2)[0]
                        latest[(dataset, task, (effective_model, task), test_case)] = path
    rows = []
    for (dataset, task, (model, _task), test_case), path in sorted(latest.items()):
        rows.append((dataset, task, model, test_case, path))
    return rows


def parse_cases(logs_dir: Path, max_workers: int, datasets: set[str] | None = None) -> list[dict[str, object]]:
    parsed = []
    expressions = []
    if any((logs_dir / name).is_dir() for name in ("llm-srbench", "aifeynman")):
        effective_logs = iter_statistics_effective_logs(logs_dir, datasets)
    else:
        effective_logs = [
            ("", task, model, test_case, path)
            for task, model, test_case, path in iter_effective_logs(logs_dir)
            if task not in BASELINE_HEURISTIC_EXCLUDED_TASKS
        ]
    print(f"Parsing {len(effective_logs)} effective logs", flush=True)
    for i, (dataset, task, model, test_case, path) in enumerate(effective_logs, start=1):
        content = path.read_text(encoding="utf-8", errors="ignore")
        candidates = parse_final_candidates(content)
        if not candidates:
            continue
        expressions.extend(cand.expression for cand in candidates)
        matched_ids = parse_matched_candidate_ids(content)
        label_source = "matched_ids"
        if matched_ids is None:
            label_source = "status_fallback"
            matched_ids = [1] if parse_final_status(content) == "1" else []
        parsed.append({
            "dataset": dataset,
            "task": task,
            "model": model,
            "test_case": test_case,
            "source_file": str(path.relative_to(logs_dir.parent)),
            "raw_candidates": candidates,
            "matched_set": set(matched_ids),
            "label_source": label_source,
        })
        if i % 250 == 0:
            print(f"  parsed {i}/{len(effective_logs)}", flush=True)

    complexity_map = compute_complexity_map(expressions, max_workers=max_workers)
    cases = []
    for row in parsed:
        candidates = apply_complexities(row["raw_candidates"], complexity_map)
        cases.append({**row, "candidates": candidates})
    print(f"Prepared {len(cases)} cases", flush=True)
    return cases


def baseline_row(cases: list[dict[str, object]]) -> dict[str, object]:
    row = {"selector": "baseline", "params": "original_order", "n_cases": len(cases)}
    for name, n in (("top1", 1), ("top5", 5), ("top10", 10), ("top50", 50), ("all", None)):
        hits = 0
        for case in cases:
            candidates = case["candidates"]
            ids = [cand.cid for cand in candidates] if n is None else [cand.cid for cand in candidates[:n]]
            hits += int(hit(ids, case["matched_set"]))
        row[f"{name}_hits"] = hits
        row[f"{name}_rate"] = round(hits / len(cases), 6)
    return row


def evaluate_variant(
    cases: list[dict[str, object]],
    selector: str,
    params: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    hits_by_group = defaultdict(int)
    total_by_group = defaultdict(int)
    hit_count = 0

    for case in cases:
        candidates = case["candidates"]
        if selector == "occam":
            selected = select_occam_variant(candidates, **params)
        elif selector == "pareto":
            selected = select_pareto_variant(candidates, **params)
        elif selector == "mdl":
            selected = select_mdl_variant(candidates, **params)
        else:
            raise ValueError(selector)

        did_hit = int(hit(selected, case["matched_set"]))
        hit_count += did_hit
        key = (case["dataset"], case["model"], case["task"])
        hits_by_group[key] += did_hit
        total_by_group[key] += 1

    params_text = ";".join(f"{k}={v}" for k, v in params.items())
    overall = {
        "selector": selector,
        "params": params_text,
        "n_cases": len(cases),
        "hits": hit_count,
        "rate": round(hit_count / len(cases), 6),
    }
    by_group = []
    for (dataset, model, task), total in sorted(total_by_group.items()):
        hits = hits_by_group[(dataset, model, task)]
        by_group.append({
            "selector": selector,
            "params": params_text,
            "dataset": dataset,
            "model": model,
            "task": task,
            "n_cases": total,
            "hits": hits,
            "rate": round(hits / total, 6),
        })
    return overall, by_group


def init_evaluate_worker(cases: list[dict[str, object]]) -> None:
    global _WORKER_CASES
    _WORKER_CASES = cases


def evaluate_variant_worker(job: tuple[str, dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    selector, params = job
    return evaluate_variant(_WORKER_CASES, selector, params)


def occam_param_grid() -> list[dict[str, object]]:
    values = []
    for log10_delta in (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        for param_weight in (0.0, 1.0, 3.0, 6.0, 10.0):
            for special_weight in (0.0, 2.0, 5.0, 10.0):
                for op_weight in (0.0, 0.25, 0.5):
                    for force_top1 in (False, True):
                        values.append({
                            "log10_delta": log10_delta,
                            "param_weight": param_weight,
                            "special_weight": special_weight,
                            "op_weight": op_weight,
                            "force_top1": force_top1,
                        })
    return values


def mdl_param_grid() -> list[dict[str, object]]:
    values = []
    for alpha in (0.0, 0.01, 0.03, 0.06, 0.1, 0.2):
        for beta in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04):
            for gamma in (0.0,):
                for op_beta in (0.0, 0.002, 0.01):
                    for force_top1 in (False, True):
                        values.append({
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "op_beta": op_beta,
                            "force_top1": force_top1,
                        })
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=THIS_DIR.parents[1] / "logs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset directory to tune on; repeat for multiple datasets. Defaults to all datasets.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="Parallel workers for parameter-grid evaluation; defaults to --max-workers.",
    )
    parser.add_argument(
        "--select-n",
        type=int,
        default=SELECT_N,
        help="Number of candidate IDs selected by each tuned ranker.",
    )
    args = parser.parse_args()
    eval_workers = args.eval_workers if args.eval_workers is not None else args.max_workers
    eval_workers = max(1, eval_workers)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = THIS_DIR / f"tuning_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.dataset) if args.dataset else None
    cases = parse_cases(args.logs_dir, args.max_workers, datasets=datasets)
    baseline = baseline_row(cases)
    print("Baseline:", baseline, flush=True)

    overall_rows = []
    group_rows = []

    grids = [
        ("occam", occam_param_grid()),
        ("mdl", mdl_param_grid()),
    ]
    for selector, grid in grids:
        print(f"Evaluating {selector}: {len(grid)} variants with {eval_workers} workers", flush=True)
        jobs = [(selector, {**params, "n": args.select_n}) for params in grid]
        if eval_workers == 1:
            for i, job in enumerate(jobs, start=1):
                overall, by_group = evaluate_variant(cases, *job)
                overall_rows.append(overall)
                group_rows.extend(by_group)
                if i % 100 == 0:
                    print(f"  {selector}: {i}/{len(grid)}", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=eval_workers,
                initializer=init_evaluate_worker,
                initargs=(cases,),
            ) as executor:
                futures = [executor.submit(evaluate_variant_worker, job) for job in jobs]
                for i, future in enumerate(as_completed(futures), start=1):
                    overall, by_group = future.result()
                    overall_rows.append(overall)
                    group_rows.extend(by_group)
                    if i % 100 == 0:
                        print(f"  {selector}: {i}/{len(grid)}", flush=True)

    overall_sorted = sorted(overall_rows, key=lambda row: row["hits"], reverse=True)
    group_sorted = sorted(group_rows, key=lambda row: (row["model"], row["task"], -row["hits"]))
    write_csv(output_dir / "baseline.csv", [baseline])
    write_csv(output_dir / "overall_grid.csv", overall_sorted)
    write_csv(output_dir / "by_model_task_grid.csv", group_sorted)

    best_by_group = {}
    for row in group_rows:
        key = (row.get("dataset", ""), row["model"], row["task"])
        if key not in best_by_group or row["hits"] > best_by_group[key]["hits"]:
            best_by_group[key] = row
    write_csv(
        output_dir / "best_by_model_task.csv",
        sorted(best_by_group.values(), key=lambda r: (r.get("dataset", ""), r["model"], r["task"])),
    )

    lines = [
        "# Ranker Parameter Tuning",
        "",
        f"- cases: {len(cases)}",
        f"- top5 baseline: {baseline['top5_hits']}/{baseline['n_cases']}",
        f"- top10 baseline: {baseline['top10_hits']}/{baseline['n_cases']}",
        f"- top50 baseline: {baseline['top50_hits']}/{baseline['n_cases']}",
        "",
        "## Best Overall",
        "",
    ]
    for row in overall_sorted[:20]:
        lines.append(f"- {row['selector']} {row['hits']}/{row['n_cases']} ({row['rate']:.3f}) {row['params']}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote tuning outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
