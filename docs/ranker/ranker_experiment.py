#!/usr/bin/env python3
"""Try small candidate selectors over the full formula pool in logs.

The selectors choose five candidates from the complete Final output pool.
They use only training NMSE and expression complexity; test/OOD NMSE are parsed
for reporting but are not used for selection.

1. occam: near-best NMSE candidates, then simpler first.
2. mdl: a BIC/MDL-flavored score using train NMSE, params, and tree size.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import sympy as sp


TOP_KS = (1, 5, 10, 50)
SPECIAL_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "asin", "acos", "atan"}
SYMPY_FUNCS = {name: getattr(sp, name) for name in SPECIAL_FUNCS if hasattr(sp, name)}


@dataclass
class Candidate:
    cid: int
    expression: str
    params: str
    train_nmse: float
    test_nmse: float
    ood_nmse: float
    original_rank: int
    param_count: int
    tree_size: int
    op_count: int
    special_count: int

    @property
    def complexity(self) -> float:
        return self.tree_size + 3.0 * self.param_count + 2.0 * self.special_count


def parse_nmse(value: str) -> float:
    value = value.strip().rstrip(",;")
    if value.lower() in {"inf", "+inf", "infinity", "+infinity"} or value == "∞":
        return math.inf
    try:
        parsed = float(value)
    except ValueError:
        return math.inf
    return parsed if math.isfinite(parsed) and parsed >= 0 else math.inf


def safe_log(value: float) -> float:
    if not math.isfinite(value):
        return 1e9
    return math.log(max(value, 1e-300))


def parse_matched_candidate_ids(content: str) -> list[int] | None:
    pattern = re.compile(r"Matched\s+candidate\s+IDs?\s*[:：]\s*\[([^\]\n]*)\]", re.I)
    matches = list(pattern.finditer(content))
    if not matches:
        return None
    return [int(value) for value in re.findall(r"\d+", matches[-1].group(1))]


def parse_final_status(content: str) -> str:
    ids = parse_matched_candidate_ids(content)
    if ids is not None:
        return str(min(ids)) if ids else "0"

    for line in reversed(content.splitlines()):
        if "Conclusion:" in line:
            if "Match found" in line:
                return "1"
            if "No match" in line:
                return "0"
        if "[MATCH FOUND]" in line:
            return "1"
        if "[NO MATCH]" in line:
            return "0"
    return ""


def sympy_complexity(expression: str) -> tuple[int, int]:
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", expression))
    locals_map = dict(SYMPY_FUNCS)
    for name in identifiers - set(SYMPY_FUNCS):
        locals_map[name] = sp.Symbol(name)
    try:
        expr = sp.sympify(expression, locals=locals_map, evaluate=False)
        tree_size = sum(1 for _ in sp.preorder_traversal(expr))
        op_count = int(sp.count_ops(expr, visual=False))
        return tree_size, op_count
    except Exception:
        # Fallback keeps the experiment robust for odd strings.
        op_count = len(re.findall(r"\*\*|[+\-*/]", expression))
        tree_size = op_count + len(re.findall(r"\b[A-Za-z_]\w*\b|\d+(?:\.\d+)?", expression))
        return tree_size, op_count


def expression_complexity_task(expression: str) -> tuple[str, int, int]:
    tree_size, op_count = sympy_complexity(expression)
    return expression, tree_size, op_count


def candidate_features(cid: int, expression: str, params: str, metrics: dict[str, float], rank: int) -> Candidate:
    param_names = set(re.findall(r"\bc\d+\b", params + " " + expression))
    special_count = sum(len(re.findall(rf"\b{name}\s*\(", expression)) for name in SPECIAL_FUNCS)
    return Candidate(
        cid=cid,
        expression=expression,
        params=params,
        train_nmse=metrics.get("train", math.inf),
        test_nmse=metrics.get("test", math.inf),
        ood_nmse=metrics.get("ood", math.inf),
        original_rank=rank,
        param_count=len(param_names),
        tree_size=0,
        op_count=0,
        special_count=special_count,
    )


def parse_final_candidates(content: str) -> list[Candidate]:
    lines = content.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"\s*Final output\s+\d+\s+formulas", line):
            start = i
    if start is None:
        return []

    candidates: list[Candidate] = []
    current: dict[str, object] | None = None
    candidate_pattern = re.compile(
        r"^\s*(\d+)\.\s*(?:\[Mature\]\s*)?<<<FORMULA>>>(.*?)<<<END_FORMULA>>>"
    )
    metrics_pattern = re.compile(r"\b(train|test|ood)\s*=\s*([^\s]+)", re.I)

    def flush() -> None:
        if not current:
            return
        candidates.append(
            candidate_features(
                cid=int(current["cid"]),
                expression=str(current["expression"]),
                params=str(current.get("params", "")),
                metrics=dict(current.get("metrics", {})),
                rank=len(candidates) + 1,
            )
        )

    for line in lines[start + 1 :]:
        if line.startswith("=" * 10):
            break
        candidate_match = candidate_pattern.match(line)
        if candidate_match:
            flush()
            current = {
                "cid": int(candidate_match.group(1)),
                "expression": candidate_match.group(2).strip(),
                "params": "",
                "metrics": {},
            }
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("params:"):
            current["params"] = stripped.removeprefix("params:").strip()
        for metric, value in metrics_pattern.findall(stripped):
            current_metrics = current.setdefault("metrics", {})
            current_metrics[metric.lower()] = parse_nmse(value)

    flush()
    return candidates


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    objectives = {
        cand.cid: (
            safe_log(cand.train_nmse),
            cand.complexity,
            cand.param_count,
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


def select_occam(candidates: list[Candidate], n: int = 5, log10_delta: float = 0.5) -> list[int]:
    if not candidates:
        return []
    finite_train = [cand.train_nmse for cand in candidates if math.isfinite(cand.train_nmse)]
    if not finite_train:
        return [cand.cid for cand in candidates[:n]]
    best_log10 = math.log10(max(min(finite_train), 1e-300))

    selected: list[Candidate] = []
    seen: set[int] = set()
    for delta in (log10_delta, 1.0, 2.0, math.inf):
        near = [
            cand
            for cand in candidates
            if math.log10(max(cand.train_nmse, 1e-300)) - best_log10 <= delta
        ]
        front = pareto_front(near)
        ranked = sorted(front, key=lambda cand: (cand.complexity, cand.param_count, safe_log(cand.train_nmse), cand.original_rank))
        for cand in ranked:
            if cand.cid not in seen:
                selected.append(cand)
                seen.add(cand.cid)
            if len(selected) >= n:
                return [cand.cid for cand in selected]

    for cand in sorted(candidates, key=lambda cand: (cand.complexity, safe_log(cand.train_nmse), cand.original_rank)):
        if cand.cid not in seen:
            selected.append(cand)
            seen.add(cand.cid)
        if len(selected) >= n:
            break
    return [cand.cid for cand in selected]


def mdl_score(cand: Candidate, alpha: float, beta: float, gamma: float) -> float:
    # ``gamma`` is accepted for compatibility with older calls, but the
    # selection score is train-only and does not use test/OOD NMSE.
    return safe_log(cand.train_nmse) + alpha * cand.param_count + beta * cand.tree_size


def select_mdl(
    candidates: list[Candidate],
    n: int = 5,
    alpha: float = 0.08,
    beta: float = 0.015,
    gamma: float = 0.0,
) -> list[int]:
    ranked = sorted(candidates, key=lambda cand: (mdl_score(cand, alpha, beta, gamma), cand.original_rank))
    return [cand.cid for cand in ranked[:n]]


def hit(selected_ids: Iterable[int], matched_ids: set[int]) -> bool:
    return bool(set(selected_ids).intersection(matched_ids))


def candidate_ids(candidates: list[Candidate]) -> list[int]:
    return [cand.cid for cand in candidates]


def iter_effective_logs(logs_dir: Path) -> Iterable[tuple[str, str, str, Path]]:
    """Yield the same effective task/model/test_case file that statistics.py keeps."""
    latest: dict[tuple[str, str, str], Path] = {}
    for task_dir in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        task = task_dir.name
        for model_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            model = model_dir.name
            if "legacy" in model.lower():
                continue
            for date_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                if date_dir.name == "checkpoint":
                    continue
                for path in sorted(date_dir.iterdir()):
                    if path.suffix not in {".txt", ".log"}:
                        continue
                    test_case = path.stem.split("_")[0]
                    latest[(task, model, test_case)] = path
    for (task, model, test_case), path in sorted(latest.items()):
        yield task, model, test_case, path


def compute_complexity_map(expressions: Iterable[str], max_workers: int) -> dict[str, tuple[int, int]]:
    unique_expressions = sorted(set(expressions))
    if not unique_expressions:
        return {}

    complexity_map: dict[str, tuple[int, int]] = {}
    worker_count = max(1, max_workers)
    chunksize = max(1, len(unique_expressions) // max(worker_count * 16, 1))
    print(
        f"Computing complexity for {len(unique_expressions)} unique expressions "
        f"with {worker_count} workers, chunksize={chunksize}",
        flush=True,
    )

    if worker_count == 1:
        iterator = map(expression_complexity_task, unique_expressions)
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        iterator = executor.map(expression_complexity_task, unique_expressions, chunksize=chunksize)

    try:
        for i, (expression, tree_size, op_count) in enumerate(iterator, start=1):
            complexity_map[expression] = (tree_size, op_count)
            if i % 1000 == 0:
                print(f"  complexity: {i}/{len(unique_expressions)}", flush=True)
    finally:
        if worker_count != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    return complexity_map


def apply_complexities(candidates: list[Candidate], complexity_map: dict[str, tuple[int, int]]) -> list[Candidate]:
    enriched = []
    for cand in candidates:
        tree_size, op_count = complexity_map.get(cand.expression, (0, 0))
        enriched.append(replace(cand, tree_size=tree_size, op_count=op_count))
    return enriched


def evaluate(logs_dir: Path, output_dir: Path, max_workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_logs = []
    expressions = []
    skipped_rows = []

    effective_logs = list(iter_effective_logs(logs_dir))
    print(f"Parsing {len(effective_logs)} effective logs from {logs_dir}", flush=True)
    for i, (task, model, test_case, path) in enumerate(effective_logs, start=1):
        content = path.read_text(encoding="utf-8", errors="ignore")
        candidates = parse_final_candidates(content)
        if not candidates:
            skipped_rows.append({
                "task": task,
                "model": model,
                "test_case": test_case,
                "source_file": str(path.relative_to(logs_dir.parent)),
                "reason": "no_final_candidates",
            })
            continue
        expressions.extend(cand.expression for cand in candidates)
        parsed_logs.append((task, model, test_case, path, content, candidates))
        if i % 100 == 0:
            print(f"  parsed logs: {i}/{len(effective_logs)}", flush=True)

    print(
        f"Parsed {len(parsed_logs)} logs with {len(expressions)} candidate expressions",
        flush=True,
    )
    complexity_map = compute_complexity_map(expressions, max_workers=max_workers)

    case_rows = []
    selected_rows = []

    for task, model, test_case, path, content, raw_candidates in parsed_logs:
        candidates = apply_complexities(raw_candidates, complexity_map)

        matched_ids = parse_matched_candidate_ids(content)
        label_source = "matched_ids"
        if matched_ids is None:
            label_source = "status_fallback"
            matched_ids = [1] if parse_final_status(content) == "1" else []
        matched_set = set(matched_ids)
        ordered_ids = candidate_ids(candidates)

        selections = {
            "top1": ordered_ids[:1],
            "top5": ordered_ids[:5],
            "top10": ordered_ids[:10],
            "top50": ordered_ids[:50],
            "all": ordered_ids,
            "occam": select_occam(candidates),
            "mdl": select_mdl(candidates),
        }

        row = {
            "task": task,
            "model": model,
            "test_case": test_case,
            "source_file": str(path.relative_to(logs_dir.parent)),
            "pool_size": len(candidates),
            "matched_ids": " ".join(map(str, matched_ids)),
            "label_source": label_source,
        }
        for name, ids in selections.items():
            row[f"{name}_ids"] = " ".join(map(str, ids))
            row[f"{name}_hit"] = int(hit(ids, matched_set))
        case_rows.append(row)

        for selector_name in ("occam", "mdl"):
            by_id = {cand.cid: cand for cand in candidates}
            for slot, cid in enumerate(selections[selector_name], start=1):
                cand = by_id[cid]
                selected_rows.append({
                    "task": task,
                    "model": model,
                    "test_case": test_case,
                    "selector": selector_name,
                    "slot": slot,
                    "candidate_id": cid,
                    "is_match": int(cid in matched_set),
                    "original_rank": cand.original_rank,
                    "train_nmse": cand.train_nmse,
                    "test_nmse": cand.test_nmse,
                    "ood_nmse": cand.ood_nmse,
                    "param_count": cand.param_count,
                    "tree_size": cand.tree_size,
                    "op_count": cand.op_count,
                    "special_count": cand.special_count,
                    "expression": cand.expression,
                })

    case_fieldnames = [
        "task",
        "model",
        "test_case",
        "source_file",
        "pool_size",
        "matched_ids",
        "label_source",
        "top1_ids",
        "top1_hit",
        "top5_ids",
        "top5_hit",
        "top10_ids",
        "top10_hit",
        "top50_ids",
        "top50_hit",
        "all_ids",
        "all_hit",
        "occam_ids",
        "occam_hit",
        "mdl_ids",
        "mdl_hit",
    ]
    write_csv(output_dir / "case_results.csv", case_rows, case_fieldnames)
    write_csv(output_dir / "selected_candidates.csv", selected_rows, list(selected_rows[0]) if selected_rows else [])
    write_csv(output_dir / "skipped_logs.csv", skipped_rows, list(skipped_rows[0]) if skipped_rows else [])

    summary_rows = summarize(case_rows, ("model", "task"))
    write_csv(output_dir / "summary_by_model_task.csv", summary_rows, list(summary_rows[0]) if summary_rows else [])
    total_rows = summarize(case_rows, tuple())
    write_csv(output_dir / "summary_total.csv", total_rows, list(total_rows[0]) if total_rows else [])
    write_readme(output_dir, total_rows, summary_rows)


def summarize(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows = []
    for key, group in sorted(grouped.items()):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        total = len(group)
        out["n_cases"] = total
        out["avg_pool_size"] = round(sum(int(row["pool_size"]) for row in group) / total, 2) if total else 0
        for name in ("top1", "top5", "top10", "top50", "all", "occam", "mdl"):
            count = sum(int(row[f"{name}_hit"]) for row in group)
            out[f"{name}_hits"] = count
            out[f"{name}_rate"] = round(count / total, 6) if total else 0
        out["occam_delta_vs_top5"] = out["occam_hits"] - out["top5_hits"]
        out["mdl_delta_vs_top5"] = out["mdl_hits"] - out["top5_hits"]
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir: Path, total_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    total = total_rows[0] if total_rows else {}
    best_rows = sorted(
        summary_rows,
        key=lambda row: max(row["occam_delta_vs_top5"], row["mdl_delta_vs_top5"]),
        reverse=True,
    )[:10]
    lines = [
        "# Ranker Experiment",
        "",
        "Selectors choose 5 formulas from the complete `Final output` pool, not just the first 50.",
        "",
        "## Overall",
        "",
    ]
    if total:
        for name in ("top1", "top5", "top10", "top50", "all", "occam", "mdl"):
            lines.append(f"- {name}: {total[f'{name}_hits']}/{total['n_cases']} ({total[f'{name}_rate']:.3f})")
    lines.extend(["", "## Largest Improvements Vs Top5", ""])
    for row in best_rows:
        lines.append(
            "- {model}/{task}: top5={top5_hits}, occam={occam_hits} "
            "(delta {occam_delta_vs_top5:+}), mdl={mdl_hits} "
            "(delta {mdl_delta_vs_top5:+}), top50={top50_hits}, n={n_cases}".format(**row)
        )
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "logs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Parallel workers for per-expression complexity evaluation.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / f"results_{stamp}"
    evaluate(args.logs_dir, output_dir, max_workers=args.max_workers)
    print(f"Wrote ranker experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
