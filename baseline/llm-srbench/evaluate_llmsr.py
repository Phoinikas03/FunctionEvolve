#!/usr/bin/env python3
"""
Evaluate LLMSR symbolic regression results using LLM-as-judge verification.

This script:
1. Finds all results.jsonl files under logs/
2. Extracts discovered programs for each equation
3. Uses verify.py's verification pipeline to check mathematical equivalence with GT
4. Outputs per-equation results and a summary table

Usage:
    python evaluate_llmsr.py [--llm-config CONFIG] [--split SPLIT] [--model MODEL]
                             [--output OUTPUT] [--parallel N] [--metrics-only]
                             [--results-dir DIR]

Examples:
    # Full verification (LLM judge + numerical metrics)
    # Saves per-equation results to results/ next to each results.jsonl
    python evaluate_llmsr.py --llm-config /home/xaa5sgh/symregression/verify.yaml

    # Only show numerical metrics (no LLM calls)
    python evaluate_llmsr.py --metrics-only

    # Evaluate a specific split
    python evaluate_llmsr.py --split chem_react --metrics-only

    # Evaluate a specific model
    python evaluate_llmsr.py --model Llmsr-opus-4-6 --metrics-only

    # Save all LLM results to a custom directory
    python evaluate_llmsr.py --results-dir ./my_results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SYMREG_DIR = SCRIPT_DIR.parent.parent  # /home/xaa5sgh/symregression
GT_CSV = SYMREG_DIR / "gt_expressions.csv"
LOGS_DIR = SCRIPT_DIR / "logs"

# Add symregression to path for importing verify
sys.path.insert(0, str(SYMREG_DIR))


def load_gt_map() -> dict:
    """Load GT expressions CSV into a dict keyed by equation_name."""
    gt_map = {}
    with open(GT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_map[row["equation_name"]] = row
    return gt_map


def find_results_files(
    split: Optional[str] = None, model: Optional[str] = None
) -> list[Path]:
    """Find all results.jsonl files, optionally filtered by split and model."""
    if not LOGS_DIR.is_dir():
        print(f"Error: logs directory not found: {LOGS_DIR}")
        sys.exit(1)

    pattern_parts = []
    if split:
        pattern_parts.append(split)
    else:
        pattern_parts.append("*")

    if model:
        pattern_parts.append(model)
    else:
        pattern_parts.append("*")

    pattern_parts.extend(["*", "results.jsonl"])
    pattern = "/".join(pattern_parts)

    results = sorted(LOGS_DIR.glob(pattern))
    return results


def parse_results_file(path: Path) -> list[dict]:
    """Parse a results.jsonl file into a list of equation results."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_candidate_from_entry(entry: dict) -> Optional[str]:
    """Extract the best discovered program/equation from a result entry."""
    eval_results = entry.get("eval_results", [])
    if not eval_results:
        return None

    best = eval_results[0]
    # Prefer discovered_program over discovered_equation
    if best.get("discovered_program"):
        return best["discovered_program"]
    if best.get("discovered_equation"):
        return best["discovered_equation"]
    return None


def extract_metrics(entry: dict) -> dict:
    """Extract key numerical metrics from a result entry."""
    eval_results = entry.get("eval_results", [])
    if not eval_results:
        return {}

    best = eval_results[0]
    result = {
        "search_time": best.get("search_time"),
        "best_program_sample_order": best.get("best_program_sample_order"),
        "best_program_score": best.get("best_program_score"),
    }

    for prefix in ("id_metrics", "ood_metrics"):
        metrics = best.get(prefix, {})
        if metrics:
            for key in ("nmse", "r2", "mape"):
                result[f"{prefix}_{key}"] = metrics.get(key)

    return result


def verify_single(
    eq_id: str, candidate: str, llm_config: dict,
    results_dir: Optional[Path] = None, verbose: bool = False,
) -> tuple[str, str]:
    """Verify a single candidate against GT using LLM judge.

    Returns:
        (verdict, full_text) where verdict is 'match'/'no_match'/'unknown'
    """
    from verify import verify_candidates

    result = verify_candidates(eq_id, [candidate], llm_config, verbose=verbose)

    if "Match found" in result:
        verdict = "match"
    elif "No match" in result:
        verdict = "no_match"
    else:
        verdict = "unknown"

    # Save full LLM output to file
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{eq_id}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)

    return verdict, result


def r2_threshold_check(entry: dict, threshold: float = 0.999) -> dict:
    """Check if id/ood R² exceed threshold (quick heuristic)."""
    eval_results = entry.get("eval_results", [])
    if not eval_results:
        return {"id_pass": False, "ood_pass": False}

    best = eval_results[0]
    id_r2 = best.get("id_metrics", {}).get("r2", 0)
    ood_r2 = best.get("ood_metrics", {}).get("r2", 0)
    return {
        "id_pass": id_r2 >= threshold,
        "ood_pass": ood_r2 >= threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMSR symbolic regression results")
    parser.add_argument(
        "--llm-config",
        type=str,
        default=str(SYMREG_DIR / "verify.yaml"),
        help="Path to LLM config YAML for verification (default: verify.yaml)",
    )
    parser.add_argument("--split", type=str, default=None, help="Filter by dataset split (e.g., chem_react)")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., Llmsr-opus-4-6)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel LLM verification calls")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only print numerical metrics, skip LLM verification",
    )
    parser.add_argument("--r2-threshold", type=float, default=0.999, help="R² threshold for pass/fail (default: 0.999)")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save per-equation LLM verification results (default: results/ next to results.jsonl)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed LLM verification output")
    args = parser.parse_args()

    # Find all results files
    results_files = find_results_files(split=args.split, model=args.model)
    if not results_files:
        print("No results.jsonl files found.")
        sys.exit(1)

    print(f"Found {len(results_files)} results file(s):")
    for f in results_files:
        rel = f.relative_to(LOGS_DIR)
        print(f"  {rel}")
    print()

    # Load GT map
    gt_map = load_gt_map()

    # Load LLM config if needed
    llm_config = None
    if not args.metrics_only:
        import yaml

        cfg_path = Path(args.llm_config)
        if not cfg_path.is_file():
            print(f"Error: LLM config file not found: {cfg_path}")
            sys.exit(1)
        with open(cfg_path, "r", encoding="utf-8") as f:
            llm_config = yaml.safe_load(f) or {}

    # Process all results
    all_rows = []
    for results_file in results_files:
        # Extract split and model from path: logs/{split}/{model}/{timestamp}/results.jsonl
        parts = results_file.relative_to(LOGS_DIR).parts
        split_name = parts[0] if len(parts) > 0 else "unknown"
        model_name = parts[1] if len(parts) > 1 else "unknown"

        print(f"Processing: {split_name}/{model_name}")
        entries = parse_results_file(results_file)
        print(f"  Equations: {len(entries)}")

        for entry in entries:
            eq_id = entry.get("equation_id", "???")
            gt_eq = entry.get("gt_equation", "")
            candidate = extract_candidate_from_entry(entry)
            metrics = extract_metrics(entry)
            r2_check = r2_threshold_check(entry, args.r2_threshold)

            row = {
                "split": split_name,
                "model": model_name,
                "equation_id": eq_id,
                "gt_equation": gt_eq[:80] + ("..." if len(gt_eq) > 80 else ""),
                "has_candidate": candidate is not None,
                "id_r2": metrics.get("id_metrics_r2"),
                "ood_r2": metrics.get("ood_metrics_r2"),
                "id_nmse": metrics.get("id_metrics_nmse"),
                "ood_nmse": metrics.get("ood_metrics_nmse"),
                "id_mape": metrics.get("id_metrics_mape"),
                "ood_mape": metrics.get("ood_metrics_mape"),
                "search_time": metrics.get("search_time"),
                "id_pass": r2_check["id_pass"],
                "ood_pass": r2_check["ood_pass"],
                "llm_verdict": None,
            }
            all_rows.append((row, eq_id, candidate, results_file))

    # LLM verification (if not metrics-only)
    if not args.metrics_only and llm_config:
        to_verify = [
            (i, row, eq_id, cand, rf)
            for i, (row, eq_id, cand, rf) in enumerate(all_rows)
            if cand is not None
        ]
        print(f"\nRunning LLM verification for {len(to_verify)} equations (parallel={args.parallel})...")

        def _verify_task(item):
            idx, row, eq_id, cand, rf = item
            # Determine results dir: explicit flag or next to results.jsonl
            if args.results_dir:
                res_dir = Path(args.results_dir)
            else:
                res_dir = rf.parent / "results"
            try:
                verdict, _ = verify_single(
                    eq_id, cand, llm_config,
                    results_dir=res_dir, verbose=args.verbose,
                )
            except Exception as e:
                verdict = f"error: {e}"
            return idx, verdict

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_verify_task, item): item for item in to_verify}
            done_count = 0
            for future in as_completed(futures):
                idx, verdict = future.result()
                all_rows[idx][0]["llm_verdict"] = verdict
                done_count += 1
                if done_count % 10 == 0:
                    print(f"  Verified {done_count}/{len(to_verify)}")

        print(f"  Verified {len(to_verify)}/{len(to_verify)} done.")

    # Build final rows list
    rows = [r for r, _, _, *_ in all_rows]

    # Print summary table
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)

    # Header
    if args.metrics_only:
        header = f"{'Split':<16} {'EqID':<8} {'ID R²':>12} {'OOD R²':>12} {'ID NMSE':>14} {'OOD NMSE':>14} {'ID Pass':>8} {'OOD Pass':>9}"
    else:
        header = f"{'Split':<16} {'EqID':<8} {'ID R²':>12} {'OOD R²':>12} {'ID NMSE':>14} {'OOD NMSE':>14} {'ID Pass':>8} {'OOD Pass':>9} {'LLM':>10}"
    print(header)
    print("-" * len(header))

    for row in rows:
        id_r2 = f"{row['id_r2']:.6f}" if row["id_r2"] is not None else "N/A"
        ood_r2 = f"{row['ood_r2']:.6f}" if row["ood_r2"] is not None else "N/A"
        id_nmse = f"{row['id_nmse']:.2e}" if row["id_nmse"] is not None else "N/A"
        ood_nmse = f"{row['ood_nmse']:.2e}" if row["ood_nmse"] is not None else "N/A"
        id_pass = "Yes" if row["id_pass"] else "No"
        ood_pass = "Yes" if row["ood_pass"] else "No"

        line = f"{row['split']:<16} {row['equation_id']:<8} {id_r2:>12} {ood_r2:>12} {id_nmse:>14} {ood_nmse:>14} {id_pass:>8} {ood_pass:>9}"
        if not args.metrics_only:
            verdict = row.get("llm_verdict") or "N/A"
            line += f" {verdict:>10}"
        print(line)

    # Aggregate stats per split
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    splits = sorted(set(r["split"] for r in rows))
    for split_name in splits:
        split_rows = [r for r in rows if r["split"] == split_name]
        n = len(split_rows)
        id_pass_count = sum(1 for r in split_rows if r["id_pass"])
        ood_pass_count = sum(1 for r in split_rows if r["ood_pass"])
        both_pass_count = sum(1 for r in split_rows if r["id_pass"] and r["ood_pass"])

        id_r2_vals = [r["id_r2"] for r in split_rows if r["id_r2"] is not None]
        ood_r2_vals = [r["ood_r2"] for r in split_rows if r["ood_r2"] is not None]
        avg_id_r2 = sum(id_r2_vals) / len(id_r2_vals) if id_r2_vals else 0
        avg_ood_r2 = sum(ood_r2_vals) / len(ood_r2_vals) if ood_r2_vals else 0

        # Median R²
        id_r2_sorted = sorted(id_r2_vals)
        ood_r2_sorted = sorted(ood_r2_vals)
        median_id_r2 = id_r2_sorted[len(id_r2_sorted) // 2] if id_r2_sorted else 0
        median_ood_r2 = ood_r2_sorted[len(ood_r2_sorted) // 2] if ood_r2_sorted else 0

        search_times = [r["search_time"] for r in split_rows if r["search_time"] is not None]
        avg_time = sum(search_times) / len(search_times) if search_times else 0

        print(f"\n--- {split_name} ({n} equations) ---")
        print(f"  ID  R² pass (>={args.r2_threshold}): {id_pass_count}/{n} ({100*id_pass_count/n:.1f}%)")
        print(f"  OOD R² pass (>={args.r2_threshold}): {ood_pass_count}/{n} ({100*ood_pass_count/n:.1f}%)")
        print(f"  Both pass:                  {both_pass_count}/{n} ({100*both_pass_count/n:.1f}%)")
        print(f"  Avg  ID  R²: {avg_id_r2:.6f}   Median: {median_id_r2:.6f}")
        print(f"  Avg  OOD R²: {avg_ood_r2:.6f}   Median: {median_ood_r2:.6f}")
        print(f"  Avg search time: {avg_time:.1f}s")

        if not args.metrics_only:
            match_count = sum(1 for r in split_rows if r.get("llm_verdict") == "match")
            no_match_count = sum(1 for r in split_rows if r.get("llm_verdict") == "no_match")
            unknown_count = sum(1 for r in split_rows if r.get("llm_verdict") not in ("match", "no_match", None))
            print(f"  LLM Match: {match_count}/{n}  No Match: {no_match_count}/{n}  Unknown: {unknown_count}/{n}")

    # Overall
    total = len(rows)
    total_id_pass = sum(1 for r in rows if r["id_pass"])
    total_ood_pass = sum(1 for r in rows if r["ood_pass"])
    total_both = sum(1 for r in rows if r["id_pass"] and r["ood_pass"])

    print(f"\n{'='*80}")
    print(f"OVERALL ({total} equations)")
    print(f"  ID  R² pass: {total_id_pass}/{total} ({100*total_id_pass/total:.1f}%)")
    print(f"  OOD R² pass: {total_ood_pass}/{total} ({100*total_ood_pass/total:.1f}%)")
    print(f"  Both pass:   {total_both}/{total} ({100*total_both/total:.1f}%)")

    if not args.metrics_only:
        total_match = sum(1 for r in rows if r.get("llm_verdict") == "match")
        print(f"  LLM Match:   {total_match}/{total} ({100*total_match/total:.1f}%)")

    # Write CSV output
    if args.output:
        output_path = Path(args.output)
        fieldnames = [
            "split", "model", "equation_id", "gt_equation", "has_candidate",
            "id_r2", "ood_r2", "id_nmse", "ood_nmse", "id_mape", "ood_mape",
            "search_time", "id_pass", "ood_pass",
        ]
        if not args.metrics_only:
            fieldnames.append("llm_verdict")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out = {k: row.get(k) for k in fieldnames}
                writer.writerow(out)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
