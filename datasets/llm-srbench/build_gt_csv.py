#!/usr/bin/env python
"""
Build Ground Truth CSV files for llm-srbench 4 splits.

Strategy:
  1. Clean expressions: P(t)->P, 0.123_z -> 0.123*_z
  2. Replace numeric constants with c0, c1, ... via regex -> symbolic skeleton
  3. Use sympify to find non-feature free symbols -> named parameters (_s, beta, etc.)
  4. Fit named parameters using Evaluator
  5. Merge skeleton parameters (directly extracted) + named parameters (fitted)
  6. Verify train_nmse < 1e-10

Output columns:
  split, equation_name, feature_names, raw_expression,
  symbolic_expression, param_names, gt_params, numerical_expression,
  train_nmse, test_nmse, ood_nmse

Usage:
  python build_gt_csv.py [--output gt_expressions.csv] [--nmse-threshold 1e-10]
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset import SRDataset
from src.evaluator import Evaluator

SPLITS = ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]

SYMPY_BUILTINS = frozenset({
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs',
    'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
    'sign', 'floor', 'ceiling', 'pi', 'E', 'I',
})

_NUM_RE = re.compile(
    r'(?<![a-zA-Z_\d.])'
    r'((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    r'(?![\w.])'
)


def _clean_expression(expr: str, symbols: list, feature_names: list) -> str:
    """P(t)→P, v(t)→v, 0.123_z → 0.123*_z"""
    all_known = set(symbols) | set(feature_names)
    for sym in all_known:
        expr = re.sub(rf'\b{re.escape(sym)}\s*\([^)]*\)', sym, expr)
    expr = re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)(_[A-Za-z]\w*)', r'\1*\2', expr)
    return expr


def _make_skeleton(
    expr_str: str,
) -> Tuple[str, List[str], List[float]]:
    """Replace numeric constants with c0, c1, ... Return (skeleton, param_names, param_values)."""
    counter = [0]
    names: List[str] = []
    values: List[float] = []

    def _replacer(m: re.Match) -> str:
        name = f"c{counter[0]}"
        counter[0] += 1
        names.append(name)
        values.append(float(m.group(0)))
        return name

    skeleton = _NUM_RE.sub(_replacer, expr_str)
    return skeleton, names, values


def _safe_sympify(expr_str: str) -> sp.Expr:
    """Register all non-builtin identifiers as Symbols before sympify."""
    identifiers = set(re.findall(r'\b([A-Za-z_]\w*)\b', expr_str))
    local_dict = {
        name: sp.Symbol(name)
        for name in identifiers if name not in SYMPY_BUILTINS
    }
    return sp.sympify(expr_str, locals=local_dict)


def _find_named_params(
    expr_str: str, feature_names: List[str],
) -> List[str]:
    """Find non-feature free symbols in the expression as named parameters."""
    sympy_expr = _safe_sympify(expr_str)
    feature_set = set(feature_names)
    all_syms = {str(s) for s in sympy_expr.free_symbols}
    return sorted(all_syms - feature_set)


def _make_numerical_expr(
    sympy_expr: sp.Expr, param_names: List[str], values: List[float],
) -> str:
    """Substitute named parameters with their numeric values."""
    subs = {sp.Symbol(n): sp.Float(v) for n, v in zip(param_names, values)}
    return str(sympy_expr.subs(subs))


def process_equation(
    split: str, eq_name: str, threshold: float,
) -> Optional[dict]:
    try:
        ds = SRDataset.from_srbench(split, eq_name)
        ds.load(quiet=True)
    except Exception as e:
        print(f"  [SKIP] {split}/{eq_name}: load failed - {e}")
        return None

    raw_expr = ds.expression
    if not raw_expr:
        print(f"  [SKIP] {split}/{eq_name}: no GT expression")
        return None

    feature_names = ds.feature_names
    cleaned = _clean_expression(raw_expr, ds.symbols, feature_names)

    # ---- skeleton: numeric constants -> c0, c1, ... ----
    skeleton, skel_names, skel_values = _make_skeleton(cleaned)

    # ---- named parameters (_s, _m, beta, omega0 ...) ----
    try:
        named_params = _find_named_params(cleaned, feature_names)
    except Exception as e:
        print(f"  [SKIP] {split}/{eq_name}: parse failed - {e}")
        return None

    # ---- fit named parameters ----
    CONFIGS = [
        dict(n_restarts=8, bound=200.0, de_timeout=180.0,
             cma_timeout=180.0, lbfgs_timeout=90.0),
        dict(n_restarts=16, bound=1000.0, de_timeout=300.0,
             cma_timeout=300.0, lbfgs_timeout=120.0),
    ]

    fitted_values: List[float] = []
    train_nmse = float("inf")
    test_nmse = float("inf")
    ood_nmse = float("inf")

    for attempt, cfg in enumerate(CONFIGS):
        evaluator = Evaluator(
            feature_names=feature_names,
            X_train=ds.X_train, y_train=ds.y_train,
            X_test=ds.X_test, y_test=ds.y_test,
            X_ood_test=ds.X_ood_test, y_ood_test=ds.y_ood_test,
            **cfg,
        )

        parent = fitted_values if fitted_values else None
        result = evaluator.evaluate_skeleton(
            cleaned, named_params, parent_params=parent)
        fitted_values = result.best_params
        train_nmse = result.train_nmse
        test_nmse = result.test_nmse
        ood_nmse = result.ood_test_nmse

        if train_nmse <= threshold:
            break
        if attempt < len(CONFIGS) - 1:
            print(f"    [RETRY] {split}/{eq_name}: "
                  f"attempt {attempt+1} nmse={train_nmse:.2e}, retrying...")

    if train_nmse > threshold:
        print(f"  [FAIL] {split}/{eq_name}: train_nmse={train_nmse:.2e}")
        return None

    # ---- merge all parameters ----
    all_param_names = skel_names + named_params
    all_param_values = skel_values + list(fitted_values)

    # ---- numerical expression: substitute named parameters back ----
    if named_params:
        sympy_expr = _safe_sympify(cleaned)
        numerical_expr = _make_numerical_expr(
            sympy_expr, named_params, fitted_values)
    else:
        numerical_expr = cleaned

    print(f"  [OK]   {split}/{eq_name}: "
          f"{len(skel_names)} consts + {len(named_params)} named, "
          f"train={train_nmse:.2e}")

    return {
        "split": split,
        "equation_name": eq_name,
        "feature_names": ";".join(feature_names),
        "raw_expression": raw_expr,
        "symbolic_expression": skeleton,
        "param_names": ";".join(all_param_names),
        "gt_params": ";".join(f"{v:.15e}" for v in all_param_values),
        "numerical_expression": numerical_expr,
        "train_nmse": train_nmse,
        "test_nmse": test_nmse,
        "ood_nmse": ood_nmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Build llm-srbench GT CSV")
    parser.add_argument("--output", default=str(Path(__file__).parent / "gt_expressions.csv"))
    parser.add_argument("--nmse-threshold", type=float, default=1e-10)
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()

    splits = args.splits or SPLITS
    threshold = args.nmse_threshold

    print(f"=== Build GT CSV ===")
    print(f"Splits: {splits}")
    print(f"NMSE threshold: {threshold:.0e}")
    print(f"Output: {args.output}\n")

    rows: List[dict] = []
    failures: List[str] = []

    for split in splits:
        print(f"\n--- {split} ---")
        try:
            ds_tmp = SRDataset.from_srbench(split)
            equations = ds_tmp.list_equations()
        except Exception as e:
            print(f"  [ERROR] failed to load split: {e}")
            continue

        print(f"  {len(equations)} equations total")
        for eq in equations:
            result = process_equation(split, eq, threshold)
            if result is not None:
                rows.append(result)
            else:
                failures.append(f"{split}/{eq}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\n=== Done ===")
    print(f"Success: {len(rows)} / {len(rows) + len(failures)} equations")
    if failures:
        print(f"Failed: {len(failures)} equations")
        for f in failures:
            print(f"  - {f}")
    print(f"Output: {args.output}")
    if len(df):
        print(f"Max train_nmse: {df['train_nmse'].max():.2e}")


if __name__ == "__main__":
    main()
