"""
GT expression extraction + variant generator.

Extract 15 GT expressions from each of the MatSci / BPG / CRK splits
of nnheui/llm-srbench, replace numeric constants with symbolic parameters
c0..cN, and record GT parameter values.
Then apply 4 transforms (2 variants each) to each GT, generating 8 augmented cases.
"""

from __future__ import annotations

import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp

# ---------------------------------------------------------------------------
# Data paths (local repo dataset snapshot)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "datasets" / "llm-srbench"

def _find_snapshot_dir() -> Path:
    if not (_DATA_DIR / "lsr_bench_data.hdf5").is_file():
        raise FileNotFoundError(
            f"llm-srbench data not found at {_DATA_DIR} "
            "(expected lsr_bench_data.hdf5 and data/*.parquet)"
        )
    return _DATA_DIR

_SPLITS = {
    "matsci":  {"hf_split": "lsr_synth_matsci",        "hdf5_prefix": "lsr_synth/matsci"},
    "bpg":     {"hf_split": "lsr_synth_bio_pop_growth", "hdf5_prefix": "lsr_synth/bio_pop_growth"},
    "crk":     {"hf_split": "lsr_synth_chem_react",     "hdf5_prefix": "lsr_synth/chem_react"},
    "po":      {"hf_split": "lsr_synth_phys_osc",       "hdf5_prefix": "lsr_synth/phys_osc"},
}

_CSV_SPLIT_MAP = {
    "bio_pop_growth": "bpg",
    "chem_react":     "crk",
    "matsci":         "matsci",
    "phys_osc":       "po",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchCase:
    case_id: str                  # e.g. "MatSci0_T1a"
    split: str                    # matsci / bpg / crk
    equation_name: str            # original equation name
    skeleton: str                 # skeleton expression (with c0, c1, ...)
    param_names: List[str]        # ["c0", "c1", ...]
    gt_params: List[float]        # GT parameter values
    feature_names: List[str]      # ["epsilon", "T"] etc.
    transform: str                # "base" / "T1a" / "T1b" / ...
    n_params: int = 0

    def __post_init__(self):
        self.n_params = len(self.param_names)


@dataclass
class GTInfo:
    equation_name: str
    split: str
    raw_expression: str
    skeleton: str
    param_names: List[str]
    gt_params: List[float]
    feature_names: List[str]


# ---------------------------------------------------------------------------
# Expression parsing: numeric values -> symbolic parameters
# ---------------------------------------------------------------------------

# Match floats (including scientific notation) possibly followed by _letter suffix (CRK undetermined coefficients)
_NUM_PATTERN = re.compile(
    r'(?<![a-zA-Z_])'           # must not follow a letter/underscore (avoid matching digits in variable names)
    r'('
    r'-?(?:\d+\.?\d*|\.\d+)'    # basic numeric value
    r'(?:[eE][+-]?\d+)?'        # scientific notation
    r')'
    r'(_[a-zA-Z]\w*)?'          # optional _z / _w / _s suffix
)


def _clean_expression(expr: str, split: str) -> str:
    """Clean formatting issues in raw expressions."""
    if split == "bpg":
        expr = expr.replace("P(t)", "P")
        expr = expr.replace("A(t)", "A")
    elif split == "crk":
        expr = expr.replace("A(t)", "A")
        expr = expr.replace("P(t)", "P")
    return expr


def _get_feature_names(symbols: list, split: str) -> List[str]:
    """Extract feature variable names from the symbols list (drop the target variable = first one)."""
    syms = [str(s) for s in symbols]
    if len(syms) > 1:
        return syms[1:]
    return syms


def _parse_expression(raw_expr: str, feature_names: List[str]
                      ) -> Tuple[str, List[str], List[Optional[float]]]:
    """
    Replace numeric constants in the expression with c0, c1, ...

    Returns (skeleton, param_names, gt_values)
    where gt_values[i] = float means known GT, = None means undetermined coefficient (CRK's _z etc.).
    """
    feature_set = set(feature_names)
    param_names: List[str] = []
    gt_values: List[Optional[float]] = []
    counter = [0]

    def _replacer(m: re.Match) -> str:
        num_str = m.group(1)
        suffix = m.group(2)

        # Check if this is a special constant context that should be preserved
        # e.g. 8.314 (gas constant R) in MatSci expressions
        # We still replace it as a parameter since the optimizer needs to discover it

        pname = f"c{counter[0]}"
        counter[0] += 1
        param_names.append(pname)

        if suffix:
            # Coefficient with suffix (CRK), the numeric part is not the GT value
            gt_values.append(None)
        else:
            gt_values.append(float(num_str))

        return pname

    # Protect feature variables and function names with numbered placeholders to avoid regex false matches
    placeholders: Dict[str, str] = {}
    protected_expr = raw_expr
    idx = 0

    for fname in sorted(feature_names, key=len, reverse=True):
        ph = f"__V{idx}__"
        placeholders[ph] = fname
        protected_expr = re.sub(
            r'(?<![a-zA-Z_])' + re.escape(fname) + r'(?![a-zA-Z_0-9])',
            ph, protected_expr)
        idx += 1

    func_names = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
    for fn in func_names:
        ph = f"__V{idx}__"
        placeholders[ph] = fn
        protected_expr = re.sub(
            r'(?<![a-zA-Z_])' + re.escape(fn) + r'(?![a-zA-Z_0-9])',
            ph, protected_expr)
        idx += 1

    skeleton = _NUM_PATTERN.sub(_replacer, protected_expr)

    for ph, orig in placeholders.items():
        skeleton = skeleton.replace(ph, orig)

    return skeleton, param_names, gt_values


# ---------------------------------------------------------------------------
# Reference fitting: determine GT values of undetermined coefficients
# ---------------------------------------------------------------------------

def _reference_fit(skeleton: str, param_names: List[str],
                   gt_values: List[Optional[float]],
                   feature_names: List[str],
                   X_train: np.ndarray, y_train: np.ndarray,
                   ) -> List[float]:
    """
    Perform reference fitting for parameters with unknown GT values.
    Fix parameters with known values and only optimize unknown ones.
    """
    from scipy.optimize import differential_evolution, minimize

    unknown_idx = [i for i, v in enumerate(gt_values) if v is None]
    if not unknown_idx:
        return [v for v in gt_values]  # type: ignore

    known_values = {i: v for i, v in enumerate(gt_values) if v is not None}

    all_symbols = [sp.Symbol(f) for f in feature_names] + \
                  [sp.Symbol(p) for p in param_names]
    try:
        local_dict = {s.name: s for s in all_symbols}
        sympy_expr = sp.sympify(skeleton, locals=local_dict)
        func = sp.lambdify(all_symbols, sympy_expr, modules=["numpy"])
    except Exception as e:
        raise RuntimeError(f"Failed to compile skeleton expression: {skeleton}\nError: {e}")

    X_cols = [X_train[:, i] for i in range(X_train.shape[1])]

    def objective(unknown_vals):
        full_params = list(gt_values)
        for idx_in_unknown, orig_idx in enumerate(unknown_idx):
            full_params[orig_idx] = unknown_vals[idx_in_unknown]
        for i, v in known_values.items():
            full_params[i] = v
        try:
            import warnings
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = np.asarray(func(*X_cols, *full_params), dtype=float)
                if np.any(~np.isfinite(y_pred)):
                    return 1e20
                return float(np.mean((y_pred - y_train) ** 2))
        except Exception:
            return 1e20

    n_unknown = len(unknown_idx)
    bounds = [(-100.0, 100.0)] * n_unknown

    best_mse = float("inf")
    best_vals = [1.0] * n_unknown

    for seed in range(5):
        try:
            res = differential_evolution(
                objective, bounds=bounds, maxiter=2000,
                seed=seed, tol=1e-12, popsize=15,
            )
            if res.fun < best_mse:
                best_mse = res.fun
                best_vals = list(res.x)
        except Exception:
            pass

    if best_mse < 1e-3:
        try:
            res2 = minimize(objective, best_vals, method="L-BFGS-B",
                            options={"maxiter": 5000, "ftol": 1e-15})
            if res2.fun < best_mse:
                best_vals = list(res2.x)
        except Exception:
            pass

    result = list(gt_values)
    for idx_in_unknown, orig_idx in enumerate(unknown_idx):
        result[orig_idx] = best_vals[idx_in_unknown]
    return result  # type: ignore


# ---------------------------------------------------------------------------
# Parameter decoupling: eliminate c_i*c_j / c_i/c_j coupling
# ---------------------------------------------------------------------------

_FUNC_TYPES = (sp.exp, sp.sin, sp.cos, sp.log, sp.tan,
               sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan)


def _is_pure_param(factor: sp.Expr, param_syms: set) -> bool:
    """True when *factor* depends only on parameter symbols (no features)."""
    if factor.is_Number:
        return True
    fs = factor.free_symbols
    return bool(fs) and fs <= param_syms


def _collapse_arg_coupling(
    arg: sp.Expr,
    param_syms: set,
    gt_map: Dict[sp.Symbol, float],
    counter: List[int],
) -> Tuple[sp.Expr, List[Tuple[str, float]]]:
    """Expand *arg* and merge any >=2 pure-param multiplicative factors."""
    expanded = sp.expand_mul(arg)
    terms = sp.Add.make_args(expanded)

    if not any(
        sum(1 for f in sp.Mul.make_args(t) if _is_pure_param(f, param_syms)) >= 2
        for t in terms
    ):
        return arg, []

    new_params: List[Tuple[str, float]] = []
    new_terms: List[sp.Expr] = []
    for term in terms:
        factors = sp.Mul.make_args(term)
        p_facs = [f for f in factors if _is_pure_param(f, param_syms)]
        o_facs = [f for f in factors if not _is_pure_param(f, param_syms)]

        if len(p_facs) >= 2:
            prod = sp.Mul(*p_facs)
            gt_val = float(prod.xreplace(gt_map))
            name = f"c{counter[0]}"; counter[0] += 1
            sym = sp.Symbol(name)
            param_syms.add(sym)
            gt_map[sym] = gt_val
            new_params.append((name, gt_val))
            rest = sp.Mul(*o_facs) if o_facs else sp.Integer(1)
            new_terms.append(sym * rest)
        else:
            new_terms.append(term)

    return sp.Add(*new_terms), new_params


def _normalise_subtree(
    expr: sp.Expr,
    param_syms: set,
    gt_map: Dict[sp.Symbol, float],
    counter: List[int],
) -> Tuple[sp.Expr, List[Tuple[str, float]]]:
    """Bottom-up: collapse param coupling inside func args & Pow exponents."""
    if expr.is_Atom:
        return expr, []

    collected: List[Tuple[str, float]] = []

    new_args = []
    for a in expr.args:
        na, np_ = _normalise_subtree(a, param_syms, gt_map, counter)
        new_args.append(na)
        collected.extend(np_)
    expr = expr.func(*new_args)

    if isinstance(expr, _FUNC_TYPES):
        new_arg, ap = _collapse_arg_coupling(
            expr.args[0], param_syms, gt_map, counter)
        collected.extend(ap)
        if ap:
            expr = expr.func(new_arg)

    elif isinstance(expr, sp.Pow):
        _base, _exp = expr.args
        if not _exp.is_Atom and (_exp.free_symbols & param_syms):
            new_exp, ep = _collapse_arg_coupling(
                _exp, param_syms, gt_map, counter)
            collected.extend(ep)
            if ep:
                expr = sp.Pow(_base, new_exp)

    return expr, collected


def _decouple_skeleton(
    skeleton: str,
    param_names: List[str],
    gt_values: List[float],
    feature_names: List[str],
) -> Tuple[str, List[str], List[float]]:
    """
    Rewrite *skeleton* so that no two param symbols are multiplied / divided.

    Phase 1 – recursively normalise function args & Pow exponents.
    Phase 2 – ``expand_mul`` at top level, group additive terms by monomial,
              one linear coefficient param per monomial group.

    Returns ``(new_skeleton, new_param_names, new_gt_values)``.
    """
    all_names = feature_names + param_names
    local_dict = {n: sp.Symbol(n) for n in all_names}
    expr = sp.sympify(skeleton, locals=local_dict)

    param_syms: set = {sp.Symbol(p) for p in param_names}
    gt_map: Dict[sp.Symbol, float] = {
        sp.Symbol(p): v for p, v in zip(param_names, gt_values)
    }
    max_idx = max(
        (int(p[1:]) for p in param_names
         if p.startswith("c") and p[1:].isdigit()),
        default=-1,
    )
    counter = [max_idx + 1]

    # ---- Phase 1 ----
    expr, _ = _normalise_subtree(expr, param_syms, gt_map, counter)

    # ---- Phase 2 ----
    expanded = sp.expand_mul(expr)
    terms = sp.Add.make_args(expanded)

    groups: Dict[str, Tuple[sp.Expr, List[sp.Expr]]] = {}
    for term in terms:
        factors = sp.Mul.make_args(term)
        p_facs = [f for f in factors if _is_pure_param(f, param_syms)]
        o_facs = [f for f in factors if not _is_pure_param(f, param_syms)]
        coeff = sp.Mul(*p_facs) if p_facs else sp.Integer(1)
        mono = sp.Mul(*o_facs) if o_facs else sp.Integer(1)
        key = str(mono)
        if key not in groups:
            groups[key] = (mono, [])
        groups[key][1].append(coeff)

    nl_syms: set = set()
    for mono, _ in groups.values():
        nl_syms |= (mono.free_symbols & param_syms)

    out_names: List[str] = []
    out_gt: List[float] = []
    rename: Dict[sp.Symbol, sp.Symbol] = {}

    for s in sorted(nl_syms, key=lambda x: str(x)):
        new = f"c{len(out_names)}"
        out_names.append(new)
        out_gt.append(float(gt_map[s]))
        rename[s] = sp.Symbol(new)

    parts: List[sp.Expr] = []
    for _key, (mono, coeffs) in groups.items():
        total = sp.Add(*coeffs)
        gv = float(total.xreplace(gt_map))
        new = f"c{len(out_names)}"
        out_names.append(new)
        out_gt.append(gv)
        parts.append(sp.Symbol(new) * mono.xreplace(rename))

    new_expr = sp.Add(*parts) if parts else sp.Integer(0)
    return str(new_expr), out_names, out_gt


def _verify_decouple(
    old_skeleton: str, old_params: List[str], old_gt: List[float],
    new_skeleton: str, new_params: List[str], new_gt: List[float],
    feature_names: List[str],
) -> float:
    """Return max relative error between old and new at several test points."""
    feat_syms = [sp.Symbol(f) for f in feature_names]

    old_local = {n: sp.Symbol(n) for n in feature_names + old_params}
    new_local = {n: sp.Symbol(n) for n in feature_names + new_params}
    old_expr = sp.sympify(old_skeleton, locals=old_local)
    new_expr = sp.sympify(new_skeleton, locals=new_local)

    for p, v in zip(old_params, old_gt):
        old_expr = old_expr.subs(sp.Symbol(p), sp.Float(v))
    for p, v in zip(new_params, new_gt):
        new_expr = new_expr.subs(sp.Symbol(p), sp.Float(v))

    old_fn = sp.lambdify(feat_syms, old_expr, modules=["numpy"])
    new_fn = sp.lambdify(feat_syms, new_expr, modules=["numpy"])

    rng = np.random.default_rng(42)
    max_rel = 0.0
    for _ in range(5):
        vals = rng.uniform(0.5, 5.0, len(feature_names))
        try:
            ov = float(old_fn(*vals))
            nv = float(new_fn(*vals))
            if not (np.isfinite(ov) and np.isfinite(nv)):
                continue
            rel = abs(ov - nv) / (abs(ov) + 1e-10)
            max_rel = max(max_rel, rel)
        except Exception:
            continue
    return max_rel


# ---------------------------------------------------------------------------
# GT extraction (from CSV)
# ---------------------------------------------------------------------------

_GT_CSV = Path(__file__).parent.parent / "datasets" / "llm-srbench" / "gt_expressions.csv"


def extract_gt_from_csv(
    csv_path: Optional[Path] = None,
    max_per_split: int = 0,
) -> List[GTInfo]:
    """Read GT expressions from ``gt_expressions.csv`` and perform parameter decoupling.

    Parameters
    ----------
    csv_path : Optional, defaults to ``gt_expressions.csv`` in the same directory.
    max_per_split : Max number of GTs per split, 0 means all.
    """
    csv_path = Path(csv_path) if csv_path else _GT_CSV
    df = pd.read_csv(csv_path)

    split_counts: Dict[str, int] = {}
    all_gts: List[GTInfo] = []

    for _, row in df.iterrows():
        raw_split = str(row["split"])
        short = _CSV_SPLIT_MAP.get(raw_split, raw_split)

        if max_per_split > 0:
            split_counts.setdefault(short, 0)
            if split_counts[short] >= max_per_split:
                continue
            split_counts[short] += 1

        eq_name = str(row["equation_name"])
        feature_names = str(row["feature_names"]).split(";")
        skeleton = str(row["symbolic_expression"])
        param_names = str(row["param_names"]).split(";")
        gt_params = [float(x) for x in str(row["gt_params"]).split(";")]
        raw_expression = str(row.get("raw_expression", ""))

        old_n = len(param_names)
        try:
            new_skel, new_pnames, new_gv = _decouple_skeleton(
                skeleton, param_names, gt_params, feature_names)
            rel_err = _verify_decouple(
                skeleton, param_names, gt_params,
                new_skel, new_pnames, new_gv,
                feature_names)
            if rel_err <= 1e-4:
                skeleton = new_skel
                param_names = new_pnames
                gt_params = new_gv
                if len(param_names) != old_n:
                    print(f"  [decouple] {eq_name}: {old_n} → {len(param_names)} params")
        except Exception as e:
            print(f"  [WARN] {eq_name}: decoupling failed ({e}), keeping original skeleton")

        all_gts.append(GTInfo(
            equation_name=eq_name,
            split=short,
            raw_expression=raw_expression,
            skeleton=skeleton,
            param_names=param_names,
            gt_params=gt_params,
            feature_names=feature_names,
        ))

    print(f"[INFO] Read {len(all_gts)} GT expressions from CSV")
    return all_gts


# ---------------------------------------------------------------------------
# GT extraction (from parquet + HDF5, legacy path)
# ---------------------------------------------------------------------------

def extract_gt_cases(max_per_split: int = 15) -> List[GTInfo]:
    """Extract max_per_split GT expressions from each of the three splits."""
    import h5py

    snap_dir = _find_snapshot_dir()
    data_dir = snap_dir / "data"
    hdf5_path = snap_dir / "lsr_bench_data.hdf5"

    all_gts: List[GTInfo] = []

    for split_short, split_info in _SPLITS.items():
        hf_prefix = split_info["hf_split"]
        hdf5_prefix = split_info["hdf5_prefix"]
        pq_files = list(data_dir.glob(f"{hf_prefix}-*.parquet"))
        if not pq_files:
            print(f"[WARN] {split_short}: parquet files not found, skipping")
            continue

        df = pd.read_parquet(pq_files[0])
        print(f"[INFO] {split_short}: {len(df)} equations total, taking first {max_per_split}")

        with h5py.File(hdf5_path, "r") as hf:
            count = 0
            for _, row in df.iterrows():
                if count >= max_per_split:
                    break

                eq_name = str(row["name"])
                raw_expr = str(row["expression"])
                symbols = row["symbols"]
                feature_names = _get_feature_names(symbols, split_short)

                cleaned = _clean_expression(raw_expr, split_short)
                skeleton, param_names, gt_values = _parse_expression(
                    cleaned, feature_names)

                if not param_names:
                    print(f"  [SKIP] {eq_name}: no parameters")
                    continue

                # Load training data (for reference fitting)
                has_unknown = any(v is None for v in gt_values)
                if has_unknown:
                    eq_path = f"/{hdf5_prefix}/{eq_name}"
                    try:
                        item = hf[eq_path]
                        train_key = list(item.keys())[0]
                        data = np.array(item[train_key], dtype=np.float64)
                        y_train = data[:, 0]
                        X_train = data[:, 1:]
                    except Exception as e:
                        print(f"  [SKIP] {eq_name}: failed to load data - {e}")
                        continue

                    try:
                        final_gt = _reference_fit(
                            skeleton, param_names, gt_values,
                            feature_names, X_train, y_train)
                    except Exception as e:
                        print(f"  [SKIP] {eq_name}: reference fitting failed - {e}")
                        continue

                    # Verify fitting quality
                    all_syms = [sp.Symbol(f) for f in feature_names] + \
                               [sp.Symbol(p) for p in param_names]
                    local_dict = {s.name: s for s in all_syms}
                    sympy_expr = sp.sympify(skeleton, locals=local_dict)
                    func = sp.lambdify(all_syms, sympy_expr, modules=["numpy"])
                    X_cols = [X_train[:, i] for i in range(X_train.shape[1])]
                    import warnings
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y_pred = np.asarray(func(*X_cols, *final_gt), dtype=float)
                    var_y = float(np.var(y_train))
                    mse = float(np.mean((y_pred - y_train) ** 2))
                    nmse = mse / var_y if var_y > 0 else float("inf")
                    if nmse > 1e-4:
                        print(f"  [SKIP] {eq_name}: reference fit NMSE={nmse:.6f} too large")
                        continue
                    print(f"  [OK]   {eq_name}: {len(param_names)} params, "
                          f"NMSE={nmse:.2e} (with reference fit)")
                else:
                    final_gt = gt_values  # type: ignore
                    print(f"  [OK]   {eq_name}: {len(param_names)} params")

                # ---- parameter decoupling ----
                old_n = len(param_names)
                try:
                    new_skel, new_pnames, new_gv = _decouple_skeleton(
                        skeleton, param_names, final_gt, feature_names)
                    rel_err = _verify_decouple(
                        skeleton, param_names, final_gt,
                        new_skel, new_pnames, new_gv,
                        feature_names)
                    if rel_err > 1e-4:
                        print(f"         [WARN] decoupling verification error {rel_err:.2e}, keeping original skeleton")
                    else:
                        skeleton = new_skel
                        param_names = new_pnames
                        final_gt = new_gv
                        if len(param_names) != old_n:
                            print(f"         -> decoupled: {old_n} -> {len(param_names)} params")
                except Exception as e:
                    print(f"         [WARN] decoupling failed ({e}), keeping original skeleton")

                all_gts.append(GTInfo(
                    equation_name=eq_name,
                    split=split_short,
                    raw_expression=raw_expr,
                    skeleton=skeleton,
                    param_names=param_names,
                    gt_params=final_gt,
                    feature_names=feature_names,
                ))
                count += 1

    print(f"\n[INFO] Extracted {len(all_gts)} GT expressions total")
    return all_gts


# ---------------------------------------------------------------------------
# Variant generator (4 transforms x 2)
# ---------------------------------------------------------------------------

def _next_param(existing: List[str]) -> str:
    """Return the next available parameter name cN."""
    max_idx = -1
    for p in existing:
        if p.startswith("c") and p[1:].isdigit():
            max_idx = max(max_idx, int(p[1:]))
    return f"c{max_idx + 1}"


def _alloc_params(existing: List[str], n: int) -> List[str]:
    """Allocate n new parameter names."""
    result = []
    cur = list(existing)
    for _ in range(n):
        p = _next_param(cur)
        result.append(p)
        cur.append(p)
    return result


def _get_first_feature(gt: GTInfo) -> str:
    return gt.feature_names[0]


def _get_second_feature(gt: GTInfo) -> str:
    if len(gt.feature_names) > 1:
        return gt.feature_names[1]
    return gt.feature_names[0]


def transform_T1a(gt: GTInfo) -> BenchCase:
    """Add composite zero term: + p1 * sin(log(1 + p2*x) + p3)"""
    x = _get_first_feature(gt)
    new_p = _alloc_params(gt.param_names, 3)
    p1, p2, p3 = new_p
    added = f" + {p1}*sin(log(1 + {p2}*{x}) + {p3})"
    return BenchCase(
        case_id=f"{gt.equation_name}_T1a",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=gt.skeleton + added,
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [0.0, 1.0, 0.0],
        feature_names=gt.feature_names,
        transform="T1a",
    )


def transform_T1b(gt: GTInfo) -> BenchCase:
    """Add composite zero term: + p1 * exp(p2 * cos(p3 * x))"""
    x = _get_first_feature(gt)
    new_p = _alloc_params(gt.param_names, 3)
    p1, p2, p3 = new_p
    added = f" + {p1}*exp({p2}*cos({p3}*{x}))"
    return BenchCase(
        case_id=f"{gt.equation_name}_T1b",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=gt.skeleton + added,
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [0.0, 0.1, 1.0],
        feature_names=gt.feature_names,
        transform="T1b",
    )


def _has_param_pow(expr: sp.Expr, var_sym: sp.Symbol, param_syms: set) -> bool:
    """Check if *var_sym* appears as base of a Pow whose exponent contains params."""
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.args[0] == var_sym:
            if node.args[1].free_symbols & param_syms:
                return True
    return False


def _pick_linear_feature(
    gt: GTInfo, expr: sp.Expr, param_syms: set, prefer: List[str],
) -> Optional[sp.Symbol]:
    """Return the first feature in *prefer* that appears without param-Pow."""
    seen = set()
    for name in prefer:
        if name in seen:
            continue
        seen.add(name)
        sym = sp.Symbol(name)
        if sym in expr.free_symbols and not _has_param_pow(expr, sym, param_syms):
            return sym
    return None


def transform_T2a(gt: GTInfo) -> Optional[BenchCase]:
    """Add power parameter to linear feature: var -> var**c_new, GT: c_new=1 (prefer second feature)"""
    local_dict = {n: sp.Symbol(n) for n in gt.feature_names + gt.param_names}
    try:
        expr = sp.sympify(gt.skeleton, locals=local_dict)
    except Exception:
        return None

    param_syms = {sp.Symbol(p) for p in gt.param_names}
    prefer = [_get_second_feature(gt), _get_first_feature(gt)]
    target = _pick_linear_feature(gt, expr, param_syms, prefer)
    if target is None:
        return None

    new_p = _alloc_params(gt.param_names, 1)
    new_expr = expr.subs(target, target ** sp.Symbol(new_p[0]))

    return BenchCase(
        case_id=f"{gt.equation_name}_T2a",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=str(new_expr),
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [1.0],
        feature_names=gt.feature_names,
        transform="T2a",
    )


def transform_T2b(gt: GTInfo) -> BenchCase:
    """Add power parameter to entire expression: (expr)**c_new, GT: c_new=1"""
    new_p = _alloc_params(gt.param_names, 1)
    return BenchCase(
        case_id=f"{gt.equation_name}_T2b",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=f"({gt.skeleton})**{new_p[0]}",
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [1.0],
        feature_names=gt.feature_names,
        transform="T2b",
    )


def transform_T4a(gt: GTInfo) -> BenchCase:
    """Add rational zero term: + (ca*x + cb) / (cc*x + cd), GT: ca=0, cb=0, cc=1, cd=1"""
    x = _get_first_feature(gt)
    new_p = _alloc_params(gt.param_names, 4)
    ca, cb, cc, cd = new_p
    added = f" + ({ca}*{x} + {cb})/({cc}*{x} + {cd})"

    return BenchCase(
        case_id=f"{gt.equation_name}_T4a",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=gt.skeleton + added,
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [0.0, 0.0, 1.0, 1.0],
        feature_names=gt.feature_names,
        transform="T4a",
    )


def transform_T4b(gt: GTInfo) -> BenchCase:
    """Add rational zero term (using second feature): + (ca*x2 + cb) / (cc*x + cd)"""
    x = _get_first_feature(gt)
    x2 = _get_second_feature(gt)
    new_p = _alloc_params(gt.param_names, 4)
    ca, cb, cc, cd = new_p
    added = f" + ({ca}*{x2} + {cb})/({cc}*{x} + {cd})"

    return BenchCase(
        case_id=f"{gt.equation_name}_T4b",
        split=gt.split,
        equation_name=gt.equation_name,
        skeleton=gt.skeleton + added,
        param_names=gt.param_names + new_p,
        gt_params=gt.gt_params + [0.0, 0.0, 1.0, 1.0],
        feature_names=gt.feature_names,
        transform="T4b",
    )


_TRANSFORMS = [
    transform_T1a, transform_T1b,
    transform_T2a, transform_T2b,
    transform_T4a, transform_T4b,
]


def generate_all_cases(
    max_per_split: int = 0,
    csv_path: Optional[Path] = None,
) -> List[BenchCase]:
    """Generate all benchmark test cases: N GT x (1 base + 6 variants).

    Prefer reading from CSV; fall back to parquet + HDF5 when CSV does not exist.
    """
    use_csv = csv_path or _GT_CSV.exists()
    if use_csv:
        gts = extract_gt_from_csv(csv_path, max_per_split=max_per_split)
    else:
        gts = extract_gt_cases(max_per_split=max_per_split or 15)
    cases: List[BenchCase] = []

    for gt in gts:
        # T0: original formula, no perturbation terms
        cases.append(BenchCase(
            case_id=f"{gt.equation_name}_T0",
            split=gt.split,
            equation_name=gt.equation_name,
            skeleton=gt.skeleton,
            param_names=gt.param_names,
            gt_params=gt.gt_params,
            feature_names=gt.feature_names,
            transform="T0",
        ))

        for tfn in _TRANSFORMS:
            result = tfn(gt)
            if result is None:
                # T2a/T2b may skip due to no Pow node, use fallback variant
                x = _get_first_feature(gt)
                new_p = _alloc_params(gt.param_names, 1)
                fallback_skeleton = f"({gt.skeleton})**{new_p[0]}"
                result = BenchCase(
                    case_id=f"{gt.equation_name}_{tfn.__name__[-3:]}",
                    split=gt.split,
                    equation_name=gt.equation_name,
                    skeleton=fallback_skeleton,
                    param_names=gt.param_names + new_p,
                    gt_params=gt.gt_params + [1.0],
                    feature_names=gt.feature_names,
                    transform=tfn.__name__[-3:],
                )
            cases.append(result)

    print(f"\n[INFO] Generated {len(cases)} test cases total "
          f"({len(gts)} GT x (1 T0 + {len(_TRANSFORMS)} variants))")
    return cases


# ---------------------------------------------------------------------------
# CLI entry point: extract and print info
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = generate_all_cases(max_per_split=15)
    print(f"\n{'='*60}")
    print(f"Total: {len(cases)} test cases")
    print(f"{'='*60}")

    from collections import Counter
    split_counts = Counter(c.split for c in cases)
    transform_counts = Counter(c.transform for c in cases)
    param_counts = [c.n_params for c in cases]

    print(f"\nDistribution by split: {dict(split_counts)}")
    print(f"Distribution by transform: {dict(transform_counts)}")
    print(f"Parameter count: min={min(param_counts)}, max={max(param_counts)}, "
          f"mean={np.mean(param_counts):.1f}")

    print("\nFirst 10 cases:")
    for c in cases[:10]:
        print(f"  {c.case_id}: {c.n_params} params, "
              f"skeleton={c.skeleton[:60]}...")
