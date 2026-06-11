"""Constant optimizer base class and shared utilities."""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency fallback
    threadpool_limits = None


# ---------------------------------------------------------------------------
# Safe power function (handles negative base + non-integer exponent)
# ---------------------------------------------------------------------------

_RATIONAL_DENOM = 3


def _snap_rational(x: float) -> float:
    return round(x * _RATIONAL_DENOM) / _RATIONAL_DENOM


def _real_pow_np(base, exp):
    base = np.asarray(base, dtype=float)
    exp_val = float(exp)
    p = int(round(exp_val * _RATIONAL_DENOM))
    snapped = p / _RATIONAL_DENOM
    abs_result = np.power(np.abs(base), snapped)
    neg_mask = base < 0
    if np.any(neg_mask):
        if p % 2 != 0:
            abs_result = np.where(neg_mask, -abs_result, abs_result)
    return abs_result


class _RealPow(sp.Function):
    nargs = 2


_SAFE_MODULES = [{"_RealPow": _real_pow_np}, "numpy"]

_RATIONAL_INIT_POOL = [
    -3, -2, -5 / 3, -4 / 3, -1, -2 / 3, -1 / 3,
    1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2, 3,
]

_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_THREAD_LIMITER = None
_THREAD_LIMITS_APPLIED = False


def _configure_numeric_threads(limit: int = 1) -> None:
    """Force BLAS/OpenMP-style backends to stay single-threaded per process."""
    global _THREAD_LIMITER, _THREAD_LIMITS_APPLIED
    if _THREAD_LIMITS_APPLIED:
        return

    limit_str = str(limit)
    for env_name in _THREAD_ENV_VARS:
        os.environ[env_name] = limit_str

    if threadpool_limits is not None:
        try:
            _THREAD_LIMITER = threadpool_limits(limits=limit)
            _THREAD_LIMITER.__enter__()
        except Exception:
            _THREAD_LIMITER = None

    _THREAD_LIMITS_APPLIED = True


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    best_params: np.ndarray
    best_mse: float
    n_feval: int
    rational_idx: List[int] = field(default_factory=list)
    safe_expr: Optional[sp.Expr] = None


# ---------------------------------------------------------------------------
# Internal timeout signal
# ---------------------------------------------------------------------------

class _OptTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Expression analysis utilities (pure functions / static methods)
# ---------------------------------------------------------------------------

def parse_expr(skeleton: str, param_names: List[str],
               feature_names: List[str]) -> sp.Expr:
    local_syms = {n: sp.Symbol(n) for n in feature_names + list(param_names)}
    return sp.sympify(skeleton, locals=local_syms)


def compile_expr(sympy_expr: sp.Expr, param_names: List[str],
                 feature_names: List[str], X: np.ndarray
                 ) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a SymPy expression into a ``params -> y_pred`` numpy function."""
    all_symbols = [sp.Symbol(n) for n in feature_names + list(param_names)]
    lambdified = sp.lambdify(all_symbols, sympy_expr, modules=_SAFE_MODULES)
    X_cols = [X[:, i] for i in range(len(feature_names))]

    def func(params: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lambdified(*X_cols, *list(params))
        return np.asarray(result, dtype=float)

    return func


def build_mse_fn(predict_fn: Callable, y: np.ndarray,
                 penalty: float = 1e10) -> Callable[[np.ndarray], float]:
    def mse_fn(params: np.ndarray) -> float:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y_pred = predict_fn(params)
                if np.any(~np.isfinite(y_pred)):
                    return penalty
                mse = float(np.mean((y_pred - y) ** 2))
                return mse if np.isfinite(mse) else penalty
            except TimeoutError:
                raise
            except Exception:
                return penalty
    return mse_fn


def build_residual_fn(predict_fn: Callable, y: np.ndarray,
                      penalty_per_sample: float = 1e10
                      ) -> Callable[[np.ndarray], np.ndarray]:
    n = len(y)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y_pred = predict_fn(params)
                if np.any(~np.isfinite(y_pred)):
                    return np.full(n, penalty_per_sample)
                return y_pred - y
            except TimeoutError:
                raise
            except Exception:
                return np.full(n, penalty_per_sample)
    return residual_fn


def detect_rational_constrained(sympy_expr: sp.Expr, param_names: List[str],
                                feature_names: List[str],
                                X_train: np.ndarray) -> set:
    """Return the set of parameter indices that require rational snap."""
    param_sym_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
    param_syms = set(param_sym_to_idx.keys())
    feature_syms = {sp.Symbol(f) for f in feature_names}
    feat_to_col = {sp.Symbol(f): i for i, f in enumerate(feature_names)}

    constrained: set = set()
    for node in sp.preorder_traversal(sympy_expr):
        if not (isinstance(node, sp.Pow) and len(node.args) == 2):
            continue
        base, exponent = node.args
        exp_params = exponent.free_symbols & param_syms
        if not exp_params:
            continue
        base_features = base.free_symbols & feature_syms
        if not base_features:
            continue
        has_negative_base = any(
            np.any(X_train[:, feat_to_col[f]] < 0)
            for f in base_features if f in feat_to_col
        )
        if has_negative_base:
            for s in exp_params:
                constrained.add(param_sym_to_idx[s])
    return constrained


def make_safe_expr(sympy_expr: sp.Expr, param_names: List[str],
                   rational_idx: set, feature_names: List[str]) -> sp.Expr:
    """Replace Pow nodes at risk of negative base with _RealPow."""
    if not rational_idx:
        return sympy_expr
    param_syms = {sp.Symbol(p) for i, p in enumerate(param_names) if i in rational_idx}
    feature_syms = {sp.Symbol(f) for f in feature_names}

    def _replace(node):
        if isinstance(node, sp.Pow) and len(node.args) == 2:
            base, exponent = node.args
            if (exponent.free_symbols & param_syms
                    and base.free_symbols & feature_syms):
                new_base = base.replace(lambda n: isinstance(n, sp.Pow), _replace)
                new_exp = exponent.replace(lambda n: isinstance(n, sp.Pow), _replace)
                return _RealPow(new_base, new_exp)
        return node
    return sympy_expr.replace(lambda n: isinstance(n, sp.Pow), _replace)


def param_hints(sympy_expr: sp.Expr, param_names: List[str]
                ) -> Tuple[set, set]:
    """Return (positive_idx, small_idx)."""
    sym_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
    param_syms = set(sym_to_idx.keys())
    positive: set = set()
    small: set = set()

    denom_exp_ids: set = set()
    for node in sp.preorder_traversal(sympy_expr):
        if isinstance(node, sp.Pow) and len(node.args) == 2:
            if getattr(node.args[1], 'is_negative', False):
                for sub in sp.preorder_traversal(node.args[0]):
                    if isinstance(sub, sp.exp):
                        denom_exp_ids.add(id(sub))

    for node in sp.preorder_traversal(sympy_expr):
        if isinstance(node, sp.log):
            for s in node.args[0].free_symbols & param_syms:
                positive.add(sym_to_idx[s])
                small.add(sym_to_idx[s])
        elif isinstance(node, sp.exp):
            if id(node) not in denom_exp_ids:
                inner = node.args[0]
                expanded = sp.expand(inner)
                for s in inner.free_symbols & param_syms:
                    if expanded.coeff(s) != 0:
                        small.add(sym_to_idx[s])
        elif isinstance(node, (sp.sin, sp.cos, sp.tan)):
            arg = node.args[0]
            for s in arg.free_symbols & param_syms:
                coeff = arg.coeff(s)
                has_features = bool(coeff.free_symbols - param_syms)
                if has_features:
                    positive.add(sym_to_idx[s])
                    small.add(sym_to_idx[s])
                else:
                    small.add(sym_to_idx[s])
        elif isinstance(node, (sp.Pow, _RealPow)) and len(node.args) == 2:
            base, exp = node.args
            for s in exp.free_symbols & param_syms:
                small.add(sym_to_idx[s])
            if getattr(exp, 'is_number', False) and getattr(exp, 'is_negative', False):
                nested_in_func: set = set()
                for sub in sp.preorder_traversal(base):
                    if isinstance(sub, (sp.exp, sp.sin, sp.cos, sp.tan, sp.log, _RealPow)):
                        nested_in_func |= sub.free_symbols
                for s in (base.free_symbols & param_syms) - nested_in_func:
                    positive.add(sym_to_idx[s])
    return positive, small


def _find_feat_offset_in_add(
    add_node: sp.Add,
    param_syms: set,
    param_to_idx: Dict,
    feat_syms_set: set,
    feature_names: List[str],
) -> Dict[int, int]:
    """Find (feat ± c) patterns in an Add node, return {param_idx: feature_col_idx}.

    Match condition: the Add contains a term with only parameters (no features)
    and a term with features.
    """
    result: Dict[int, int] = {}
    for p in add_node.free_symbols & param_syms:
        has_pure_param_term = False
        feat_col = -1
        for term in add_node.args:
            tsyms = term.free_symbols
            if p in tsyms and not (tsyms & feat_syms_set):
                other_params = (tsyms & param_syms) - {p}
                if not other_params:
                    has_pure_param_term = True
            elif tsyms & feat_syms_set and p not in tsyms:
                for fi, fname in enumerate(feature_names):
                    if sp.Symbol(fname) in tsyms:
                        feat_col = fi
                        break
        if has_pure_param_term and feat_col >= 0:
            result[param_to_idx[p]] = feat_col
    return result


def compute_exp_param_bounds(
    sympy_expr: sp.Expr,
    param_names: List[str],
    feature_names: List[str],
    X_train: np.ndarray,
    max_exp_arg: float = 10.0,
) -> Dict[int, Tuple[float, float]]:
    """Compute adaptive search bounds for exp() parameters and Pow offset parameters based on training data range.

    Handles three patterns:
    1. exp(c_i * f(X)): constrain c_i to [-M/f_max, M/f_max]
    2. exp(g(feat - c_i)): constrain c_i to the feature's data range (with margin)
    3. Pow(feat - c_i, n) / _RealPow(feat - c_i, n): same as 2

    Skips exp in denominators (e.g. ``1/(c + exp(...))``), because large negative
    values are legitimate there (exp→0 makes denominator→c), no extra constraints needed.
    """
    param_syms = {sp.Symbol(p) for p in param_names}
    param_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
    feat_syms_set = {sp.Symbol(f) for f in feature_names}
    feat_syms_list = [sp.Symbol(f) for f in feature_names]

    denom_exps: set = set()
    for node in sp.preorder_traversal(sympy_expr):
        if isinstance(node, sp.Pow) and node.args[1].is_negative:
            for sub in sp.preorder_traversal(node.args[0]):
                if isinstance(sub, sp.exp):
                    denom_exps.add(id(sub))

    tight: Dict[int, Tuple[float, float]] = {}

    def _set_feat_range_bounds(p_idx: int, fi: int):
        if p_idx in tight:
            return
        col = X_train[:, fi]
        margin = max(float(np.ptp(col)) * 0.2, 10.0)
        lo = float(np.min(col)) - margin
        hi = float(np.max(col)) + margin
        tight[p_idx] = (lo, hi)

    # --- Pattern 1: exp(c_i * f(X)) → constrain c_i ---
    for node in sp.preorder_traversal(sympy_expr):
        if not isinstance(node, sp.exp):
            continue
        if id(node) in denom_exps:
            continue
        arg = node.args[0]
        arg_params = arg.free_symbols & param_syms
        if not arg_params:
            continue

        expanded_arg = sp.expand(arg)

        for p in arg_params:
            coeff = expanded_arg.coeff(p)
            p_idx = param_to_idx[p]

            if coeff != 0 and not (coeff.free_symbols & param_syms):
                try:
                    if not coeff.free_symbols:
                        f_absmax = abs(float(coeff))
                    else:
                        coeff_fn = sp.lambdify(feat_syms_list, coeff, modules=["numpy"])
                        X_cols = [X_train[:, i] for i in range(len(feature_names))]
                        with np.errstate(all="ignore"), warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            vals = np.asarray(coeff_fn(*X_cols), dtype=float)
                        if not np.all(np.isfinite(vals)):
                            continue
                        f_absmax = float(np.max(np.abs(vals)))
                except Exception:
                    continue

                if f_absmax < 1e-10:
                    continue

                half = max_exp_arg / f_absmax
                if p_idx in tight:
                    old_lo, old_hi = tight[p_idx]
                    tight[p_idx] = (max(old_lo, -half), min(old_hi, half))
                else:
                    tight[p_idx] = (-half, half)

    # --- Pattern 2: Gaussian offset inside exp: exp(g(feat - c)) ---
    for node in sp.preorder_traversal(sympy_expr):
        if not isinstance(node, sp.exp):
            continue
        if id(node) in denom_exps:
            continue
        arg = node.args[0]
        expanded_arg = sp.expand(arg)
        for p in arg.free_symbols & param_syms:
            if expanded_arg.coeff(p) != 0:
                continue
            for sub in sp.preorder_traversal(arg):
                if not isinstance(sub, sp.Add):
                    continue
                offsets = _find_feat_offset_in_add(
                    sub, param_syms, param_to_idx,
                    feat_syms_set, feature_names)
                for p_idx, fi in offsets.items():
                    _set_feat_range_bounds(p_idx, fi)

    # --- Pattern 3: bare Pow/RealPow offset (feat - c)**n ---
    for node in sp.preorder_traversal(sympy_expr):
        if not isinstance(node, (sp.Pow, _RealPow)):
            continue
        if len(node.args) != 2:
            continue
        base, _ = node.args
        if not ((base.free_symbols & param_syms) and
                (base.free_symbols & feat_syms_set)):
            continue
        if isinstance(base, sp.Add):
            offsets = _find_feat_offset_in_add(
                base, param_syms, param_to_idx,
                feat_syms_set, feature_names)
            for p_idx, fi in offsets.items():
                _set_feat_range_bounds(p_idx, fi)

    return tight


def detect_gaussian_offsets(
    sympy_expr: sp.Expr,
    param_names: List[str],
    feature_names: List[str],
    X_train: np.ndarray,
    n_grid: int = 100,
) -> Dict[int, np.ndarray]:
    """Detect (feat - c) offset patterns, return {param_idx: dense scan grid}.

    For offset parameters like exp(-(T - c0)**n) and (T - c1)**c2,
    generate a dense grid from the corresponding feature's data range
    for OLS scanning within the Pow Grid.
    """
    param_syms = {sp.Symbol(p) for p in param_names}
    param_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
    feat_syms_set = {sp.Symbol(f) for f in feature_names}

    all_offsets: Dict[int, int] = {}

    # Pattern A: Gaussian offset inside exp
    for node in sp.preorder_traversal(sympy_expr):
        if not isinstance(node, sp.exp):
            continue
        arg = node.args[0]
        expanded_arg = sp.expand(arg)
        for p in arg.free_symbols & param_syms:
            if expanded_arg.coeff(p) != 0:
                continue
            for sub in sp.preorder_traversal(arg):
                if not isinstance(sub, sp.Add):
                    continue
                offsets = _find_feat_offset_in_add(
                    sub, param_syms, param_to_idx,
                    feat_syms_set, feature_names)
                all_offsets.update(offsets)

    # Pattern B: bare Pow/RealPow offset (feat - c)**n
    for node in sp.preorder_traversal(sympy_expr):
        if not isinstance(node, (sp.Pow, _RealPow)):
            continue
        if len(node.args) != 2:
            continue
        base, _ = node.args
        if not ((base.free_symbols & param_syms) and
                (base.free_symbols & feat_syms_set)):
            continue
        if isinstance(base, sp.Add):
            offsets = _find_feat_offset_in_add(
                base, param_syms, param_to_idx,
                feat_syms_set, feature_names)
            all_offsets.update(offsets)

    result: Dict[int, np.ndarray] = {}
    for p_idx, fi in all_offsets.items():
        col = X_train[:, fi]
        lo, hi = float(np.min(col)), float(np.max(col))
        result[p_idx] = np.linspace(lo, hi, n_grid)

    return result


def smart_x0(n: int, positive_idx: set = frozenset(),
             small_idx: set = frozenset(),
             rational_idx: set = frozenset(),
             exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
             gaussian_hints: Optional[Dict[int, np.ndarray]] = None,
             ) -> np.ndarray:
    _eb = exp_bounds or {}
    _gh = gaussian_hints or {}
    x0 = np.random.uniform(-5.0, 5.0, size=n)
    for i in range(n):
        if i in rational_idx:
            x0[i] = float(np.random.choice(_RATIONAL_INIT_POOL))
        elif i in _gh:
            x0[i] = float(np.random.choice(_gh[i]))
        elif i in _eb:
            lo, hi = _eb[i]
            x0[i] = np.random.uniform(lo, hi)
        elif i in positive_idx and i in small_idx:
            x0[i] = np.random.uniform(1e-6, 2.0)
        elif i in positive_idx:
            x0[i] = np.random.uniform(1e-6, 5.0)
        elif i in small_idx:
            x0[i] = np.random.uniform(-2.0, 2.0)
    return x0


def make_bounds(n_params: int, bound: float,
                pos_idx: set = frozenset(),
                small_idx: set = frozenset(),
                rational_idx: set = frozenset(),
                exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
                ) -> List[Tuple[float, float]]:
    _eb = exp_bounds or {}
    bounds = []
    for i in range(n_params):
        if i in rational_idx:
            bounds.append((-10.0, 10.0))
        elif i in _eb:
            bounds.append(_eb[i])
        elif i in pos_idx and i in small_idx:
            bounds.append((1e-6, 5.0))
        elif i in pos_idx:
            bounds.append((1e-6, bound))
        elif i in small_idx:
            bounds.append((-5.0, 5.0))
        else:
            bounds.append((-bound, bound))
    return bounds


def wrap_rational(mse_fn: Callable, rational_idx: set) -> Callable:
    """If rational_idx is non-empty, wrap mse_fn to snap parameters at specified positions."""
    if not rational_idx:
        return mse_fn

    def wrapped(p):
        p2 = np.array(p, dtype=float)
        for i in rational_idx:
            p2[i] = _snap_rational(p2[i])
        return mse_fn(p2)
    return wrapped


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseOptimizer(ABC):
    """Constant optimizer base class.

    All subclasses implement the ``optimize`` method with the same parameter signature.
    """

    def __init__(self, *,
                 timeout: float = 60.0,
                 max_iter: int = 1000,
                 n_restarts: int = 1,
                 bound: float = 100.0,
                 penalty: float = 1e10):
        _configure_numeric_threads()
        self.timeout = timeout
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.bound = bound
        self.penalty = penalty

    @abstractmethod
    def optimize(
        self,
        skeleton: Union[str, sp.Expr],
        param_names: List[str],
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        parent_params: Optional[List[float]] = None,
    ) -> OptResult:
        ...
