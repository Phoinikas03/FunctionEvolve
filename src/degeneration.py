"""Post-fit degeneration analysis.

Degeneration is distinct from mutation: it asks whether a fitted expression is
effectively a simpler expression under the training-domain scale.  The engine
produces simplified children, and search decides whether those children dominate
the raw node after they are evaluated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from .normalization import ExpressionNormalizer, NormalizationOptions


def _to_real_or_none(raw, *, tol=1e-9):
    """Cast an eval result to a real float array, or return None if genuinely complex.

    Avoids the implicit complex->float cast (which emits ``ComplexWarning`` and
    silently drops the imaginary part): a result whose imaginary part is not
    negligible relative to its real magnitude means the expression is not
    real-valued over the domain and should be rejected rather than truncated.
    """
    arr = np.asarray(raw)
    if np.iscomplexobj(arr):
        imag = np.abs(arr.imag)
        real = np.abs(arr.real)
        finite = np.isfinite(imag) & np.isfinite(real)
        if finite.any():
            scale = 1.0 + float(real[finite].max())
            if float(imag[finite].max()) > tol * scale:
                return None
        arr = arr.real
    return np.asarray(arr, dtype=float)


@dataclass(frozen=True)
class DegenerationConfig:
    base_tol: float = 1e-3
    dominance_ratio: float = 2.0
    max_rounds: int = 10
    large_coeff: float = 1e3
    overfit_min_depth: int = 10
    snap_abs_tol: float = 1e-2
    merge_tol: float = 1e-3


@dataclass
class DegeneratedCandidate:
    expression: str
    params: List[str]
    reason: str
    kind: str = "degeneration"


@dataclass
class DegenerationReport:
    status: str = "ok"  # ok | simplified | overfit
    canonical_expression: Optional[str] = None
    canonical_key: Optional[str] = None
    canonical_sp_expr: Optional[sp.Expr] = None
    children: List[DegeneratedCandidate] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def first_child(self) -> Optional[DegeneratedCandidate]:
        return self.children[0] if self.children else None


class DegenerationEngine:
    """Data-scale-aware post-fit simplifier."""

    _RATIONAL_CANDIDATES: List[Tuple[float, sp.Rational]] = []
    _seen_rats = set()
    for _q in range(-5, 6):
        _r = sp.Rational(_q)
        if _r not in _seen_rats:
            _seen_rats.add(_r)
            _RATIONAL_CANDIDATES.append((float(_r), _r))
    for _q in range(-5, 6):
        if _q == 0:
            continue
        _r = sp.Rational(1, _q)
        if _r not in _seen_rats:
            _seen_rats.add(_r)
            _RATIONAL_CANDIDATES.append((float(_r), _r))
    del _seen_rats, _q, _r

    def __init__(
        self,
        feature_names: List[str],
        X_train: Optional[np.ndarray] = None,
        config: Optional[DegenerationConfig] = None,
        normalizer: Optional[ExpressionNormalizer] = None,
    ):
        self.feature_names = list(feature_names)
        self.X_train = X_train
        self.config = config or DegenerationConfig()
        self.normalizer = normalizer or ExpressionNormalizer(feature_names)

    def analyze(
        self,
        expr_str: str,
        param_names: List[str],
        fitted_params: List[float],
    ) -> DegenerationReport:
        if not param_names or not fitted_params:
            return DegenerationReport()

        param_vals = dict(zip(param_names, fitted_params))
        try:
            expr = sp.sympify(expr_str, locals=self.normalizer.make_locals(expr_str))
        except Exception:
            return DegenerationReport()

        if self._sympy_ast_depth(expr) >= self.config.overfit_min_depth:
            for pname, pval in param_vals.items():
                if abs(pval) > self.config.large_coeff:
                    return DegenerationReport(
                        status="overfit",
                        reasons=[f"{pname}={pval:.3e}->overfitting(|c|>{self.config.large_coeff})"],
                    )

        original_str = str(expr)
        reasons: List[str] = []
        try:
            simplified = self._iterative_simplify(expr, param_vals, reasons)
            simplified = self._merge_like_terms(simplified, param_vals, reasons)
            simplified = sp.simplify(simplified)
        except (TypeError, ValueError, ZeroDivisionError):
            return DegenerationReport()

        if sp.zoo in simplified.atoms() or sp.oo in simplified.atoms():
            return DegenerationReport(
                status="overfit",
                reasons=reasons + ["infinity (zoo/oo) after simplification->degenerated"],
            )

        if str(simplified) == original_str:
            return DegenerationReport(
                canonical_expression=original_str,
                canonical_key=self.normalizer.structural_key(original_str),
                canonical_sp_expr=expr,
            )

        feature_syms = {sp.Symbol(v) for v in self.feature_names}
        if not (simplified.free_symbols & feature_syms):
            return DegenerationReport(
                status="overfit",
                reasons=reasons + ["no feature variables after simplification->overfitting"],
            )

        norm_expr = self.normalizer.normalize_parameterization(
            simplified,
            NormalizationOptions(
                expand_products=False,
                ensure_elementary_params=False,
            ),
        )
        norm_expression = str(norm_expr)
        norm_key = self.normalizer.structural_key(norm_expression)
        norm_sp_expr = self.normalizer.parse(norm_expression)
        child = DegeneratedCandidate(
            expression=norm_expression,
            params=self.normalizer.collect_params(norm_expr),
            reason="; ".join(reasons),
        )
        return DegenerationReport(
            status="simplified",
            canonical_expression=norm_expression,
            canonical_key=norm_key,
            canonical_sp_expr=norm_sp_expr,
            children=[child],
            reasons=reasons,
        )

    def check_degeneracy(
        self,
        expr_str: str,
        param_names: List[str],
        fitted_params: List[float],
    ) -> Tuple[str, Optional[str], Optional[List[str]], List[str]]:
        report = self.analyze(expr_str, param_names, fitted_params)
        if report.status == "simplified" and report.first_child is not None:
            child = report.first_child
            return report.status, child.expression, child.params, report.reasons
        return report.status, None, None, report.reasons

    # ------------------------------------------------------------------ #
    # Cascading simplification
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sympy_ast_depth(expr: sp.Expr) -> int:
        if not expr.args:
            return 1
        return 1 + max(DegenerationEngine._sympy_ast_depth(a) for a in expr.args)

    def _iterative_simplify(
        self,
        expr: sp.Expr,
        param_vals: Dict[str, float],
        reasons: List[str],
    ) -> sp.Expr:
        for _ in range(self.config.max_rounds):
            new_expr = self._apply_degen_rules(expr, param_vals, reasons)
            try:
                new_expr = sp.simplify(new_expr)
            except (TypeError, ValueError):
                break
            if sp.zoo in new_expr.atoms() or sp.oo in new_expr.atoms():
                break
            try:
                if new_expr.equals(expr):
                    break
            except (TypeError, ValueError):
                break
            expr = new_expr
        return expr

    def _apply_degen_rules(
        self,
        expr: sp.Expr,
        param_vals: Dict[str, float],
        reasons: List[str],
    ) -> sp.Expr:
        feature_syms = {sp.Symbol(v) for v in self.feature_names}

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _param_val(s):
            return param_vals.get(str(s))

        def _has_feature(e):
            return bool(e.free_symbols & feature_syms)

        def _direct_coeff_value(args) -> Optional[float]:
            coeff = 1.0
            saw_coeff = False
            for a in args:
                if a.is_Number:
                    coeff *= float(a)
                    saw_coeff = True
                elif _is_param(a):
                    v = _param_val(a)
                    if v is None:
                        return None
                    coeff *= float(v)
                    saw_coeff = True
            return coeff if saw_coeff else 1.0

        def _direct_coeff_expr(args):
            coeff_args = [a for a in args if a.is_Number or _is_param(a)]
            return sp.Mul(*coeff_args) if coeff_args else sp.Integer(1)

        def _snap_unit_value(value: float) -> Optional[sp.Integer]:
            if abs(value - 1.0) < self.config.snap_abs_tol:
                return sp.Integer(1)
            if abs(value + 1.0) < self.config.snap_abs_tol:
                return sp.Integer(-1)
            return None

        def _eval_max_abs(expr_to_eval) -> Optional[float]:
            expr_sub = expr_to_eval
            try:
                for name, value in param_vals.items():
                    expr_sub = expr_sub.subs(sp.Symbol(name), sp.Float(value))
                if not (expr_sub.free_symbols & feature_syms):
                    val = complex(expr_sub.evalf())
                    if abs(val.imag) > 1e-9:
                        return None
                    return abs(float(val.real))
                if self.X_train is None or getattr(self.X_train, "size", 0) == 0:
                    return None
                func = sp.lambdify(
                    [sp.Symbol(n) for n in self.feature_names],
                    expr_sub,
                    modules={"sin": np.sin, "cos": np.cos, "tan": np.tan,
                             "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                             "Abs": np.abs, "numpy": np},
                )
                cols = [self.X_train[:, i] for i in range(len(self.feature_names))]
                with np.errstate(all="ignore"):
                    vals = _to_real_or_none(func(*cols))
                if vals is None:
                    return None
                if vals.shape == ():
                    vals = np.full(self.X_train.shape[0], float(vals))
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return None
                return float(np.max(np.abs(vals)))
            except Exception:
                return None

        def _is_scaled_small(e) -> bool:
            max_abs = _eval_max_abs(e)
            return max_abs is not None and max_abs < self.config.base_tol

        def _split_const_plus_small_feature(node):
            terms = list(node.args) if node.func == sp.Add else [node]
            const_terms = []
            scaled_terms = []
            for term in terms:
                if _has_feature(term):
                    if _is_scaled_small(term):
                        scaled_terms.append(term)
                    else:
                        return None
                else:
                    const_terms.append(term)
            if not scaled_terms:
                return None
            const = sp.Add(*const_terms) if const_terms else sp.Integer(0)
            scaled = sp.Add(*scaled_terms)
            return const, scaled

        def _linearize_unary(inner, func_builder, label: str):
            split = _split_const_plus_small_feature(inner)
            if split is None:
                return None
            const, scaled_expr = split
            z = sp.Symbol("__taylor_z__")
            try:
                func_z = func_builder(z)
                value = func_z.subs(z, const)
                deriv = sp.diff(func_z, z).subs(z, const)
            except Exception:
                return None
            reasons.append(f"Taylor absorbed ({label}), max|delta|<={self.config.base_tol:g}")
            return value + deriv * scaled_expr

        def _linearize_pow(node):
            if node.func != sp.Pow or len(node.args) != 2:
                return None
            base, exponent = node.args
            split = _split_const_plus_small_feature(exponent)
            if split is None:
                return None
            const, scaled_expr = split
            if _eval_max_abs(sp.log(base) * scaled_expr) is None:
                # Unsafe bases (negative/zero) often fail here; skip rather than
                # introducing invalid log(base) terms.
                return None
            reasons.append(f"Taylor absorbed (pow exponent), max|delta|<={self.config.base_tol:g}")
            return base ** const + base ** const * sp.log(base) * scaled_expr

        def _absorb_first_order_in_mul(node):
            if node.func != sp.Mul:
                return None
            args = list(node.args)
            for i, arg in enumerate(args):
                replacement = None
                if arg.func in (sp.log, sp.exp, sp.sin, sp.cos, sp.tan):
                    replacement = _linearize_unary(
                        arg.args[0],
                        arg.func,
                        f"{arg.func.__name__}(const+delta)",
                    )
                elif arg.func == sp.Pow:
                    replacement = _linearize_pow(arg)
                if replacement is None:
                    continue
                new_node = sp.Mul(*(args[:i] + [replacement] + args[i + 1:]))
                try:
                    # Keep original parameter products visible for later rounds so
                    # small-term pruning can use the fitted values of each factor.
                    return sp.expand_mul(new_node)
                except Exception:
                    return new_node
            return None

        def _walk(node, parent_func=None):
            if node.is_Atom:
                if _is_param(node) and parent_func in (sp.Add, None):
                    v = _param_val(node)
                    snapped = _snap_unit_value(v) if v is not None else None
                    if snapped is not None:
                        reasons.append(f"{node}={v:.4f}->snapped to {snapped}")
                        return snapped
                return node

            if node.func == sp.Mul:
                absorbed = _absorb_first_order_in_mul(node)
                if absorbed is not None:
                    return absorbed

            new_args = [_walk(a, node.func) for a in node.args]
            node = node.func(*new_args)

            if node.func in (sp.log, sp.exp, sp.sin, sp.cos, sp.tan):
                replacement = _linearize_unary(
                    node.args[0],
                    node.func,
                    f"{node.func.__name__}(const+delta)",
                )
                if replacement is not None:
                    return replacement

            if node.func == sp.Pow:
                replacement = _linearize_pow(node)
                if replacement is not None:
                    return replacement
                base, exp_node = node.args
                if _is_param(exp_node):
                    v = _param_val(exp_node)
                    if v is not None:
                        best_rat = self._snap_to_rational(v)
                        if best_rat is not None:
                            reasons.append(f"{exp_node}={v:.4f}->snapped to {best_rat}")
                            return sp.Pow(base, best_rat)

            if node.func == sp.Mul:
                coeff_value = _direct_coeff_value(node.args)
                has_param_coeff = any(_is_param(a) for a in node.args)
                if coeff_value is not None and has_param_coeff:
                    if abs(coeff_value) < self.config.base_tol or _is_scaled_small(node):
                        coeff_expr = _direct_coeff_expr(node.args)
                        reasons.append(f"{coeff_expr}~{coeff_value:.3e}->small term,{node}->0")
                        return sp.Integer(0)
                    snapped = _snap_unit_value(coeff_value)
                    if snapped is not None:
                        coeff_expr = _direct_coeff_expr(node.args)
                        core_args = [
                            a for a in node.args
                            if not (a.is_Number or _is_param(a))
                        ]
                        if core_args:
                            core = sp.Mul(*core_args)
                            reasons.append(
                                f"{coeff_expr}={coeff_value:.4f}->snapped to {snapped}"
                            )
                            return snapped * core

            if node.func == sp.Add:
                kept = []
                for term in node.args:
                    if _is_scaled_small(term):
                        reasons.append(f"max|{term}|<={self.config.base_tol:g}->term->0")
                        continue
                    kept.append(term)
                if len(kept) != len(node.args):
                    return sp.Add(*kept) if kept else sp.Integer(0)

            if node.func == sp.exp and node.args[0] == 0:
                reasons.append("exp(0)->1")
                return sp.Integer(1)
            if node.func == sp.log and node.args[0] == 1:
                reasons.append("log(1)->0")
                return sp.Integer(0)

            if node.func == sp.sin:
                node = self._simplify_sin(node.args[0], _is_param, _param_val, reasons)

            return node

        return _walk(expr)

    def _simplify_sin(self, inner, _is_param, _param_val, reasons: List[str]) -> sp.Expr:
        if inner.func == sp.Add:
            phase_param = None
            phase_val = None
            rest_terms = []
            for term in inner.args:
                if _is_param(term):
                    v = _param_val(term)
                    if v is not None:
                        k = round(v / math.pi)
                        if k != 0 and abs(v - k * math.pi) < 0.05:
                            phase_param = term
                            phase_val = k
                            continue
                rest_terms.append(term)

            if phase_param is not None:
                new_inner = sp.Add(*rest_terms) if rest_terms else sp.Integer(0)
                sign = sp.Integer(1) if phase_val % 2 == 0 else sp.Integer(-1)
                reasons.append(f"{phase_param}={_param_val(phase_param):.4f}~{phase_val}*pi->phase absorbed")
                return sign * sp.sin(new_inner)

        if inner.func == sp.Mul:
            for a in inner.args:
                if _is_param(a):
                    v = _param_val(a)
                    if v is not None and abs(v) < self.config.base_tol:
                        reasons.append(f"{a}={v:.3e}->small-angle approx,sin->{inner}")
                        return inner
        elif _is_param(inner):
            v = _param_val(inner)
            if v is not None and abs(v) < self.config.base_tol:
                reasons.append(f"{inner}={v:.3e}->small-angle approx,sin->{inner}")
                return inner
        return sp.sin(inner)

    # ------------------------------------------------------------------ #
    # Like-term merging and helpers
    # ------------------------------------------------------------------ #

    def _merge_like_terms(
        self,
        expr: sp.Expr,
        param_vals: Dict[str, float],
        reasons: List[str],
    ) -> sp.Expr:
        if expr.func != sp.Add:
            new_args = [self._merge_like_terms(a, param_vals, reasons) for a in expr.args] if not expr.is_Atom else []
            return expr.func(*new_args) if new_args else expr

        feature_syms = {sp.Symbol(v) for v in self.feature_names}

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _pval(s):
            return param_vals.get(str(s))

        def _extract_coeff_and_core(term):
            coeff = sp.Integer(1)
            core = term
            if term.func == sp.Mul:
                coeff_factors = []
                core_factors = []
                for a in term.args:
                    if not (a.free_symbols & feature_syms):
                        coeff_factors.append(a)
                    else:
                        core_factors.append(a)
                if core_factors:
                    coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
                    core = sp.Mul(*core_factors) if len(core_factors) > 1 else core_factors[0]
                else:
                    return None
            if not (core.free_symbols & feature_syms):
                return None
            nl_vals = self._get_nonlinear_param_vals(core, _is_param, _pval)
            if nl_vals is None:
                return None
            template = self._get_core_template(core, _is_param)
            return coeff, core, template, nl_vals

        processed_args = [self._merge_like_terms(a, param_vals, reasons) for a in expr.args]
        expr = sp.Add(*processed_args)
        if expr.func != sp.Add:
            return expr

        groups: List[Tuple[str, tuple, list]] = []
        ungrouped = []
        for term in expr.args:
            info = _extract_coeff_and_core(term)
            if info is None:
                ungrouped.append(term)
                continue
            coeff, core, template, nl_vals = info
            merged = False
            for g_template, g_vals, g_items in groups:
                if (
                    g_template == template
                    and len(nl_vals) == len(g_vals)
                    and all(abs(a - b) < self.config.merge_tol for a, b in zip(nl_vals, g_vals))
                ):
                    g_items.append(info)
                    merged = True
                    break
            if not merged:
                groups.append((template, nl_vals, [info]))

        result_terms = list(ungrouped)
        for _, _, items in groups:
            if len(items) == 1:
                coeff, core, _, _ = items[0]
                result_terms.append(coeff * core)
            else:
                ref_core = items[0][1]
                total_coeff = sp.Add(*[it[0] for it in items])
                merged_term = total_coeff * ref_core
                terms_str = " + ".join(str(it[0] * it[1]) for it in items)
                reasons.append(f"Like-term merge: {terms_str} -> {merged_term}")
                result_terms.append(merged_term)

        return sp.Add(*result_terms) if result_terms else sp.Integer(0)

    @staticmethod
    def _get_core_template(core, _is_param) -> str:
        def _templ(node):
            if node.is_Atom:
                if _is_param(node):
                    return "C"
                return str(node)
            return f"{node.func.__name__}({','.join(_templ(a) for a in node.args)})"
        return _templ(core)

    @staticmethod
    def _get_nonlinear_param_vals(core, _is_param, _pval):
        vals = []

        def _extract(node):
            if node.is_Atom:
                return
            if node.func == sp.Pow:
                exp_node = node.args[1]
                if _is_param(exp_node):
                    v = _pval(exp_node)
                    if v is not None:
                        vals.append(v)
                _extract(node.args[0])
            elif node.func in (sp.sin, sp.exp, sp.log):
                inner = node.args[0]
                is_sin = node.func == sp.sin
                terms = inner.args if inner.func in (sp.Mul, sp.Add) else [inner]
                for term in terms:
                    factors = term.args if getattr(term, "func", None) == sp.Mul else [term]
                    for a in factors:
                        if _is_param(a):
                            if is_sin and term == a:
                                continue
                            v = _pval(a)
                            if v is not None:
                                vals.append(abs(v) if is_sin else v)
            else:
                for a in node.args:
                    _extract(a)

        _extract(core)
        return tuple(vals) if vals else None

    def _snap_to_rational(self, value: float) -> Optional[sp.Rational]:
        best_dist = float("inf")
        best_rat = None
        for fval, rat in self._RATIONAL_CANDIDATES:
            dist = abs(value - fval)
            if dist < self.config.snap_abs_tol and dist < best_dist:
                best_dist = dist
                best_rat = rat
        return best_rat
