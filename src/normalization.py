"""Expression normalization utilities.

This module is the single entry point for normalizing symbolic formulas before
they enter the evolution tree.  It intentionally preserves the behavior that
used to live in ``ASTMutator`` so the refactor can be staged safely.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import sympy as sp


@dataclass(frozen=True)
class NormalizedExpression:
    expression: str
    param_names: List[str]
    structural_key: str


@dataclass(frozen=True)
class NormalizationOptions:
    """Controls which normalization stages are allowed to reshape the formula."""

    expand_products: bool = True
    ensure_elementary_params: bool = True
    collapse_parameters: bool = True
    renumber_parameters: bool = True


class ExpressionNormalizer:
    """Canonical expression and fitted-parameter normalization."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)

    def normalize(self, expr_str: str, *, compact: bool = False) -> NormalizedExpression:
        expr, params = self.normalize_expression(expr_str, compact=compact)
        return NormalizedExpression(
            expression=expr,
            param_names=params,
            structural_key=self.structural_key(expr),
        )

    def normalize_expression(
        self,
        expr_str: str,
        *,
        compact: bool = False,
        deadline: Optional[float] = None,
    ) -> Tuple[str, List[str]]:
        expr = self.parse(expr_str)
        if expr is None:
            return expr_str, self.collect_params_from_str(expr_str)

        options = self.options_for(compact=compact)
        input_param_count = len(self.collect_params(expr))
        expr = self.canonicalize_identities(expr)
        equivalent_options = NormalizationOptions(
            expand_products=options.expand_products,
            ensure_elementary_params=False,
            collapse_parameters=options.collapse_parameters,
            renumber_parameters=options.renumber_parameters,
        )
        expr = self.select_normal_form(
            expr,
            equivalent_options,
            max_params=input_param_count,
            deadline=deadline,
        )
        if options.ensure_elementary_params:
            expr = self.complete_elementary_parameterization(expr, options)
        return str(expr), self.collect_params(expr)

    def structural_key(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        if expr is None:
            return expr_str
        expr = self.reparameterize_local_constants(expr)
        return self.struct_hash(expr)

    def parse(self, expr_str: str) -> Optional[sp.Expr]:
        try:
            expr = sp.sympify(expr_str, locals=self.make_locals(expr_str))
        except Exception:
            return None
        if not isinstance(expr, sp.Expr):
            return None
        return expr

    @staticmethod
    def _is_param_symbol(sym) -> bool:
        return isinstance(sym, sp.Symbol) and re.match(r"^c\d+$", str(sym))

    def _has_feature(self, expr: sp.Expr) -> bool:
        feature_syms = {sp.Symbol(v) for v in self.feature_names}
        return bool(expr.free_symbols & feature_syms)

    @staticmethod
    def options_for(*, compact: bool = False) -> NormalizationOptions:
        return NormalizationOptions(expand_products=not compact)

    def normalize_parameterization(
        self,
        expr: sp.Expr,
        options: NormalizationOptions,
    ) -> sp.Expr:
        if options.expand_products:
            expr = self.expand_products(expr)
        if options.collapse_parameters:
            expr = self.reparameterize_local_constants(expr)
        if options.ensure_elementary_params:
            expr = self.ensure_elementary_params(expr)
        if options.expand_products:
            expr = self.expand_products(expr)
        if options.collapse_parameters:
            expr = self.reparameterize_local_constants(expr)
        if options.renumber_parameters:
            expr = self.renumber_params(expr)
            expr = self.renumber_params(expr)
        return expr

    def complete_elementary_parameterization(
        self,
        expr: sp.Expr,
        options: NormalizationOptions,
    ) -> sp.Expr:
        if options.ensure_elementary_params:
            expr = self.ensure_elementary_params(expr)
        if options.collapse_parameters:
            expr = self.reparameterize_local_constants(expr)
        if options.renumber_parameters:
            expr = self.renumber_params(expr)
            expr = self.renumber_params(expr)
        return expr

    def select_normal_form(
        self,
        expr: sp.Expr,
        options: NormalizationOptions,
        max_params: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> sp.Expr:
        best: Optional[sp.Expr] = None
        best_score: Optional[Tuple[float, ...]] = None
        for candidate in self.generate_equivalent_candidates(expr, deadline=deadline):
            # C (事中兜底): once a usable normal form exists, stop exploring more
            # equivalents when the time budget is spent and return the best found
            # so far, instead of letting the whole call run past its deadline and
            # get discarded.  Candidates are ordered cheap-first (the raw input
            # comes before cancel/factor), so the first iteration already yields a
            # valid form.
            if (
                deadline is not None
                and best is not None
                and time.monotonic() > deadline
            ):
                break
            for candidate_param in self.parameterization_candidates(candidate, options):
                if (
                    max_params is not None
                    and len(self.collect_params(candidate_param)) > max_params
                ):
                    continue
                score = self.complexity_score(candidate_param)
                if best is None or best_score is None or score < best_score:
                    best = candidate_param
                    best_score = score
        return best if best is not None else self.normalize_parameterization(expr, options)

    def _rewrite_complexity(self, expr: sp.Expr) -> Tuple[int, int, int]:
        """Cheap structural predictors of how expensive the heavy rewrites are.

        ``cancel``/``factor``/``together`` on a sum of rationals must build a
        common denominator and run multivariate polynomial GCD/factorization;
        the cost is driven by how many *distinct denominators* are combined, the
        highest feature *degree*, and how many symbolic *parameters* become
        polynomial coefficients -- none of which the raw op-count guard sees (a
        17-op input can explode to 255 ops after a single ``cancel``).
        """
        denoms: Set[str] = set()
        for term in sp.Add.make_args(expr):
            try:
                _, den = sp.fraction(term)
            except Exception:
                den = sp.Integer(1)
            if den != 1:
                denoms.add(str(den))
        max_pow = 1
        for node in sp.preorder_traversal(expr):
            if node.func == sp.Pow:
                base, exp = node.args
                if getattr(exp, "is_Integer", False) and self._has_feature(base):
                    max_pow = max(max_pow, abs(int(exp)))
        n_params = sum(1 for s in expr.free_symbols if self._is_param_symbol(s))
        return len(denoms), max_pow, n_params

    def generate_equivalent_candidates(
        self, expr: sp.Expr, deadline: Optional[float] = None
    ) -> List[sp.Expr]:
        candidates = [expr]

        # B (事前剪枝): only attempt the expensive rewrites when the form is
        # genuinely simple.  Multivariate / high-degree / multi-denominator sums
        # make cancel+factor+compress blow up the intermediate expression, which
        # is what drives the normalization timeouts, so fall back to cancel-only
        # (GCD-based, cheap) in that regime.
        n_denoms, max_pow, n_params = self._rewrite_complexity(expr)
        # ``cancel`` is cheap to parameterize only when the result stays small:
        # that needs a single, low-degree denominator.  Multiple denominators get
        # multiplied into a high-degree common denominator, and high feature
        # powers (e.g. (c1+t)**5) expand -- either way, parameterizing the
        # cancelled form costs tens of seconds (measured 14-72s on real
        # candidates), which is what drives the normalization timeouts.
        cancel_safe = self._rewrite_budget_allows(expr) and n_denoms <= 1 and max_pow <= 2
        if cancel_safe and n_params <= 15:
            rewrite_fns = [sp.cancel, sp.factor]
            if not self._has_elementary(expr):
                rewrite_fns.append(sp.together)
        elif cancel_safe:
            rewrite_fns = [sp.cancel]
        else:
            # Multivariate / high-degree / multi-denominator: use the raw input
            # only -- its compact parameterization is sub-0.1s.
            rewrite_fns = []

        for fn in rewrite_fns:
            if deadline is not None and time.monotonic() > deadline:
                break
            try:
                candidates.append(fn(expr))
            except Exception:
                pass

        if cancel_safe:
            expanded = []
            for candidate in candidates:
                if deadline is not None and time.monotonic() > deadline:
                    break
                compressed = self.compress_rational_coefficients(candidate)
                if compressed is not None:
                    expanded.append(compressed)
                    try:
                        expanded.append(sp.factor(compressed))
                    except Exception:
                        pass
            candidates.extend(expanded)
        return self._dedup_exprs(candidates)

    def _rewrite_budget_allows(self, expr: sp.Expr) -> bool:
        return self._op_count(expr) <= 120 and len(str(expr)) <= 2500

    def parameterization_candidates(
        self,
        expr: sp.Expr,
        options: NormalizationOptions,
    ) -> List[sp.Expr]:
        variants = [options]
        alternate = NormalizationOptions(
            expand_products=not options.expand_products,
            ensure_elementary_params=options.ensure_elementary_params,
            collapse_parameters=options.collapse_parameters,
            renumber_parameters=options.renumber_parameters,
        )
        if alternate != options:
            variants.append(alternate)

        candidates = []
        for variant in variants:
            try:
                candidates.append(self.normalize_parameterization(expr, variant))
            except Exception:
                continue
        return self._dedup_exprs(candidates)

    def compress_rational_coefficients(self, expr: sp.Expr) -> Optional[sp.Expr]:
        """Compress coefficients of rational polynomials in feature variables.

        This keeps transformations strictly equivalent up to free
        reparameterization.  It prevents forms like
        P*c0*(c1 + c2*(P + c3))/(P + c3) from carrying redundant nested
        parameter products; the result is represented as
        P*(P*c_new + c_new)/(P + c_new).
        """
        if self._op_count(expr) > 160 or len(str(expr)) > 3000:
            return None

        feature_syms = [sp.Symbol(v) for v in self.feature_names]
        if not feature_syms:
            return None

        try:
            num, den = sp.fraction(sp.cancel(expr))
            num_poly = sp.Poly(num, *feature_syms)
            den_poly = sp.Poly(den, *feature_syms)
        except Exception:
            return None

        if not num_poly.terms() or not den_poly.terms():
            return None

        scale = self._choose_denominator_scale(den_poly)
        if scale is None or scale == 0:
            return None

        start = max(
            (int(str(s)[1:]) for s in expr.free_symbols if self._is_param_symbol(s)),
            default=-1,
        ) + 1000
        counter = [start]

        def _new_param() -> sp.Symbol:
            sym = sp.Symbol(f"c{counter[0]}")
            counter[0] += 1
            return sym

        def _compress_coeff(coeff: sp.Expr) -> sp.Expr:
            coeff = sp.cancel(coeff / scale)
            if coeff == 0:
                return sp.Integer(0)
            if coeff == 1:
                return sp.Integer(1)
            if coeff == -1:
                return sp.Integer(-1)
            if coeff.free_symbols and not self._has_feature(coeff):
                return _new_param()
            return coeff

        new_num = self._poly_from_compressed_terms(num_poly, _compress_coeff)
        new_den = self._poly_from_compressed_terms(den_poly, _compress_coeff)
        if new_den == 0:
            return None
        compressed = sp.cancel(new_num / new_den)
        if compressed == expr:
            return None
        return compressed

    def _choose_denominator_scale(self, den_poly: sp.Poly) -> Optional[sp.Expr]:
        terms = den_poly.terms()
        if not terms:
            return None
        for _, coeff in terms:
            if coeff == 1 or coeff == -1:
                return coeff
        for _, coeff in terms:
            if coeff.free_symbols and not self._has_feature(coeff):
                return coeff
        return terms[0][1]

    def _poly_from_compressed_terms(
        self,
        poly: sp.Poly,
        compress_coeff,
    ) -> sp.Expr:
        gens = poly.gens
        terms = []
        for powers, coeff in poly.terms():
            new_coeff = compress_coeff(coeff)
            if new_coeff == 0:
                continue
            monomial = sp.Integer(1)
            for gen, power in zip(gens, powers):
                if power:
                    monomial *= gen ** power
            terms.append(new_coeff * monomial)
        return sp.Add(*terms) if terms else sp.Integer(0)

    def complexity_score(self, expr: sp.Expr) -> Tuple[float, ...]:
        params = self.collect_params(expr)
        return (
            float(len(params)),
            float(self._elementary_rational_penalty(expr)),
            float(self._repeated_denominator_penalty(expr)),
            float(self._add_term_count(expr)),
            float(self._op_count(expr)),
            float(self._ast_depth(expr)),
            float(len(str(expr))) / 1000.0,
        )

    @staticmethod
    def _has_elementary(expr: sp.Expr) -> bool:
        elem_funcs = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)
        return any(node.func in elem_funcs for node in sp.preorder_traversal(expr))

    def _elementary_rational_penalty(self, expr: sp.Expr) -> int:
        if not self._has_elementary(expr):
            return 0
        try:
            _, den = sp.fraction(expr)
        except Exception:
            return 0
        return 1 if den != 1 and self._has_feature(den) else 0

    def _repeated_denominator_penalty(self, expr: sp.Expr) -> int:
        denoms = []
        for term in sp.Add.make_args(expr):
            try:
                _, den = sp.fraction(term)
            except Exception:
                continue
            if den != 1:
                denoms.append(str(den))
        return max(0, len(denoms) - len(set(denoms)))

    @staticmethod
    def _add_term_count(expr: sp.Expr) -> int:
        if expr.func == sp.Add:
            return len(expr.args)
        return 1

    @staticmethod
    def _op_count(expr: sp.Expr) -> int:
        try:
            return int(sp.count_ops(expr, visual=False))
        except Exception:
            return len(str(expr))

    @staticmethod
    def _ast_depth(expr: sp.Expr) -> int:
        if not expr.args:
            return 1
        return 1 + max(ExpressionNormalizer._ast_depth(arg) for arg in expr.args)

    @staticmethod
    def _dedup_exprs(exprs: Iterable[sp.Expr]) -> List[sp.Expr]:
        seen: Set[str] = set()
        unique = []
        for expr in exprs:
            key = str(expr)
            if key in seen:
                continue
            seen.add(key)
            unique.append(expr)
        return unique

    @staticmethod
    def expand_products(expr: sp.Expr) -> sp.Expr:
        return sp.expand_mul(expr)

    def normalize_fitted_params(
        self,
        expr_str: str,
        param_names: List[str],
        fitted_params: List[float],
    ) -> List[float]:
        """Normalize equivalent fitted parameter values without changing structure."""
        import math

        try:
            expr = sp.sympify(expr_str, locals=self.make_locals(expr_str))
        except Exception:
            return list(fitted_params)

        pvals = dict(zip(param_names, fitted_params))

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _find_sin_params(node, outer_coeff_sym=None):
            results = []
            if node.is_Atom:
                return results

            if node.func == sp.Mul:
                coeff_syms = [a for a in node.args if _is_param(a)]
                non_coeff = [a for a in node.args if not _is_param(a)]
                amp = coeff_syms[0] if coeff_syms else None
                for sub in non_coeff:
                    results.extend(_find_sin_params(sub, amp))
                return results

            if node.func == sp.sin:
                inner = node.args[0]
                freq_sym = None
                phase_sym = None

                if inner.func == sp.Add:
                    for term in inner.args:
                        if _is_param(term) and phase_sym is None:
                            phase_sym = term
                        elif term.func == sp.Mul:
                            for a in term.args:
                                if _is_param(a) and freq_sym is None:
                                    freq_sym = a
                elif inner.func == sp.Mul:
                    for a in inner.args:
                        if _is_param(a) and freq_sym is None:
                            freq_sym = a

                if freq_sym is not None:
                    results.append((outer_coeff_sym, freq_sym, phase_sym))
                return results

            for a in node.args:
                results.extend(_find_sin_params(a))
            return results

        two_pi = 2 * math.pi
        for amp_sym, freq_sym, phase_sym in _find_sin_params(expr):
            freq_name = str(freq_sym)
            freq_val = pvals.get(freq_name)
            if freq_val is None:
                continue

            if freq_val < 0:
                pvals[freq_name] = -freq_val
                if amp_sym is not None and str(amp_sym) in pvals:
                    pvals[str(amp_sym)] = -pvals[str(amp_sym)]
                if phase_sym is not None and str(phase_sym) in pvals:
                    pvals[str(phase_sym)] = -pvals[str(phase_sym)]

            if phase_sym is not None and str(phase_sym) in pvals:
                pvals[str(phase_sym)] = pvals[str(phase_sym)] % two_pi

        return [pvals[n] for n in param_names]

    # ------------------------------------------------------------------ #
    # Core normalization implementation
    # ------------------------------------------------------------------ #

    def algebraic_simplify(self, expr: sp.Expr) -> sp.Expr:
        """Backward-compatible alias for identity-only canonicalization."""
        return self.canonicalize_identities(expr)

    def canonicalize_identities(self, expr: sp.Expr) -> sp.Expr:
        feature_syms = {sp.Symbol(v) for v in self.feature_names}

        def _has_feature(e):
            return bool(e.free_symbols & feature_syms)

        def _walk(node):
            if node.is_Atom:
                return node
            new_args = [_walk(a) for a in node.args]
            node = node.func(*new_args)

            if node.func == sp.exp:
                inner = node.args[0]
                log_part, coeff = self.extract_log_from_mul(inner)
                if log_part is not None:
                    return log_part.args[0] ** coeff

                if inner.func == sp.Add:
                    const_terms = []
                    log_terms = []
                    rest_terms = []
                    for a in inner.args:
                        if not _has_feature(a):
                            const_terms.append(a)
                        else:
                            lp, lc = self.extract_log_from_mul(a)
                            if lp is not None:
                                log_terms.append((lp, lc))
                            else:
                                rest_terms.append(a)

                    if const_terms or log_terms:
                        factors = []
                        if const_terms:
                            c = sp.Add(*const_terms) if len(const_terms) > 1 else const_terms[0]
                            factors.append(sp.exp(c))
                        for lp, lc in log_terms:
                            factors.append(lp.args[0] ** lc)
                        if rest_terms:
                            rest = sp.Add(*rest_terms) if len(rest_terms) > 1 else rest_terms[0]
                            factors.append(sp.exp(rest))
                        if len(factors) > 1 or log_terms:
                            return sp.Mul(*factors)

            if node.func == sp.log:
                inner = node.args[0]
                if hasattr(inner, "func") and inner.func == sp.exp:
                    return inner.args[0]

            return node

        prev = None
        result = expr
        for _ in range(5):
            result = _walk(result)
            if result == prev:
                break
            prev = result
        return result

    @staticmethod
    def extract_log_from_mul(expr) -> Tuple[Optional[sp.Expr], sp.Expr]:
        if expr.func == sp.log:
            return expr, sp.Integer(1)
        if expr.func == sp.Mul:
            log_parts = [a for a in expr.args if hasattr(a, "func") and a.func == sp.log]
            if len(log_parts) == 1:
                rest = [a for a in expr.args if a is not log_parts[0]]
                coeff = sp.Mul(*rest) if rest else sp.Integer(1)
                feature_syms = [
                    v.name for v in log_parts[0].free_symbols
                    if not re.match(r"^c\d+$", str(v))
                ]
                if not any(
                    isinstance(s, sp.Symbol) and s.name in feature_syms
                    for s in coeff.free_symbols
                ):
                    return log_parts[0], coeff
        return None, sp.Integer(1)

    def collapse_params(self, expr: sp.Expr) -> sp.Expr:
        """Backward-compatible alias for local constant reparameterization."""
        return self.reparameterize_local_constants(expr)

    def reparameterize_local_constants(self, expr: sp.Expr) -> sp.Expr:
        feature_syms = {sp.Symbol(v) for v in self.feature_names}
        start = max(
            (int(str(s)[1:]) for s in expr.free_symbols if re.match(r"^c\d+$", str(s))),
            default=-1,
        ) + 100
        counter = [start]

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _has_feature(e):
            return bool(e.free_symbols & feature_syms)

        def _has_param(e):
            return any(_is_param(s) for s in e.free_symbols)

        def _new_param():
            sym = sp.Symbol(f"c{counter[0]}")
            counter[0] += 1
            return sym

        def _walk(node):
            if node.is_Atom:
                return node
            new_args = [_walk(a) for a in node.args]
            node = node.func(*new_args)

            if node.func not in (sp.Mul, sp.Add):
                if not _has_feature(node) and _has_param(node):
                    return _new_param()

            if node.func == sp.Mul:
                params = [a for a in node.args if _is_param(a)]
                nums = [a for a in node.args if a.is_Number]
                others = [a for a in node.args if not _is_param(a) and not a.is_Number]
                const_others = [a for a in others if not _has_feature(a)]
                feat_others = [a for a in others if _has_feature(a)]
                absorbable = list(params) + [n for n in nums if n != 1] + const_others
                if len(params) >= 1 and len(absorbable) >= 2:
                    p = _new_param()
                    return sp.Mul(p, *feat_others) if feat_others else p

            elif node.func == sp.Add:
                const_terms = [a for a in node.args if not _has_feature(a)]
                var_terms = [a for a in node.args if _has_feature(a)]
                has_param = any(_has_param(a) for a in const_terms)
                if has_param and len(const_terms) >= 2:
                    p = _new_param()
                    return sp.Add(p, *var_terms) if var_terms else p

            return node

        return _walk(expr)

    def ensure_elementary_params(self, expr: sp.Expr) -> sp.Expr:
        max_idx = max(
            (int(str(s)[1:]) for s in expr.free_symbols if re.match(r"^c\d+$", str(s))),
            default=-1,
        )
        counter = [max_idx + 1]

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _new_p():
            sym = sp.Symbol(f"c{counter[0]}")
            counter[0] += 1
            return sym

        def _has_inner_scale(inner):
            if _is_param(inner):
                return True
            if inner.func == sp.Mul:
                return any(_is_param(a) for a in inner.args)
            return False

        def _walk(node):
            if node.is_Atom:
                return node
            new_args = [_walk(a) for a in node.args]
            node = node.func(*new_args)

            if node.func == sp.sin:
                inner = node.args[0]
                if inner.func == sp.Add:
                    terms = list(inner.args)
                    phase_terms = [t for t in terms if _is_param(t)]
                    non_phase = [t for t in terms if not _is_param(t)]
                    main = sp.Add(*non_phase) if non_phase else sp.Integer(0)
                else:
                    phase_terms = []
                    main = inner

                if not _has_inner_scale(main):
                    main = _new_p() * main
                phase = sp.Add(*phase_terms) if phase_terms else _new_p()
                node = sp.sin(main + phase)

            elif node.func == sp.exp:
                inner = node.args[0]
                if not _has_inner_scale(inner):
                    inner = _new_p() * inner
                node = sp.exp(inner)

            elif node.func == sp.log:
                inner = node.args[0]
                if inner.func == sp.Add:
                    const_parts = [
                        a for a in inner.args
                        if a.is_Number or _is_param(a)
                    ]
                    has_one = any(a == sp.Integer(1) for a in inner.args)
                    var_parts = [a for a in inner.args if a not in const_parts]
                    if has_one and var_parts:
                        core = sp.Add(*var_parts) if len(var_parts) > 1 else var_parts[0]
                        if not _has_inner_scale(core):
                            core = _new_p() * core
                        node = sp.log(1 + core)
                    else:
                        if not _has_inner_scale(inner):
                            inner = _new_p() * inner
                        node = sp.log(1 + inner)
                else:
                    if not _has_inner_scale(inner):
                        inner = _new_p() * inner
                    node = sp.log(1 + inner)

            return node

        return self.ensure_outer_scale(_walk(expr))

    def ensure_outer_scale(self, expr: sp.Expr) -> sp.Expr:
        max_idx = max(
            (int(str(s)[1:]) for s in expr.free_symbols if re.match(r"^c\d+$", str(s))),
            default=-1,
        )
        counter = [max_idx + 1]

        def _is_param(s):
            return isinstance(s, sp.Symbol) and re.match(r"^c\d+$", str(s))

        def _new_p():
            sym = sp.Symbol(f"c{counter[0]}")
            counter[0] += 1
            return sym

        elem_funcs = (sp.sin, sp.cos, sp.exp, sp.log)

        def _is_elem(node):
            return hasattr(node, "func") and node.func in elem_funcs

        def _walk(node, inside_mul_with_param=False):
            if node.is_Atom:
                return node
            if node.func == sp.Mul:
                has_param = any(_is_param(a) for a in node.args)
                new_args = [_walk(a, inside_mul_with_param=has_param) for a in node.args]
                return node.func(*new_args)

            new_args = [_walk(a, inside_mul_with_param=False) for a in node.args]
            node = node.func(*new_args)
            if _is_elem(node) and not inside_mul_with_param:
                return _new_p() * node
            return node

        return _walk(expr)

    @staticmethod
    def renumber_params(expr: sp.Expr) -> sp.Expr:
        canonical = str(expr)
        seen: Set[str] = set()
        ordered: List[str] = []
        for m in re.finditer(r"\bc(\d+)\b", canonical):
            name = m.group()
            if name not in seen:
                seen.add(name)
                ordered.append(name)

        if not ordered or all(f"c{i}" == old for i, old in enumerate(ordered)):
            return expr

        tmp = {sp.Symbol(old): sp.Symbol(f"__tmp{i}__") for i, old in enumerate(ordered)}
        expr = expr.subs(tmp)
        final = {sp.Symbol(f"__tmp{i}__"): sp.Symbol(f"c{i}") for i in range(len(ordered))}
        return expr.subs(final)

    def struct_hash(self, expr: sp.Expr) -> str:
        if expr.is_Symbol:
            name = str(expr)
            return "C" if re.match(r"^c\d+$", name) else f"V:{name}"
        if expr.is_Number:
            return f"N:{expr}"
        children = [self.struct_hash(a) for a in expr.args]
        fname = expr.func.__name__
        if fname in ("Add", "Mul"):
            children.sort()
        return f"{fname}({','.join(children)})"

    def make_locals(self, expr_str: str) -> Dict[str, sp.Symbol]:
        loc = {n: sp.Symbol(n) for n in self.feature_names}
        for m in re.finditer(r"c\d+", expr_str):
            loc[m.group()] = sp.Symbol(m.group())
        return loc

    @staticmethod
    def collect_params(expr) -> List[str]:
        params = []
        for sym in expr.free_symbols:
            name = str(sym)
            if re.match(r"^c\d+$", name):
                params.append(name)
        return sorted(params, key=lambda s: int(s[1:]))

    @staticmethod
    def collect_params_from_str(expr_str: str) -> List[str]:
        return sorted(set(re.findall(r"\bc\d+\b", expr_str)), key=lambda s: int(s[1:]))
