"""Structure optimizer: VARPRO decomposition + DE/CMA global search + L-BFGS-B local refinement."""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from scipy.optimize import minimize, differential_evolution, least_squares as scipy_lstsq

from .base import (
    BaseOptimizer, OptResult, _OptTimeout, _RealPow,
    _SAFE_MODULES, _RATIONAL_INIT_POOL, _snap_rational, _RATIONAL_DENOM,
    parse_expr, compile_expr, build_mse_fn, build_residual_fn,
    detect_rational_constrained, make_safe_expr,
    param_hints, smart_x0, wrap_rational, compute_exp_param_bounds,
    detect_gaussian_offsets, make_bounds,
)


def _to_real_or_none(raw, *, tol=1e-9):
    """Cast an eval result to a real float array, or return None if genuinely complex.

    Avoids the implicit complex->float cast (which emits ``ComplexWarning`` and
    silently drops the imaginary part): a result whose imaginary part is not
    negligible relative to its real magnitude means the candidate is not
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


class StructureOptimizer(BaseOptimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        skeleton: Union[str, sp.Expr],
        param_names: List[str],
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        parent_params: Optional[List[float]] = None,
    ) -> OptResult:
        n_params = len(param_names)
        if isinstance(skeleton, sp.Expr):
            sympy_expr = skeleton
        else:
            sympy_expr = parse_expr(skeleton, param_names, feature_names)
        rational_idx = detect_rational_constrained(
            sympy_expr, param_names, feature_names, X_train)
        safe_expr = make_safe_expr(sympy_expr, param_names, rational_idx, feature_names)

        try:
            predict_fn = compile_expr(safe_expr, param_names, feature_names, X_train)
        except Exception:
            return OptResult(np.ones(n_params), float("inf"), 0)

        pos_idx, small_idx = param_hints(sympy_expr, param_names)
        exp_bounds = compute_exp_param_bounds(
            sympy_expr, param_names, feature_names, X_train)
        gaussian_hints = detect_gaussian_offsets(
            sympy_expr, param_names, feature_names, X_train)
        raw_mse = build_mse_fn(predict_fn, y_train, self.penalty)
        raw_objective = wrap_rational(raw_mse, rational_idx)
        total_feval = 0

        def _count_feval(n: int = 1) -> None:
            nonlocal total_feval
            total_feval += n

        def objective(p):
            _count_feval()
            return raw_objective(p)

        y_var = float(np.var(y_train))
        if y_var < 1e-30:
            y_var = 1.0
        _early_mse = 1e-10 * y_var

        deadline = time.perf_counter() + self.timeout

        raw_residual = build_residual_fn(predict_fn, y_train, np.sqrt(self.penalty))
        if rational_idx:
            _raw_res = raw_residual
            def residual(p):
                _count_feval()
                p2 = np.array(p, dtype=float)
                for i in rational_idx:
                    p2[i] = _snap_rational(p2[i])
                return _raw_res(p2)
        else:
            def residual(p):
                _count_feval()
                return raw_residual(p)

        best_mse = float("inf")
        best_params: List[float] = [1.0] * n_params

        lbfgs_bounds = make_bounds(n_params, self.bound, pos_idx, small_idx,
                                   rational_idx, exp_bounds=exp_bounds)
        lb_arr = np.array([b[0] for b in lbfgs_bounds])
        ub_arr = np.array([b[1] for b in lbfgs_bounds])

        start_time = deadline - self.timeout
        varpro_deadline = start_time + self.timeout * 0.45
        fallback_deadline = start_time + self.timeout * 0.90
        lbfgs_deadline = deadline

        def _run_lbfgs(ws: np.ndarray) -> None:
            nonlocal best_mse, best_params
            ws = np.clip(ws, lb_arr, ub_arr)
            remaining = max(lbfgs_deadline - time.perf_counter(), 0)
            lbfgs_dl = time.perf_counter() + remaining / 2.0
            def _tobj(p, _dl=lbfgs_dl):
                if time.perf_counter() >= _dl:
                    raise _OptTimeout
                return objective(p)
            try:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        _tobj, ws, method="L-BFGS-B",
                        bounds=lbfgs_bounds,
                        options={"maxiter": self.max_iter, "ftol": 1e-12})
                if res.fun < best_mse:
                    best_mse, best_params = float(res.fun), list(res.x)
            except _OptTimeout:
                pass
            except TimeoutError:
                raise
            except Exception:
                pass

        # 0) Compound-Pow pre-search — before VARPRO, shares the 45% time budget
        compound_pow = self._detect_compound_pow_params(sympy_expr, param_names)
        varpro_x0: Optional[List[float]] = None

        if compound_pow and best_mse > _early_mse:
            presearch_budget = (varpro_deadline - time.perf_counter()) * 0.70
            presearch_deadline = time.perf_counter() + presearch_budget
            try:
                ps_params, ps_mse = self._try_pow_presearch(
                    sympy_expr, safe_expr, param_names, feature_names,
                    X_train, y_train, parent_params, rational_idx,
                    compound_pow, exp_bounds, _early_mse, objective,
                    presearch_deadline, gaussian_hints)
                if ps_params is not None and np.isfinite(ps_mse) and ps_mse < best_mse:
                    best_mse, best_params = ps_mse, list(ps_params)
                    varpro_x0 = list(ps_params)
            except TimeoutError:
                raise
            except Exception:
                pass

        # 1) VARPRO — remaining 45% time (gets full 45% when no compound-pow)
        if best_mse > _early_mse and time.perf_counter() < varpro_deadline:
            try:
                varpro_result = self._try_varpro(
                    safe_expr, param_names, feature_names,
                    X_train, y_train, parent_params, rational_idx,
                    exp_bounds=exp_bounds, early_mse=_early_mse,
                    global_timeout=varpro_deadline,
                    gaussian_hints=gaussian_hints,
                    feval_counter=_count_feval)
                if varpro_result is not None:
                    vp_x0, varpro_mse = varpro_result
                    if vp_x0 is not None and np.isfinite(varpro_mse):
                        if varpro_mse < best_mse:
                            best_mse, best_params = varpro_mse, list(vp_x0)
                            varpro_x0 = list(vp_x0)
            except TimeoutError:
                raise
            except Exception:
                pass

        # 2) Fallback: full-parameter TRF / CMA / DE — 45% time
        #    When VARPRO finishes early, remaining time goes to Fallback
        if (varpro_x0 is None or best_mse > _early_mse) and time.perf_counter() < fallback_deadline:
            remaining = fallback_deadline - time.perf_counter()
            stage_budget = remaining / 3.0

            # TRF — independent starting point, deadline-driven multi-restart
            if best_mse > _early_mse and time.perf_counter() < fallback_deadline:
                trf_dl = min(time.perf_counter() + stage_budget, fallback_deadline)
                x0_trf = smart_x0(n_params, pos_idx, small_idx, rational_idx,
                                  exp_bounds=exp_bounds,
                                  gaussian_hints=gaussian_hints)
                try:
                    rp, rm = self._run_trf(
                        residual, x0_trf, pos_idx, small_idx, rational_idx,
                        exp_bounds=exp_bounds, global_deadline=trf_dl,
                        gaussian_hints=gaussian_hints)
                    if rm < best_mse:
                        best_mse, best_params = rm, rp
                except TimeoutError:
                    raise
                except Exception:
                    pass

            # CMA — independent starting point
            if best_mse > _early_mse and time.perf_counter() < fallback_deadline:
                cma_dl = min(time.perf_counter() + stage_budget, fallback_deadline)
                x0_cma = smart_x0(n_params, pos_idx, small_idx, rational_idx,
                                  exp_bounds=exp_bounds,
                                  gaussian_hints=gaussian_hints)
                try:
                    rp, rm = self._run_cma(objective, x0_cma,
                                           pos_idx, small_idx, rational_idx,
                                           exp_bounds=exp_bounds,
                                           global_deadline=cma_dl,
                                           early_mse=_early_mse,
                                           gaussian_hints=gaussian_hints)
                    if rm < best_mse:
                        best_mse, best_params = rm, rp
                except TimeoutError:
                    raise
                except Exception:
                    pass

            # DE — independent starting point
            if best_mse > _early_mse and time.perf_counter() < fallback_deadline:
                de_dl = min(time.perf_counter() + stage_budget, fallback_deadline)
                x0_de = smart_x0(n_params, pos_idx, small_idx, rational_idx,
                                 exp_bounds=exp_bounds,
                                 gaussian_hints=gaussian_hints)
                try:
                    rp, rm = self._run_de(
                        objective, x0_de, pos_idx, small_idx, rational_idx,
                        exp_bounds=exp_bounds,
                        global_deadline=de_dl)
                    if rm < best_mse:
                        best_mse, best_params = rm, rp
                except TimeoutError:
                    raise
                except Exception:
                    pass

        # 3) L-BFGS-B refinement — 10% time
        if best_mse > _early_mse and time.perf_counter() < lbfgs_deadline:
            _run_lbfgs(np.array(best_params, dtype=float))

        if best_mse > _early_mse and parent_params and time.perf_counter() < lbfgs_deadline:
            warm_x0 = np.zeros(n_params)
            for i in range(min(len(parent_params), n_params)):
                warm_x0[i] = parent_params[i]
            for i in rational_idx:
                if i < len(warm_x0):
                    warm_x0[i] = _snap_rational(warm_x0[i])
            _run_lbfgs(warm_x0)

        # 4) Rational snap
        for i in rational_idx:
            best_params[i] = _snap_rational(best_params[i])

        # 5) Pow exponent integer/rational snap + refit
        pow_exp_idx = self._detect_pow_exponents(sympy_expr, param_names)
        if pow_exp_idx and best_mse > _early_mse:
            snap_params, snap_mse = self._snap_pow_and_refit(
                best_params, best_mse, pow_exp_idx,
                objective, lbfgs_bounds, exp_bounds or {})
            if snap_mse < best_mse:
                best_mse, best_params = snap_mse, snap_params

        return OptResult(np.array(best_params), best_mse, total_feval,
                         rational_idx=list(rational_idx), safe_expr=safe_expr)

    # ------------------------------------------------------------------ #
    # Compound-Pow pre-search (Phase 0)
    # ------------------------------------------------------------------ #

    _POW_PRESEARCH_CANDIDATES = [
        sp.Integer(1), sp.Integer(2), sp.Rational(1, 2),
        sp.Integer(3), sp.Rational(1, 3), sp.Integer(-1),
        sp.Rational(3, 2), sp.Integer(4), sp.Integer(5),
        sp.Rational(-1, 2), sp.Integer(-2), sp.Integer(-3),
    ]

    @staticmethod
    def _detect_compound_pow_params(
        sympy_expr: sp.Expr, param_names: List[str],
    ) -> Dict[int, sp.Symbol]:
        """Detect Pow exponent parameters where the base contains other parameters (compound-pow).

        Returns {param_index: param_symbol} mapping.
        Only matches Pow nodes where the exponent is a single parameter and
        the base contains other parameters; excludes simple x**c (base has no parameters).
        """
        sym_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
        param_syms = set(sym_to_idx.keys())
        result: Dict[int, sp.Symbol] = {}
        for node in sp.preorder_traversal(sympy_expr):
            if not isinstance(node, (sp.Pow, _RealPow)) or len(node.args) != 2:
                continue
            base, exponent = node.args
            exp_params = exponent.free_symbols & param_syms
            base_params = base.free_symbols & param_syms
            if len(exp_params) == 1 and base_params:
                p_sym = next(iter(exp_params))
                result[sym_to_idx[p_sym]] = p_sym
        return result

    def _try_pow_presearch(
        self,
        sympy_expr: sp.Expr,
        safe_expr: sp.Expr,
        param_names: List[str],
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        parent_params: Optional[List[float]],
        rational_idx: set,
        compound_pow: Dict[int, sp.Symbol],
        exp_bounds: Optional[Dict[int, Tuple[float, float]]],
        early_mse: float,
        objective,
        presearch_deadline: float,
        gaussian_hints: Optional[Dict[int, np.ndarray]],
    ) -> Tuple[Optional[List[float]], float]:
        """Phase 0: enumerate rational candidates for compound-pow parameters, run VARPRO for each.

        Returns (best_full_params, best_mse); returns (None, inf) when no valid solution is found.
        """
        best_params: Optional[List[float]] = None
        best_mse = float("inf")
        n_params = len(param_names)

        n_candidates = len(self._POW_PRESEARCH_CANDIDATES)
        per_candidate_budget = max(
            (presearch_deadline - time.perf_counter()) / max(n_candidates, 1), 0.5)

        for cand_val in self._POW_PRESEARCH_CANDIDATES:
            if time.perf_counter() >= presearch_deadline:
                break
            if best_mse <= early_mse:
                break

            cand_deadline = min(
                time.perf_counter() + per_candidate_budget, presearch_deadline)

            subs_dict = {sym: cand_val for sym in compound_pow.values()}
            cand_float = float(cand_val)
            try:
                reduced_expr = sympy_expr.subs(subs_dict)
                reduced_safe = safe_expr.subs(subs_dict)
            except Exception:
                continue

            reduced_param_names = [
                p for i, p in enumerate(param_names) if i not in compound_pow]
            reduced_rational_idx = set()
            idx_map: Dict[int, int] = {}
            new_i = 0
            for old_i, p in enumerate(param_names):
                if old_i in compound_pow:
                    continue
                if old_i in rational_idx:
                    reduced_rational_idx.add(new_i)
                idx_map[old_i] = new_i
                new_i += 1

            reduced_exp_bounds: Dict[int, Tuple[float, float]] = {}
            if exp_bounds:
                for old_i, bounds in exp_bounds.items():
                    if old_i in idx_map:
                        reduced_exp_bounds[idx_map[old_i]] = bounds

            reduced_parent = None
            if parent_params:
                reduced_parent = [
                    parent_params[i] for i in range(len(parent_params))
                    if i not in compound_pow and i < len(parent_params)]

            reduced_gaussian: Optional[Dict[int, np.ndarray]] = None
            if gaussian_hints:
                reduced_gaussian = {}
                for old_i, arr in gaussian_hints.items():
                    if old_i in idx_map:
                        reduced_gaussian[idx_map[old_i]] = arr
                if not reduced_gaussian:
                    reduced_gaussian = None

            try:
                vp_result = self._try_varpro(
                    reduced_safe, reduced_param_names, feature_names,
                    X_train, y_train, reduced_parent, reduced_rational_idx,
                    exp_bounds=reduced_exp_bounds, early_mse=early_mse,
                    global_timeout=cand_deadline,
                    gaussian_hints=reduced_gaussian)
            except _OptTimeout:
                continue
            except TimeoutError:
                raise
            except Exception:
                continue

            if vp_result is None:
                continue

            reduced_x0, reduced_mse = vp_result
            if reduced_x0 is None or not np.isfinite(reduced_mse):
                continue

            full_params = [0.0] * n_params
            ri = 0
            for oi in range(n_params):
                if oi in compound_pow:
                    full_params[oi] = cand_float
                else:
                    full_params[oi] = reduced_x0[ri]
                    ri += 1

            full_mse = objective(np.array(full_params))
            if np.isfinite(full_mse) and full_mse < best_mse:
                best_mse = full_mse
                best_params = full_params

        return best_params, best_mse

    # ------------------------------------------------------------------ #
    # Pow exponent snap
    # ------------------------------------------------------------------ #

    _POW_GRID_CANDIDATES = [
        1/3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
        -1/3, -0.5, -1.0, -2.0, -3.0,
    ]

    @staticmethod
    def _detect_pow_exponents(
        sympy_expr: sp.Expr, param_names: List[str],
    ) -> set:
        """Return the set of parameter indices appearing as Pow exponents (excluding exp parameters)."""
        sym_to_idx = {sp.Symbol(p): i for i, p in enumerate(param_names)}
        param_syms = set(sym_to_idx.keys())
        result: set = set()
        for node in sp.preorder_traversal(sympy_expr):
            if isinstance(node, (sp.Pow, _RealPow)) and len(node.args) == 2:
                for s in node.args[1].free_symbols & param_syms:
                    result.add(sym_to_idx[s])
        return result

    def _snap_pow_and_refit(
        self,
        params: List[float],
        current_mse: float,
        pow_idx: set,
        objective,
        bounds: List[Tuple[float, float]],
        exp_bounds: Dict[int, Tuple[float, float]],
    ) -> Tuple[List[float], float]:
        """Snap Pow exponent parameters to integer/rational values, then refine with L-BFGS-B.

        VARPRO has already performed global search with DE; this is lightweight local refinement only.
        """
        best_params = list(params)
        best_mse = current_mse
        n = len(params)

        for idx in sorted(pow_idx):
            cur_val = best_params[idx]
            candidates = sorted(self._POW_GRID_CANDIDATES,
                                key=lambda c: abs(c - cur_val))
            candidates = [c for c in candidates if abs(c - cur_val) <= 3.0]

            for snap_val in candidates:
                trial = list(best_params)
                trial[idx] = snap_val
                trial_mse = objective(np.array(trial))
                if not np.isfinite(trial_mse):
                    continue

                fixed_bounds = list(bounds)
                fixed_bounds[idx] = (snap_val - 1e-15, snap_val + 1e-15)
                try:
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = minimize(
                            objective, trial, method="L-BFGS-B",
                            bounds=fixed_bounds,
                            options={"maxiter": self.max_iter, "ftol": 1e-15})
                    if res.fun < best_mse:
                        best_mse = float(res.fun)
                        best_params = list(res.x)
                        best_params[idx] = snap_val
                except Exception:
                    pass

        return best_params, best_mse

    _POW_GRID_N_RESTARTS = 5

    def _pow_rational_grid(
        self,
        n_nl: int,
        nl_pow_idx: set,
        ols_objective,
        ols_residual,
        nl_pos_idx: set,
        nl_small_idx: set,
        nl_rational_idx: set,
        nl_exp_bounds: Dict[int, Tuple[float, float]],
        varpro_deadline: Optional[float] = None,
        nl_gaussian: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[Optional[List[float]], float]:
        """Phase 1: rational candidate grid search for Pow exponents.

        For each Pow exponent parameter (sequential greedy), enumerate candidates
        from _POW_GRID_CANDIDATES, fix the exponent and use multi-start
        least_squares (TRF) to search the remaining nonlinear parameters.
        Multiple Pow exponents are fixed sequentially (sequential greedy).
        """
        best_nl: Optional[List[float]] = None
        best_mse = float("inf")

        fixed_pow: Dict[int, float] = {}
        
        def _check_timeout():
            if varpro_deadline is not None and time.perf_counter() >= varpro_deadline:
                raise _OptTimeout

        for pidx in sorted(nl_pow_idx):
            _check_timeout()
            
            search_idx = [j for j in range(n_nl)
                          if j != pidx and j not in fixed_pow]

            if not search_idx:
                for sv in self._POW_GRID_CANDIDATES:
                    _check_timeout()
                    trial = np.empty(n_nl)
                    for fp_i, fp_v in fixed_pow.items():
                        trial[fp_i] = fp_v
                    trial[pidx] = sv
                    trial_mse = ols_objective(trial)
                    if trial_mse < best_mse:
                        best_mse = trial_mse
                        best_nl = list(trial)
                if best_nl is not None:
                    fixed_pow[pidx] = best_nl[pidx]
                continue

            reduced_pos = {k for k, si in enumerate(search_idx)
                           if si in nl_pos_idx}
            reduced_small = {k for k, si in enumerate(search_idx)
                             if si in nl_small_idx}
            reduced_rational = {k for k, si in enumerate(search_idx)
                                if si in nl_rational_idx}
            reduced_eb: Dict[int, Tuple[float, float]] = {}
            for k, si in enumerate(search_idx):
                if si in nl_exp_bounds:
                    reduced_eb[k] = nl_exp_bounds[si]

            n_reduced = len(search_idx)
            reduced_bounds_list = make_bounds(
                n_reduced, self.bound,
                reduced_pos, reduced_small, reduced_rational,
                exp_bounds=reduced_eb)
            lb = np.array([b[0] for b in reduced_bounds_list])
            ub = np.array([b[1] for b in reduced_bounds_list])

            best_cand_val: Optional[float] = None
            best_cand_mse = float("inf")
            best_cand_reduced: Optional[List[float]] = None

            for sv in self._POW_GRID_CANDIDATES:
                _check_timeout()
                _fp = dict(fixed_pow)
                _si = list(search_idx)

                def _fixed_residual(reduced_p, _sv=sv, _pidx=pidx,
                                    _fp=_fp, _si=_si):
                    full_p = np.empty(n_nl)
                    full_p[_pidx] = _sv
                    for fp_i, fp_v in _fp.items():
                        full_p[fp_i] = fp_v
                    for k, si in enumerate(_si):
                        full_p[si] = reduced_p[k]
                    return ols_residual(full_p)

                _ng = nl_gaussian or {}
                reduced_gh: Dict[int, np.ndarray] = {}
                for k, si in enumerate(search_idx):
                    if si in _ng:
                        reduced_gh[k] = _ng[si]

                # Gaussian offset: dense OLS grid scan (fast)
                if reduced_gh:
                    base_r = smart_x0(n_reduced, reduced_pos, reduced_small,
                                      reduced_rational, exp_bounds=reduced_eb)
                    if best_nl is not None:
                        for k2, si in enumerate(search_idx):
                            base_r[k2] = best_nl[si]
                    for gh_k, gh_grid in reduced_gh.items():
                        for gv in gh_grid:
                            if varpro_deadline is not None and \
                               time.perf_counter() >= varpro_deadline:
                                raise _OptTimeout
                            trial_r = base_r.copy()
                            trial_r[gh_k] = gv
                            full_trial = np.empty(n_nl)
                            full_trial[pidx] = sv
                            for fp_i, fp_v in fixed_pow.items():
                                full_trial[fp_i] = fp_v
                            for k2, si in enumerate(search_idx):
                                full_trial[si] = trial_r[k2]
                            m = ols_objective(full_trial)
                            if m < best_cand_mse:
                                best_cand_mse = m
                                best_cand_val = sv
                                best_cand_reduced = list(trial_r)

                trf_starts = []
                if best_cand_reduced is not None:
                    trf_starts.append(np.array(best_cand_reduced, dtype=float))
                for _ in range(self._POW_GRID_N_RESTARTS):
                    trf_starts.append(smart_x0(
                        n_reduced, reduced_pos, reduced_small,
                        reduced_rational, exp_bounds=reduced_eb,
                        gaussian_hints=reduced_gh if reduced_gh else None))

                for x0 in trf_starts:
                    if varpro_deadline is not None and time.perf_counter() >= varpro_deadline:
                        raise _OptTimeout
                    x0 = np.clip(x0, lb + 1e-10, ub - 1e-10)
                    try:
                        with np.errstate(all="ignore"), warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = scipy_lstsq(
                                _fixed_residual, x0,
                                bounds=(lb, ub), method="trf",
                                max_nfev=min(200 * n_reduced, 1000),
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
                            mse = float(np.mean(res.fun ** 2))
                            if mse < best_cand_mse:
                                best_cand_mse = mse
                                best_cand_val = sv
                                best_cand_reduced = list(res.x)
                    except _OptTimeout:
                        raise
                    except Exception:
                        pass

            if best_cand_val is not None:
                fixed_pow[pidx] = best_cand_val
                trial = np.empty(n_nl)
                trial[pidx] = best_cand_val
                for fp_i, fp_v in fixed_pow.items():
                    if fp_i != pidx:
                        trial[fp_i] = fp_v
                for k, si in enumerate(search_idx):
                    trial[si] = best_cand_reduced[k]
                if best_cand_mse < best_mse:
                    best_mse = best_cand_mse
                    best_nl = list(trial)

        return best_nl, best_mse

    # ------------------------------------------------------------------ #
    # Internal TRF / DE / CMA
    # ------------------------------------------------------------------ #

    def _run_trf(
        self, residual_fn, x0: np.ndarray,
        pos_idx: set = frozenset(), small_idx: set = frozenset(),
        rational_idx: set = frozenset(),
        exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        global_deadline: Optional[float] = None,
        gaussian_hints: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[List[float], float]:
        """Multi-start least_squares (TRF) search."""
        _eb = exp_bounds or {}
        n = len(x0)
        lb = np.full(n, -self.bound)
        ub = np.full(n, self.bound)
        for i in range(n):
            if i in rational_idx:
                lb[i], ub[i] = -10.0, 10.0
            elif i in _eb:
                lb[i], ub[i] = _eb[i]
            elif i in pos_idx and i in small_idx:
                lb[i], ub[i] = 1e-6, 5.0
            elif i in pos_idx:
                lb[i], ub[i] = 1e-6, self.bound
            elif i in small_idx:
                lb[i], ub[i] = -5.0, 5.0

        best_x, best_fun = list(x0), float("inf")
        trf_deadline = global_deadline if global_deadline is not None \
            else time.perf_counter() + self.timeout

        def timed_residual(p):
            if time.perf_counter() >= trf_deadline:
                raise _OptTimeout
            return residual_fn(p)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # First try from the provided x0
            x0_clipped = np.clip(x0, lb + 1e-10, ub - 1e-10)
            try:
                res = scipy_lstsq(
                    timed_residual, x0_clipped,
                    bounds=(lb, ub), method="trf",
                    max_nfev=min(self.max_iter * n, 1000),
                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
                mse = float(np.mean(res.fun ** 2))
                if mse < best_fun:
                    best_x, best_fun = list(res.x), mse
            except _OptTimeout:
                return best_x, best_fun
            except Exception:
                pass
            
            # Then restart with random starting points until deadline
            while time.perf_counter() < trf_deadline:
                x0_trial = smart_x0(n, pos_idx, small_idx, rational_idx,
                                    exp_bounds=_eb,
                                    gaussian_hints=gaussian_hints)
                x0_trial = np.clip(x0_trial, lb + 1e-10, ub - 1e-10)
                try:
                    res = scipy_lstsq(
                        timed_residual, x0_trial,
                        bounds=(lb, ub), method="trf",
                        max_nfev=min(self.max_iter * n, 1000),
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    mse = float(np.mean(res.fun ** 2))
                    if mse < best_fun:
                        best_x, best_fun = list(res.x), mse
                except _OptTimeout:
                    break
                except TimeoutError:
                    raise
                except Exception:
                    pass
        return best_x, best_fun

    def _run_de(
        self, objective, x0: np.ndarray,
        pos_idx: set = frozenset(), small_idx: set = frozenset(),
        rational_idx: set = frozenset(),
        exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        global_deadline: Optional[float] = None,
    ) -> Tuple[List[float], float]:
        _eb = exp_bounds or {}
        bounds = []
        for i in range(len(x0)):
            if i in rational_idx:
                bounds.append((-10.0, 10.0))
            elif i in _eb:
                bounds.append(_eb[i])
            elif i in pos_idx and i in small_idx:
                bounds.append((1e-6, 5.0))
            elif i in pos_idx:
                bounds.append((1e-6, self.bound))
            elif i in small_idx:
                bounds.append((-5.0, 5.0))
            else:
                bounds.append((-self.bound, self.bound))

        best_x, best_fun = list(x0), float("inf")
        de_deadline = global_deadline if global_deadline is not None \
            else time.perf_counter() + self.timeout

        def timed_objective(p):
            if time.perf_counter() >= de_deadline:
                raise _OptTimeout
            return objective(p)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(10000):
                if time.perf_counter() >= de_deadline:
                    break
                try:
                    result = differential_evolution(
                        timed_objective, bounds=bounds,
                        maxiter=self.max_iter, seed=seed, tol=1e-8,
                        popsize=10, polish=True,
                    )
                    if result.fun < best_fun:
                        best_x, best_fun = list(result.x), float(result.fun)
                except _OptTimeout:
                    break
                except Exception:
                    pass
        return best_x, best_fun

    def _run_cma(
        self, objective, x0: np.ndarray,
        pos_idx: set = frozenset(), small_idx: set = frozenset(),
        rational_idx: set = frozenset(),
        exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        global_deadline: Optional[float] = None,
        early_mse: float = 1e-20,
        gaussian_hints: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[List[float], float]:
        from cmaes import CMA

        _eb = exp_bounds or {}
        n = len(x0)
        lb = np.full(n, -self.bound)
        ub = np.full(n, self.bound)
        for i in range(n):
            if i in rational_idx:
                lb[i], ub[i] = -10.0, 10.0
            elif i in _eb:
                lb[i], ub[i] = _eb[i]
            elif i in pos_idx and i in small_idx:
                lb[i], ub[i] = 1e-6, 5.0
            elif i in pos_idx:
                lb[i], ub[i] = 1e-6, self.bound
            elif i in small_idx:
                lb[i], ub[i] = -5.0, 5.0

        cma_bounds = np.column_stack([lb, ub])
        best_x, best_fun = list(x0), objective(x0)

        cma_deadline = global_deadline if global_deadline is not None \
            else time.perf_counter() + self.timeout

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while time.perf_counter() < cma_deadline and best_fun > early_mse:
                x0_cma = smart_x0(n, pos_idx, small_idx, rational_idx,
                                  exp_bounds=_eb,
                                  gaussian_hints=gaussian_hints)
                cma_opt = CMA(
                    mean=np.clip(x0_cma, lb + 1e-6, ub - 1e-6).astype(float),
                    sigma=2.0,
                    population_size=20,
                    bounds=cma_bounds,
                )
                for _ in range(self.max_iter):
                    if time.perf_counter() >= cma_deadline:
                        break
                    solutions = []
                    for _ in range(cma_opt.population_size):
                        x = cma_opt.ask()
                        val = objective(x)
                        solutions.append((x, val))
                        if val < best_fun:
                            best_x, best_fun = list(x), val
                    cma_opt.tell(solutions)
                    if best_fun <= early_mse or cma_opt.should_stop():
                        break
        return best_x, best_fun

    # ------------------------------------------------------------------ #
    # VARPRO core
    # ------------------------------------------------------------------ #

    @staticmethod
    def _identify_explicit_nonlinear_params(
        sympy_expr: sp.Expr, param_names: List[str],
    ) -> set:
        """Return parameters that appear in explicitly nonlinear positions.

        This intentionally does not mark parameters that merely multiply other
        parameters as nonlinear. Joint affine checks below decide which of the
        remaining parameters are safe to solve by OLS.
        """
        param_syms = {sp.Symbol(p) for p in param_names}
        nonlinear: set = set()

        for node in sp.preorder_traversal(sympy_expr):
            if isinstance(node, (sp.sin, sp.cos, sp.tan)):
                for s in node.args[0].free_symbols & param_syms:
                    nonlinear.add(s)
            elif isinstance(node, sp.exp):
                for s in node.args[0].free_symbols & param_syms:
                    nonlinear.add(s)
            elif isinstance(node, sp.log):
                for s in node.args[0].free_symbols & param_syms:
                    nonlinear.add(s)
            elif isinstance(node, (sp.Pow, _RealPow)) and len(node.args) == 2:
                _, exponent = node.args
                for s in exponent.free_symbols & param_syms:
                    nonlinear.add(s)
        return nonlinear

    @staticmethod
    def _expr_is_zero(expr: sp.Expr) -> bool:
        if expr == 0:
            return True
        try:
            return bool(sp.simplify(expr) == 0)
        except TimeoutError:
            raise
        except Exception:
            return False

    @classmethod
    def _select_joint_affine_params(
        cls,
        sympy_expr: sp.Expr,
        param_names: List[str],
        explicit_nonlinear: set,
    ) -> List[str]:
        """Select a maximal safe subset of parameters that are jointly affine.

        A parameter is eligible for OLS only if the expression has zero second
        derivative with respect to that parameter. Two eligible parameters can
        be solved together only if their mixed second derivative is also zero.
        """
        candidates: List[sp.Symbol] = []
        for name in param_names:
            sym = sp.Symbol(name)
            if sym in explicit_nonlinear:
                continue
            first = sp.diff(sympy_expr, sym)
            if cls._expr_is_zero(first):
                continue
            if cls._expr_is_zero(sp.diff(first, sym)):
                candidates.append(sym)

        if not candidates:
            return []

        conflicts: Dict[sp.Symbol, set] = {sym: set() for sym in candidates}
        for i, left in enumerate(candidates):
            left_grad = sp.diff(sympy_expr, left)
            for right in candidates[i + 1:]:
                mixed = sp.diff(left_grad, right)
                if not cls._expr_is_zero(mixed):
                    conflicts[left].add(right)
                    conflicts[right].add(left)

        # Parameter counts are usually small. Brute force gives the largest
        # jointly affine subset; fall back to greedy for unusually large cases.
        if len(candidates) <= 18:
            best: List[sp.Symbol] = []
            n = len(candidates)
            for mask in range(1, 1 << n):
                if mask.bit_count() <= len(best):
                    continue
                subset = [candidates[i] for i in range(n) if mask & (1 << i)]
                ok = True
                for i, left in enumerate(subset):
                    if any(right in conflicts[left] for right in subset[i + 1:]):
                        ok = False
                        break
                if ok:
                    best = subset
            return [str(sym) for sym in best]

        selected: List[sp.Symbol] = []
        for sym in sorted(candidates, key=lambda s: len(conflicts[s])):
            if all(sym not in conflicts[chosen] for chosen in selected):
                selected.append(sym)
        return [str(sym) for sym in selected]

    def _try_varpro(
        self,
        sympy_expr: sp.Expr,
        param_names: List[str],
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        parent_params: Optional[List[float]] = None,
        rational_idx: set = frozenset(),
        exp_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        early_mse: float = 0.0,
        global_timeout: Optional[float] = None,
        gaussian_hints: Optional[Dict[int, np.ndarray]] = None,
        feval_counter: Optional[Callable[[int], None]] = None,
    ) -> Optional[Tuple[List[float], float]]:
        try:
            explicit_nonlinear = self._identify_explicit_nonlinear_params(
                sympy_expr, param_names)
            linear_names = self._select_joint_affine_params(
                sympy_expr, param_names, explicit_nonlinear)
        except TimeoutError:
            raise
        except Exception:
            return None

        linear_set = {sp.Symbol(p) for p in linear_names}
        nonlinear_names = [p for p in param_names if sp.Symbol(p) not in linear_set]

        if not linear_names:
            return None

        nonlinear_syms = [sp.Symbol(p) for p in nonlinear_names]
        feature_syms = [sp.Symbol(f) for f in feature_names]

        # ---- Decompose original expression into a jointly affine OLS block ----
        try:
            zero_linear = {sp.Symbol(name): 0 for name in linear_names}
            monomials = [sp.Symbol(name) for name in linear_names]
            basis_exprs = [
                sp.diff(sympy_expr, sp.Symbol(name)).subs(zero_linear)
                for name in linear_names
            ]
            fixed_part = sympy_expr.subs(zero_linear)

            if not basis_exprs:
                return None
        except TimeoutError:
            raise
        except Exception:
            return None

        # ---- Compile basis functions ----
        all_input_syms = feature_syms + nonlinear_syms
        try:
            basis_block_func = sp.lambdify(
                all_input_syms, sp.Tuple(*basis_exprs), modules=_SAFE_MODULES)
            fixed_func = (
                sp.lambdify(all_input_syms, fixed_part, modules=_SAFE_MODULES)
                if fixed_part != sp.S.Zero
                else None
            )
        except TimeoutError:
            raise
        except Exception:
            return None

        X_cols = [X_train[:, i] for i in range(len(feature_names))]
        n_samples = len(y_train)
        n_basis = len(basis_exprs)

        def _normalize_col(raw):
            col = _to_real_or_none(raw)
            if col is None:
                return None
            if col.ndim == 0:
                col = np.full(n_samples, float(col))
            else:
                col = col.ravel()
                if col.shape[0] == 1:
                    col = np.full(n_samples, col[0])
            if col.shape[0] != n_samples or not np.all(np.isfinite(col)):
                return None
            return col

        def _eval_basis_block(inputs):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_block = basis_block_func(*inputs)
            if isinstance(raw_block, tuple):
                raw_items = raw_block
            else:
                raw_items = (raw_block,)
            if len(raw_items) != n_basis:
                return None
            A = np.empty((n_samples, n_basis))
            for j, raw in enumerate(raw_items):
                col = _normalize_col(raw)
                if col is None:
                    return None
                A[:, j] = col
            return A

        penalty_residual = np.full(n_samples, np.sqrt(self.penalty))

        last_ols_key: Optional[bytes] = None
        last_ols_result: Optional[Tuple[np.ndarray, float, np.ndarray]] = None

        def _ols_core(nl_values):
            """Return (residual_vector, mse_scalar, ols_coeffs) or None."""
            nonlocal last_ols_key, last_ols_result
            try:
                nl_array = np.asarray(nl_values, dtype=float).ravel()
                cache_key = nl_array.tobytes()
                if cache_key == last_ols_key:
                    return last_ols_result
                if feval_counter is not None:
                    feval_counter()

                inputs = X_cols + nl_array.tolist()
                A = _eval_basis_block(inputs)
                if A is None:
                    return None

                y_adj = y_train
                if fixed_func is not None:
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fixed_raw = fixed_func(*inputs)
                    fv = _normalize_col(fixed_raw)
                    if fv is None:
                        return None
                    y_adj = y_train - fv

                coeffs, _, _, _ = np.linalg.lstsq(A, y_adj, rcond=None)
                if not np.all(np.isfinite(coeffs)):
                    return None

                y_pred = A @ coeffs
                if fixed_func is not None:
                    y_pred = y_pred + fv

                residuals = y_pred - y_train
                if not np.all(np.isfinite(residuals)):
                    return None
                with np.errstate(all="ignore"):
                    mse = float(np.mean(residuals ** 2))
                result = (residuals, mse, coeffs)
                last_ols_key = cache_key
                last_ols_result = result
                return result
            except TimeoutError:
                raise
            except Exception:
                return None

        def ols_objective(nl_values):
            result = _ols_core(nl_values)
            if result is None:
                return self.penalty
            return result[1] if np.isfinite(result[1]) else self.penalty

        def ols_residual(nl_values):
            result = _ols_core(nl_values)
            if result is None:
                return penalty_residual
            return result[0]

        # ---- Pure OLS (no nonlinear parameters) ----
        if not nonlinear_names:
            ols_result = _ols_core([])
            if ols_result is None:
                return None
            _, pure_mse, ols_coeffs = ols_result
            if pure_mse >= self.penalty or not np.isfinite(pure_mse):
                return None
            return self._recover_varpro_params(
                param_names, linear_names, nonlinear_names,
                monomials, ols_coeffs, [],
            )

        # ---- Outer search for nonlinear parameters ----
        n_nl = len(nonlinear_names)
        nl_pos_idx, nl_small_idx = param_hints(sympy_expr, nonlinear_names)

        nl_rational_idx: set = set()
        nl_exp_bounds: Dict[int, Tuple[float, float]] = {}
        nl_gaussian: Optional[Dict[int, np.ndarray]] = None
        _eb = exp_bounds or {}
        _gh = gaussian_hints or {}
        for i, name in enumerate(nonlinear_names):
            orig_idx = param_names.index(name)
            if orig_idx in rational_idx:
                nl_rational_idx.add(i)
            if orig_idx in _eb:
                nl_exp_bounds[i] = _eb[orig_idx]
            if orig_idx in _gh:
                if nl_gaussian is None:
                    nl_gaussian = {}
                nl_gaussian[i] = _gh[orig_idx]

        # Rational snap is only applied as post-processing after selecting the best in Phase 3,
        # not wrapped inside the objective function, to avoid gradient blinding

        # ---- Detect Pow exponents ----
        nl_pow_idx = self._detect_pow_exponents(sympy_expr, nonlinear_names)

        varpro_deadline = global_timeout if global_timeout is not None \
            else time.perf_counter() + self.timeout

        # ============================================================
        # Phase 1: Rational candidate grid search (Pow exponents)
        # Fix exponent at each candidate rational value, short DE search for remaining params
        # ============================================================
        best_rational_nl: Optional[List[float]] = None
        best_rational_mse = float("inf")

        if nl_pow_idx:
            try:
                best_rational_nl, best_rational_mse = self._pow_rational_grid(
                    n_nl, nl_pow_idx, ols_objective, ols_residual,
                    nl_pos_idx, nl_small_idx, nl_rational_idx, nl_exp_bounds,
                    varpro_deadline=varpro_deadline,
                    nl_gaussian=nl_gaussian)
            except _OptTimeout:
                # Timed out but may have partial results, continue
                pass
            except TimeoutError:
                raise
            except Exception:
                pass

        # ============================================================
        # Phase 1b: Gaussian offset grid scan (only when no Pow Grid)
        # ============================================================
        if nl_gaussian and not nl_pow_idx and best_rational_mse > early_mse:
            _scan_best_nl = list(best_rational_nl) if best_rational_nl is not None \
                else [1.0] * n_nl
            _scan_best_mse = best_rational_mse

            for g_idx, g_quantiles in nl_gaussian.items():
                if varpro_deadline is not None and time.perf_counter() >= varpro_deadline:
                    break
                other_idx = [j for j in range(n_nl) if j != g_idx]
                if not other_idx:
                    for qv in g_quantiles:
                        trial = np.array(_scan_best_nl, dtype=float)
                        trial[g_idx] = qv
                        m = ols_objective(trial)
                        if m < _scan_best_mse:
                            _scan_best_mse, _scan_best_nl = m, list(trial)
                    continue

                r_pos = {k for k, si in enumerate(other_idx) if si in nl_pos_idx}
                r_small = {k for k, si in enumerate(other_idx) if si in nl_small_idx}
                r_rat = {k for k, si in enumerate(other_idx) if si in nl_rational_idx}
                r_eb: Dict[int, Tuple[float, float]] = {}
                for k, si in enumerate(other_idx):
                    if si in nl_exp_bounds:
                        r_eb[k] = nl_exp_bounds[si]
                n_r = len(other_idx)
                r_bounds = make_bounds(n_r, self.bound, r_pos, r_small, r_rat, exp_bounds=r_eb)
                r_lb = np.array([b[0] for b in r_bounds])
                r_ub = np.array([b[1] for b in r_bounds])

                for qv in g_quantiles:
                    if varpro_deadline is not None and time.perf_counter() >= varpro_deadline:
                        break

                    def _g_residual(rp, _qv=qv, _gi=g_idx, _oi=other_idx):
                        full = np.empty(n_nl)
                        full[_gi] = _qv
                        for k, si in enumerate(_oi):
                            full[si] = rp[k]
                        return ols_residual(full)

                    x0_r = smart_x0(n_r, r_pos, r_small, r_rat, exp_bounds=r_eb)
                    x0_r = np.clip(x0_r, r_lb + 1e-10, r_ub - 1e-10)
                    try:
                        res = scipy_lstsq(
                            _g_residual, x0_r,
                            bounds=(r_lb, r_ub), method="trf",
                            max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                        m = float(np.mean(res.fun ** 2))
                        if m < _scan_best_mse:
                            full_p = np.empty(n_nl)
                            full_p[g_idx] = qv
                            for k, si in enumerate(other_idx):
                                full_p[si] = res.x[k]
                            _scan_best_mse = m
                            _scan_best_nl = list(full_p)
                    except Exception:
                        pass

            if _scan_best_mse < best_rational_mse:
                best_rational_mse = _scan_best_mse
                best_rational_nl = _scan_best_nl

        # ============================================================
        # Phase 2: Continuous search (TRF → CMA → DE, free continuous exponent search)
        # ============================================================
        best_cont_mse = float("inf")
        best_cont_nl: Optional[List[float]] = None

        if best_rational_mse <= early_mse:
            pass  # Phase 1 already good enough, skip continuous search
        elif time.perf_counter() < varpro_deadline:
            vp_remaining = varpro_deadline - time.perf_counter()
            vp_stage = vp_remaining / 3.0

            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Phase 1 result used only as TRF starting point (if available)
                if best_rational_nl is not None and best_rational_mse < self.penalty / 10:
                    x0_trf = np.array(best_rational_nl, dtype=float)
                else:
                    x0_trf = smart_x0(n_nl, nl_pos_idx, nl_small_idx, nl_rational_idx,
                                      exp_bounds=nl_exp_bounds,
                                      gaussian_hints=nl_gaussian)

                # TRF
                trf_vp_dl = min(time.perf_counter() + vp_stage, varpro_deadline)
                try:
                    rp, rm = self._run_trf(
                        ols_residual, x0_trf, nl_pos_idx, nl_small_idx, nl_rational_idx,
                        exp_bounds=nl_exp_bounds, global_deadline=trf_vp_dl)
                    if rm < best_cont_mse:
                        best_cont_mse, best_cont_nl = rm, rp
                except _OptTimeout:
                    pass
                except Exception:
                    pass

                # CMA-ES — independent starting point
                if time.perf_counter() < varpro_deadline and best_cont_mse > early_mse:
                    cma_vp_dl = min(time.perf_counter() + vp_stage, varpro_deadline)
                    x0_cma = smart_x0(n_nl, nl_pos_idx, nl_small_idx, nl_rational_idx,
                                      exp_bounds=nl_exp_bounds,
                                      gaussian_hints=nl_gaussian)
                    try:
                        rp, rm = self._run_cma(ols_objective, x0_cma,
                                               nl_pos_idx, nl_small_idx, nl_rational_idx,
                                               exp_bounds=nl_exp_bounds,
                                               global_deadline=cma_vp_dl,
                                               early_mse=early_mse,
                                               gaussian_hints=nl_gaussian)
                        if rm < best_cont_mse:
                            best_cont_mse, best_cont_nl = rm, rp
                    except _OptTimeout:
                        pass
                    except Exception:
                        pass

                # DE — independent starting point
                if time.perf_counter() < varpro_deadline and best_cont_mse > early_mse:
                    de_vp_dl = min(time.perf_counter() + vp_stage, varpro_deadline)
                    x0_de = smart_x0(n_nl, nl_pos_idx, nl_small_idx, nl_rational_idx,
                                     exp_bounds=nl_exp_bounds,
                                     gaussian_hints=nl_gaussian)
                    try:
                        rp, rm = self._run_de(
                            ols_objective, x0_de, nl_pos_idx, nl_small_idx, nl_rational_idx,
                            exp_bounds=nl_exp_bounds,
                            global_deadline=de_vp_dl)
                        if rm < best_cont_mse:
                            best_cont_mse, best_cont_nl = rm, rp
                    except _OptTimeout:
                        pass
                    except Exception:
                        pass

        # ============================================================
        # Phase 3: Select best result (rational grid vs continuous search)
        # ============================================================
        best_nl: Optional[List[float]] = None
        best_nl_mse = float("inf")

        if best_rational_nl is not None and best_rational_mse < best_nl_mse:
            best_nl_mse, best_nl = best_rational_mse, list(best_rational_nl)

        if best_cont_nl is not None and best_cont_mse < best_nl_mse:
            best_nl_mse, best_cont_nl = best_cont_mse, list(best_cont_nl)
            best_nl = list(best_cont_nl)

        if best_nl is not None:
            for i in nl_rational_idx:
                best_nl[i] = _snap_rational(best_nl[i])

        if best_nl is None or best_nl_mse >= self.penalty:
            return None

        # ---- Final OLS to recover linear parameters ----
        final_ols_result = _ols_core(best_nl)
        if final_ols_result is None:
            return None
        _, final_mse, ols_coeffs = final_ols_result

        final_params = self._recover_varpro_params(
            param_names, linear_names, nonlinear_names,
            monomials, ols_coeffs, best_nl,
        )
        
        if final_params is None:
            return None
            
        return final_params, final_mse

    # ------------------------------------------------------------------ #
    # VARPRO parameter recovery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recover_varpro_params(
        param_names: List[str],
        linear_names: List[str],
        nonlinear_names: List[str],
        monomials: List[sp.Expr],
        ols_coeffs: np.ndarray,
        nl_values: List[float],
    ) -> Optional[List[float]]:
        solution: Dict[str, float] = {}

        for name, val in zip(nonlinear_names, nl_values):
            solution[name] = float(val)

        unsolved = set(linear_names)

        for mono, coeff in zip(monomials, ols_coeffs):
            name = str(mono)
            if name in unsolved:
                solution[name] = float(coeff)
                unsolved.discard(name)

        for _ in range(10):
            if not unsolved:
                break
            progress = False
            for mono, coeff in zip(monomials, ols_coeffs):
                remaining = [s for s in mono.free_symbols if str(s) in unsolved]
                if len(remaining) != 1:
                    continue
                sym = remaining[0]
                mono_sub = mono
                for name, val in solution.items():
                    mono_sub = mono_sub.subs(sp.Symbol(name), val)
                try:
                    sols = sp.solve(sp.Eq(mono_sub, float(coeff)), sym)
                    if sols:
                        val = complex(sols[0])
                        if abs(val.imag) < 1e-10:
                            solution[str(sym)] = val.real
                            unsolved.discard(str(sym))
                            progress = True
                except TimeoutError:
                    raise
                except Exception:
                    pass
            if not progress:
                break

        for name in unsolved:
            solution[name] = 1.0

        try:
            return [float(solution[name]) for name in param_names]
        except TimeoutError:
            raise
        except Exception:
            return None
