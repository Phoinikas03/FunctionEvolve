"""Differential Evolution optimizer."""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Union

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution

from .base import (
    BaseOptimizer, OptResult, _OptTimeout,
    parse_expr, compile_expr, build_mse_fn,
    detect_rational_constrained, make_safe_expr,
    param_hints, smart_x0, make_bounds, wrap_rational, _snap_rational,
)


class DEOptimizer(BaseOptimizer):

    def __init__(self, *, popsize: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.popsize = popsize

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
        sympy_expr = skeleton if isinstance(skeleton, sp.Expr) else parse_expr(skeleton, param_names, feature_names)
        rational_idx = detect_rational_constrained(
            sympy_expr, param_names, feature_names, X_train)
        safe_expr = make_safe_expr(sympy_expr, param_names, rational_idx, feature_names)
        predict_fn = compile_expr(safe_expr, param_names, feature_names, X_train)
        raw_mse = build_mse_fn(predict_fn, y_train, self.penalty)
        objective = wrap_rational(raw_mse, rational_idx)

        pos_idx, small_idx = param_hints(sympy_expr, param_names)
        bounds = make_bounds(n_params, self.bound, pos_idx, small_idx, rational_idx)

        best_mse = float("inf")
        best_params = np.zeros(n_params)
        total_feval = 0
        deadline = time.perf_counter() + self.timeout

        y_var = float(np.var(y_train))
        _early_mse = 1e-10 * max(y_var, 1e-30)

        def timed_objective(p):
            if time.perf_counter() >= deadline:
                raise _OptTimeout
            return objective(p)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.n_restarts):
                x0 = smart_x0(n_params, pos_idx, small_idx, rational_idx)
                try:
                    res = differential_evolution(
                        timed_objective, bounds=bounds,
                        maxiter=self.max_iter, seed=None, tol=1e-8,
                        popsize=self.popsize, x0=x0,
                    )
                    total_feval += res.nfev
                    if res.fun < best_mse:
                        best_mse = float(res.fun)
                        best_params = res.x.copy()
                except _OptTimeout:
                    break
                except TimeoutError:
                    raise
                except Exception:
                    pass
                if best_mse < _early_mse:
                    break

        for i in rational_idx:
            best_params[i] = _snap_rational(best_params[i])

        return OptResult(best_params, best_mse, total_feval,
                         rational_idx=list(rational_idx), safe_expr=safe_expr)
