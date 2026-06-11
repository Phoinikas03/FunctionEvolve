"""CMA-ES optimizer."""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Union

import numpy as np
import sympy as sp

from .base import (
    BaseOptimizer, OptResult,
    parse_expr, compile_expr, build_mse_fn,
    detect_rational_constrained, make_safe_expr,
    param_hints, smart_x0, wrap_rational, _snap_rational,
)


class CMAOptimizer(BaseOptimizer):

    def __init__(self, *, sigma: float = 2.0, popsize: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
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
        from cmaes import CMA

        n_params = len(param_names)
        sympy_expr = skeleton if isinstance(skeleton, sp.Expr) else parse_expr(skeleton, param_names, feature_names)
        rational_idx = detect_rational_constrained(
            sympy_expr, param_names, feature_names, X_train)
        safe_expr = make_safe_expr(sympy_expr, param_names, rational_idx, feature_names)
        predict_fn = compile_expr(safe_expr, param_names, feature_names, X_train)
        raw_mse = build_mse_fn(predict_fn, y_train, self.penalty)
        objective = wrap_rational(raw_mse, rational_idx)

        pos_idx, small_idx = param_hints(sympy_expr, param_names)

        best_mse = float("inf")
        best_params = np.zeros(n_params)
        total_feval = 0
        deadline = time.perf_counter() + self.timeout

        y_var = float(np.var(y_train))
        _early_mse = 1e-10 * max(y_var, 1e-30)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.n_restarts):
                x0 = smart_x0(n_params, pos_idx, small_idx, rational_idx)
                cma = CMA(
                    mean=x0.astype(float),
                    sigma=self.sigma,
                    population_size=self.popsize,
                )
                for _ in range(self.max_iter):
                    if time.perf_counter() >= deadline:
                        break
                    solutions = []
                    for _ in range(cma.population_size):
                        x = cma.ask()
                        val = objective(x)
                        total_feval += 1
                        solutions.append((x, val))
                        if val < best_mse:
                            best_mse, best_params = val, x.copy()
                    cma.tell(solutions)
                    if best_mse < _early_mse or cma.should_stop():
                        break
                if time.perf_counter() >= deadline or best_mse < _early_mse:
                    break

        for i in rational_idx:
            best_params[i] = _snap_rational(best_params[i])

        return OptResult(best_params, best_mse, total_feval,
                         rational_idx=list(rational_idx), safe_expr=safe_expr)
