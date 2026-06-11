"""
Constant optimization and evaluator.

After refactoring, the Evaluator is only responsible for:
  1. Calling optimizer.optimize() to obtain OptResult
  2. Computing NMSE on test / ood datasets

All optimization logic has been moved to the optimizer/ directory.
"""

from __future__ import annotations

import warnings
from typing import List, NamedTuple, Optional

import numpy as np
import sympy as sp

from src.optimizer.base import (
    _SAFE_MODULES,
)
from src.optimizer import BaseOptimizer, StructureOptimizer, OptResult, get_optimizer


class EvalResult(NamedTuple):
    """Complete result of a single formula evaluation (NMSE = MSE / Var(Y))."""
    filled_expr: Optional[sp.Expr]
    best_params: List[float]
    train_nmse: float
    test_nmse: float
    ood_test_nmse: float
    fail_reason: str = ""


class Evaluator:
    """
    Constant optimization and evaluator.
    Fits optimal values of c0, c1... in the formula on the training set,
    then computes NMSE on train / test / ood_test datasets using the same optimal constants.
    """

    def __init__(
        self,
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_ood_test: Optional[np.ndarray] = None,
        y_ood_test: Optional[np.ndarray] = None,
        penalty_value: float = 1e10,
        optimizer: str = "Structure",
        **optimizer_kwargs,
    ):
        self.feature_names = feature_names
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test if X_test is not None else np.array([])
        self.y_test = y_test if y_test is not None else np.array([])
        self.X_ood_test = X_ood_test if X_ood_test is not None else np.array([])
        self.y_ood_test = y_ood_test if y_ood_test is not None else np.array([])
        self.penalty_value = penalty_value

        self._var_train = float(np.var(y_train)) if y_train.size > 0 else 1.0
        self._var_test = float(np.var(self.y_test)) if self.y_test.size > 0 else 1.0
        self._var_ood = float(np.var(self.y_ood_test)) if self.y_ood_test.size > 0 else 1.0

        optimizer_kwargs.setdefault("penalty", penalty_value)
        self._optimizer = get_optimizer(optimizer, **optimizer_kwargs)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def evaluate_skeleton(
        self,
        expr: sp.Expr,
        param_names: List[str],
        parent_params: Optional[List[float]] = None,
    ) -> EvalResult:
        def fail(reason: str) -> EvalResult:
            return EvalResult(
                None,
                [],
                float("inf"),
                float("inf"),
                float("inf"),
                reason,
            )

        if not param_names:
            train_mse = self._compute_mse(expr, self.X_train, self.y_train)
            if train_mse == float("inf"):
                return fail("numeric_invalid")
            test_mse = self._compute_mse(expr, self.X_test, self.y_test)
            ood_mse = self._compute_mse(expr, self.X_ood_test, self.y_ood_test)
            return EvalResult(
                expr, [],
                self._mse_to_nmse(train_mse, self._var_train),
                self._mse_to_nmse(test_mse, self._var_test),
                self._mse_to_nmse(ood_mse, self._var_ood),
            )

        try:
            opt_result = self._optimizer.optimize(
                skeleton=expr,
                param_names=list(param_names),
                feature_names=self.feature_names,
                X_train=self.X_train,
                y_train=self.y_train,
                parent_params=parent_params,
            )
        except TimeoutError:
            raise
        except Exception:
            return fail("optimizer_exception")

        best_mse = opt_result.best_mse
        best_params = list(opt_result.best_params)

        if best_mse >= self.penalty_value:
            return fail("optimizer_penalty")

        try:
            filled_expr = opt_result.safe_expr
            for name, val in zip(param_names, best_params):
                filled_expr = filled_expr.subs(sp.Symbol(name), sp.Float(val))
        except TimeoutError:
            raise
        except Exception:
            return fail("postprocess_exception")

        test_mse = self._compute_mse(filled_expr, self.X_test, self.y_test)
        ood_mse = self._compute_mse(filled_expr, self.X_ood_test, self.y_ood_test)

        return EvalResult(
            filled_expr, best_params,
            self._mse_to_nmse(best_mse, self._var_train),
            self._mse_to_nmse(test_mse, self._var_test),
            self._mse_to_nmse(ood_mse, self._var_ood),
        )

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mse_to_nmse(mse: float, var_y: float) -> float:
        if mse == float("inf") or var_y <= 0:
            return float("inf")
        nmse = mse / var_y
        return nmse if np.isfinite(nmse) else float("inf")

    def _compute_mse(
        self, sympy_expr: sp.Expr, X: np.ndarray, y: np.ndarray,
    ) -> float:
        if X.size == 0 or y.size == 0:
            return float("inf")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                feature_symbols = [sp.Symbol(n) for n in self.feature_names]
                func = sp.lambdify(feature_symbols, sympy_expr, modules=_SAFE_MODULES)
                X_cols = [X[:, i] for i in range(len(self.feature_names))]
                y_pred = np.asarray(func(*X_cols), dtype=float)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return float("inf")
                sq = (y_pred - y) ** 2
                if np.any(np.isinf(sq)):
                    return float("inf")
                mse = float(np.mean(sq))
                return mse if np.isfinite(mse) else float("inf")
            except TimeoutError:
                raise
            except Exception:
                return float("inf")
