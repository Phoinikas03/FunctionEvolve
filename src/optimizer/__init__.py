"""Unified optimizer package.

Usage::

    from src.optimizer import get_optimizer, OptResult

    opt = get_optimizer("Structure", de_timeout=30)
    result: OptResult = opt.optimize(skeleton, param_names, feature_names, X, y)
"""

from .base import BaseOptimizer, OptResult
from .de import DEOptimizer
from .cma import CMAOptimizer
from .lbfgs import LBFGSOptimizer
from .least_squares import LeastSquaresOptimizer
from .structure import StructureOptimizer

_REGISTRY = {
    "DE": DEOptimizer,
    "CMA-ES": CMAOptimizer,
    "L-BFGS-B": LBFGSOptimizer,
    "least_squares": LeastSquaresOptimizer,
    "Structure": StructureOptimizer,
}

OPTIMIZER_NAMES = list(_REGISTRY.keys())


def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """Create an optimizer instance by name; extra keyword arguments are passed to the constructor."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {', '.join(_REGISTRY)}")
    return cls(**kwargs)


__all__ = [
    "BaseOptimizer", "OptResult",
    "DEOptimizer", "CMAOptimizer", "LBFGSOptimizer",
    "LeastSquaresOptimizer", "StructureOptimizer",
    "get_optimizer", "OPTIMIZER_NAMES",
]
