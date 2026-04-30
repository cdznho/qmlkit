"""Fast NumPy simulation helpers used by qmlkit core objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmlkit.core.backend import Backend


class SimulatorBackend(Backend):
    """Lightweight CPU backend for quantum-inspired feature simulation.

    The simulator is intentionally classical. It provides vectorized feature
    maps that mimic some useful properties of quantum embeddings without
    constructing circuits or requiring quantum dependencies.
    """

    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Run a callable circuit-like object.

        Parameters
        ----------
        circuit:
            A callable that accepts a parameter mapping. Circuit DSL execution
            is outside the v0.1 scope.
        params:
            Optional parameter mapping passed to the callable.
        """

        if callable(circuit):
            return circuit(params or {})
        raise NotImplementedError(
            "SimulatorBackend.run only supports callables in qmlkit v0.1. "
            "Use the feature_map helpers for built-in simulations."
        )

    def iqp_features(self, X: ArrayLike, scale: float = 1.0) -> NDArray[np.float64]:
        """Return deterministic IQP-inspired trigonometric features."""

        X_arr = _as_2d_float(X)
        Z = scale * X_arr
        interactions = _pairwise_products(Z, include_diagonal=False)
        blocks = [np.sin(Z), np.cos(Z)]
        if interactions.shape[1] > 0:
            blocks.extend([np.sin(interactions), np.cos(interactions)])
        return np.concatenate(blocks, axis=1)

    def random_features(
        self,
        X: ArrayLike,
        weights: NDArray[np.float64],
        bias: NDArray[np.float64],
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Return random Fourier-style nonlinear projection features."""

        X_arr = _as_2d_float(X)
        projection = (scale * X_arr) @ weights + bias
        return np.sqrt(2.0 / weights.shape[1]) * np.cos(projection)

    def tensor_features(self, X: ArrayLike, scale: float = 1.0) -> NDArray[np.float64]:
        """Return compact tensor-product-inspired polynomial features."""

        X_arr = _as_2d_float(X)
        Z = scale * X_arr
        pairwise = _pairwise_products(Z, include_diagonal=False)
        blocks = [Z, Z**2]
        if pairwise.shape[1] > 0:
            blocks.append(pairwise)
        return np.concatenate(blocks, axis=1)


def _as_2d_float(X: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array-like input.")
    if arr.shape[0] == 0:
        raise ValueError("Expected at least one sample.")
    return np.ascontiguousarray(arr)


def _pairwise_products(
    X: NDArray[np.float64], include_diagonal: bool = False
) -> NDArray[np.float64]:
    n_features = X.shape[1]
    cols: list[NDArray[np.float64]] = []
    for i in range(n_features):
        start = i if include_diagonal else i + 1
        for j in range(start, n_features):
            cols.append((X[:, i] * X[:, j])[:, None])
    if not cols:
        return np.empty((X.shape[0], 0), dtype=np.float64)
    return np.concatenate(cols, axis=1)
