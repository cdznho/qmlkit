"""Quantum-inspired kernels and feature maps."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmlkit.core.simulator import SimulatorBackend


class QuantumFeatureMap:
    """scikit-learn-style quantum-inspired feature transformer.

    Parameters
    ----------
    method:
        Feature map to use. Supported values are ``"iqp"``, ``"random"``,
        and ``"tensor"``.
    n_components:
        Optional output feature count. If omitted, each method chooses a
        practical default based on the input dimensionality.
    random_state:
        Seed used by the ``"random"`` feature map.
    scale:
        Multiplicative input scale applied before nonlinear transformations.
    cache:
        Whether to cache transformed feature matrices by content hash.
    """

    _VALID_METHODS = {"iqp", "random", "tensor"}

    def __init__(
        self,
        method: str = "iqp",
        n_components: int | None = None,
        random_state: int | None = None,
        scale: float = 1.0,
        cache: bool = True,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.scale = scale
        self.cache = cache

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> QuantumFeatureMap:
        """Fit transformer state from input data.

        The deterministic methods only record the input width. The random
        method samples and stores projection parameters.
        """

        del y
        self._validate_hyperparameters()
        X_arr = self._validate_X(X)
        self.n_features_in_ = X_arr.shape[1]
        self.backend_ = SimulatorBackend()
        self._cache_: dict[str, NDArray[np.float64]] = {}

        if self.method == "random":
            output_dim = self.n_components or max(8, 2 * self.n_features_in_)
            rng = np.random.default_rng(self.random_state)
            self.random_weights_ = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(self.n_features_in_, output_dim),
            )
            self.random_bias_ = rng.uniform(0.0, 2.0 * np.pi, size=output_dim)
            self.n_output_features_ = output_dim
        else:
            raw = self._raw_feature_map(X_arr)
            self.n_output_features_ = self.n_components or raw.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform input samples into quantum-inspired explicit features."""

        self._check_is_fitted()
        X_arr = self._validate_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input feature width does not match fit data: "
                f"expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )

        key = self._cache_key(X_arr)
        if self.cache and key in self._cache_:
            return self._cache_[key].copy()

        features = self._raw_feature_map(X_arr)
        features = self._resize_features(features, self.n_output_features_)
        features = np.ascontiguousarray(features, dtype=np.float64)

        if self.cache:
            self._cache_[key] = features.copy()
        return features

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit the transformer and return transformed features."""

        return self.fit(X, y).transform(X)

    def kernel_matrix(
        self, X: ArrayLike, Y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Compute a dot-product Gram matrix from transformed features.

        If the transformer is not already fitted, it is fitted on ``X``.
        """

        if not self._is_fitted():
            self.fit(X)
        X_features = self.transform(X)
        Y_features = X_features if Y is None else self.transform(Y)
        return X_features @ Y_features.T

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-compatible estimator parameters."""

        del deep
        return {
            "method": self.method,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "scale": self.scale,
            "cache": self.cache,
        }

    def set_params(self, **params: Any) -> QuantumFeatureMap:
        """Set sklearn-compatible estimator parameters."""

        valid = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for QuantumKernel.")
            setattr(self, key, value)
        for fitted_name in (
            "n_features_in_",
            "n_output_features_",
            "backend_",
            "random_weights_",
            "random_bias_",
            "_cache_",
        ):
            if hasattr(self, fitted_name):
                delattr(self, fitted_name)
        return self

    def _raw_feature_map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.method == "iqp":
            return self.backend_.iqp_features(X, scale=float(self.scale))
        if self.method == "tensor":
            return self.backend_.tensor_features(X, scale=float(self.scale))
        if self.method == "random":
            return self.backend_.random_features(
                X,
                weights=self.random_weights_,
                bias=self.random_bias_,
                scale=float(self.scale),
            )
        raise ValueError(f"Unsupported method {self.method!r}.")

    def _resize_features(
        self, features: NDArray[np.float64], output_dim: int
    ) -> NDArray[np.float64]:
        if features.shape[1] == output_dim:
            return features
        if features.shape[1] > output_dim:
            indices = np.linspace(0, features.shape[1] - 1, output_dim, dtype=int)
            return features[:, indices]

        blocks = [features]
        current = features.shape[1]
        step = 1
        while current < output_dim:
            needed = output_dim - current
            block = np.sin((step + 1) * features)
            blocks.append(block[:, :needed])
            current += min(needed, block.shape[1])
            step += 1
        return np.concatenate(blocks, axis=1)

    def _cache_key(self, X: NDArray[np.float64]) -> str:
        h = hashlib.sha1()
        h.update(str(X.shape).encode("utf-8"))
        h.update(str(X.dtype).encode("utf-8"))
        h.update(X.tobytes())
        h.update(repr(self.get_params(deep=False)).encode("utf-8"))
        if self.method == "random":
            h.update(self.random_weights_.tobytes())
            h.update(self.random_bias_.tobytes())
        return h.hexdigest()

    def _validate_hyperparameters(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"Unsupported method {self.method!r}. "
                f"Expected one of {sorted(self._VALID_METHODS)}."
            )
        if self.n_components is not None and self.n_components <= 0:
            raise ValueError("n_components must be a positive integer or None.")
        if not np.isfinite(float(self.scale)):
            raise ValueError("scale must be finite.")

    def _validate_X(self, X: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D array-like input.")
        if arr.shape[0] == 0:
            raise ValueError("Expected at least one sample.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input contains NaN or infinite values.")
        return np.ascontiguousarray(arr)

    def _is_fitted(self) -> bool:
        return hasattr(self, "n_features_in_")

    def _check_is_fitted(self) -> None:
        if not self._is_fitted():
            raise RuntimeError(f"{self.__class__.__name__} is not fitted. Call fit first.")


class QuantumKernel(QuantumFeatureMap):
    """Backward-compatible name for :class:`QuantumFeatureMap`.

    In qmlkit v0.1 the sklearn pipeline object exposes explicit nonlinear
    features. ``QuantumKernel`` remains the user-facing import requested by the
    MVP spec, while ``QuantumFeatureMap`` is the clearer name for new code.
    """


__all__ = ["QuantumFeatureMap", "QuantumKernel"]
