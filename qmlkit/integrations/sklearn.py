"""scikit-learn integration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qmlkit.core.kernel import QuantumFeatureMap, QuantumKernel

if TYPE_CHECKING:

    class _BaseEstimator:
        """Type-checking fallback sklearn base class."""

    class _TransformerMixin:
        """Type-checking fallback sklearn transformer mixin."""

else:
    try:  # pragma: no cover - class shape depends on optional dependency.
        from sklearn.base import BaseEstimator as _BaseEstimator
        from sklearn.base import TransformerMixin as _TransformerMixin

    except ImportError:  # pragma: no cover - depends on environment.

        class _BaseEstimator:
            """Fallback sklearn base class."""

        class _TransformerMixin:
            """Fallback sklearn transformer mixin."""


class SklearnQuantumFeatureMap(QuantumFeatureMap, _BaseEstimator, _TransformerMixin):
    """QuantumFeatureMap variant with sklearn mixins when sklearn is installed."""


SklearnQuantumKernel = SklearnQuantumFeatureMap
QuantumKernelTransformer = SklearnQuantumFeatureMap

__all__ = [
    "QuantumFeatureMap",
    "QuantumKernel",
    "QuantumKernelTransformer",
    "SklearnQuantumFeatureMap",
    "SklearnQuantumKernel",
]
