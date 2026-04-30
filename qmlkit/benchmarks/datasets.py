"""Toy dataset loaders used by qmlkit benchmarks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Dataset = dict[str, NDArray[np.float64] | NDArray[np.int_]]


def load_moons(
    n_samples: int = 300,
    noise: float = 0.22,
    random_state: int = 0,
    test_size: float = 0.25,
) -> Dataset:
    """Load a two-class moons dataset split into train and test arrays."""

    _require_sklearn()
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return _split(X, y, test_size=test_size, random_state=random_state)


def load_classification(
    n_samples: int = 300,
    n_features: int = 8,
    random_state: int = 0,
    test_size: float = 0.25,
) -> Dataset:
    """Load a compact synthetic classification dataset."""

    _require_sklearn()
    from sklearn.datasets import make_classification

    informative = max(2, n_features // 2)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )
    return _split(X, y, test_size=test_size, random_state=random_state)


def load_toy_dataset(
    kind: Literal["moons", "classification"] = "moons",
    n_samples: int = 300,
    noise: float = 0.22,
    n_features: int = 8,
    random_state: int = 0,
    test_size: float = 0.25,
) -> Dataset:
    """Load a named toy dataset."""

    if kind == "moons":
        return load_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
            test_size=test_size,
        )
    if kind == "classification":
        return load_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            test_size=test_size,
        )
    raise ValueError("kind must be 'moons' or 'classification'.")


def _split(
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    test_size: float,
    random_state: int,
) -> Dataset:
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return {
        "X_train": np.asarray(X_train, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float64),
        "y_train": np.asarray(y_train, dtype=np.int_),
        "y_test": np.asarray(y_test, dtype=np.int_),
    }


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "Benchmark datasets require scikit-learn. Install with "
            "`pip install qmlkit[sklearn]` or `pip install scikit-learn`."
        ) from exc
