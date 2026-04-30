"""Simple benchmark runner for qmlkit models."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from qmlkit import QuantumKernel
from qmlkit.benchmarks.datasets import load_toy_dataset


def run_benchmark(
    dataset: Literal["moons", "classification"] = "moons",
    n_samples: int = 300,
    random_state: int = 0,
    include_torch: bool = True,
) -> dict[str, Any]:
    """Compare standard SVM, qmlkit kernel SVM, and a small MLP.

    Returns a plain dictionary with accuracy and runtime for each model.
    """

    _require_sklearn()
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    data = load_toy_dataset(
        kind=dataset, n_samples=n_samples, random_state=random_state
    )
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results: dict[str, Any] = {
        "dataset": dataset,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "models": {},
    }

    svm = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(gamma="scale")),
    ])
    results["models"]["svm"] = _time_model(
        svm, X_train, y_train, X_test, y_test, accuracy_score
    )

    qsvm = Pipeline([
        ("scale", StandardScaler()),
        ("embed", QuantumKernel(method="iqp", random_state=random_state)),
        ("clf", SVC(gamma="scale")),
    ])
    results["models"]["quantum_kernel_svm"] = _time_model(
        qsvm, X_train, y_train, X_test, y_test, accuracy_score
    )

    if include_torch:
        results["models"]["mlp"] = _run_torch_mlp(
            X_train, y_train, X_test, y_test, random_state=random_state
        )

    return results


def _time_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    accuracy_score: Any,
) -> dict[str, float]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    runtime = time.perf_counter() - start
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "runtime_sec": float(runtime),
    }


def _run_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> dict[str, float | str]:
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover - depends on environment.
        return {"skipped": "PyTorch is not installed."}

    start = time.perf_counter()
    torch.manual_seed(random_state)
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    Xte = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(Xte).argmax(dim=1).cpu().numpy()
    runtime = time.perf_counter() - start
    accuracy = float((predictions == y_test).mean())
    return {"accuracy": accuracy, "runtime_sec": float(runtime)}


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "run_benchmark requires scikit-learn. Install with "
            "`pip install qmlkit[sklearn]` or `pip install scikit-learn`."
        ) from exc
