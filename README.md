# qmlkit

`qmlkit` is a quantum-inspired and hybrid ML SDK for AI engineers.

It is not a quantum physics SDK and it does not require quantum hardware. The
goal is to give developers simple, composable ML abstractions that feel natural
in PyTorch and scikit-learn workflows while leaving room for future quantum
backend adapters.

> Status: public alpha. qmlkit is useful for experimentation, demos, and early
> feedback. It does not make quantum advantage claims.

## What is included in v0.1

- `QuantumFeatureMap`: a scikit-learn-style transformer for nonlinear quantum-inspired feature maps.
- `QuantumKernel`: compatibility alias for the v0.1 sklearn API.
- `HybridLayer`: a differentiable PyTorch layer with simulated quantum-like rotations and mixing.
- `SimulatorBackend`: a lightweight NumPy backend for CPU execution.
- Placeholder `QiskitBackend` and `PennyLaneBackend` classes for future integrations.
- Toy benchmark utilities and two runnable examples.

## Install

Core install:

```bash
pip install .
```

With scikit-learn examples and benchmarks:

```bash
pip install ".[sklearn]"
```

With PyTorch support:

```bash
pip install ".[torch]"
```

For development:

```bash
pip install ".[dev]"
```

## Example: sklearn pipeline

```python
from qmlkit import QuantumKernel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

model = Pipeline([
    ("embed", QuantumKernel(method="iqp")),
    ("clf", SVC()),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

`QuantumKernel` supports:

- `"iqp"`: deterministic trigonometric interaction features.
- `"random"`: random Fourier-style nonlinear projections.
- `"tensor"`: compact tensor-product-inspired features.

You can also compute a Gram matrix directly:

```python
kernel = QuantumKernel(method="iqp").fit(X_train)
K_train = kernel.kernel_matrix(X_train)
K_test = kernel.kernel_matrix(X_test, X_train)
```

For new code, `QuantumFeatureMap` is the clearer name for the same explicit
feature-map API:

```python
from qmlkit import QuantumFeatureMap

features = QuantumFeatureMap(method="tensor").fit_transform(X_train)
```

## Example: PyTorch layer

```python
import torch
from torch import nn
from qmlkit import HybridLayer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = HybridLayer(n_qubits=8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.q(x))


model = Model()
out = model(torch.randn(16, 4))
```

## Run examples

```bash
python -m qmlkit.examples.kernel_svm
python -m qmlkit.examples.hybrid_nn
```

## Run benchmarks

```python
from qmlkit.benchmarks.runner import run_benchmark

results = run_benchmark(dataset="moons")
print(results)
```

## Run tests

```bash
python3 -m pytest qmlkit/tests
```

## Project quality checks

```bash
python -m pytest
python -m ruff check .
python -m mypy qmlkit
```

## Non-goals

`qmlkit` does not build a quantum circuit DSL, execute on real quantum hardware,
or compete with low-level frameworks such as Qiskit, PennyLane, or Cirq. Those
projects are excellent circuit-level tools. `qmlkit` sits above that layer and
focuses on ML-friendly abstractions.

## Contributing

Contributions are welcome once the repository is public. The most useful early
work is example notebooks, integrations with existing ML pipelines, and honest
benchmarks against classical baselines.
