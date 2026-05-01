# qmlkit

**Quantum-inspired ML primitives for PyTorch and scikit-learn. Classical by default.**

`qmlkit` is a small Python SDK for AI engineers who want to experiment with
quantum-inspired kernels, feature maps, and hybrid neural layers without
learning circuit DSLs or needing quantum hardware.

It is intentionally ML-first:

- use it inside normal `sklearn` pipelines;
- drop `HybridLayer` into a PyTorch model;
- run everything locally on CPU by default;
- optionally run small Qiskit and PennyLane circuits through local simulators.

> Status: public alpha. qmlkit is useful for experimentation, demos, and early
> feedback. It does not make quantum advantage claims.

## Quick Start

Install from GitHub:

```bash
pip install "qmlkit @ git+https://github.com/cdznho/qmlkit.git"
```

Use a quantum-inspired feature map in a standard sklearn classifier:

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

Use a simulated hybrid layer in PyTorch:

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

## Why qmlkit?

Most quantum ML tooling starts at the circuit level. That is powerful, but it
can feel far away from everyday ML workflows.

qmlkit starts where ML engineers already work:

- `fit`, `transform`, and `Pipeline` for sklearn users;
- `nn.Module` for PyTorch users;
- NumPy simulation for zero-hardware experimentation;
- optional Qiskit and PennyLane adapters when you want circuit-level interop.

The goal is not to replace Qiskit or PennyLane. The goal is to provide a higher
level ML abstraction that can eventually sit above them.

## What Is Included In v0.1

| API | What it does |
| --- | --- |
| `QuantumFeatureMap` | sklearn-style transformer for nonlinear quantum-inspired feature maps. |
| `QuantumKernel` | Compatibility alias for the v0.1 sklearn API. |
| `HybridLayer` | Differentiable PyTorch layer with simulated quantum-like rotations and mixing. |
| `SimulatorBackend` | Lightweight NumPy backend for local CPU execution. |
| `QiskitBackend` | Optional local Qiskit statevector execution for circuits and probabilities. |
| `PennyLaneBackend` | Optional local PennyLane execution for QNodes and quantum functions. |
| `qmlkit.benchmarks` | Toy datasets and honest baseline comparisons. |

Feature-map methods:

| Method | Description |
| --- | --- |
| `"iqp"` | Deterministic trigonometric interaction features. |
| `"random"` | Random Fourier-style nonlinear projections. |
| `"tensor"` | Compact tensor-product-inspired feature expansion. |

## Installation Options

Core package only:

```bash
pip install "qmlkit @ git+https://github.com/cdznho/qmlkit.git"
```

With scikit-learn support:

```bash
pip install "qmlkit[sklearn] @ git+https://github.com/cdznho/qmlkit.git"
```

With PyTorch support:

```bash
pip install "qmlkit[torch] @ git+https://github.com/cdznho/qmlkit.git"
```

With Qiskit and PennyLane adapters:

```bash
pip install "qmlkit[quantum] @ git+https://github.com/cdznho/qmlkit.git"
```

For local development:

```bash
git clone https://github.com/cdznho/qmlkit.git
cd qmlkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## More Examples

Compute explicit features:

```python
from qmlkit import QuantumFeatureMap

features = QuantumFeatureMap(method="tensor").fit_transform(X_train)
```

Compute a Gram matrix:

```python
from qmlkit import QuantumKernel

kernel = QuantumKernel(method="iqp").fit(X_train)
K_train = kernel.kernel_matrix(X_train)
K_test = kernel.kernel_matrix(X_test, X_train)
```

Run bundled examples:

```bash
python -m qmlkit.examples.kernel_svm
python -m qmlkit.examples.hybrid_nn
```

Run a small benchmark:

```python
from qmlkit.benchmarks.runner import run_benchmark

results = run_benchmark(dataset="moons")
print(results)
```

Run a Qiskit circuit locally:

```python
from qiskit import QuantumCircuit
from qmlkit import QiskitBackend

qc = QuantumCircuit(1)
qc.h(0)

probabilities = QiskitBackend().run(qc)
print(probabilities)  # {"0": 0.5, "1": 0.5}
```

Run a PennyLane quantum function locally:

```python
import pennylane as qml
from qmlkit import PennyLaneBackend


def circuit(theta):
    qml.RX(theta, wires=0)
    return qml.probs(wires=0)


probabilities = PennyLaneBackend(wires=1).run(circuit, {"theta": 0.0})
print(probabilities)  # [1. 0.]
```

## Development

Run tests:

```bash
python -m pytest
```

Run quality checks:

```bash
python -m ruff check .
python -m mypy qmlkit
python -m build
```

## Roadmap

- Add example notebooks with honest wins and failures.
- Improve benchmark coverage across more classical baselines.
- Add richer sklearn utilities around precomputed kernels.
- Expand Qiskit and PennyLane adapters beyond local simulation.
- Publish stable docs once the v0.1 API gets user feedback.

## Non-Goals

qmlkit does not:

- build a quantum circuit DSL;
- execute on real quantum hardware in v0.1;
- claim quantum advantage;
- compete directly with Qiskit, PennyLane, or Cirq.

Those projects are excellent circuit-level tools. qmlkit focuses on simple,
developer-friendly ML abstractions above that layer.

## Contributing

Contributions are welcome. The most useful early work is:

- example notebooks;
- integrations with existing ML workflows;
- honest benchmarks against classical baselines;
- API feedback from people actually trying to build with it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development checks.
