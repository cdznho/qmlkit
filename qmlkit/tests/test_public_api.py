import sys
import types
from typing import Any, cast

import numpy as np
import pytest

import qmlkit
from qmlkit import PennyLaneBackend, QiskitBackend, SimulatorBackend


def test_top_level_public_exports():
    expected = {
        "Backend",
        "HybridLayer",
        "PennyLaneBackend",
        "QiskitBackend",
        "QuantumFeatureMap",
        "QuantumKernel",
        "SimulatorBackend",
    }
    assert expected.issubset(set(qmlkit.__all__))


def test_simulator_backend_runs_callable():
    backend = SimulatorBackend()
    result = backend.run(lambda params: params["value"] + 1, {"value": 41})
    assert result == 42


def test_qiskit_backend_clear_error_when_dependency_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "qiskit", None)
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", None)
    with pytest.raises(ImportError, match="requires Qiskit"):
        QiskitBackend().run(object())


def test_qiskit_backend_runs_statevector_probabilities_with_fake_module(monkeypatch):
    quantum_info = types.ModuleType("qiskit.quantum_info")

    class FakeStatevector:
        @classmethod
        def from_instruction(cls, instruction):
            assert instruction.assigned == {"theta": 0.5}
            assert instruction.measurements_removed is True
            return cls()

        def probabilities_dict(self, qargs=None, decimals=None):
            assert qargs is None
            assert decimals == 3
            return {"0": 0.5, "1": 0.5}

    class FakeParameter:
        def __init__(self, name):
            self.name = name

    class FakeCircuit:
        def __init__(self):
            self.assigned = {}
            self.measurements_removed = False
            self.parameters = [FakeParameter("theta")]

        def assign_parameters(self, parameters, inplace=False):
            assert inplace is False
            assigned = FakeCircuit()
            assigned.assigned = {parameter.name: value for parameter, value in parameters.items()}
            return assigned

        def remove_final_measurements(self, inplace=False):
            assert inplace is False
            self.measurements_removed = True
            return self

    cast(Any, quantum_info).Statevector = FakeStatevector
    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", quantum_info)

    result = QiskitBackend().run(FakeCircuit(), {"theta": 0.5, "decimals": 3})
    assert result == {"0": 0.5, "1": 0.5}


def test_qiskit_backend_samples_counts_with_fake_module(monkeypatch):
    quantum_info = types.ModuleType("qiskit.quantum_info")

    class FakeStatevector:
        @classmethod
        def from_instruction(cls, instruction):
            return cls()

        def probabilities_dict(self, qargs=None, decimals=None):
            return {"0": 1.0, "1": 0.0}

    cast(Any, quantum_info).Statevector = FakeStatevector
    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    monkeypatch.setitem(sys.modules, "qiskit.quantum_info", quantum_info)

    result = QiskitBackend(return_type="counts", shots=32, seed=7).run(object())
    assert result == {"0": 32}


def test_pennylane_backend_clear_error_when_dependency_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pennylane", None)
    with pytest.raises(ImportError, match="requires PennyLane"):
        PennyLaneBackend(wires=1).run(lambda: None)


def test_pennylane_backend_wraps_quantum_function_with_fake_module(monkeypatch):
    qml = types.ModuleType("pennylane")
    calls: dict[str, Any] = {}

    def device(name, wires, shots=None, **kwargs):
        calls["device"] = (name, wires, shots, kwargs)
        return {"name": name, "wires": wires}

    def qnode(fn, dev, interface="auto", diff_method="best"):
        calls["qnode"] = (dev, interface, diff_method)

        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped

    class Math:
        @staticmethod
        def toarray(value):
            return np.asarray(value)

    qml_any = cast(Any, qml)
    qml_any.device = device
    qml_any.qnode = qnode
    qml_any.math = Math()
    monkeypatch.setitem(sys.modules, "pennylane", qml)

    def circuit(theta):
        return np.array([np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2])

    result = PennyLaneBackend(wires=1).run(circuit, {"theta": 0.0})
    np.testing.assert_allclose(result, np.array([1.0, 0.0]))
    assert calls["device"] == ("default.qubit", 1, None, {})
    assert calls["qnode"][1:] == ("auto", "best")


def test_pennylane_backend_runs_existing_qnode_like_callable(monkeypatch):
    qml = types.ModuleType("pennylane")

    class Math:
        @staticmethod
        def toarray(value):
            return np.asarray(value)

    cast(Any, qml).math = Math()
    monkeypatch.setitem(sys.modules, "pennylane", qml)

    class FakeQNode:
        device = object()
        qtape = object()

        def __call__(self, value):
            return value

    assert PennyLaneBackend().run(FakeQNode(), {"args": [0.25]}) == 0.25
