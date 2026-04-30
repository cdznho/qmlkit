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


@pytest.mark.parametrize("backend_cls", [QiskitBackend, PennyLaneBackend])
def test_placeholder_backends_raise_clear_errors(backend_cls):
    with pytest.raises(NotImplementedError, match="placeholder"):
        backend_cls().run(None)
