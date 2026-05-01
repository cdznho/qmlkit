"""Backend interfaces for simulated and optional quantum-framework execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Backend(ABC):
    """Abstract execution backend.

    Backends intentionally expose a tiny interface in v0.1. Higher-level
    qmlkit objects should depend on this method instead of importing a
    specific external quantum framework.
    """

    @abstractmethod
    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Run a circuit-like object with optional parameters."""


class QiskitBackend(Backend):
    """Run Qiskit circuits with local statevector simulation.

    This adapter intentionally avoids hardware execution in v0.1. It accepts a
    ``qiskit.QuantumCircuit`` or instruction, optionally binds parameters, and
    returns probabilities, counts, or the raw ``Statevector``.

    Parameters
    ----------
    return_type:
        Default output type. One of ``"probabilities"``, ``"counts"``, or
        ``"statevector"``.
    shots:
        Number of samples to draw when returning ``"counts"``. If omitted,
        counts use 1024 shots.
    seed:
        Optional random seed for count sampling.
    strip_measurements:
        Whether to remove final measurements before statevector simulation.
    """

    _RETURN_TYPES = {"probabilities", "counts", "statevector"}

    def __init__(
        self,
        return_type: str = "probabilities",
        shots: int | None = None,
        seed: int | None = None,
        strip_measurements: bool = True,
    ) -> None:
        if return_type not in self._RETURN_TYPES:
            raise ValueError(f"return_type must be one of {sorted(self._RETURN_TYPES)}.")
        if shots is not None and shots <= 0:
            raise ValueError("shots must be positive or None.")
        self.return_type = return_type
        self.shots = shots
        self.seed = seed
        self.strip_measurements = strip_measurements

    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Run a Qiskit circuit locally.

        Parameters
        ----------
        circuit:
            Qiskit ``QuantumCircuit`` or instruction-like object supported by
            ``qiskit.quantum_info.Statevector``.
        params:
            Optional execution settings and parameter bindings. Supported
            reserved keys are ``parameters``, ``return_type``, ``shots``,
            ``seed``, ``qargs``, and ``decimals``. If ``parameters`` is absent,
            remaining keys are treated as circuit parameter bindings.
        """

        Statevector = _import_qiskit_statevector()
        run_params = dict(params or {})
        return_type = run_params.pop("return_type", self.return_type)
        if return_type not in self._RETURN_TYPES:
            raise ValueError(f"return_type must be one of {sorted(self._RETURN_TYPES)}.")
        shots = run_params.pop("shots", self.shots)
        seed = run_params.pop("seed", self.seed)
        qargs = run_params.pop("qargs", None)
        decimals = run_params.pop("decimals", None)
        parameter_values = run_params.pop("parameters", run_params)

        prepared = self._prepare_circuit(circuit, parameter_values)
        statevector = Statevector.from_instruction(prepared)
        if return_type == "statevector":
            return statevector

        probabilities = _normalize_probabilities(
            statevector.probabilities_dict(qargs=qargs, decimals=decimals)
        )
        if return_type == "probabilities":
            return probabilities

        return _sample_counts(probabilities, shots=shots or 1024, seed=seed)

    def _prepare_circuit(self, circuit: Any, parameters: Mapping[str, Any]) -> Any:
        if parameters:
            circuit = _assign_qiskit_parameters(circuit, parameters)
        if self.strip_measurements and hasattr(circuit, "remove_final_measurements"):
            return circuit.remove_final_measurements(inplace=False)
        return circuit


class PennyLaneBackend(Backend):
    """Run PennyLane QNodes or quantum functions on a local device.

    This adapter uses PennyLane's local simulator devices, defaulting to
    ``"default.qubit"``. It accepts an existing QNode or a PennyLane quantum
    function that returns measurements such as ``qml.probs`` or ``qml.expval``.

    Parameters
    ----------
    wires:
        Wires to allocate when wrapping a plain quantum function.
    device:
        PennyLane device name.
    shots:
        Optional shot count passed to ``qml.device``.
    interface:
        PennyLane QNode interface.
    diff_method:
        PennyLane QNode differentiation method.
    device_kwargs:
        Additional keyword arguments passed to ``qml.device``.
    """

    def __init__(
        self,
        wires: int | Sequence[int] | None = None,
        device: str = "default.qubit",
        shots: int | None = None,
        interface: str = "auto",
        diff_method: str = "best",
        device_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if shots is not None and shots <= 0:
            raise ValueError("shots must be positive or None.")
        self.wires = wires
        self.device = device
        self.shots = shots
        self.interface = interface
        self.diff_method = diff_method
        self.device_kwargs = dict(device_kwargs or {})

    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Run a PennyLane QNode or quantum function locally.

        Parameters
        ----------
        circuit:
            PennyLane QNode or quantum function. Plain quantum functions are
            wrapped with ``qml.qnode`` using this backend's configured device.
        params:
            Optional execution settings and function arguments. Use ``args`` for
            positional arguments and ``kwargs`` for keyword arguments. Remaining
            keys are passed as keyword arguments.
        """

        qml = _import_pennylane()
        run_params = dict(params or {})
        args = tuple(run_params.pop("args", ()))
        explicit_kwargs = dict(run_params.pop("kwargs", {}))
        wires = run_params.pop("wires", self.wires)
        device = run_params.pop("device", self.device)
        shots = run_params.pop("shots", self.shots)
        interface = run_params.pop("interface", self.interface)
        diff_method = run_params.pop("diff_method", self.diff_method)
        device_kwargs = {**self.device_kwargs, **dict(run_params.pop("device_kwargs", {}))}
        kwargs = {**run_params, **explicit_kwargs}

        executable = circuit
        if not _is_pennylane_qnode(circuit):
            if not callable(circuit):
                raise TypeError("PennyLaneBackend.run expects a QNode or quantum function.")
            if wires is None:
                raise ValueError(
                    "wires must be provided when running a plain PennyLane quantum function."
                )
            dev = qml.device(device, wires=wires, shots=shots, **device_kwargs)
            executable = _make_pennylane_qnode(
                qml,
                circuit,
                dev,
                interface=interface,
                diff_method=diff_method,
            )

        return _to_python_result(executable(*args, **kwargs), qml)


def _import_qiskit_statevector() -> Any:
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:  # pragma: no cover - depends on optional dependency.
        raise ImportError(
            "QiskitBackend requires Qiskit. Install it with "
            "`pip install qmlkit[qiskit]` or `pip install qiskit`."
        ) from exc
    return Statevector


def _import_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - depends on optional dependency.
        raise ImportError(
            "PennyLaneBackend requires PennyLane. Install it with "
            "`pip install qmlkit[pennylane]` or `pip install pennylane`."
        ) from exc
    return qml


def _assign_qiskit_parameters(circuit: Any, parameters: Mapping[str, Any]) -> Any:
    if not hasattr(circuit, "assign_parameters"):
        raise TypeError("Qiskit parameter binding requires a circuit with assign_parameters.")

    try:
        return circuit.assign_parameters(dict(parameters), inplace=False)
    except Exception as original_error:
        named_parameters = {
            parameter: parameters[parameter.name]
            for parameter in getattr(circuit, "parameters", [])
            if parameter.name in parameters
        }
        if not named_parameters:
            raise original_error
        return circuit.assign_parameters(named_parameters, inplace=False)


def _sample_counts(
    probabilities: Mapping[str, float], shots: int, seed: int | None
) -> dict[str, int]:
    states = list(probabilities)
    weights = np.asarray([probabilities[state] for state in states], dtype=np.float64)
    if weights.size == 0:
        return {}
    weights = weights / weights.sum()
    sampled = np.random.default_rng(seed).multinomial(shots, weights)
    return {state: int(count) for state, count in zip(states, sampled) if count > 0}


def _normalize_probabilities(probabilities: Mapping[Any, Any]) -> dict[str, float]:
    return {str(state): float(probability) for state, probability in probabilities.items()}


def _make_pennylane_qnode(
    qml: Any,
    circuit: Callable[..., Any],
    device: Any,
    interface: str,
    diff_method: str,
) -> Callable[..., Any]:
    if hasattr(qml, "QNode"):
        return qml.QNode(circuit, device, interface=interface, diff_method=diff_method)
    return qml.qnode(circuit, device, interface=interface, diff_method=diff_method)


def _is_pennylane_qnode(circuit: Any) -> bool:
    return callable(circuit) and hasattr(circuit, "device") and (
        hasattr(circuit, "func") or hasattr(circuit, "qtape") or hasattr(circuit, "_tape")
    )


def _to_python_result(result: Any, qml: Any) -> Any:
    if isinstance(result, tuple):
        return tuple(_to_python_result(item, qml) for item in result)
    if isinstance(result, list):
        return [_to_python_result(item, qml) for item in result]
    try:
        array = qml.math.toarray(result)
    except Exception:
        return result
    return _normalize_array_result(np.asarray(array))


def _normalize_array_result(array: NDArray[Any]) -> float | NDArray[Any]:
    if array.ndim == 0:
        return float(array)
    return array
