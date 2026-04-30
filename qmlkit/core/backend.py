"""Backend interfaces for simulated and future quantum execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


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
    """Placeholder adapter for a future Qiskit integration."""

    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Raise until the Qiskit adapter is implemented."""

        raise NotImplementedError(
            "QiskitBackend is a placeholder in qmlkit v0.1. "
            "Use SimulatorBackend for local execution."
        )


class PennyLaneBackend(Backend):
    """Placeholder adapter for a future PennyLane integration."""

    def run(self, circuit: Any, params: Mapping[str, Any] | None = None) -> Any:
        """Raise until the PennyLane adapter is implemented."""

        raise NotImplementedError(
            "PennyLaneBackend is a placeholder in qmlkit v0.1. "
            "Use SimulatorBackend for local execution."
        )
