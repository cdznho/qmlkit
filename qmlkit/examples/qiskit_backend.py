"""Run a small Qiskit circuit through qmlkit.QiskitBackend."""

from __future__ import annotations

from qmlkit import QiskitBackend


def main() -> None:
    """Run the Qiskit backend example."""

    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "This example requires Qiskit. Install it with `pip install qmlkit[qiskit]`."
        ) from exc

    circuit = QuantumCircuit(1)
    circuit.h(0)

    probabilities = QiskitBackend().run(circuit, {"decimals": 3})
    print(f"qiskit probabilities={probabilities}")


if __name__ == "__main__":
    main()
