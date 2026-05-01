"""Run a small PennyLane circuit through qmlkit.PennyLaneBackend."""

from __future__ import annotations

from qmlkit import PennyLaneBackend


def main() -> None:
    """Run the PennyLane backend example."""

    try:
        import pennylane as qml
    except ImportError as exc:
        raise SystemExit(
            "This example requires PennyLane. Install it with `pip install qmlkit[pennylane]`."
        ) from exc

    def circuit(theta: float):
        qml.RX(theta, wires=0)
        return qml.probs(wires=0)

    probabilities = PennyLaneBackend(wires=1).run(circuit, {"theta": 0.0})
    print(f"pennylane probabilities={probabilities}")


if __name__ == "__main__":
    main()
