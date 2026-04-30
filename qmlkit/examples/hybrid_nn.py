"""Train a tiny neural network with qmlkit.HybridLayer."""

from __future__ import annotations

from qmlkit import HybridLayer


def main() -> None:
    """Run the HybridLayer example."""

    import torch
    from torch import nn

    torch.manual_seed(7)
    X = torch.randn(256, 4)
    y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).long()

    model = nn.Sequential(
        HybridLayer(n_qubits=8),
        nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        accuracy = (model(X).argmax(dim=1) == y).float().mean().item()
    print(f"hybrid_nn training_accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
