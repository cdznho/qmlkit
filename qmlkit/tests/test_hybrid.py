import pytest

torch = pytest.importorskip("torch")

from qmlkit import HybridLayer  # noqa: E402


def test_hybrid_layer_output_shape():
    layer = HybridLayer(n_qubits=4)
    x = torch.randn(6, 3)
    y = layer(x)
    assert y.shape == (6, 4)


def test_hybrid_layer_gradients_flow():
    layer = HybridLayer(n_qubits=4)
    x = torch.randn(6, 3, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None


def test_hybrid_layer_optimizer_before_first_forward():
    layer = HybridLayer(n_qubits=4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    x = torch.randn(6, 3)
    loss = layer(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    assert tuple(layer.weight.shape) == (3, 4)


def test_hybrid_layer_rejects_backend_placeholder():
    with pytest.raises(NotImplementedError, match="backend='sim'"):
        HybridLayer(n_qubits=4, backend="qiskit")
