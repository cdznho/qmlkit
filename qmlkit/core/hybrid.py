"""PyTorch hybrid neural network layers."""

from __future__ import annotations

try:  # pragma: no cover - exercised by tests when torch is installed.
    import torch
    from torch import Tensor, nn
    from torch.nn.parameter import UninitializedParameter

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class HybridLayer(nn.Module):
        """Differentiable quantum-inspired PyTorch layer.

        Parameters
        ----------
        n_qubits:
            Number of simulated qubit channels to emit.
        in_features:
            Optional input feature width. If omitted, parameters are lazily
            initialized on the first forward pass.
        backend:
            Backend selector. Only ``"sim"`` is implemented in v0.1.
        """

        def __init__(
            self,
            n_qubits: int,
            in_features: int | None = None,
            backend: str = "sim",
        ) -> None:
            super().__init__()
            if n_qubits <= 0:
                raise ValueError("n_qubits must be positive.")
            if in_features is not None and in_features <= 0:
                raise ValueError("in_features must be positive or None.")
            if backend != "sim":
                raise NotImplementedError(
                    "HybridLayer only supports backend='sim' in qmlkit v0.1."
                )

            self.n_qubits = n_qubits
            self.in_features = in_features
            self.backend = backend
            if in_features is not None:
                self._initialize_parameters(
                    in_features,
                    device=None,
                    dtype=torch.get_default_dtype(),
                )
            else:
                self.weight = UninitializedParameter()
                self.phase = nn.Parameter(torch.zeros(self.n_qubits))
                self.entangle = nn.Parameter(torch.empty(self.n_qubits, self.n_qubits))
                self.readout = nn.Parameter(torch.ones(self.n_qubits))
                self._reset_non_lazy_parameters()

        def forward(self, x: Tensor) -> Tensor:
            """Apply simulated rotations, mixing, and measurement-like readout."""

            if x.ndim == 1:
                x = x.unsqueeze(0)
            if x.ndim != 2:
                raise ValueError("HybridLayer expects a 2D tensor shaped (batch, features).")
            if not torch.is_floating_point(x):
                x = x.float()
            if isinstance(self.weight, UninitializedParameter):
                self._materialize_weight(
                    x.shape[-1],
                    device=x.device,
                    dtype=x.dtype,
                )
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    "Input feature width does not match layer parameters: "
                    f"expected {self.in_features}, got {x.shape[-1]}."
                )

            weight = self.weight.to(device=x.device, dtype=x.dtype)
            phase = self.phase.to(device=x.device, dtype=x.dtype)
            entangle = self.entangle.to(device=x.device, dtype=x.dtype)
            readout = self.readout.to(device=x.device, dtype=x.dtype)

            angles = x @ weight + phase
            rotated = torch.sin(angles) * torch.cos(0.5 * angles)
            mixed = rotated + torch.tanh(rotated @ entangle) / (self.n_qubits**0.5)
            return torch.tanh(mixed * readout)

        def extra_repr(self) -> str:
            """Return a compact module representation."""

            return (
                f"n_qubits={self.n_qubits}, "
                f"in_features={self.in_features}, backend={self.backend!r}"
            )

        def _initialize_parameters(
            self,
            in_features: int,
            device: torch.device | None,
            dtype: torch.dtype,
        ) -> None:
            self.in_features = in_features
            self.weight = nn.Parameter(
                torch.empty(in_features, self.n_qubits, device=device, dtype=dtype)
            )
            self.phase = nn.Parameter(torch.zeros(self.n_qubits, device=device, dtype=dtype))
            self.entangle = nn.Parameter(
                torch.empty(self.n_qubits, self.n_qubits, device=device, dtype=dtype)
            )
            self.readout = nn.Parameter(torch.ones(self.n_qubits, device=device, dtype=dtype))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            """Reset layer parameters."""

            if not isinstance(self.weight, UninitializedParameter):
                nn.init.xavier_uniform_(self.weight)
            self._reset_non_lazy_parameters()

        def _materialize_weight(
            self,
            in_features: int,
            device: torch.device | None,
            dtype: torch.dtype,
        ) -> None:
            self.in_features = in_features
            self.weight.materialize(
                (in_features, self.n_qubits),
                device=device,
                dtype=dtype,
            )
            nn.init.xavier_uniform_(self.weight)

        def _reset_non_lazy_parameters(self) -> None:
            with torch.no_grad():
                self.phase.uniform_(-0.1, 0.1)
                self.entangle.zero_()
                self.entangle.add_(0.05 * torch.randn_like(self.entangle))
                self.entangle.add_(
                    torch.eye(
                        self.n_qubits,
                        device=self.entangle.device,
                        dtype=self.entangle.dtype,
                    )
                )
                self.readout.fill_(1.0)

else:

    class HybridLayer:  # type: ignore[no-redef]
        """Unavailable placeholder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "HybridLayer requires PyTorch. Install it with "
                "`pip install qmlkit[torch]` or `pip install torch`."
            )


__all__ = ["HybridLayer", "TORCH_AVAILABLE"]
