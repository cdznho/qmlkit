# qmlkit launch notes

## Positioning

qmlkit is a public-alpha SDK for AI engineers who want quantum-inspired ML
building blocks without leaving PyTorch or scikit-learn.

## Suggested GitHub description

Quantum-inspired and hybrid ML tools for PyTorch and scikit-learn. Classical by
default, backend-extensible later.

## Suggested README tagline

Quantum-inspired ML, without the quantum hardware requirement.

## First promotion post

I just open-sourced qmlkit, a tiny Python SDK for quantum-inspired ML workflows.

The idea: most AI engineers do not want to start with circuits. They want
sklearn transformers, PyTorch layers, toy benchmarks, and clean extension points.

v0.1 includes:

- sklearn-compatible quantum-inspired feature maps
- a differentiable PyTorch `HybridLayer`
- a NumPy simulator backend
- placeholder Qiskit/PennyLane adapters for future integrations

This is a public alpha, not a quantum advantage claim. The goal is to make the
interface useful first, then let the community help shape what belongs next.
