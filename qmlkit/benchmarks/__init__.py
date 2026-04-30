"""Benchmark utilities for qmlkit."""

from qmlkit.benchmarks.datasets import load_classification, load_moons, load_toy_dataset
from qmlkit.benchmarks.runner import run_benchmark

__all__ = ["load_classification", "load_moons", "load_toy_dataset", "run_benchmark"]
