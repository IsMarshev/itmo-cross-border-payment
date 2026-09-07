"""CBSB-1 — the cross-border signal benchmark.

Public surface::

    from signal_layer.benchmark import BenchmarkSpec, run_benchmark

``spec`` holds the contract, ``labels`` the ground truth, ``strategies`` the
field, ``stats`` the tests and ``runner`` the execution. Read ``BENCHMARK.md``
first: it explains what the numbers mean before this package explains how they
are produced.
"""

from ..labels import build_labels
from .runner import BenchmarkResult, run_benchmark
from .spec import GATES, BenchmarkSpec, Gate
from .strategies import DEFAULT_STRATEGY_NAMES, STRATEGIES, Strategy, get_strategy

__all__ = [
    "BenchmarkResult",
    "BenchmarkSpec",
    "DEFAULT_STRATEGY_NAMES",
    "GATES",
    "Gate",
    "STRATEGIES",
    "Strategy",
    "build_labels",
    "get_strategy",
    "run_benchmark",
]
