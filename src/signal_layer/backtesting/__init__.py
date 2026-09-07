"""Leakage-safe offline evaluation of signal policies."""

from signal_layer.backtesting.engine import BacktestConfig, BacktestResult, run_backtest
from signal_layer.backtesting.outcomes import build_outcomes
from signal_layer.backtesting.policy import PolicyConfig, apply_policy
from signal_layer.backtesting.scoring import build_baseline_scores, build_model_scores

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PolicyConfig",
    "apply_policy",
    "build_baseline_scores",
    "build_model_scores",
    "build_outcomes",
    "run_backtest",
]
