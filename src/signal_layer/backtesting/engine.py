"""Orchestration of scores, policy decisions, outcomes and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from signal_layer.backtesting.outcomes import build_outcomes
from signal_layer.backtesting.policy import PolicyConfig, apply_policy
from signal_layer.backtesting.reporting import build_report, matched_random_schedules


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    horizon: int = 20
    epsilon_bps: float = 30.0
    random_trials: int = 200
    bootstrap_trials: int = 1_000
    seed: int = 0
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.random_trials <= 0 or self.bootstrap_trials <= 0:
            raise ValueError("trial counts must be positive")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    decision_log: pd.DataFrame
    random_log: pd.DataFrame
    report: pd.DataFrame


def run_backtest(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a chronological policy and attach future outcomes only afterwards."""
    resolved = config or BacktestConfig()
    decisions = apply_policy(scores, resolved.policy)
    outcomes = build_outcomes(
        panel,
        horizon=resolved.horizon,
        epsilon_bps=resolved.epsilon_bps,
    )
    outcome_columns = [
        "quote_date",
        "iso",
        "future_median",
        "future_minimum",
        "advantage_bps",
        "early_send",
        "regret_bps",
        "outcome_available_on",
        "outcome_complete",
    ]
    decision_log = decisions.merge(
        outcomes[outcome_columns],
        on=["quote_date", "iso"],
        how="left",
        validate="one_to_one",
    )
    decision_log["outcome_complete"] = decision_log["outcome_complete"].fillna(False)
    random_log = matched_random_schedules(
        decision_log,
        trials=resolved.random_trials,
        seed=resolved.seed,
    )
    report = build_report(
        decision_log,
        random_log,
        block_length=resolved.horizon,
        bootstrap_trials=resolved.bootstrap_trials,
        seed=resolved.seed,
    )
    return BacktestResult(decision_log=decision_log, random_log=random_log, report=report)
