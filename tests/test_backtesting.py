from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.backtesting import BacktestConfig, PolicyConfig, apply_policy, run_backtest
from signal_layer.backtesting.outcomes import build_outcomes
from signal_layer.backtesting.reporting import matched_random_schedules


def _panel(rates: list[float], iso: str = "USD") -> pd.DataFrame:
    quote_dates = pd.bdate_range("2026-01-01", periods=len(rates))
    return pd.DataFrame(
        {
            "quote_date": quote_dates,
            "available_on": quote_dates + pd.offsets.Day(1),
            "iso": iso,
            "rub_per_unit": rates,
        }
    )


def _scores(values: list[float]) -> pd.DataFrame:
    panel = _panel([100.0] * len(values))
    panel["score"] = values
    panel["score_source"] = "test"
    return panel


def test_outcomes_match_stage_four_definitions() -> None:
    panel = _panel([100.0, 90.0, 110.0, 120.0])

    outcomes = build_outcomes(panel, horizon=2, epsilon_bps=30.0)

    first = outcomes.iloc[0]
    assert first["future_median"] == pytest.approx(100.0)
    assert first["advantage_bps"] == pytest.approx(0.0)
    assert bool(first["early_send"])
    assert first["regret_bps"] == pytest.approx(1_000.0)
    assert first["outcome_available_on"] == panel.iloc[2]["available_on"]
    assert outcomes["outcome_complete"].tolist() == [True, True, False, False]


def test_policy_respects_budget_cooldown_and_does_not_force_weak_scores() -> None:
    scores = _scores([0.1, 0.2, 0.9, 0.95, 0.96, 0.99])
    policy = PolicyConfig(
        window="month",
        max_signals_per_window=2,
        expected_observations_per_window=10,
        cooldown_observations=1,
        minimum_threshold_history=2,
        minimum_score=0.8,
    )

    decisions = apply_policy(scores, policy)

    assert decisions.index[decisions["decision"]].tolist() == [2, 4]
    assert decisions.iloc[3]["decision_reason"] == "cooldown"
    assert decisions.iloc[5]["decision_reason"] == "window_budget_exhausted"

    weak = _scores([0.1, 0.2, 0.3, 0.4])
    weak_decisions = apply_policy(weak, policy)
    assert not weak_decisions["decision"].any()


def test_policy_history_is_unchanged_when_future_scores_are_appended() -> None:
    policy = PolicyConfig(minimum_threshold_history=2, minimum_score=0.2)
    original = _scores([0.1, 0.2, 0.9, 0.3, 0.8])
    extended = _scores([0.1, 0.2, 0.9, 0.3, 0.8, 1.0, 1.0])

    before = apply_policy(original, policy)
    after = apply_policy(extended, policy).iloc[: len(original)]

    columns = ["threshold", "decision", "decision_reason", "slots_after"]
    pd.testing.assert_frame_equal(
        before[columns].reset_index(drop=True),
        after[columns].reset_index(drop=True),
    )


def test_random_baseline_matches_every_window_signal_count() -> None:
    panel = _panel(list(np.linspace(100.0, 80.0, 90)))
    scores = panel.copy()
    scores["score"] = np.linspace(0.0, 1.0, len(scores))
    scores["score_source"] = "test"
    result = run_backtest(
        panel,
        scores,
        BacktestConfig(
            horizon=5,
            random_trials=10,
            bootstrap_trials=20,
            policy=PolicyConfig(
                window="month",
                max_signals_per_window=2,
                cooldown_observations=1,
                minimum_threshold_history=5,
                minimum_score=0.2,
            ),
        ),
    )

    random_log = matched_random_schedules(result.decision_log, trials=10, seed=0)
    selected = result.decision_log.loc[
        result.decision_log["decision"] & result.decision_log["outcome_complete"]
    ]
    expected = selected.groupby(["iso", "window_id"]).size().sort_index()
    for _, trial in random_log.groupby("trial"):
        actual = trial.groupby(["iso", "window_id"]).size().sort_index()
        pd.testing.assert_series_equal(actual, expected)

    assert {
        "mean_advantage_bps",
        "early_send_rate",
        "advantage_delta_bps",
        "advantage_ci_low",
        "advantage_ci_high",
    }.issubset(result.report.columns)
