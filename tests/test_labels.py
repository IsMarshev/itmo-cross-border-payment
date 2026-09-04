"""The ground-truth labels are the one place look-ahead is allowed.

These tests pin down exactly how much of the future each column may see, so a
later change cannot quietly widen it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.labels import build_labels


def _panel(rates: list[float], iso: str = "TJS") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(rates))
    return pd.DataFrame(
        {
            "quote_date": dates,
            "available_on": dates + pd.Timedelta(days=1),
            "iso": iso,
            "rub_per_unit": rates,
        }
    )


def test_forward_window_starts_after_the_execution_row():
    # A V shape: the dip is at position 5, execution happens one row later.
    rates = [10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
    labels = build_labels(_panel(rates), horizon=2, execution_offset=1)
    row = labels[labels["quote_date"].eq(labels["quote_date"].iloc[4])].iloc[0]

    # Signal on row 4 executes at row 5 (the dip, 9.0) and looks at rows 6-7.
    assert row["exec_rate"] == pytest.approx(9.0)
    assert row["fwd_advantage_bps"] == pytest.approx((11.0 - 9.0) / 9.0 * 10_000)
    assert row["regret_bps"] == pytest.approx(0.0)
    assert bool(row["held_window_closing"]) is True
    assert bool(row["held_favourable"]) is False


def test_adverse_and_bad_push_track_the_downside_of_the_advantage():
    # Rate keeps falling after execution: the client should have waited.
    rates = [10.0] * 5 + [10.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    labels = build_labels(
        _panel(rates), horizon=3, execution_offset=1, bad_push_bps=100.0
    )
    row = labels[labels["outcome_complete"]].iloc[4]
    assert row["fwd_advantage_bps"] < 0
    assert row["adverse_bps"] == pytest.approx(-row["fwd_advantage_bps"])
    assert bool(row["bad_push"]) is True


def test_currency_gain_uses_currency_per_rouble_not_the_rate():
    rates = [12.0, 8.0, 12.0, 8.0, 12.0, 10.0, 12.0, 8.0, 12.0, 8.0, 12.0]
    labels = build_labels(_panel(rates), horizon=3, execution_offset=0)
    row = labels[labels["quote_date"].eq(labels["quote_date"].iloc[5])].iloc[0]

    window = np.array(rates[2:9], dtype=float)
    expected = ((1 / 10.0) / np.mean(1 / window) - 1.0) * 10_000
    assert row["currency_gain_bps"] == pytest.approx(expected)
    # The rate-based version is a different number: averaging rates and
    # averaging what they buy are not the same operation.
    assert row["window_advantage_bps"] != pytest.approx(expected)


def test_incomplete_tail_is_kept_but_flagged():
    labels = build_labels(_panel([10.0] * 12), horizon=4, execution_offset=1)
    assert len(labels) == 12
    # Last offset + horizon rows cannot have a full forward window.
    assert (~labels["outcome_complete"]).sum() == 5
    assert labels.loc[~labels["outcome_complete"], "fwd_advantage_bps"].isna().all()


def test_labels_of_a_date_do_not_change_when_later_data_arrives():
    rng = np.random.default_rng(7)
    rates = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    full = build_labels(_panel(rates), horizon=5, execution_offset=1)
    truncated = build_labels(_panel(rates[:150]), horizon=5, execution_offset=1)

    merged = truncated[truncated["outcome_complete"]].merge(
        full, on=["iso", "quote_date"], suffixes=("_short", "_full")
    )
    assert len(merged) > 100
    for column in ("fwd_advantage_bps", "window_advantage_bps", "currency_gain_bps"):
        np.testing.assert_allclose(
            merged[f"{column}_short"], merged[f"{column}_full"], rtol=1e-12
        )


def test_local_minimum_needs_the_centred_window():
    rates = [10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0]
    labels = build_labels(_panel(rates), horizon=2, execution_offset=0)
    flags = labels.set_index("quote_date")["is_local_min"]
    assert bool(flags.iloc[3]) is True
    assert bool(flags.iloc[4]) is False
    # Row 1 has no full window before it, so it is never labelled a minimum.
    assert bool(flags.iloc[1]) is False


# --- rule indicators ---------------------------------------------------------


def test_rule_matrix_carries_no_look_ahead():
    """A rule column for a date must not move when later data arrives.

    The reversal rule vetoes days with a large negative sentinel. Collapsing
    that sentinel onto the series minimum would read the whole series to pick
    the floor, so the feature version splits it into two leak-free columns.
    """
    from signal_layer.features import compute_features
    from signal_layer.rules import RULE_FEATURE_COLUMNS, rule_matrix

    rng = np.random.default_rng(5)
    rates = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    full = rule_matrix(compute_features(_panel(rates)))
    short = rule_matrix(compute_features(_panel(rates[:300])))

    overlap = short.index.intersection(full.index)
    assert len(overlap) > 200
    for column in RULE_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            short.loc[overlap, column].to_numpy(dtype=float),
            full.loc[overlap, column].to_numpy(dtype=float),
            rtol=1e-12,
            equal_nan=True,
        )


def test_rule_matrix_has_no_sentinel_values():
    """The veto sentinel belongs in a score, never in a design matrix."""
    from signal_layer.features import compute_features
    from signal_layer.rules import BLOCKED, rule_matrix

    rng = np.random.default_rng(6)
    rates = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    matrix = rule_matrix(compute_features(_panel(rates))).to_numpy(dtype=float)
    filled = matrix[np.isfinite(matrix)]  # warm-up rows are legitimately NaN
    assert len(filled) > 0
    assert (filled > BLOCKED / 2).all()


def test_rule_scores_keep_the_veto_sentinel():
    """As a contender score, the reversal rule must still be able to refuse."""
    from signal_layer.features import compute_features
    from signal_layer.rules import BLOCKED, rule_score

    rates = [10.0, 9.0, 8.0, 7.0, 6.0] * 60  # monotone falls, then a jump up
    scores = rule_score("reversal", compute_features(_panel(rates)))
    assert (scores == BLOCKED).any(), "days that did not turn up must be vetoed"
