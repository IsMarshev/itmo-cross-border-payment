"""The live signal layer: the contract a client-facing push depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.signals import (
    INDICATOR,
    SCENARIO,
    SIGNAL_COLUMNS,
    SignalLayerConfig,
    latest_signal,
    signal_table,
    signals_asof,
)


def _panel(n: int = 1500, seed: int = 4) -> pd.DataFrame:
    """Two corridors sharing a mean-reverting rouble move, as the real ones do."""
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for i in range(1, n):
        level[i] = 0.97 * level[i - 1] + rng.normal(0, 0.012)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "quote_date": dates,
                    "available_on": dates + pd.Timedelta(days=1),
                    "iso": iso,
                    "rub_per_unit": scale * np.exp(level),
                }
            )
            for iso, scale in (("TJS", 9.0), ("USD", 85.0))
        ],
        ignore_index=True,
    )


CONFIG = SignalLayerConfig()


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return signal_table(_panel(), ["TJS"], CONFIG)


def test_table_has_the_briefs_columns(table):
    assert list(table.columns) == list(SIGNAL_COLUMNS)
    assert not table.empty
    assert (table["indicator"] == INDICATOR).all()
    assert (table["scenario"] == SCENARIO).all()
    assert table["direction"].isin({"down", "up"}).all()
    assert table["speed"].isin({"fast", "medium", "slow"}).all()


def test_every_signal_carries_a_message_whose_claim_is_true():
    """The push states a fact, so the fact must hold when it is sent."""
    result = signal_table(_panel(), ["TJS"], CONFIG)
    assert (result["message"].str.len() > 0).all()
    # The message claims the rate sits below its trend; that must be so.
    assert (result["deviation_pct"] < 0).all()
    for _, row in result.head(50).iterrows():
        claimed = float(row["message"].split("на ")[1].split("%")[0])
        assert claimed == pytest.approx(abs(float(row["deviation_pct"])), abs=0.05)


def test_a_message_never_promises_anything():
    """Compliance: facts about the past and present only, no forecast."""
    result = signal_table(_panel(), ["TJS"], CONFIG)
    banned = (
        "вырастет", "подорожает", "успейте", "гарантируем", "заработайте",
        "скоро", "прогноз", "будет", "ожидается",
    )
    joined = " ".join(result["message"]).lower()
    for word in banned:
        assert word not in joined, f"push text must not contain {word!r}"


def test_turning_the_truth_gate_off_admits_days_with_nothing_to_say():
    """The gate is what makes every signal describable; without it some are not."""
    ungated = signal_table(
        _panel(), ["TJS"], SignalLayerConfig(require_true_fact=False)
    )
    gated = signal_table(_panel(), ["TJS"], CONFIG)
    assert (ungated["deviation_pct"] >= 0).any()
    assert (ungated["message"] == "").any()
    assert len(gated) < len(ungated)


def test_the_layer_respects_the_weekly_push_budget(table):
    weeks = table["signal_date"].dt.to_period("W-SUN")
    assert weeks.value_counts().max() <= CONFIG.max_signals_per_window


def test_signals_asof_match_the_historical_run_exactly():
    """The brief's disqualifying condition, checked by deleting the future."""
    panel = _panel()
    full = signal_table(panel, ["TJS"], CONFIG)
    asof = pd.Timestamp(panel["quote_date"].iloc[1000])

    truncated = signals_asof(panel, ["TJS"], asof, CONFIG)
    expected = full[full["signal_date"] <= asof]

    assert len(truncated) == len(expected) > 20
    merged = truncated.merge(expected, on=["signal_date", "iso"], suffixes=("_a", "_b"))
    assert len(merged) == len(truncated)
    np.testing.assert_allclose(merged["strength_a"], merged["strength_b"], rtol=1e-12)
    assert (merged["message_a"] == merged["message_b"]).all()


def test_latest_signal_returns_none_on_a_hold():
    panel = _panel()
    table = signal_table(panel, ["TJS"], CONFIG)
    fired = set(table["signal_date"])
    quiet = [d for d in panel["quote_date"].iloc[900:1000] if d not in fired]
    assert quiet, "the layer should not fire every day"
    assert latest_signal(panel, "TJS", quiet[0], CONFIG) is None


def test_latest_signal_returns_the_row_for_a_firing_day():
    panel = _panel()
    table = signal_table(panel, ["TJS"], CONFIG)
    day = table["signal_date"].iloc[len(table) // 2]
    signal = latest_signal(panel, "TJS", day, CONFIG)
    assert signal is not None
    assert signal["signal_date"] == day
    assert signal["indicator"] == INDICATOR


def test_short_history_yields_no_signals_rather_than_guesses():
    assert signal_table(_panel(n=200), ["TJS"], CONFIG).empty
