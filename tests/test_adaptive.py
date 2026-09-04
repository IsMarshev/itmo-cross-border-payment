"""Walk-forward calibration: it must select, and it must not peek."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.adaptive import (
    TuningConfig,
    _spearman,
    percentile_candidates,
    walk_forward_tuned,
    zscore_candidates,
)


def _panel(rates: list[float], iso: str = "TJS") -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=len(rates))
    return pd.DataFrame(
        {
            "quote_date": dates,
            "available_on": dates + pd.Timedelta(days=1),
            "iso": iso,
            "rub_per_unit": rates,
        }
    )


def _mean_reverting(n: int = 1400, seed: int = 3) -> list[float]:
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for i in range(1, n):
        level[i] = 0.97 * level[i - 1] + rng.normal(0, 0.01)
    return list(10.0 * np.exp(level))


CONFIG = TuningConfig(horizon=5, min_train=400, refit_every=40, lookback=400)


def test_spearman_is_rank_based_and_nan_safe():
    x = np.arange(100, dtype=float)
    assert _spearman(x, x**3) == pytest.approx(1.0)  # monotone but very non-linear
    assert _spearman(x, -(x**3)) == pytest.approx(-1.0)
    assert np.isnan(_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))  # too few
    assert np.isnan(_spearman(np.full(100, 1.0), x))  # no variation to rank


def test_calibration_produces_a_score_and_names_its_choice():
    panel = _panel(_mean_reverting())
    scored = walk_forward_tuned(panel, "TJS", zscore_candidates((10, 60, 250)), CONFIG)

    assert not scored.empty
    assert scored["score"].notna().all()
    assert set(scored["chosen"]) <= {"span=10", "span=60", "span=250"}
    assert scored["chosen"].ne("").all()


def test_the_choice_for_a_date_cannot_change_when_later_data_arrives():
    """The whole point of walk-forward: truncating the future changes nothing."""
    rates = _mean_reverting()
    full = walk_forward_tuned(
        _panel(rates), "TJS", zscore_candidates((10, 60, 250)), CONFIG
    )
    short = walk_forward_tuned(
        _panel(rates[:1100]), "TJS", zscore_candidates((10, 60, 250)), CONFIG
    )

    merged = short.merge(full, on="quote_date", suffixes=("_short", "_full"))
    assert len(merged) > 300
    np.testing.assert_allclose(
        merged["score_short"], merged["score_full"], rtol=1e-12
    )
    assert (merged["chosen_short"] == merged["chosen_full"]).all()


def test_calibration_prefers_the_candidate_that_tracked_client_money():
    """A candidate that is pure noise must lose to one that carries signal."""
    panel = _panel(_mean_reverting())
    rng = np.random.default_rng(0)

    def noise(values: pd.Series) -> pd.Series:
        return pd.Series(rng.normal(size=len(values)), index=values.index)

    real = zscore_candidates((10,))
    scored = walk_forward_tuned(
        panel, "TJS", {**real, "noise": noise}, CONFIG
    )
    share_real = (scored["chosen"] == "span=10").mean()
    assert share_real > 0.5, "calibration should mostly avoid the noise candidate"


def test_percentile_candidates_are_bounded():
    panel = _panel(_mean_reverting())
    scored = walk_forward_tuned(panel, "TJS", percentile_candidates((30, 90)), CONFIG)
    assert scored["score"].between(0.0, 1.0).all()


def test_missing_corridor_returns_an_empty_frame():
    scored = walk_forward_tuned(_panel(_mean_reverting()), "KGS", None, CONFIG)
    assert scored.empty
