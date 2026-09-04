"""The MVP model: leakage contract, head behaviour and the lambda dial."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.utility_risk import (
    LogisticIRLS,
    UtilityRiskConfig,
    _training_prefix_lengths,
    rescore,
    scores_asof,
    walk_forward_scores,
)


def _synthetic_panel(n: int = 1400, seed: int = 3) -> pd.DataFrame:
    """A mean-reverting series with enough history to clear ``min_train``."""
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for i in range(1, n):
        level[i] = 0.97 * level[i - 1] + rng.normal(0, 0.01)
    rates = 10.0 * np.exp(level)
    dates = pd.bdate_range("2015-01-01", periods=n)
    frames = []
    for offset, iso in ((0.0, "TJS"), (0.5, "USD")):
        frames.append(
            pd.DataFrame(
                {
                    "quote_date": dates,
                    "available_on": dates + pd.Timedelta(days=1),
                    "iso": iso,
                    "rub_per_unit": rates * (1.0 + offset),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


CONFIG = UtilityRiskConfig(horizon=5, min_train=400, refit_every=40)


def test_logistic_recovers_a_known_boundary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] * 2.0 - 1.0 + rng.normal(0, 0.3, 2000) > 0).astype(float)
    model = LogisticIRLS(l2=1e-6).fit(X, y)

    assert model.coef_[0] > 1.0  # the informative feature dominates
    assert abs(model.coef_[1]) < 0.3
    predictions = model.predict_proba(X) > 0.5
    assert (predictions == y.astype(bool)).mean() > 0.9


def test_logistic_falls_back_to_the_base_rate_on_a_degenerate_target():
    X = np.random.default_rng(1).normal(size=(50, 4))
    model = LogisticIRLS().fit(X, np.zeros(50))
    assert np.allclose(model.predict_proba(X), model.constant_)
    assert model.constant_ < 1e-5


def test_training_prefix_never_includes_an_unmatured_label():
    decision = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    ).to_numpy()
    matured = pd.to_datetime(
        ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
    ).to_numpy()
    lengths = _training_prefix_lengths(matured, decision)

    # Row 0 decides on 01-02, when no label has matured yet.
    assert lengths[0] == 0
    # Row 2 decides on 01-06; only row 0's label (due 01-06) is available.
    assert lengths[2] == 1
    assert list(lengths) == sorted(lengths)


def test_scores_asof_reproduce_the_historical_run_exactly():
    panel = _synthetic_panel()
    full, _ = walk_forward_scores(panel, "TJS", CONFIG)
    assert not full.empty

    for asof in full["quote_date"].iloc[[0, len(full) // 2, -1]]:
        live = scores_asof(panel, "TJS", asof, CONFIG)
        live_row = live[live["quote_date"].eq(asof)]
        historical = full[full["quote_date"].eq(asof)]
        assert len(live_row) == 1
        for column in ("p_min", "p_bad", "u_bps", "risk_bps", "score"):
            assert live_row[column].iloc[0] == pytest.approx(
                historical[column].iloc[0], abs=1e-12
            )


def test_deleting_the_future_cannot_change_a_past_score():
    """The strongest leakage check: truncate the panel, not just the features."""
    panel = _synthetic_panel()
    cut = panel["quote_date"].iloc[900]
    full, _ = walk_forward_scores(panel, "TJS", CONFIG)
    truncated, _ = walk_forward_scores(panel[panel["quote_date"] <= cut], "TJS", CONFIG)

    merged = truncated.merge(full, on="quote_date", suffixes=("_cut", "_full"))
    assert len(merged) > 100
    np.testing.assert_allclose(merged["score_cut"], merged["score_full"], atol=1e-12)


def test_probabilities_stay_in_range_and_risk_is_non_negative():
    scores, _ = walk_forward_scores(_synthetic_panel(), "TJS", CONFIG)
    assert scores["p_min"].between(0.0, 1.0).all()
    assert scores["p_bad"].between(0.0, 1.0).all()
    assert (scores["risk_bps"] >= 0).all()


def test_rescore_matches_a_refit_at_the_same_lambda():
    panel = _synthetic_panel()
    base, _ = walk_forward_scores(panel, "TJS", CONFIG)
    other = UtilityRiskConfig(
        horizon=CONFIG.horizon, min_train=CONFIG.min_train,
        refit_every=CONFIG.refit_every, lam=4.0,
    )
    refitted, _ = walk_forward_scores(panel, "TJS", other)
    np.testing.assert_allclose(
        rescore(base, 4.0)["score"].to_numpy(),
        refitted["score"].to_numpy(),
        atol=1e-12,
    )


def test_a_higher_price_of_error_never_favours_a_riskier_day():
    """Raising lambda may only push risky days down the ranking."""
    scores, _ = walk_forward_scores(_synthetic_panel(), "TJS", CONFIG)
    low = rescore(scores, 0.0)["score"].to_numpy()
    high = rescore(scores, 5.0)["score"].to_numpy()
    risk = scores["risk_bps"].to_numpy()
    baseline_risk = scores["base_risk_bps"].to_numpy()

    # score(lam) is linear in lam with slope (base_risk - risk): days riskier
    # than an ordinary day lose, days safer than one gain.
    np.testing.assert_allclose(high - low, 5.0 * (baseline_risk - risk), atol=1e-9)


def test_config_rejects_a_negative_price_of_error():
    with pytest.raises(ValueError, match="lam"):
        UtilityRiskConfig(lam=-1.0)
