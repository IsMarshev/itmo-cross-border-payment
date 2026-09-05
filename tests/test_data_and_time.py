from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.config import Config, config_from_dict
from signal_layer.data import daily_rates, normalize
from signal_layer.features import build_features
from signal_layer.targets import build_targets, mature_rows
from signal_layer.training import temporal_partitions


def test_nominal_change_does_not_create_fx_move():
    f = normalize(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "iso": ["TJS"] * 2,
                "nominal": [10, 100],
                "rate": [100, 1000],
            }
        )
    )
    assert f.rub_per_unit.tolist() == [10, 10]
    assert f.date.equals(f.effective_date)


@pytest.mark.parametrize("column,value", [("rate", 0), ("nominal", -1), ("rate", np.inf)])
def test_invalid_rates_fail(column, value):
    raw = {"date": ["2024-01-02"], "iso": ["TJS"], "nominal": [10], "rate": [100]}
    raw[column] = [value]
    with pytest.raises(ValueError):
        normalize(pd.DataFrame(raw))


def test_duplicate_and_negative_lag_fail():
    f = pd.DataFrame(
        {"date": ["2024-01-02"] * 2, "iso": ["TJS"] * 2, "nominal": [10] * 2, "rate": [100] * 2}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        normalize(f)
    with pytest.raises(ValueError, match="negative"):
        normalize(f.iloc[:1], -1)


def test_carry_forward_is_not_an_observation():
    f = normalize(
        pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-09"],
                "iso": ["TJS"] * 2,
                "nominal": [1] * 2,
                "rate": [10, 9],
            }
        )
    )
    d = daily_rates(f, max_stale_days=2)
    assert len(f) == 2 and len(d) == 5
    assert d.loc["2024-01-07", "rub_per_unit"] == 10
    assert pd.isna(d.loc["2024-01-08", "rub_per_unit"])
    assert d.index.min() == f.date.min() and d.index.max() == f.date.max()


def test_features_equal_prefix_and_ignore_mutated_future(prepared, synthetic_config):
    panel, full, _, cols = prepared
    cutoff = pd.Timestamp("2022-05-14")
    prefix, prefix_cols = build_features(panel.loc[panel.date <= cutoff], synthetic_config)
    pd.testing.assert_frame_equal(full.loc[full.date <= cutoff].reset_index(drop=True), prefix)
    changed = panel.copy()
    changed.loc[changed.date > cutoff, "rub_per_unit"] *= 7
    changed_features, _ = build_features(changed, synthetic_config)
    pd.testing.assert_frame_equal(
        prefix, changed_features.loc[changed_features.date <= cutoff].reset_index(drop=True)
    )
    assert prefix_cols == cols


def test_calendar_targets_match_brute_force():
    c = Config()
    c.data.corridors, c.data.context = ["TJS"], []
    c.targets.horizons, c.targets.primary_horizon = [1, 3], 3
    dates = pd.date_range("2024-01-01", periods=31)
    rates = np.array([10 + 0.3 * np.sin(i) for i in range(31)])
    p = normalize(pd.DataFrame({"date": dates, "iso": "TJS", "nominal": 1, "rate": rates}))
    y = build_targets(p, p[["date", "iso", "rub_per_unit"]], c)
    for i in range(3, 28):
        window, future, price = rates[i - 3 : i + 4], rates[i + 1 : i + 4], rates[i]
        regret = (price / min(price, future.min()) - 1) * 10000
        stale = (1 - price / max(price, future.max())) * 10000
        assert y.loc[i, "gain_bps_h3"] == pytest.approx((1 - price / window.mean()) * 10000)
        assert y.loc[i, "regret_bps_h3"] == pytest.approx(regret)
        assert y.loc[i, "stale_bps_h3"] == pytest.approx(stale)
        assert y.loc[i, "local_min_h3"] == (
            (price / window.min() - 1) * 10000 <= c.targets.near_min_bps
        )
    assert y.tail(3).gain_bps_h3.isna().all()


def test_holding_and_no_regret_have_opposite_price_directions():
    c = Config()
    c.data.corridors, c.data.context = ["TJS"], []
    dates = pd.date_range("2024-01-01", periods=70)
    for increasing in (True, False):
        values = np.linspace(10, 20, 70) if increasing else np.linspace(20, 10, 70)
        p = normalize(pd.DataFrame({"date": dates, "iso": "TJS", "nominal": 1, "rate": values}))
        y = build_targets(p, p[["date", "iso", "rub_per_unit"]], c).iloc[30]
        assert y.no_regret_h5 == float(increasing)
        assert y.hold_h5 == float(not increasing)


def test_partitions_purge_unmatured_labels(prepared, synthetic_config):
    _, f, y, _ = prepared
    d = f.loc[f.eligible].merge(y, on=["date", "iso"])
    cutoff = pd.Timestamp("2022-05-31")
    train, cal, tune = temporal_partitions(d, cutoff, synthetic_config)
    tune_start = cutoff - pd.Timedelta(days=synthetic_config.model.tuning_days - 1)
    cal_start = tune_start - pd.Timedelta(days=synthetic_config.model.calibration_days)
    assert train.label_known_on.max() < cal_start
    assert cal.label_known_on.max() < tune_start
    assert tune.label_known_on.max() <= cutoff
    recent = d.loc[d.date <= cutoff].tail(5).copy()
    recent["label_known_on"] = cutoff + pd.Timedelta(days=1)
    assert mature_rows(recent, cutoff).empty


def test_unknown_config_fails_instead_of_silently_ignoring_typo():
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"model": {"iteration": 100}})
