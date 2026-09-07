from __future__ import annotations

import numpy as np
import pandas as pd

from signal_layer.features import FEATURE_COLUMNS, compute_features, features_asof


def _toy_panel(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Random walk with a slight drift so features vary.
    rates = 10.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame(
        {
            "quote_date": dates,
            "iso": "TJS",
            "rub_per_unit": rates,
        }
    )


def test_features_have_no_lookahead() -> None:
    """Adding future observations must not change features available at date T."""
    panel = _toy_panel(300)
    feats = compute_features(panel)

    cutoff = pd.Timestamp("2020-06-01")
    snapshot_before = features_asof(feats, cutoff, "TJS")

    # Append 60 future observations and recompute.
    more = _toy_panel(60, seed=1)
    more["quote_date"] = pd.bdate_range("2021-01-01", periods=60)
    panel_extended = pd.concat([panel, more], ignore_index=True)
    feats_ext = compute_features(panel_extended)
    snapshot_after = features_asof(feats_ext, cutoff, "TJS")

    pd.testing.assert_frame_equal(
        snapshot_before.reset_index(drop=True),
        snapshot_after.reset_index(drop=True),
        check_dtype=False,
    )


def test_features_columns_present() -> None:
    panel = _toy_panel(300)
    feats = compute_features(panel)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    # Each backward-looking feature has a warm-up window only at the start.
    # rub_strength is NaN when USD is not in the panel; the rest must be filled.
    non_usd_cols = tuple(c for c in FEATURE_COLUMNS if c != "rub_strength")
    for col in non_usd_cols:
        assert feats[col].notna().sum() > 200


def test_features_asof_uses_publication_date_when_present() -> None:
    panel = _toy_panel(100)
    panel["available_on"] = panel["quote_date"] + pd.offsets.Day(2)
    features = compute_features(panel)
    as_of = panel.iloc[50]["quote_date"] + pd.offsets.Day(1)

    snapshot = features_asof(features, as_of, "TJS")

    assert snapshot["available_on"].max() <= as_of
    assert panel.iloc[50]["quote_date"] not in set(snapshot["quote_date"])
