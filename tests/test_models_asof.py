from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.models import build_dataset, predict_asof, walk_forward_predict


def _model_panel(n: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    position = np.arange(n)
    usd = 90.0 + 0.01 * position + np.sin(position / 17)
    tjs = 8.2 + 0.002 * position + 0.1 * np.sin(position / 13)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "quote_date": dates,
                    "available_on": dates + pd.offsets.Day(1),
                    "iso": iso,
                    "rub_per_unit": rates,
                }
            )
            for iso, rates in (("USD", usd), ("TJS", tjs))
        ],
        ignore_index=True,
    )


def test_walk_forward_uses_only_targets_matured_by_decision_time() -> None:
    panel = _model_panel()
    dataset = build_dataset(panel, "TJS", h=20, include_unlabelled=True)

    predictions = walk_forward_predict(panel, "TJS", h=20, min_train=50)

    first = predictions.iloc[0]
    eligible = dataset.loc[
        dataset["advantage"].notna()
        & (dataset["target_available_on"] <= first["available_on"])
        & (dataset["quote_date"] < first["quote_date"])
    ]
    assert first["training_observations"] == len(eligible)
    assert first["training_observations"] >= 50
    assert eligible["target_available_on"].max() <= first["available_on"]


def test_predict_asof_is_unchanged_by_unavailable_future_rows() -> None:
    panel = _model_panel()
    as_of = panel.loc[panel["iso"].eq("TJS"), "available_on"].iloc[350]

    snapshot_prediction = predict_asof(
        panel.loc[panel["available_on"] <= as_of],
        "TJS",
        as_of,
        h=20,
        min_train=50,
    )
    full_prediction = predict_asof(panel, "TJS", as_of, h=20, min_train=50)

    assert full_prediction.quote_date == snapshot_prediction.quote_date
    assert full_prediction.training_observations == snapshot_prediction.training_observations
    assert full_prediction.predicted_advantage_bps == pytest.approx(
        snapshot_prediction.predicted_advantage_bps
    )
