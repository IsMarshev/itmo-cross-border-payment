from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from signal_layer import SignalEngine
from signal_layer.models import fit_model
from signal_layer.training import temporal_partitions


def test_backtest_outputs_and_budget(completed_run):
    required = [
        "dashboard.html",
        "summary.csv",
        "fold_metrics.csv",
        "signals.csv",
        "decisions.csv.gz",
        "outcomes.csv.gz",
        "diagnostics.csv",
        "calibration.csv",
        "waiting_episodes.csv",
        "random_day_draws.csv",
        "random_policy_draws.csv",
    ]
    assert all((completed_run / p).exists() for p in required)
    manifest = json.loads((completed_run / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["folds"]) == 2
    signals = pd.read_csv(completed_run / "signals.csv", parse_dates=["date"])
    assert not signals.empty
    for _, g in signals.groupby(["method", "iso"]):
        assert (g.date.sort_values().diff().dropna().dt.days >= 3).all()
        for dt in g.date:
            assert g.date.between(dt - pd.Timedelta(days=6), dt).sum() <= 2
    html = (completed_run / "dashboard.html").read_text()
    assert "__PAYLOAD__" not in html and "__PLOTLY__" not in html
    assert "<script src=" not in html and "window.__FX_REPORT_READY__" in html


@pytest.mark.parametrize("method", ["linear", "random_walk", "rule_value", "random_policy"])
def test_asof_replay_matches_archived_decisions(completed_run, method):
    e = SignalEngine.load(completed_run)
    actual = e.signals("2022-07-19", method=method)
    stored = pd.read_csv(completed_run / "decisions.csv.gz", parse_dates=["date"])
    expected = stored.loc[(stored.method == method) & (stored.date == "2022-07-19")]
    cols = [
        "date",
        "iso",
        "decision",
        "reason",
        "utility_bps",
        "upper_regret_bps",
        "upper_stale_bps",
    ]
    pd.testing.assert_frame_equal(
        actual[cols].reset_index(drop=True),
        expected[cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_future_artifact_cannot_be_used_for_past(completed_run):
    model = sorted((completed_run / "folds").glob("*/model.joblib"))[-1]
    engine = SignalEngine.load(model)
    with pytest.raises(ValueError, match="retrospectively"):
        engine.signals("2022-06-01", method="linear")


def test_non_update_day_abstains(completed_run):
    result = SignalEngine.load(completed_run).signals("2022-07-17", method="random_policy")
    assert (result.reason == "no_new_observation").all()
    assert (result.decision == "abstain").all()


def test_catboost_heads_and_calibration(prepared, synthetic_config):
    _, f, y, columns = prepared
    dataset = f.loc[f.eligible].merge(y, on=["date", "iso"])
    train, cal, test = temporal_partitions(dataset, "2022-06-01", synthetic_config)
    model = fit_model("catboost", train, cal, columns, synthetic_config)
    predictions = model.predict(test[f.columns].tail(5).reset_index(drop=True))
    for head in ["local_min", "no_regret", "hold", "close"]:
        assert predictions[f"pred_{head}"].between(0, 1).all()
    for head in ["gain_bps", "regret_bps", "stale_bps", "wait_delta_bps"]:
        assert np.isfinite(predictions[f"pred_{head}"]).all()
    assert model.risk_seed and model.predictor.importance()


def test_random_day_counts_match_actual_signal_counts(completed_run):
    random = pd.read_csv(completed_run / "random_day_draws.csv")
    sent = pd.read_csv(completed_run / "signals.csv")
    for r in random.itertuples():
        assert r.n == len(sent.loc[(sent.method == r.method) & (sent.iso == r.iso)])
