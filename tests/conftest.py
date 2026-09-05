from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_layer.config import Config
from signal_layer.engine import SignalEngine


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("rates")
    dates = pd.date_range("2020-01-01", "2023-03-01")
    dates = dates[dates.dayofweek.isin([1, 2, 3, 4, 5])]
    for j, iso in enumerate(["TJS", "KZT", "USD"]):
        t = np.arange(len(dates))
        price = (10 + j * 5) * np.exp(0.025 * np.sin(t / 5) + 0.012 * np.sin(t / 21) + t * 0.00002)
        nominal = np.where(t < 350, 10, 100)
        pd.DataFrame(
            {
                "date": dates,
                "iso": iso,
                "nominal": nominal,
                "rate": price * nominal,
                "rate_per_unit": price,
            }
        ).to_csv(root / f"rates_{iso}.csv", index=False)
    c = Config()
    c.data.directory, c.data.start = str(root), "2020-01-01"
    c.data.corridors, c.data.context = ["TJS", "KZT"], ["USD"]
    c.model.methods = ["linear", "random_walk", "rule_value", "random_policy"]
    c.model.iterations, c.model.depth, c.model.threads = 8, 3, 1
    c.model.min_train_rows, c.model.min_calibration_rows = 100, 20
    c.model.train_window_days = 500
    c.model.calibration_days = c.model.tuning_days = 100
    c.model.simulation_paths = 24
    c.policy.level_thresholds, c.policy.probability_thresholds = [0.35], [0.25]
    c.policy.contact_costs_bps, c.policy.min_tuning_signals = [0.0], 1
    c.backtest.start, c.backtest.end = "2022-06-01", "2022-07-30"
    c.backtest.fold_days, c.backtest.holdout_days = 30, 30
    c.backtest.random_repeats, c.backtest.bootstrap_samples = 2, 20
    c.backtest.ablations = False
    return c.validate()


@pytest.fixture(scope="session")
def prepared(synthetic_config):
    return SignalEngine(synthetic_config).prepare()


@pytest.fixture(scope="session")
def completed_run(synthetic_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("run_parent") / "backtest"
    SignalEngine(synthetic_config).backtest(out, progress=lambda _: None)
    return out
