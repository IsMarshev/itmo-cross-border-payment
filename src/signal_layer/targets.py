"""Future outcomes are stored separately and carry an explicit maturity date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import daily_rates

CLASS_HEADS = ["local_min", "no_regret", "hold", "close"]
VALUE_HEADS = ["gain_bps", "regret_bps", "stale_bps", "wait_delta_bps"]


def build_targets(panel: pd.DataFrame, features: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = features[["date", "iso"]].copy()
    tc = config.targets
    for h in tc.horizons:
        for name in [
            "local_min",
            "no_regret",
            "hold",
            "close",
            "gain_bps",
            "regret_bps",
            "stale_bps",
        ]:
            out[f"{name}_h{h}"] = np.nan
    out["label_known_on"] = pd.NaT
    for iso, group in panel.loc[panel.iso.isin(config.data.corridors)].groupby("iso"):
        daily = daily_rates(group, config.data.max_stale_days).rub_per_unit
        ids = out.index[out.iso == iso]
        dates = out.loc[ids, "date"]
        for h in tc.horizons:
            centered = daily.rolling(2 * h + 1, center=True, min_periods=2 * h + 1)
            future = daily.shift(-1).iloc[::-1].rolling(h, min_periods=h)
            future_min = future.min().iloc[::-1]
            future_max = future.max().iloc[::-1]
            regret = (daily / np.minimum(daily, future_min) - 1).clip(lower=0) * 10000
            stale = (1 - daily / np.maximum(daily, future_max)).clip(lower=0) * 10000
            valid = centered.mean().notna()
            values = {
                "local_min": ((daily / centered.min() - 1) * 10000 <= tc.near_min_bps).astype(
                    float
                ),
                "no_regret": (regret <= tc.regret_tolerance_bps).astype(float),
                "hold": (stale <= tc.hold_tolerance_bps).astype(float),
                "close": ((daily.shift(-h) / daily - 1) * 10000 >= tc.closing_bps).astype(float),
                "gain_bps": (1 - daily / centered.mean()) * 10000,
                "regret_bps": regret,
                "stale_bps": stale,
            }
            for name, series in values.items():
                out.loc[ids, f"{name}_h{h}"] = series.where(valid).reindex(dates).to_numpy()
        out.loc[ids, "label_known_on"] = (dates + pd.Timedelta(days=max(tc.horizons))).to_numpy()
    for name in CLASS_HEADS + VALUE_HEADS[:-1]:
        h = tc.opening_horizon if name in ("hold", "stale_bps") else tc.primary_horizon
        out[f"y_{name}"] = out[f"{name}_h{h}"]
    out["y_wait_delta_bps"] = np.nan
    for iso, ids in out.groupby("iso").groups.items():
        g = out.loc[ids].sort_values("date")
        price = features.loc[features.iso == iso].set_index("date").rub_per_unit.reindex(g.date)
        cost = (1 - price / price.shift(-1)).to_numpy() * 10000
        risk_reduction = config.risk.regret_penalty * (g.y_regret_bps - g.y_regret_bps.shift(-1))
        stale_reduction = config.risk.stale_penalty * (g.y_stale_bps - g.y_stale_bps.shift(-1))
        valid = (g.date.shift(-1) - g.date).dt.days <= config.policy.max_wait_days
        out.loc[g.index, "y_wait_delta_bps"] = (-cost + risk_reduction + stale_reduction).where(
            valid
        )
        maturity = g.label_known_on.shift(-1).where(valid, g.label_known_on)
        out.loc[g.index, "label_known_on"] = maturity
    return out


def mature_rows(frame: pd.DataFrame, cutoff) -> pd.DataFrame:
    """A previous row is NOT trainable until its complete future label is observed."""
    return frame.loc[
        (frame.label_known_on <= pd.Timestamp(cutoff))
        & frame[[f"y_{h}" for h in CLASS_HEADS + VALUE_HEADS]].notna().all(axis=1)
    ].copy()
