"""Past-only features on actual source updates; calendar facts use calendar windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import daily_rates


def _streak(values):
    out, count = [], 0
    for value in values:
        count = count + 1 if value < 0 else 0
        out.append(count)
    return np.array(out)


def build_features(panel: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, list[str]]:
    frames, fc = [], config.features
    holidays = None
    if config.data.holidays_file:
        holidays = pd.read_csv(config.data.holidays_file, parse_dates=["date", "known_on"])
        if not {"iso", "date", "known_on", "name"} <= set(holidays):
            raise ValueError("Holiday file needs iso,date,known_on,name")
        if holidays[["date", "known_on"]].isna().any().any():
            raise ValueError("Every holiday needs its historical known_on date")
    for iso, group in panel.groupby("iso", sort=True):
        g = group.sort_values("date").reset_index(drop=True).copy()
        p = g.rub_per_unit
        ret = np.log(p).diff()
        g["ret_1"] = ret
        for w in sorted(set(fc.windows + [fc.level_window, 5, 20, 60])):
            g[f"return_{w}"] = np.log(p / p.shift(w))
            # Compare current price to previous observations, not a high-low range.
            g[f"percentile_{w}"] = p.rolling(w + 1, min_periods=w + 1).apply(
                lambda x: (np.sum(x[:-1] < x[-1]) + 0.5 * np.sum(x[:-1] == x[-1])) / w,
                raw=True,
            )
            g[f"median_gap_{w}"] = (p / p.shift(1).rolling(w).median() - 1) * 10000
            g[f"vol_{w}"] = ret.rolling(w).std(ddof=0)
        g["down_streak"] = _streak(ret)
        g["ewma_return"] = ret.ewm(alpha=config.model.ets_alpha, adjust=False).mean()
        # Classical baselines can update on observed prices without waiting for labels.
        # These sufficient statistics also remain strictly prefix-invariant.
        sw = config.model.statistical_window
        current = ret.rolling(sw, min_periods=min(60, sw))
        lagged = ret.shift(1).rolling(sw, min_periods=min(60, sw))
        covariance = current.cov(ret.shift(1), ddof=0)
        g["stat_mean"] = current.mean()
        g["stat_sigma"] = current.std(ddof=0).clip(lower=1e-6)
        g["stat_ar_phi"] = (covariance / lagged.var(ddof=0).clip(lower=1e-12)).clip(-0.95, 0.95)
        g["stat_ar_intercept"] = current.mean() - g.stat_ar_phi * lagged.mean()
        g["stat_ar_sigma"] = np.sqrt(
            (
                current.var(ddof=0)
                + g.stat_ar_phi**2 * lagged.var(ddof=0)
                - 2 * g.stat_ar_phi * covariance
            ).clip(lower=1e-12)
        )
        g["acceleration"] = ret - ret.shift(1)
        g["rebound_bps"] = (p / p.shift(1).rolling(20).min() - 1) * 10000
        g["level_rank"] = g[f"percentile_{fc.level_window}"]
        g["vol_ratio"] = g.vol_20 / g.vol_60.clip(lower=1e-8)
        g["shock_z"] = ret.abs() / ret.shift(1).rolling(60).std(ddof=0).clip(lower=1e-8)
        g["update_gap_days"] = g.date.diff().dt.days
        g["updates_in_30_days"] = g.set_index("date").rub_per_unit.rolling("30D").count().values
        g["trend_strength"] = g.return_20 / (g.vol_20.clip(lower=1e-8) * np.sqrt(20))
        g["regime"] = np.select(
            [g.shock_z > fc.shock_z, g.vol_ratio > 1.5, g.trend_strength.abs() > 1.5],
            ["shock", "volatile", "trend"],
            default="range",
        )
        g["slow_confirmed"] = (g.ret_1 > 0) & (g.rebound_bps >= fc.reversal_bps)
        g["fast_momentum"] = g.down_streak >= fc.momentum_streak
        g["month_sin"] = np.sin(2 * np.pi * g.date.dt.month / 12)
        g["month_cos"] = np.cos(2 * np.pi * g.date.dt.month / 12)
        g["weekday"] = g.date.dt.dayofweek
        g["year_sin"] = np.sin(2 * np.pi * g.date.dt.dayofyear / 365.25)
        g["year_cos"] = np.cos(2 * np.pi * g.date.dt.dayofyear / 365.25)
        g["holiday_distance"] = 366.0
        if holidays is not None:
            h = holidays.loc[holidays.iso == iso]
            for idx, dt in g.date.items():
                known = h.loc[(h.known_on <= dt) & (h.date >= dt), "date"]
                if not known.empty:
                    g.loc[idx, "holiday_distance"] = min((known.min() - dt).days, 366)
        daily = daily_rates(group, config.data.max_stale_days).rub_per_unit
        g["week_change_bps"] = (
            p.to_numpy() / daily.reindex(g.date - pd.Timedelta(days=7)).to_numpy() - 1
        ) * 10000
        # Text percentile counts strict advantage over historical calendar days.
        text_percent = daily.rolling(
            fc.text_window_days + 1, min_periods=fc.text_window_days + 1
        ).apply(lambda x: np.mean(x[:-1] > x[-1]) * 100, raw=True)
        g["text_better_pct"] = text_percent.reindex(g.date).to_numpy()
        g["past_min_primary"] = (
            daily.rolling(config.targets.primary_horizon + 1).min().reindex(g.date).to_numpy()
        )
        g["past_sum_primary"] = (
            daily.rolling(config.targets.primary_horizon + 1).sum().reindex(g.date).to_numpy()
        )
        g["history_updates"] = np.arange(1, len(g) + 1)
        frames.append(g)
    all_features = pd.concat(frames, ignore_index=True).sort_values(["date", "iso"])
    out = all_features.loc[all_features.iso.isin(config.data.corridors)].copy()
    for iso in config.data.context:
        ctx = all_features.loc[
            all_features.iso == iso, ["date", "return_5", "return_20", "vol_20"]
        ].copy()
        ctx[f"ctx_{iso}_date"] = ctx.date
        ctx = ctx.rename(columns={c: f"ctx_{iso}_{c}" for c in ["return_5", "return_20", "vol_20"]})
        out = pd.merge_asof(
            out.sort_values("date"), ctx.sort_values("date"), on="date", direction="backward"
        )
        out[f"ctx_{iso}_age"] = (out.date - out[f"ctx_{iso}_date"]).dt.days
        out = out.drop(columns=f"ctx_{iso}_date")
    excluded = {
        "date",
        "effective_date",
        "nominal",
        "rate",
        "rub_per_unit",
        "week_change_bps",
        "text_better_pct",
        "past_min_primary",
        "past_sum_primary",
        "history_updates",
    }
    feature_columns = [c for c in out if c not in excluded]
    numeric = [c for c in feature_columns if c not in ("iso", "regime")]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    out["eligible"] = (out.history_updates > max(fc.windows + [fc.level_window, 60])) & (
        out.updates_in_30_days >= 8
    )
    out["eligible"] &= out[numeric].notna().all(axis=1)
    for iso in config.data.context:
        out["eligible"] &= out[f"ctx_{iso}_age"] <= config.data.max_stale_days
    return out.sort_values(["date", "iso"]).reset_index(drop=True), feature_columns
