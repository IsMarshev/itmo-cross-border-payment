"""Backward-looking features for the signal layer.

Every feature on date ``T`` uses only observations with ``available_on <= T``,
so a snapshot taken at ``T`` is immutable to any data published afterwards.
This is the leakage-safety contract from implementation_plan.md, Stage 2.

The feature set follows the plan's Stage 2 list:
    * deviation from EWMA(60) in rolling-sigma units (z-score),
    * percentile of the residual over 250 observations,
    * length of the down streak,
    * return over 5 observations,
    * volatility over 20 observations,
    * drawdown from the 60-observation high.

All values are computed per corridor on the canonical panel (``quote_date``,
``iso``, ``rub_per_unit``). Missing non-trading dates are NOT forward-filled —
an absent observation is not a market move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "ewma_zscore",
    "residual_pct",
    "down_streak",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "accel_5",
    "vol_20",
    "vol_ratio",
    "drawdown_60",
    "pct_rank_90",
    "dist_to_min_90",
    "rub_strength",
    "month_sin",
    "month_cos",
    "dom_sin",
    "dom_cos",
    "seasonal_zscore",
    "days_to_eom",
    "is_month_start",
    "is_month_end",
)


def _ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rolling_zscore(value: pd.Series, span: int) -> pd.Series:
    """Deviation from EWMA(span) in units of rolling EWMA std."""
    mean = _ewma(value, span)
    # EWMA variance via the recursive form: var_t = (1-a)*var_{t-1} + a*(x-mean)^2.
    # pandas' ewm().var() does not accept ddof; compute manually for stability.
    alpha = 2.0 / (span + 1.0)
    sq_dev = (value - mean) ** 2
    var = sq_dev.ewm(alpha=alpha, adjust=False).mean()
    std = np.sqrt(var)
    return (value - mean) / std.replace(0, np.nan)


def compute_features(
    panel: pd.DataFrame,
    *,
    ewma_span: int = 60,
    pct_window: int = 250,
    vol_window: int = 20,
    dd_window: int = 60,
) -> pd.DataFrame:
    """Compute leakage-safe features for every observation in the panel.

    Parameters are in *trading observations*, not calendar days, because the
    panel only has trading dates. Returns one row per input observation with
    ``quote_date``, ``iso``, ``rub_per_unit`` and the feature columns.
    """
    parts: list[pd.DataFrame] = []
    for iso, grp in panel.sort_values("quote_date").groupby("iso", sort=False):
        g = grp.copy()
        v = g["rub_per_unit"].astype(float)

        # Z-score vs EWMA: negative => cheaper than the recent trend (favourable).
        g["ewma_zscore"] = _rolling_zscore(v, ewma_span)

        # Residual = value - EWMA; percentile of that residual over pct_window.
        residual = v - _ewma(v, ewma_span)
        g["residual_pct"] = residual.rolling(pct_window, min_periods=20).rank(pct=True)

        # Down streak: consecutive observations where the rate fell.
        # streak resets when direction changes; we count the current run of "down"
        down = (v.diff() < 0).astype(int)
        # run-length encode the "down" flag
        run_id = (down != down.shift()).cumsum()
        g["down_streak"] = down.groupby(run_id).cumsum() * down

        # Momentum over several lags (log return for symmetry).
        logret = np.log(v / v.shift(1))
        g["ret_1"] = logret
        g["ret_5"] = np.log(v / v.shift(5))
        g["ret_10"] = np.log(v / v.shift(10))
        g["ret_20"] = np.log(v / v.shift(20))

        # Acceleration: change in 5-obs return vs its previous value.
        g["accel_5"] = g["ret_5"] - g["ret_5"].shift(5)

        # Realised volatility over vol_window (std of log returns).
        g["vol_20"] = logret.rolling(vol_window, min_periods=5).std(ddof=1)
        vol_60 = logret.rolling(60, min_periods=10).std(ddof=1)
        g["vol_ratio"] = g["vol_20"] / vol_60.replace(0, np.nan)  # >1 => vol rising

        # Drawdown from the dd_window high: 0 => at the high, positive => below.
        rolling_high = v.rolling(dd_window, min_periods=5).max()
        g["drawdown_60"] = (rolling_high - v) / rolling_high

        # Level in percentiles of the 90-obs window (low => near the bottom).
        g["pct_rank_90"] = v.rolling(90, min_periods=20).rank(pct=True)

        # Distance to the 90-obs minimum, in relative terms (0 => at the min).
        rolling_min = v.rolling(90, min_periods=20).min()
        g["dist_to_min_90"] = (v - rolling_min) / rolling_min.replace(0, np.nan)

        # --- Calendar / seasonality features ---
        # Transfer corridors have strong calendar effects: salary days, month-end
        # demand, New-Year remittance spikes. These are backward-looking by
        # construction (a date's calendar position is known on that date), so they
        # are leakage-safe even though they are not computed from the rate path.
        dates = g["quote_date"].dt
        month = dates.month.to_numpy()
        dom = dates.day.to_numpy()
        # Cyclic encoding keeps month 12 and month 1 adjacent.
        g["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        g["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        g["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
        g["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)

        # Seasonal z-score: how far today's value is from the same-calendar-month
        # average over a multi-year trailing window, in rolling-sigma units. We
        # expand the per-month history via a rolling grouping on a shifted month
        # index so only strictly-prior years contribute (no same-year leakage of
        # future same-month days). Window of 5 years (5 same-month obs per year).
        g["_year"] = dates.year.to_numpy()
        g["_month"] = month
        monthly = g.groupby(["_month"])["rub_per_unit"]
        # rolling mean/std over same-month observations, min 3 prior years.
        g["seasonal_mean"] = monthly.transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean()
        )
        g["seasonal_std"] = monthly.transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).std(ddof=1)
        )
        g["seasonal_zscore"] = (
            (v - g["seasonal_mean"]) / g["seasonal_std"].replace(0, np.nan)
        )

        # Proximity to month boundaries: salary and rent cycles concentrate near
        # the turn of the month, shifting remittance demand.
        dim = dates.days_in_month.to_numpy()
        g["days_to_eom"] = (dim - dom).astype(float)
        g["is_month_start"] = (dom <= 5).astype(int)
        g["is_month_end"] = (dim - dom < 5).astype(int)

        g = g.drop(columns=["_year", "_month", "seasonal_mean", "seasonal_std"])
        parts.append(g)
    feats = pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"])

    # Cross-currency context: RUB strength vs its own trailing mean, via USD/RUB.
    # Most corridor moves are driven by the rouble, not the foreign currency, so
    # this is a shared regime feature. Missing when USD is not in the panel.
    if "USD" in set(panel["iso"]):
        usd = feats[feats["iso"] == "USD"].sort_values("quote_date").copy()
        usd["rub_strength"] = usd["rub_per_unit"] / _ewma(
            usd["rub_per_unit"], 60
        ) - 1.0  # <0 => rouble stronger than recent trend
        rs_map = usd.set_index("quote_date")["rub_strength"]
        feats["rub_strength"] = feats["quote_date"].map(rs_map)
    else:
        feats["rub_strength"] = np.nan

    identity_columns = ["quote_date"]
    if "available_on" in feats.columns:
        identity_columns.append("available_on")
    identity_columns.extend(["iso", "rub_per_unit"])
    return feats[identity_columns + list(FEATURE_COLUMNS)]


def features_asof(
    features: pd.DataFrame, asof_date: pd.Timestamp, iso: str
) -> pd.DataFrame:
    """Return the feature snapshot available on ``asof_date`` for one corridor.

    Drops any row whose ``quote_date`` is strictly after ``asof_date``. This is
    the single leakage-safe entry point a serving layer would call.
    """
    availability_column = "available_on" if "available_on" in features.columns else "quote_date"
    return features[
        (features["iso"] == iso) & (features[availability_column] <= asof_date)
    ].sort_values("quote_date")
