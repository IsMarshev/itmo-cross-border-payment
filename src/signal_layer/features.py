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
    "ret_5",
    "vol_20",
    "drawdown_60",
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
    ret_lag: int = 5,
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

        # Return over ret_lag observations (log return for symmetry).
        g["ret_5"] = np.log(v / v.shift(ret_lag))

        # Realised volatility over vol_window (std of log returns).
        logret = np.log(v / v.shift(1))
        g["vol_20"] = logret.rolling(vol_window, min_periods=5).std(ddof=1)

        # Drawdown from the dd_window high: 0 => at the high, positive => below.
        rolling_high = v.rolling(dd_window, min_periods=5).max()
        g["drawdown_60"] = (rolling_high - v) / rolling_high

        parts.append(g)
    feats = pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"])
    return feats[list(("quote_date", "iso", "rub_per_unit") + FEATURE_COLUMNS)]


def features_asof(
    features: pd.DataFrame, asof_date: pd.Timestamp, iso: str
) -> pd.DataFrame:
    """Return the feature snapshot available on ``asof_date`` for one corridor.

    Drops any row whose ``quote_date`` is strictly after ``asof_date``. This is
    the single leakage-safe entry point a serving layer would call.
    """
    return features[
        (features["iso"] == iso) & (features["quote_date"] <= asof_date)
    ].sort_values("quote_date")
