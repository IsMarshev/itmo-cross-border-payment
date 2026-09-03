"""Rule-based signal indicators (the "fast/slow" library from the case brief).

These complement the Ridge model: simple, fully explainable rules that buy when
the rate is cheap by some backward-looking criterion. The brief (Stage 2, point
7) frames this as fast vs slow indicators — a *fast* one fires early on a dip
(noiser), a *slow* one waits for confirmation (later, more reliable). The
waiting cost between them is measured separately.

Every indicator is a function taking the canonical panel and returning a signals
frame ``[iso, signal_date, rub_per_unit]``. All use only past observations
(rolling windows over ``quote_date``), so a snapshot at date ``T`` is immutable
to future data — the same leakage contract as ``features.py``.

Indicators
----------
value(window, lo_pct)
    Rate in the bottom ``lo_pct`` of the rolling ``window`` high-low range.
    *Fast*: buys the dip without waiting for a turn.
momentum(streak)
    Rate has fallen ``streak`` observations in a row. *Fast*: catches a move
    already in progress.
value_reversal(window, lo_pct)
    VALUE plus the rate ticked up on the signal day (a turn from the dip).
    *Medium*: confirmation that the low is closing.
reversal_from_min(window)
    Rate was within ``pct`` of the rolling ``window`` minimum and ticked up.
    *Slow*: the brief's "window closing" scenario.
model_ridge
    Wrapper over the existing Ridge value-head (``models.walk_forward_predict``),
    included so the matrix compares rules against the learned model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import models
from .run_m0 import _signals_from_predictions


@dataclass(frozen=True)
class IndicatorSpec:
    """A named indicator with bound parameters and a callable."""

    name: str
    params: dict
    fn: Callable[..., pd.DataFrame]
    speed: str  # "fast" | "medium" | "slow" | "model"


def _signals_from_mask(
    grp: pd.DataFrame, mask: np.ndarray, iso: str
) -> pd.DataFrame:
    """Build a signals frame from a boolean mask aligned to a sorted corridor."""
    hit = grp.loc[mask, ["quote_date", "rub_per_unit"]].rename(
        columns={"quote_date": "signal_date"}
    )
    if hit.empty:
        return pd.DataFrame(columns=["iso", "signal_date", "rub_per_unit"])
    hit["iso"] = iso
    return hit[["iso", "signal_date", "rub_per_unit"]]


def value(
    panel: pd.DataFrame, iso: str, *, window: int = 60, lo_pct: float = 0.20
) -> pd.DataFrame:
    """Rate sits in the bottom ``lo_pct`` of the rolling high-low range."""
    g = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    v = g["rub_per_unit"].astype(float)
    lo = v.rolling(window, min_periods=max(5, window // 4)).min()
    hi = v.rolling(window, min_periods=max(5, window // 4)).max()
    span = (hi - lo).replace(0, np.nan)
    pct = (v - lo) / span
    mask = (pct < lo_pct).to_numpy()
    return _signals_from_mask(g, mask, iso)


def momentum(panel: pd.DataFrame, iso: str, *, streak: int = 3) -> pd.DataFrame:
    """Rate has fallen ``streak`` observations in a row."""
    g = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    v = g["rub_per_unit"].astype(float)
    down = (v.diff() < 0).astype(int)
    run_id = (down != down.shift()).cumsum()
    down_streak = down.groupby(run_id).cumsum() * down
    mask = (down_streak >= streak).to_numpy()
    return _signals_from_mask(g, mask, iso)


def value_reversal(
    panel: pd.DataFrame, iso: str, *, window: int = 60, lo_pct: float = 0.30
) -> pd.DataFrame:
    """VALUE plus the rate ticked up on the signal day (a turn from the dip)."""
    g = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    v = g["rub_per_unit"].astype(float)
    lo = v.rolling(window, min_periods=max(5, window // 4)).min()
    hi = v.rolling(window, min_periods=max(5, window // 4)).max()
    span = (hi - lo).replace(0, np.nan)
    pct = (v - lo) / span
    up = (v.diff() > 0).to_numpy()
    mask = (pct < lo_pct).to_numpy() & up
    return _signals_from_mask(g, mask, iso)


def reversal_from_min(
    panel: pd.DataFrame, iso: str, *, window: int = 60, pct: float = 0.02
) -> pd.DataFrame:
    """Rate was within ``pct`` of the rolling min and ticked up — "window closing"."""
    g = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    v = g["rub_per_unit"].astype(float)
    lo = v.rolling(window, min_periods=max(5, window // 4)).min()
    near_min = (v - lo) / lo.replace(0, np.nan) < pct
    up = (v.diff() > 0).to_numpy()
    mask = near_min.to_numpy() & up
    return _signals_from_mask(g, mask, iso)


def model_ridge(
    panel: pd.DataFrame,
    iso: str,
    *,
    h: int = 20,
    alpha: float = 1.0,
    min_train: int = 500,
    slots_per_week: float = 1.5,
) -> pd.DataFrame:
    """The existing Ridge value-head, thresholded into signals (the m0 baseline)."""
    pred = models.walk_forward_predict(panel, iso, h=h, alpha=alpha, min_train=min_train)
    signals = _signals_from_predictions(pred, slots_per_week=slots_per_week)
    if len(signals):
        rmap = panel[panel["iso"] == iso].set_index("quote_date")["rub_per_unit"]
        signals["rub_per_unit"] = signals["signal_date"].map(rmap).astype(float)
    return signals


# --- Indicator registry for the matrix CLI ---

# Parameter grids for per-corridor walk-forward calibration.
GRID_VALUE = {"window": [20, 60, 120], "lo_pct": [0.10, 0.20, 0.30]}
GRID_MOMENTUM = {"streak": [3, 5]}
GRID_VALUE_REVERSAL = {"window": [20, 60, 120], "lo_pct": [0.20, 0.30]}
GRID_REVERSAL_FROM_MIN = {"window": [60, 120], "pct": [0.01, 0.02]}


def _grid_params(grid: dict) -> list[dict]:
    """Expand a param grid into a list of param dicts."""
    keys = list(grid)
    combos = [{}]
    for k in keys:
        combos = [{**c, k: val} for c in combos for val in grid[k]]
    return combos


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("value", {}, value, "fast"),
    IndicatorSpec("momentum", {}, momentum, "fast"),
    IndicatorSpec("value_reversal", {}, value_reversal, "medium"),
    IndicatorSpec("reversal_from_min", {}, reversal_from_min, "slow"),
    IndicatorSpec("model_ridge", {}, model_ridge, "model"),
)


def grid_for(spec: IndicatorSpec) -> list[dict]:
    """The calibration grid for an indicator, or a single empty dict if none."""
    grids = {
        "value": GRID_VALUE,
        "momentum": GRID_MOMENTUM,
        "value_reversal": GRID_VALUE_REVERSAL,
        "reversal_from_min": GRID_REVERSAL_FROM_MIN,
    }
    grid = grids.get(spec.name)
    return [{}] if grid is None else _grid_params(grid)


def _trim_signals(
    signals: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.DataFrame:
    """Restrict signals to the [start, end] window (the backtest window)."""
    if signals.empty:
        return signals
    s = signals.reset_index(drop=True)
    dates = pd.to_datetime(s["signal_date"])
    mask = pd.Series(True, index=s.index)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return s[mask].reset_index(drop=True)


def evaluate_indicator(
    panel: pd.DataFrame,
    iso: str,
    spec: IndicatorSpec,
    params: dict,
    *,
    monthly_budget: float = 50_000.0,
    cadence_days: int = 5,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    """Run one indicator with one param set; return its metrics + uplift vs DCA.

    Truthfulness is scored under *both* hit rules: ``favourable_now`` (the rate
    stayed no worse — the right rule for "buy cheap" value/momentum indicators)
    and ``window_closing`` (the rate rose — the right rule for reversal/"the
    window is closing" indicators). ``lift_h5`` below is the lift under the rule
    that matches the indicator's own message; the other is reported as
    ``lift_h5_other`` for cross-comparison.
    """
    from . import metrics, simulation

    signals = spec.fn(panel, iso, **params)
    win_signals = _trim_signals(signals, start, end)
    res = simulation.simulate_strategies(
        panel, iso, signals, monthly_budget=monthly_budget,
        cadence_days=cadence_days, start=start, end=end,
    )
    m, d, r = res["model"], res["dca"], res["random"]
    uplift_dca = (
        (m.total_currency - d.total_currency) / d.total_currency * 100
        if d.total_currency else float("nan")
    )
    uplift_rand = (
        (m.total_currency - r.total_currency) / r.total_currency * 100
        if r.total_currency else float("nan")
    )

    # Truthfulness metrics on the backtest window only (no full-history leakage
    # into n_signals / frequency). total_days counts trading days in the window.
    win_days = 0
    if len(win_signals):
        grp = panel[panel["iso"] == iso]
        if start is not None:
            grp = grp[grp["quote_date"] >= start]
        if end is not None:
            grp = grp[grp["quote_date"] <= end]
        win_days = grp["quote_date"].nunique()

    # The hit rule that matches this indicator's message.
    own_scenario = "window_closing" if spec.speed in ("slow", "model") else "favourable_now"
    other_scenario = "favourable_now" if own_scenario == "window_closing" else "window_closing"

    own_df, freq_df = metrics.evaluate(
        panel, win_signals, horizons=(5, 15), eps_bps=0.0,
        total_days=win_days or None, scenario=own_scenario,
    )
    other_df, _ = metrics.evaluate(
        panel, win_signals, horizons=(5, 15), eps_bps=0.0,
        total_days=win_days or None, scenario=other_scenario,
    )
    row5 = own_df[own_df["horizon"] == 5]
    row15 = own_df[own_df["horizon"] == 15]
    lift5 = float(row5["lift"].iloc[0]) if len(row5) else float("nan")
    hit5 = float(row5["hit_rate"].iloc[0]) if len(row5) else float("nan")
    hit15 = float(row15["hit_rate"].iloc[0]) if len(row15) else float("nan")
    orow5 = other_df[other_df["horizon"] == 5]
    lift5_other = float(orow5["lift"].iloc[0]) if len(orow5) else float("nan")
    per_week = float(freq_df["per_week"].iloc[0]) if len(freq_df) else float("nan")
    series_share = float(freq_df["series_share"].iloc[0]) if len(freq_df) else float("nan")

    return {
        "indicator": spec.name,
        "iso": iso,
        "speed": spec.speed,
        "scenario": own_scenario,
        "params": params,
        "n_signals": len(win_signals),
        "uplift_vs_dca": uplift_dca,
        "uplift_vs_random": uplift_rand,
        "lift_h5": lift5,
        "lift_h5_other": lift5_other,
        "hit_rate_h5": hit5,
        "hit_rate_h15": hit15,
        "per_week": per_week,
        "series_share": series_share,
    }


def calibrate(
    panel: pd.DataFrame,
    iso: str,
    spec: IndicatorSpec,
    param_grid: list[dict],
    *,
    monthly_budget: float = 50_000.0,
    cadence_days: int = 5,
    n_folds: int = 4,
) -> dict:
    """Walk-forward grid search: pick the params with best median uplift across folds.

    Splits the trailing history into ``n_folds`` expanding windows and scores
    each param combo by median uplift-vs-DCA across folds (median, not mean, so
    one lucky fold cannot dominate). Returns the best params and the per-fold
    scores. Indicators with an empty grid (e.g. model_ridge) return ``{}``.
    """
    if not param_grid or param_grid == [{}]:
        return {"best_params": {}, "scores": {}}
    last = panel[panel["iso"] == iso]["quote_date"].max()
    fold_span = pd.DateOffset(months=6)
    fold_starts = [last - fold_span * (i + 1) for i in range(n_folds)][::-1]

    scores: dict[str, list[float]] = {}
    for params in param_grid:
        key = _params_key(params)
        scores[key] = []
        for fs in fold_starts:
            if pd.Timestamp(panel[panel["iso"] == iso]["quote_date"].min()) >= fs:
                continue
            try:
                ev = evaluate_indicator(
                    panel, iso, spec, params,
                    monthly_budget=monthly_budget, cadence_days=cadence_days,
                    start=fs, end=last,
                )
                if np.isfinite(ev["uplift_vs_dca"]):
                    scores[key].append(ev["uplift_vs_dca"])
            except (ValueError, KeyError):
                continue

    # Pick best by median uplift; require at least 2 folds with a score.
    best_key, best_med = None, -np.inf
    for key, vals in scores.items():
        if len(vals) >= 2:
            med = float(np.median(vals))
            if med > best_med:
                best_med, best_key = med, key
    best_params = _key_to_params(best_key) if best_key else param_grid[0]
    return {"best_params": best_params, "scores": scores}


def _params_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _key_to_params(key: str) -> dict:
    if not key:
        return {}
    out = {}
    for part in key.split(","):
        k, v = part.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def waiting_cost(
    panel: pd.DataFrame,
    iso: str,
    fast_signals: pd.DataFrame,
    slow_signals: pd.DataFrame,
) -> float:
    """Median bps the rate moved between a fast signal and the next slow one.

    For each fast signal, find the nearest *later* slow signal; the cost of
    waiting for confirmation is how much the rate rose (got worse for the buyer)
    in between, in basis points. Negative means waiting actually got a better
    rate. Returns the median across matched pairs.
    """
    if fast_signals.empty or slow_signals.empty:
        return float("nan")
    g = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    rate_by_date = dict(zip(g["quote_date"], g["rub_per_unit"]))
    fast_dates = pd.to_datetime(fast_signals["signal_date"]).sort_values()
    slow_dates = pd.to_datetime(slow_signals["signal_date"]).sort_values()
    costs = []
    for fd in fast_dates:
        # Strictly later: the slow confirmation comes *after* the fast fire, so
        # the wait interval (and its cost) is real. Same-day matches give 0.
        later = slow_dates[slow_dates > fd]
        if later.empty:
            continue
        sd = later.iloc[0]
        fr = rate_by_date.get(pd.Timestamp(fd))
        sr = rate_by_date.get(pd.Timestamp(sd))
        if fr is None or sr is None or fr == 0:
            continue
        costs.append((sr - fr) / fr * 10_000.0)
    if not costs:
        return float("nan")
    return float(np.median(costs))
