"""Offline evaluation metrics for the signal layer.

Implements the truthfulness metrics required by the case brief:

* **Hit rate** — share of signals after which the message statement holds on
  horizon ``h``. For the "favourable now" message the statement is: the rate
  stays no worse (``rub_per_unit`` does not rise above the signal-day level
  beyond a tolerance ``eps``) for ``h`` observations.
* **Advantage in basis points** — how much better the signal-day rate is than
  the average rate in the ``±h`` window; must be statistically significant > 0.
* **Lift over a random day** — signal precision divided by the precision of a
  random day in the same corridor and period (target >= 1.3, floor 1.0).
* **Frequency** — signals per week / month.
* **Clustering** — share of signals that follow another signal within ``gap``
  observations, and the spread of inter-signal intervals.

All metrics are computed on a decision log: a frame with one row per signal
carrying its corridor, date, rate, and the realised future path. The log is
produced by the backtester, never by the metrics module.

The per-corridor panel is pre-indexed once into sorted ``dates`` / ``rates``
arrays so that hit-rate and lift-over-random are computed vectorially; the
random baseline (200 trials) never rebuilds Python outcome objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default evaluation horizons, in trading observations, as required by the brief.
HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class SignalOutcome:
    """One signal with its realised future path sliced from the panel."""

    iso: str
    signal_date: pd.Timestamp
    rate: float  # rub_per_unit on the signal day
    future: np.ndarray  # rub_per_unit for the next H observations (<=H available)


@dataclass
class CorridorIndex:
    """Sorted (dates, rates) arrays for one corridor, with position lookup."""

    dates: np.ndarray  # datetime64
    rates: np.ndarray  # float64 rub_per_unit aligned to dates

    def position(self, signal_date: pd.Timestamp) -> int:
        """Return the integer position of ``signal_date``, or -1 if absent."""
        idx = np.searchsorted(self.dates, np.datetime64(signal_date))
        if idx < len(self.dates) and self.dates[idx] == np.datetime64(signal_date):
            return int(idx)
        return -1


def _build_index(panel: pd.DataFrame) -> dict[str, CorridorIndex]:
    """Pre-index the panel per corridor into sorted dates/rates arrays."""
    idx: dict[str, CorridorIndex] = {}
    for iso, grp in panel.groupby("iso", sort=False):
        g = grp.sort_values("quote_date")
        idx[iso] = CorridorIndex(
            dates=g["quote_date"].to_numpy(),
            rates=g["rub_per_unit"].to_numpy(dtype=float),
        )
    return idx


def _slice_future(panel: pd.DataFrame, iso: str, signal_date: pd.Timestamp, h: int) -> np.ndarray:
    """Return up to ``h`` future rub_per_unit observations for a corridor."""
    s = panel.loc[panel["iso"] == iso].sort_values("quote_date")
    after = s[s["quote_date"] > signal_date]["rub_per_unit"].to_numpy(dtype=float)
    return after[:h]


def build_outcomes(
    panel: pd.DataFrame, signals: pd.DataFrame, h: int
) -> list[SignalOutcome]:
    """Attach a realised future path of length ``h`` to every signal."""
    panel = panel.sort_values(["iso", "quote_date"])
    outcomes = []
    for _, sig in signals.iterrows():
        future = _slice_future(panel, sig["iso"], sig["signal_date"], h)
        outcomes.append(
            SignalOutcome(
                iso=sig["iso"],
                signal_date=sig["signal_date"],
                rate=float(sig["rub_per_unit"]),
                future=future,
            )
        )
    return outcomes


def _hit_mask(
    index: dict[str, CorridorIndex],
    signals: pd.DataFrame,
    h: int,
    eps_bps: float,
) -> np.ndarray:
    """Boolean vector: True where a signal's rate held for ``h`` observations.

    A signal holds when the rate never rises above the signal-day level by more
    than ``eps_bps`` basis points within the horizon. Signals with no future at
    all are counted as failures (False). Vectorial over a corridor's index.
    """
    sig = signals.reset_index(drop=True)
    mask = np.zeros(len(sig), dtype=bool)
    for iso, grp in sig.groupby("iso", sort=False):
        ci = index.get(iso)
        if ci is None:
            continue
        for i, (_, row) in enumerate(grp.iterrows()):
            pos = ci.position(row["signal_date"])
            if pos < 0:
                continue
            future = ci.rates[pos + 1 : pos + 1 + h]
            if len(future) == 0:
                continue
            thr = float(row["rub_per_unit"]) * (1 + eps_bps / 10_000.0)
            if np.all(future <= thr + 1e-12):
                mask[grp.index[i]] = True
    return mask


def hit_rate(outcomes: list[SignalOutcome], eps_bps: float = 0.0) -> float:
    """Share of "favourable now" signals that held on their horizon."""
    if not outcomes:
        return float("nan")
    hits = 0
    for o in outcomes:
        if len(o.future) == 0:
            continue
        thr = o.rate * (1 + eps_bps / 10_000.0)
        if np.all(o.future <= thr + 1e-12):
            hits += 1
    return hits / len(outcomes)


def advantage_bps(
    index: dict[str, CorridorIndex], signals: pd.DataFrame, h: int
) -> tuple[float, float, int]:
    """Mean advantage of the signal-day rate over the ±h window, in basis points.

    ``advantage = (mean(window) - signal_rate) / signal_rate * 10_000``.
    Positive => the signal day was cheaper than the surrounding window.
    Returns ``(mean, stderr, n)`` so significance can be checked.
    """
    if signals.empty:
        return float("nan"), float("nan"), 0
    advs = []
    for iso, sig_grp in signals.groupby("iso", sort=False):
        ci = index.get(iso)
        if ci is None:
            continue
        for _, sig in sig_grp.iterrows():
            pos = ci.position(sig["signal_date"])
            if pos < 0:
                continue
            lo = max(0, pos - h)
            hi = min(len(ci.rates), pos + h + 1)
            window = ci.rates[lo:hi]
            if len(window) < 2:
                continue
            sig_rate = float(sig["rub_per_unit"])
            advs.append((window.mean() - sig_rate) / sig_rate * 10_000.0)
    if not advs:
        return float("nan"), float("nan"), 0
    arr = np.asarray(advs)
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return float(arr.mean()), stderr, len(arr)


def signals_columns() -> list[str]:
    return ["iso", "signal_date", "rub_per_unit"]


def lift_over_random(
    index: dict[str, CorridorIndex],
    signals: pd.DataFrame,
    h: int,
    eps_bps: float = 0.0,
    n_random_trials: int = 200,
    seed: int = 0,
) -> float:
    """Signal hit rate divided by the hit rate of a random day, matched per corridor.

    Lift ~ 1.0 means the signal is indistinguishable from a random day.
    Target is a stable lift >= 1.3 across corridors and out-of-time windows.

    Random baselines draw, per corridor, as many random trading days as the
    model signalled, and score them with the same horizon/eps rule. The index
    is precomputed once; trials only resample positions and check ``rates``.
    """
    if signals.empty:
        return float("nan")
    model_mask = _hit_mask(index, signals, h, eps_bps)
    model_hit = float(model_mask.mean())
    if np.isnan(model_hit):
        return float("nan")

    rng = np.random.default_rng(seed)
    # Per-corridor signal counts and valid positions (those with a non-empty future).
    per_iso = []
    for iso, sig_grp in signals.groupby("iso", sort=False):
        ci = index.get(iso)
        if ci is None or len(ci.rates) <= h:
            continue
        n = len(sig_grp)
        valid_pos = np.arange(len(ci.rates) - h)  # positions with >=1 future obs
        if n == 0 or len(valid_pos) == 0:
            continue
        per_iso.append((iso, ci, n, valid_pos))

    if not per_iso:
        return float("nan")
    rand_hits = []
    for _ in range(n_random_trials):
        total_hits = 0
        total_n = 0
        for _iso, ci, n, valid_pos in per_iso:
            chosen = rng.choice(valid_pos, size=n, replace=True)
            for pos in chosen:
                future = ci.rates[pos + 1 : pos + 1 + h]
                if len(future) == 0:
                    continue
                total_n += 1
                thr = ci.rates[pos] * (1 + eps_bps / 10_000.0)
                if np.all(future <= thr + 1e-12):
                    total_hits += 1
        if total_n > 0:
            rand_hits.append(total_hits / total_n)
    if not rand_hits:
        return float("nan")
    random_hit = float(np.mean(rand_hits))
    if random_hit == 0:
        return float("inf")
    return model_hit / random_hit


def frequency(signals: pd.DataFrame, total_days: int) -> dict[str, float]:
    """Signals per week and per month across the evaluation period."""
    if signals.empty or total_days == 0:
        return {"per_week": float("nan"), "per_month": float("nan")}
    weeks = total_days / 5.0  # trading days per week
    months = total_days / 21.0
    return {
        "per_week": len(signals) / weeks,
        "per_month": len(signals) / months,
    }


def clustering(signals: pd.DataFrame, gap: int = 5) -> dict[str, float]:
    """Clustering of the signal stream.

    ``series_share`` — share of signals that follow another signal in the same
    corridor within ``gap`` observations. ``interval_cv`` — coefficient of
    variation of inter-signal intervals (in trading days); high CV means bursty.
    """
    if signals.empty:
        return {"series_share": float("nan"), "interval_cv": float("nan")}
    series_count = 0
    intervals = []
    for _iso, grp in signals.sort_values("signal_date").groupby("iso"):
        dates = grp["signal_date"].sort_values().reset_index(drop=True)
        for i in range(1, len(dates)):
            d = (dates.iloc[i] - dates.iloc[i - 1]).days
            intervals.append(d)
            if d <= gap:
                series_count += 1
    series_share = series_count / len(signals) if len(signals) else float("nan")
    if len(intervals) < 2:
        cv = float("nan")
    else:
        arr = np.asarray(intervals, dtype=float)
        cv = float(arr.std(ddof=1) / arr.mean()) if arr.mean() > 0 else float("nan")
    return {"series_share": series_share, "interval_cv": cv}


def evaluate(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    eps_bps: float = 0.0,
    total_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the full metric matrix: corridor x metric x horizon.

    Returns ``(metrics_df, frequency_df)``:
    * ``metrics_df``: ``iso, horizon, hit_rate, advantage_bps, advantage_tstat,
      lift, n_signals``.
    * ``frequency_df``: ``iso, per_week, per_month, series_share, interval_cv``.
    """
    index = _build_index(panel)
    rows = []
    if total_days is None:
        total_days = panel["quote_date"].nunique()
    for iso, sig_grp in signals.groupby("iso", sort=False):
        for h in horizons:
            hr = hit_rate(build_outcomes(panel, sig_grp, h), eps_bps=eps_bps)
            adv_mean, adv_se, n = advantage_bps(index, sig_grp, h)
            tstat = adv_mean / adv_se if adv_se and not np.isnan(adv_se) else float("nan")
            lf = lift_over_random(index, sig_grp, h, eps_bps=eps_bps)
            rows.append(
                {
                    "iso": iso,
                    "horizon": h,
                    "hit_rate": hr,
                    "advantage_bps": adv_mean,
                    "advantage_tstat": tstat,
                    "lift": lf,
                    "n_signals": n,
                }
            )
    metric_df = pd.DataFrame(rows)
    freq_rows = []
    for iso, sig_grp in signals.groupby("iso", sort=False):
        freq = frequency(sig_grp, total_days)
        clust = clustering(sig_grp)
        freq_rows.append(
            {
                "iso": iso,
                "per_week": freq["per_week"],
                "per_month": freq["per_month"],
                "series_share": clust["series_share"],
                "interval_cv": clust["interval_cv"],
            }
        )
    return metric_df, pd.DataFrame(freq_rows)
