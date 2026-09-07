"""Transfer-strategy simulator: how much currency does each timing buy?

The case brief frames the signal's value in client terms: a client spends a
fixed amount of roubles and wants the most foreign currency for it. The signal
layer's job is to pick the days to transfer; this module compares three timing
policies on the same budget:

* ``model``  — buy on the dates the model signalled.
* ``dca``    — buy on a fixed cadence (e.g. every K trading days), the
  dollar-cost-averaging baseline. If the model cannot beat "same amount once a
  week", it is not adding value.
* ``random`` — buy on uniformly random trading days (averaged over trials).

All three spend the **same total roubles** and make the **same number of
purchases** (so per-purchase amount is identical): ``per_buy = budget / N``,
where ``N`` is the number of model signals in the window. This keeps the
comparison honest — a strategy that buys every day would trivially win on
volume but is not a like-for-like comparison under the communication budget.

The output is what a client cares about: total foreign currency received, and
the uplift of model/DCA over random and over each other, in percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyResult:
    """Outcome of one timing strategy over one corridor window."""

    name: str
    n_buys: int
    total_currency: float  # foreign currency units bought
    total_rub: float  # roubles spent
    avg_rate: float  # average rub_per_unit paid (lower is better)
    buy_dates: np.ndarray  # trading dates the strategy bought on

    @property
    def currency_per_1000rub(self) -> float:
        """Currency bought per 1000 RUB spent — comparable across budgets."""
        if self.total_rub <= 0:
            return 0.0
        return self.total_currency / self.total_rub * 1000.0


def _align_rates(panel: pd.DataFrame, iso: str) -> pd.DataFrame:
    """Sorted (quote_date, rub_per_unit) for one corridor."""
    s = panel.loc[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    return s[["quote_date", "rub_per_unit"]]


def _buy(dates: np.ndarray, rates: np.ndarray, per_buy: float) -> tuple[float, np.ndarray]:
    """Buy ``per_buy`` roubles on each date; return (total_currency, dates)."""
    if len(dates) == 0:
        return 0.0, np.array([], dtype="datetime64[ns]")
    currency = (per_buy / rates).sum()
    return float(currency), dates


def simulate_strategies(
    panel: pd.DataFrame,
    iso: str,
    model_signals: pd.DataFrame,
    *,
    monthly_budget: float = 50_000.0,
    cadence_days: int = 5,
    n_random_trials: int = 500,
    seed: int = 0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, StrategyResult]:
    """Compare model / DCA / random on one corridor over a window.

    Business contract: a client transfers money home 1–3 times a month and
    allocates a fixed ``monthly_budget`` per calendar month. Over the window the
    total each strategy may spend is ``monthly_budget × months_in_window``. Each
    strategy divides that total equally between its own buys, so a strategy that
    buys more often buys a smaller amount each time — the *total roubles spent*
    is identical across strategies. The only thing that differs is *when*.

    Parameters
    ----------
    monthly_budget:
        Roubles per calendar month. Total spent = this × number of months.
    cadence_days:
        DCA buys every this many trading days (5 ≈ once a week).
    start, end:
        Window bounds (inclusive). Defaults to the span of model signals.

    Returns a dict keyed by strategy name.
    """
    rates_df = _align_rates(panel, iso)
    if start is not None:
        rates_df = rates_df[rates_df["quote_date"] >= start]
    if end is not None:
        rates_df = rates_df[rates_df["quote_date"] <= end]
    if rates_df.empty:
        raise ValueError(f"No rates for {iso} in the requested window")

    dates = rates_df["quote_date"].to_numpy()
    rates = rates_df["rub_per_unit"].to_numpy(dtype=float)

    # Total budget = monthly × months in window (calendar months, not trading).
    span_start = pd.Timestamp(dates[0])
    span_end = pd.Timestamp(dates[-1])
    months = max(1.0, (span_end - span_start).days / 30.44)
    total_budget = monthly_budget * months

    # Model signals within the window.
    sig = model_signals[model_signals["iso"] == iso] if "iso" in model_signals else model_signals
    if len(sig):
        sig_dates = pd.to_datetime(sig["signal_date"])
    else:
        sig_dates = pd.Series([], dtype="datetime64[ns]")
    if start is not None:
        sig_dates = sig_dates[sig_dates >= start]
    if end is not None:
        sig_dates = sig_dates[sig_dates <= end]
    n_model = len(sig_dates)

    # Model: split total budget across its N signals.
    if n_model > 0:
        rate_idx = np.searchsorted(dates, sig_dates.to_numpy())
        rate_idx = rate_idx[(rate_idx < len(dates))]
        m_dates = dates[rate_idx]
        m_rates = rates[rate_idx]
        m_per = total_budget / n_model
        m_currency, _ = _buy(m_dates, m_rates, m_per)
        m_avg = float(np.mean(m_rates)) if len(m_rates) else float("nan")
    else:
        m_currency, m_dates, m_avg = 0.0, np.array([]), float("nan")

    # DCA: every cadence_days trading days, split total budget across its buys.
    dca_idx = _dca_indices(len(rates), cadence_days)
    d_dates = dates[dca_idx]
    d_rates = rates[dca_idx]
    d_per = total_budget / len(dca_idx) if len(dca_idx) else 0.0
    d_currency, _ = _buy(d_dates, d_rates, d_per)
    d_avg = float(np.mean(d_rates)) if len(d_rates) else float("nan")

    # Random: n_model random trading days (matched count), averaged over trials.
    rng = np.random.default_rng(seed)
    n_rand = n_model if n_model > 0 else len(dca_idx)
    if n_rand > 0 and len(rates) > n_rand:
        r_per = total_budget / n_rand
        trial_currency = []
        for _ in range(n_random_trials):
            ridx = rng.choice(len(rates), size=n_rand, replace=False)
            ridx.sort()
            trial_currency.append((r_per / rates[ridx]).sum())
        rand_currency = float(np.mean(trial_currency))
        ridx = rng.choice(len(rates), size=n_rand, replace=False)
        ridx.sort()
        rand_dates = dates[ridx]
        rand_avg = float(np.mean(rates[ridx]))
    else:
        rand_currency, rand_dates, rand_avg = 0.0, np.array([]), float("nan")

    return {
        "model": StrategyResult("model", n_model, m_currency, total_budget, m_avg, m_dates),
        "dca": StrategyResult(
            "dca", len(dca_idx), d_currency, total_budget, d_avg, d_dates
        ),
        "random": StrategyResult(
            "random", n_rand, rand_currency, total_budget, rand_avg, rand_dates
        ),
    }


def _dca_indices(n_rates: int, cadence_days: int) -> np.ndarray:
    """Indices spaced ``cadence_days`` trading days apart, from the start.

    Every ``cadence_days``-th trading day (5 ≈ weekly). Returns all natural buys
    that fit in the window; the total budget is split equally across them.
    """
    if n_rates <= 0 or cadence_days <= 0:
        return np.array([], dtype=int)
    return np.arange(0, n_rates, cadence_days)


def uplift(a: StrategyResult, b: StrategyResult) -> float:
    """Percentage more currency ``a`` bought than ``b`` (negative = worse)."""
    if b.total_currency <= 0:
        return float("nan")
    return (a.total_currency - b.total_currency) / b.total_currency * 100.0
