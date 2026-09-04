"""Ground truth for the benchmark: what actually happened after each day.

This module owns the only look-ahead in the project. Nothing here may be read by
a strategy at decision time — the runner joins these columns to a signal *after*
the signal has already been produced, and the walk-forward trainers may only see
rows whose ``label_available_on`` has already passed.

Two conventions matter for reading the numbers:

* **Execution offset.** A signal derived from the quote of day ``T`` is acted on
  at the next published quote. Every outcome below is therefore measured from
  ``exec_rate`` (the rate the client actually pays), not from the quote that
  triggered the signal.
* **Direction.** ``rub_per_unit`` is roubles paid for one unit of the recipient's
  currency, so *lower is better for the sender*. A positive advantage means the
  executed rate was below the comparison window.

Columns
-------
``fwd_advantage_bps``
    ``(mean(next h rates) - exec_rate) / exec_rate`` — money saved versus a
    typical day in the next ``h``. This is the model's regression target.
``window_advantage_bps``
    Same against the mean of the centred ``+-h`` window. This is the brief's
    "выгода момента" and is a *metric*, not a target (it peeks backwards too).
``currency_gain_bps``
    The client-money version of the same comparison: extra foreign currency per
    rouble spent, against the average of the centred ``+-h`` window. A client
    spends roubles, so what they receive is proportional to ``1 / rate``, and
    averaging rates is not the same as averaging what those rates buy.

    The comparison window is deliberately *local*. A remittance is not deferred
    for months, so the honest alternative to "transfer on the day we wrote" is
    "transfer on some other day around then" — not "transfer on the cheapest day
    of the half-year". Scoring against a half-year of days would mostly grade the
    strategy on the rouble's trend, which no push can help a client exploit.
``adverse_bps``
    The downside half of ``fwd_advantage_bps``: ``max(0, -fwd_advantage_bps)``.
    Utility and risk have to live on the same scale to be traded off, so the
    risk side is the *shortfall of the same quantity*, not a different one.
``regret_bps``
    How far below the executed rate the market went within ``h`` days — the gap
    to the luckiest possible day. Reported as a diagnostic only: for a random
    walk it is positive almost always, which makes it useless as a decision
    input even though it reads well in a report.
``bad_push``
    ``fwd_advantage_bps < -bad_push_bps``: the average rate over the next ``h``
    days was materially *better* than what the client got by acting on our
    message. This is the expensive error of the brief, measured against a
    realistic alternative (a typical day in the horizon) rather than against
    perfect hindsight.
``is_local_min``
    The executed rate was the minimum of the centred ``+-h`` window (within a
    tolerance). The classification target the brief asks for.
``held_favourable`` / ``held_window_closing``
    The two message-specific hit rules: "сейчас выгодно" holds if the rate never
    rose above the executed level within ``h``; "окно закрывается" holds if the
    median rate over the next ``h`` rose above it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS: tuple[str, ...] = (
    "exec_date",
    "exec_rate",
    "fwd_advantage_bps",
    "fwd_median_advantage_bps",
    "window_advantage_bps",
    "currency_gain_bps",
    "adverse_bps",
    "regret_bps",
    "bad_push",
    "is_local_min",
    "held_favourable",
    "held_window_closing",
    "label_available_on",
    "outcome_complete",
)


def _forward_windows(rates: np.ndarray, start: int, horizon: int) -> np.ndarray:
    """Rows ``i`` hold ``rates[start + i + 1 : start + i + 1 + horizon]``."""
    tail = rates[start + 1 :]
    if len(tail) < horizon:
        return np.empty((0, horizon))
    return np.lib.stride_tricks.sliding_window_view(tail, horizon)


def build_labels(
    panel: pd.DataFrame,
    *,
    horizon: int = 10,
    execution_offset: int = 1,
    epsilon_bps: float = 0.0,
    bad_push_bps: float = 100.0,
    local_min_tolerance_bps: float = 10.0,
) -> pd.DataFrame:
    """Attach realised outcomes to every quote date that can carry a signal.

    The frame is keyed by ``(iso, quote_date)`` — the day a strategy *decides*.
    Rows whose full horizon has not elapsed are kept with ``outcome_complete``
    false so the decision log stays exhaustive, and must be excluded from every
    metric denominator.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if execution_offset < 0:
        raise ValueError("execution_offset must be non-negative")

    parts: list[pd.DataFrame] = []
    for iso, group in panel.sort_values(["iso", "quote_date"]).groupby("iso", sort=False):
        g = group.reset_index(drop=True)
        if "available_on" not in g.columns:
            g["available_on"] = g["quote_date"]
        n = len(g)
        rates = g["rub_per_unit"].to_numpy(dtype=float)
        dates = g["quote_date"].to_numpy()
        available = g["available_on"].to_numpy()

        out = {name: np.full(n, np.nan) for name in
               ("exec_rate", "fwd_advantage_bps", "fwd_median_advantage_bps",
                "window_advantage_bps", "currency_gain_bps", "adverse_bps",
                "regret_bps")}
        flags = {name: np.zeros(n, dtype=bool) for name in
                 ("bad_push", "is_local_min", "held_favourable", "held_window_closing")}
        complete = np.zeros(n, dtype=bool)
        exec_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        label_available = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

        # Signal rows t whose execution row e = t + offset has a full forward
        # window: e + horizon <= n - 1.
        last_signal = n - 1 - execution_offset - horizon
        if last_signal >= 0:
            t = np.arange(last_signal + 1)
            e = t + execution_offset
            exec_rate = rates[e]
            fw = _forward_windows(rates, execution_offset, horizon)[: len(t)]

            fwd_mean = fw.mean(axis=1)
            fwd_median = np.median(fw, axis=1)
            fwd_min = fw.min(axis=1)
            fwd_max = fw.max(axis=1)

            threshold = exec_rate * (1.0 + epsilon_bps / 10_000.0)
            out["exec_rate"][t] = exec_rate
            advantage = (fwd_mean - exec_rate) / exec_rate * 10_000.0
            out["fwd_advantage_bps"][t] = advantage
            out["fwd_median_advantage_bps"][t] = (fwd_median - exec_rate) / exec_rate * 10_000.0
            out["adverse_bps"][t] = np.maximum(0.0, -advantage)
            out["regret_bps"][t] = np.maximum(0.0, (exec_rate - fwd_min) / exec_rate * 10_000.0)
            flags["bad_push"][t] = advantage < -bad_push_bps
            flags["held_favourable"][t] = fwd_max <= threshold
            flags["held_window_closing"][t] = fwd_median > threshold
            complete[t] = True
            exec_date[t] = dates[e]
            label_available[t] = available[e + horizon]

            # Centred +-h window: needs `horizon` observations before execution too.
            centred_ok = e >= horizon
            if centred_ok.any():
                ct = t[centred_ok]
                ce = e[centred_ok]
                cw = np.lib.stride_tricks.sliding_window_view(rates, 2 * horizon + 1)
                windows = cw[ce - horizon]
                base = rates[ce]
                out["window_advantage_bps"][ct] = (
                    (windows.mean(axis=1) - base) / base * 10_000.0
                )
                # What a rouble buys here, against what it buys on an average
                # day of the same window.
                out["currency_gain_bps"][ct] = (
                    (1.0 / base) / (1.0 / windows).mean(axis=1) - 1.0
                ) * 10_000.0
                # Nothing in the +-h window was more than the tolerance cheaper.
                floor = base * (1.0 - local_min_tolerance_bps / 10_000.0)
                flags["is_local_min"][ct] = windows.min(axis=1) >= floor

        frame = pd.DataFrame(
            {
                "quote_date": g["quote_date"],
                "available_on": g["available_on"],
                "iso": iso,
                "rub_per_unit": g["rub_per_unit"],
                "exec_date": exec_date,
                "label_available_on": label_available,
                "outcome_complete": complete,
                **out,
                **flags,
            }
        )
        parts.append(frame)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"]).reset_index(
        drop=True
    )


def hit_column(scenario: str) -> str:
    """Map a message scenario to the label column that decides its hit."""
    if scenario == "favourable_now":
        return "held_favourable"
    if scenario == "window_closing":
        return "held_window_closing"
    raise ValueError(f"Unknown scenario {scenario!r}")


def horizon_hit_rates(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    scenario: str,
    execution_offset: int = 1,
    epsilon_bps: float = 0.0,
) -> dict[int, float]:
    """Hit rate of the same signal set re-scored at several horizons.

    The brief asks for h in {1,3,5,10,20}; the headline horizon lives in the
    spec, and this fills in the rest of the row without re-running any strategy.
    """
    column = hit_column(scenario)
    result: dict[int, float] = {}
    keys = signals[["iso", "quote_date"]].drop_duplicates()
    for h in horizons:
        labels = build_labels(
            panel,
            horizon=h,
            execution_offset=execution_offset,
            epsilon_bps=epsilon_bps,
        )
        joined = keys.merge(labels, on=["iso", "quote_date"], how="inner")
        joined = joined[joined["outcome_complete"]]
        result[h] = float(joined[column].mean()) if len(joined) else float("nan")
    return result
