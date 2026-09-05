"""Day-by-day simulation of the layer for the demo stand.

The stand walks a calendar forward one day at a time and asks the same question
the product asks: is today a day worth a push, and — if a push is already
sitting unread on the client's lock screen — is what it said still true.

Both answers live here rather than in the browser. The front end owns no rule:
it advances a date, renders what this service returns, and asks again.

Two things are worth stating plainly, because they are what a reviewer will
check first.

*The signals are the chronological walk-forward run, not a re-fit per day.*
:func:`signal_layer.signals.signal_table` runs the policy in order over the
panel, so the day it selects for date T uses only data up to T. Precomputing it
once at startup and serving day lookups is therefore identical to recomputing
``signals_asof(T)`` on every step — audited on four cut dates in
``demo/export_data.py`` — and it is what keeps a day step instant instead of
half a second.

*The disclosure rule is measured, not chosen by taste.* How loudly the app
should admit that a push has aged is decided against the corridor's own median
daily move: a change smaller than an ordinary day is noise, and interrupting a
transfer for noise spends trust for nothing. The rule is deliberately
asymmetric, mirroring the asymmetric cost of error in the brief — we ask the
client to confirm only when the rate moved *against* them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from signal_layer.services.rates import RateService
from signal_layer.signals import INDICATOR, SignalLayerConfig, signal_table

CORRIDORS: tuple[str, ...] = ("TJS", "UZS", "KGS", "AMD", "KZT")

#: The client-facing name of each currency. The layer's own message carries the
#: ISO code; the message shelf substitutes the word a client would recognise.
CURRENCY_NAMES: dict[str, tuple[str, str]] = {
    "TJS": ("сомони", "Таджикистан"),
    "UZS": ("сум", "Узбекистан"),
    "KGS": ("сом", "Кыргызстан"),
    "AMD": ("драм", "Армения"),
    "KZT": ("тенге", "Казахстан"),
}

#: Where the chart begins. Earlier history still feeds every indicator — this is
#: only how far back the simulation draws.
WINDOW_START = pd.Timestamp("2025-09-01")

#: Where the playhead starts: late enough that the chart opens with context,
#: early enough that a signal arrives within a few steps.
SIM_START = pd.Timestamp("2025-11-28")

FreshnessLevel = Literal["same", "better", "mild", "stale"]


@dataclass(frozen=True, slots=True)
class CorridorMeta:
    """What the front end needs to set up one corridor."""

    iso: str
    name: str
    country: str
    first_date: date
    last_date: date
    sim_start: date
    threshold_bps: float
    """Median absolute daily move: the size of an ordinary day in this corridor."""
    trend_span: int
    """The EWMA window the walk-forward tuner selected over this period."""


@dataclass(frozen=True, slots=True)
class DayDecision:
    """The layer's answer for one simulated calendar day."""

    iso: str
    day: date
    has_quote: bool
    rate: float | None
    decision: Literal["candidate", "hold"]
    signal: dict | None


@dataclass(frozen=True, slots=True)
class Freshness:
    """Whether a push that was sent on ``signal_date`` still holds on ``as_of``."""

    level: FreshnessLevel
    delta_bps: float
    threshold_bps: float
    signal_date: date
    signal_rate: float
    current_date: date
    current_rate: float


class SimulationService:
    """Serve the simulation: one corridor, one day, one freshness check at a time."""

    def __init__(
        self,
        rate_service: RateService,
        *,
        corridors: tuple[str, ...] = CORRIDORS,
        window_start: pd.Timestamp = WINDOW_START,
        sim_start: pd.Timestamp = SIM_START,
        config: SignalLayerConfig | None = None,
    ) -> None:
        self._corridors = corridors
        self._window_start = window_start
        self._sim_start = sim_start
        self._config = config or SignalLayerConfig()
        panel = rate_service.panel(list(corridors))
        table = signal_table(panel, list(corridors), self._config)
        self._meta: dict[str, CorridorMeta] = {}
        self._series: dict[str, list[dict]] = {}
        self._signals: dict[str, dict[str, dict]] = {}
        self._quotes: dict[str, pd.Series] = {}
        for iso in corridors:
            self._prepare(iso, panel, table)

    # ── setup ────────────────────────────────────────────────────────────
    def _prepare(self, iso: str, panel: pd.DataFrame, table: pd.DataFrame) -> None:
        corridor = panel[panel["iso"] == iso].sort_values("quote_date")
        quotes = corridor.set_index("quote_date")["rub_per_unit"].astype(float)
        self._quotes[iso] = quotes

        rows = table[table["iso"].eq(iso) & table["signal_date"].ge(self._window_start)]
        spans = Counter(int(str(w).split("=")[1]) for w in rows["window"] if "=" in str(w))
        trend_span = spans.most_common(1)[0][0] if spans else 10

        # The trend is taken over the whole history and then sliced: seeding an
        # EWMA at the start of the drawn window would show a warm-up the layer
        # never saw.
        trend = quotes.ewm(span=trend_span, adjust=False).mean()
        window = quotes.loc[quotes.index >= self._window_start]

        self._series[iso] = [
            {
                "d": stamp.strftime("%Y-%m-%d"),
                "r": round(float(value), 6),
                "e": round(float(trend.loc[stamp]), 6),
            }
            for stamp, value in window.items()
        ]

        moves = window.pct_change().dropna().abs() * 10_000
        threshold = float(np.median(moves)) if len(moves) else 50.0

        self._signals[iso] = {
            row["signal_date"].strftime("%Y-%m-%d"): self._signal_payload(iso, row)
            for _, row in rows.iterrows()
        }

        name, country = CURRENCY_NAMES.get(iso, (iso, ""))
        self._meta[iso] = CorridorMeta(
            iso=iso,
            name=name,
            country=country,
            first_date=window.index[0].date(),
            last_date=window.index[-1].date(),
            sim_start=self._sim_start.date(),
            threshold_bps=round(threshold, 1),
            trend_span=trend_span,
        )

    def _signal_payload(self, iso: str, row: pd.Series) -> dict:
        deviation = float(row["deviation_pct"])
        span = int(str(row["window"]).split("=")[1]) if "=" in str(row["window"]) else 0
        name = CURRENCY_NAMES.get(iso, (iso, ""))[0]
        strength = float(row["strength_pct"])
        return {
            "date": row["signal_date"].strftime("%Y-%m-%d"),
            "iso": iso,
            "indicator": INDICATOR,
            "rate": round(float(row["rub_per_unit"]), 6),
            "deviation_pct": round(deviation, 3),
            "span": span,
            "speed": str(row["speed"]),
            "scenario": str(row["scenario"]),
            "direction": str(row["direction"]),
            "strength_pct": None if not np.isfinite(strength) else round(strength, 4),
            # What the layer emits, kept verbatim so the claim is auditable.
            "message": str(row["message"]),
            # The same fact with the word a client recognises instead of the code.
            "client_message": (
                f"Курс {name} сейчас на {abs(deviation):.1f} % ниже своего "
                f"среднего за последние {span} наблюдений."
            ),
        }

    # ── queries ──────────────────────────────────────────────────────────
    def corridors(self) -> list[CorridorMeta]:
        return [self._meta[iso] for iso in self._corridors]

    def series(self, iso: str) -> list[dict]:
        """The drawn window in full. The front end reveals it as days pass."""
        return self._series[self._require(iso)]

    def day(self, iso: str, day: date) -> DayDecision:
        """What the layer decides on this calendar day.

        A day without a fresh quote is not an error: on weekends and holidays
        the rate is simply not republished, and the layer has nothing new to
        look at.
        """
        code = self._require(iso)
        stamp = pd.Timestamp(day)
        quotes = self._quotes[code]
        has_quote = bool((quotes.index == stamp).any())
        signal = self._signals[code].get(stamp.strftime("%Y-%m-%d"))
        return DayDecision(
            iso=code,
            day=day,
            has_quote=has_quote,
            rate=round(float(quotes.loc[stamp]), 6) if has_quote else None,
            decision="candidate" if signal else "hold",
            signal=signal,
        )

    def freshness(self, iso: str, signal_date: date, as_of: date) -> Freshness:
        """How far the rate has drifted since the push went out.

        ``level`` is what the app should do about it, and the thresholds are the
        corridor's own: ``mild`` is a move no larger than an ordinary day, so the
        app states the snapshot and moves on; ``stale`` is a move against the
        client larger than that, and it stops the flow for a confirmation.
        """
        code = self._require(iso)
        quotes = self._quotes[code]
        signal_stamp = pd.Timestamp(signal_date)
        if not (quotes.index == signal_stamp).any():
            raise KeyError(f"No {code} quote on {signal_date.isoformat()}")

        available = quotes.loc[quotes.index <= pd.Timestamp(as_of)]
        if available.empty:
            raise KeyError(f"No {code} quote on or before {as_of.isoformat()}")

        signal_rate = float(quotes.loc[signal_stamp])
        current_rate = float(available.iloc[-1])
        delta = (current_rate / signal_rate - 1.0) * 10_000
        threshold = self._meta[code].threshold_bps

        if abs(delta) <= 0.5:
            level: FreshnessLevel = "same"
        elif delta < 0:
            level = "better"
        elif delta <= threshold:
            level = "mild"
        else:
            level = "stale"

        return Freshness(
            level=level,
            delta_bps=round(delta, 1),
            threshold_bps=threshold,
            signal_date=signal_stamp.date(),
            signal_rate=round(signal_rate, 6),
            current_date=available.index[-1].date(),
            current_rate=round(current_rate, 6),
        )

    def _require(self, iso: str) -> str:
        code = iso.upper()
        if code not in self._meta:
            raise KeyError(f"Corridor {iso!r} is not part of the simulation")
        return code
