"""Serving the signal layer: what would we send for this corridor, on this date.

Thin by design. Every decision — which indicator, which window, whether the day
is worth a slot, what the message may claim — lives in
:mod:`signal_layer.signals`, which is what CBSB-1 selected. This module only
resolves a corridor and a date to that layer's answer and shapes it for HTTP.

There is no strategy parameter. The benchmark picked the calibrated z-score with
a send-time truth gate: 81.7 bps of client money per transfer against 23.4 for
the same rule with a fixed window and 14.0 for the learned model, significant on
all five corridors. Offering alternatives here would let something the benchmark
never blessed reach a client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from signal_layer.services.rates import RateQuote, RateService
from signal_layer.signals import INDICATOR, SignalLayerConfig, latest_signal


class InsufficientHistoryError(ValueError):
    """There is not enough history for the layer to calibrate and decide."""


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """A signal candidate. Not a delivered notification: the layer proposes."""

    currency: str
    as_of: date
    quote: RateQuote
    indicator: str
    decision: Literal["candidate", "hold"]
    reason: str
    message: str | None
    direction: str | None = None
    speed: str | None = None
    scenario: str | None = None
    window: str | None = None
    strength: float | None = None
    strength_pct: float | None = None
    deviation_pct: float | None = None
    level_percentile: float | None = None


class SignalService:
    """Resolve one corridor and date through the live signal layer."""

    def __init__(
        self,
        rate_service: RateService,
        *,
        config: SignalLayerConfig | None = None,
        context_currency: str = "USD",
    ) -> None:
        self._rate_service = rate_service
        self._config = config or SignalLayerConfig()
        self._context_currency = context_currency

    def evaluate(self, currency: str, as_of: date) -> SignalEvaluation:
        """The layer's answer for ``currency`` on ``as_of``.

        A ``hold`` is a real answer, not a failure: most days are not worth a
        scarce push, and a day whose message would not be true is refused
        outright.
        """
        quote = self._rate_service.latest_quote(currency, as_of)
        currencies = [quote.currency]
        if quote.currency != self._context_currency:
            currencies.append(self._context_currency)
        panel = self._rate_service.panel_asof(currencies, as_of)

        try:
            signal = latest_signal(
                panel, quote.currency, pd.Timestamp(as_of), self._config
            )
        except ValueError as error:
            raise InsufficientHistoryError(str(error)) from error

        if signal is None:
            return SignalEvaluation(
                currency=quote.currency,
                as_of=as_of,
                quote=quote,
                indicator=INDICATOR,
                decision="hold",
                reason=(
                    "Day not selected: either the rate is not below the trend the "
                    "indicator measured, or the communication budget is better spent "
                    "elsewhere this week"
                ),
                message=None,
            )

        return SignalEvaluation(
            currency=quote.currency,
            as_of=as_of,
            quote=quote,
            indicator=str(signal["indicator"]),
            decision="candidate",
            reason=(
                f"Rate sits {abs(float(signal['deviation_pct'])):.1f}% below its "
                f"{signal['window']} trend; signal strength is in the "
                f"{float(signal['strength_pct']) * 100:.0f}th percentile of this "
                f"corridor's own history"
            ),
            message=str(signal["message"]) or None,
            direction=str(signal["direction"]),
            speed=str(signal["speed"]),
            scenario=str(signal["scenario"]),
            window=str(signal["window"]),
            strength=float(signal["strength"]),
            strength_pct=float(signal["strength_pct"]),
            deviation_pct=float(signal["deviation_pct"]),
            level_percentile=float(signal["level_percentile"]),
        )
