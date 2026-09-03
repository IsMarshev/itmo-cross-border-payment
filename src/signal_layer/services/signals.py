"""A deterministic, explainable baseline for signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from signal_layer.models import predict_asof
from signal_layer.services.rates import RateQuote, RateService

SignalStrategy = Literal["baseline", "ridge"]


class InsufficientHistoryError(ValueError):
    """There are not enough available observations for a stable comparison."""


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """A non-stateful signal candidate; it is not a delivered notification."""

    currency: str
    as_of: date
    quote: RateQuote
    strategy: SignalStrategy
    reference_observations: int
    favourable_percentile: float
    predicted_advantage_bps: float | None
    training_observations: int | None
    decision: Literal["candidate", "hold"]
    reason: str
    message: str | None


class SignalService:
    """Evaluate an as-of factual baseline without a forecasting claim."""

    def __init__(
        self,
        rate_service: RateService,
        *,
        lookback_observations: int = 60,
        candidate_percentile: float = 85.0,
    ) -> None:
        if lookback_observations <= 0:
            raise ValueError("lookback_observations must be positive")
        if not 0 < candidate_percentile <= 100:
            raise ValueError("candidate_percentile must be in (0, 100]")
        self._rate_service = rate_service
        self._lookback_observations = lookback_observations
        self._candidate_percentile = candidate_percentile

    def evaluate(
        self,
        currency: str,
        as_of: date,
        *,
        strategy: SignalStrategy = "baseline",
    ) -> SignalEvaluation:
        """Compare the latest available quote against prior available quotes.

        The comparison set ends strictly before the current quote. Thus neither
        future rates nor a duplicate use of today's observation enter the fact
        shown to a client.
        """
        quote = self._rate_service.latest_quote(currency, as_of)
        history = self._rate_service.currency_history(quote.currency)
        available_history = history.loc[history["available_on"] <= pd.Timestamp(as_of)]
        reference = available_history.loc[
            available_history["quote_date"] < pd.Timestamp(quote.quote_date)
        ].tail(self._lookback_observations)
        if len(reference) < self._lookback_observations:
            raise InsufficientHistoryError(
                f"{quote.currency} has only {len(reference)} prior observations; "
                f"{self._lookback_observations} are required"
            )

        favourable_percentile = float((reference["rub_per_unit"] > quote.rub_per_unit).mean() * 100)
        predicted_advantage_bps = None
        training_observations = None
        if strategy == "ridge":
            context_currencies = [quote.currency]
            if quote.currency != "USD":
                context_currencies.append("USD")
            panel = self._rate_service.panel_asof(context_currencies, as_of)
            try:
                prediction = predict_asof(panel, quote.currency, as_of)
            except ValueError as error:
                raise InsufficientHistoryError(str(error)) from error
            predicted_advantage_bps = prediction.predicted_advantage_bps
            training_observations = prediction.training_observations
            is_candidate = (
                predicted_advantage_bps > 0
                and favourable_percentile >= self._candidate_percentile
            )
        else:
            is_candidate = favourable_percentile >= self._candidate_percentile
        rounded_percentile = round(favourable_percentile)
        reason = (
            f"Current rate is lower than {rounded_percentile}% of the preceding "
            f"{self._lookback_observations} available quotes"
        )
        if predicted_advantage_bps is not None:
            reason = f"Ridge score is {predicted_advantage_bps:.1f} bps; {reason.lower()}"
        message = None
        if is_candidate:
            message = (
                f"Курс {quote.currency} сейчас ниже, чем в {rounded_percentile}% "
                f"из последних {self._lookback_observations} доступных наблюдений."
            )

        return SignalEvaluation(
            currency=quote.currency,
            as_of=as_of,
            quote=quote,
            strategy=strategy,
            reference_observations=len(reference),
            favourable_percentile=favourable_percentile,
            predicted_advantage_bps=predicted_advantage_bps,
            training_observations=training_observations,
            decision="candidate" if is_candidate else "hold",
            reason=reason,
            message=message,
        )
