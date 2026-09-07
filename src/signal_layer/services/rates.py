"""Read-only access to the canonical exchange-rate panel."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from signal_layer.data import RateDataError, read_rate_directory


class RateDataUnavailableError(RuntimeError):
    """The source panel cannot be read or does not satisfy the data contract."""


class RateNotFoundError(LookupError):
    """No quote was available for a requested currency and as-of date."""


@dataclass(frozen=True, slots=True)
class RateQuote:
    """One official quote that was available when a decision was made."""

    currency: str
    quote_date: date
    available_on: date
    rub_per_unit: float


@dataclass(frozen=True, slots=True)
class DataReadiness:
    """A compact data status for the health endpoint."""

    ready: bool
    observation_count: int
    latest_available_on: date | None
    detail: str | None = None


class RateService:
    """Load official rates and enforce as-of access for every caller."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def readiness(self) -> DataReadiness:
        """Return source readiness without leaking parsing errors to clients."""
        try:
            panel = self._load_panel()
        except RateDataUnavailableError as error:
            return DataReadiness(False, 0, None, str(error))
        latest = panel["available_on"].max()
        return DataReadiness(True, len(panel), latest.date())

    def latest_quote(self, currency: str, as_of: date) -> RateQuote:
        """Return the latest quote with ``available_on <= as_of``."""
        history = self.currency_history(currency)
        available = history.loc[history["available_on"] <= pd.Timestamp(as_of)]
        if available.empty:
            raise RateNotFoundError(
                f"No {currency.upper()} quote was available on or before {as_of.isoformat()}"
            )
        return self._quote_from_row(available.iloc[-1])

    def currency_history(self, currency: str) -> pd.DataFrame:
        """Return one currency's canonical panel in quote-date order."""
        normalized_currency = self._normalize_currency(currency)
        panel = self.panel([normalized_currency])
        return panel.loc[panel["iso"] == normalized_currency].reset_index(drop=True)

    def panel(self, currencies: Iterable[str] | None = None) -> pd.DataFrame:
        """Return a validated panel for domain services and offline evaluation."""
        normalized = None
        if currencies is not None:
            normalized = [self._normalize_currency(currency) for currency in currencies]
        try:
            return read_rate_directory(self._data_dir, currencies=normalized)
        except (FileNotFoundError, RateDataError) as error:
            raise RateDataUnavailableError(str(error)) from error

    def panel_asof(self, currencies: Iterable[str], as_of: date) -> pd.DataFrame:
        """Return only observations already available on the decision date."""
        panel = self.panel(currencies)
        return panel.loc[panel["available_on"] <= pd.Timestamp(as_of)].reset_index(drop=True)

    def _load_panel(self) -> pd.DataFrame:
        return self.panel()

    @staticmethod
    def _normalize_currency(currency: str) -> str:
        normalized = currency.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", normalized):
            raise RateNotFoundError(f"Invalid ISO currency code: {currency!r}")
        return normalized

    @staticmethod
    def _quote_from_row(row: pd.Series) -> RateQuote:
        return RateQuote(
            currency=str(row["iso"]),
            quote_date=row["quote_date"].date(),
            available_on=row["available_on"].date(),
            rub_per_unit=float(row["rub_per_unit"]),
        )
