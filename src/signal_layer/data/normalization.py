"""Canonical, leakage-safe representation of official exchange-rate observations.

The source files contain an official rate for a currency *nominal*.  The signal
layer must not use that raw value directly: a rate for 100 AMD and a rate for
1 USD are not comparable.  This module converts every observation to the
number of RUB paid for one unit of the foreign currency.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = frozenset({"date", "iso", "nominal", "rate"})
CANONICAL_COLUMNS = (
    "quote_date",
    "available_on",
    "iso",
    "nominal",
    "rate",
    "rub_per_unit",
)


class RateDataError(ValueError):
    """Raised when an exchange-rate source violates the data contract."""


def normalize_rate_frame(
    frame: pd.DataFrame,
    *,
    availability_lag_days: int = 1,
    validate_source_rate_per_unit: bool = True,
) -> pd.DataFrame:
    """Return a validated canonical representation of official rate quotes.

    Parameters
    ----------
    frame:
        A table with ``date``, ``iso``, ``nominal`` and ``rate`` columns.
        ``rate`` is the number of RUB for ``nominal`` units of ``iso``.
    availability_lag_days:
        Conservative calendar-day delay between quote date and its availability
        to the model.  The default prevents same-day use of a published quote.
    validate_source_rate_per_unit:
        If the optional source column ``rate_per_unit`` is present, verify that
        it agrees with ``rate / nominal`` instead of trusting it as an input.

    Returns
    -------
    pandas.DataFrame
        One row per currency/date, sorted by currency and quote date.  A lower
        ``rub_per_unit`` is a more favourable rate for a RUB-to-currency transfer.
    """
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be non-negative")

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RateDataError(f"Missing required columns: {', '.join(sorted(missing))}")

    normalized = frame.copy()
    normalized["quote_date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if normalized["quote_date"].isna().any():
        raise RateDataError("Column 'date' contains unparsable values")
    if getattr(normalized["quote_date"].dt, "tz", None) is not None:
        normalized["quote_date"] = normalized["quote_date"].dt.tz_localize(None)
    normalized["quote_date"] = normalized["quote_date"].dt.normalize()

    normalized["iso"] = normalized["iso"].astype("string").str.strip().str.upper()
    invalid_iso = normalized["iso"].isna() | ~normalized["iso"].str.fullmatch(r"[A-Z]{3}")
    if invalid_iso.any():
        raise RateDataError("Column 'iso' must contain three-letter ISO currency codes")

    for column in ("nominal", "rate"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        values = normalized[column]
        if values.isna().any() or ~np.isfinite(values).all() or (values <= 0).any():
            raise RateDataError(f"Column '{column}' must contain finite positive numbers")

    if not np.allclose(normalized["nominal"], np.round(normalized["nominal"])):
        raise RateDataError("Column 'nominal' must contain whole numbers")
    normalized["nominal"] = normalized["nominal"].astype("int64")
    normalized["rate"] = normalized["rate"].astype("float64")
    normalized["rub_per_unit"] = normalized["rate"] / normalized["nominal"]

    if validate_source_rate_per_unit and "rate_per_unit" in normalized:
        source_per_unit = pd.to_numeric(normalized["rate_per_unit"], errors="coerce")
        if source_per_unit.isna().any() or not np.isfinite(source_per_unit).all():
            raise RateDataError("Column 'rate_per_unit' must contain finite numbers when present")
        if not np.allclose(source_per_unit, normalized["rub_per_unit"], rtol=1e-10, atol=1e-12):
            raise RateDataError("'rate_per_unit' does not match 'rate / nominal'")

    duplicate_keys = normalized.duplicated(["quote_date", "iso"], keep=False)
    if duplicate_keys.any():
        duplicates = normalized.loc[duplicate_keys, ["quote_date", "iso"]].to_dict("records")
        raise RateDataError(f"Duplicate quote_date/iso observations: {duplicates[:3]}")

    normalized["available_on"] = normalized["quote_date"] + pd.offsets.Day(
        availability_lag_days
    )
    result = normalized.loc[:, CANONICAL_COLUMNS].sort_values(
        ["iso", "quote_date"], kind="stable"
    )
    return result.reset_index(drop=True)


def read_rate_csv(
    path: str | Path,
    *,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    """Read and normalise one CSV file of official exchange-rate quotes."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Rate source does not exist: {source_path}")
    return normalize_rate_frame(
        pd.read_csv(source_path), availability_lag_days=availability_lag_days
    )


def read_rate_directory(
    directory: str | Path,
    *,
    currencies: Iterable[str] | None = None,
    availability_lag_days: int = 1,
) -> pd.DataFrame:
    """Read ``rates_<ISO>.csv`` files into one canonical trading panel.

    Missing dates are deliberately not filled: an absent date is not a trading
    observation and forward filling here would manufacture market information.
    """
    source_directory = Path(directory)
    paths = sorted(source_directory.glob("rates_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No files matching rates_*.csv in {source_directory}")

    requested = None
    if currencies is not None:
        requested = {currency.strip().upper() for currency in currencies}
        paths = [path for path in paths if path.stem.removeprefix("rates_") in requested]
        if not paths:
            raise RateDataError("None of the requested currencies has a source file")

    frames: list[pd.DataFrame] = []
    for path in paths:
        expected_iso = path.stem.removeprefix("rates_").upper()
        frame = read_rate_csv(path, availability_lag_days=availability_lag_days)
        actual_iso = set(frame["iso"].unique())
        if actual_iso != {expected_iso}:
            raise RateDataError(
                f"{path.name} must contain only {expected_iso}, received {sorted(actual_iso)}"
            )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    duplicate_keys = panel.duplicated(["quote_date", "iso"], keep=False)
    if duplicate_keys.any():
        raise RateDataError("Duplicate quote_date/iso observations across source files")
    return panel.sort_values(["iso", "quote_date"], kind="stable").reset_index(drop=True)
