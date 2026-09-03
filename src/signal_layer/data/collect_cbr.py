"""Collection of daily official exchange rates from the Central Bank of Russia.

Source: https://www.cbr.ru/development/SXML/
Endpoint: ``XML_dynamic.asp`` returns a series of one currency's rates over a
date range.

The signal layer consumes already-collected CSVs from ``currency_data/``.  This
module is the producer of those CSVs.  It is run manually and rarely, not part
of the online signal path.

Long-history handling (from 2000):
    * The CBR publishes a rate *per nominal*, and the nominal changes over time
      (cosmetically, so the published ``rate`` stays in a readable 1..100 band).
      Nominal changes are frequent but ``rate / nominal`` is continuous across
      them, so no stitching is needed.
    * Real denomination (a currency replaced by a new unit) breaks
      ``rate / nominal`` by orders of magnitude in a single day.  The clearest
      case is TJS: the Tajik ruble was redenominated into the somoni 1000:1 on
      2000-11-01 (a x934 jump).  The old and new units are different currencies
      and must not be stitched.
    * We therefore collect the full series, detect ``rate_per_unit`` breaks
      (a day-over-day ratio above ``DENOMINATION_RATIO``) and trim the series to
      keep only the part *after* the last break (the current currency).  The
      trim point is logged.

Example::

    uv run python -m signal_layer.data.collect_cbr --all
    uv run python -m signal_layer.data.collect_cbr -c TJS --from 2010-01-01
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# CBR currency codes (VAL_NM_RQ) from the XML_valFull.asp reference.
# The nominal is taken from the dynamic response itself (it changes over time);
# it is not fixed here.
CURRENCY_CODES: dict[str, str] = {
    "TJS": "R01670",  # Tajikistani somoni
    "UZS": "R01717",  # Uzbekistani sum
    "KGS": "R01370",  # Kyrgyzstani som
    "AMD": "R01060",  # Armenian dram
    "KZT": "R01335",  # Kazakhstani tenge
    "USD": "R01235",  # US dollar (context)
    "EUR": "R01239",  # Euro (context)
    "CNY": "R01375",  # Chinese yuan (context)
}

DYNAMIC_URL = "http://www.cbr.ru/scripts/XML_dynamic.asp"
ENCODING = "windows-1251"
# Year-sized chunks: the CBR serves short ranges more reliably.
CHUNK_DAYS = 366
REQUEST_TIMEOUT = 30
RETRIES = 5
RETRY_BACKOFF = 2.0
# Pauses to avoid 429 Too Many Requests: between chunks and between currencies.
INTERCHUNK_SLEEP = 0.5
INTERCURRENCY_SLEEP = 1.0
USER_AGENT = "Mozilla/5.0 (cross-border-payment-signal-layer)"

# A day-over-day rate_per_unit ratio above this threshold is treated as a break
# (real denomination, not ordinary volatility).  10x is a wide margin: even the
# 2014/2022 crisis moves stay within 1.05..1.2.
DENOMINATION_RATIO = 10.0

DEFAULT_START = date(2000, 1, 1)


@dataclass
class FetchConfig:
    """Network request parameters (extracted for testability)."""

    timeout: int = REQUEST_TIMEOUT
    retries: int = RETRIES
    backoff: float = RETRY_BACKOFF


def _fmt(d: date) -> str:
    """Date in the CBR ``dd/mm/yyyy`` format."""
    return d.strftime("%d/%m/%Y")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _to_float(s: str | None) -> float | None:
    """The CBR uses a comma as the decimal separator."""
    if s is None or s.strip() == "":
        return None
    return float(s.replace(",", ".").replace(" ", ""))


def _date_ranges(
    start: date, end: date, chunk_days: int = CHUNK_DAYS
) -> Iterator[tuple[date, date]]:
    """Split ``[start, end]`` into half-open intervals of length <= ``chunk_days``."""
    lo = start
    while lo <= end:
        hi = min(lo + timedelta(days=chunk_days - 1), end)
        yield lo, hi
        lo = hi + timedelta(days=1)


def _fetch_chunk(code: str, lo: date, hi: date, cfg: FetchConfig) -> str:
    """Download the dynamic XML for one range, with retries on 429/5xx."""
    params = {"date_req1": _fmt(lo), "date_req2": _fmt(hi), "VAL_NM_RQ": code}
    last_err: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            r = requests.get(
                DYNAMIC_URL,
                params=params,
                timeout=cfg.timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            r.encoding = ENCODING
            return r.text
        except Exception as exc:  # noqa: BLE001 - network fails in many ways
            last_err = exc
            if attempt < cfg.retries - 1:
                time.sleep(cfg.backoff * (2**attempt))
    raise RuntimeError(
        f"Failed to fetch {code} for [{lo}..{hi}] after {cfg.retries} attempts: {last_err}"
    )


def _parse_xml(xml_text: str, iso: str) -> pd.DataFrame:
    """Parse the dynamic XML into a DataFrame.

    Columns: ``date``, ``iso``, ``nominal``, ``rate`` (RUB per nominal),
    ``rate_per_unit`` (RUB per one unit).
    """
    root = ET.fromstring(xml_text)
    rows = []
    for rec in root.findall("Record"):
        nominal = rec.findtext("Nominal")
        rate = _to_float(rec.findtext("Value"))
        vunit = rec.findtext("VunitRate")
        nom = _to_float(nominal) or 1.0
        per_unit = _to_float(vunit) if vunit else (rate / nom if rate is not None else None)
        rows.append(
            {
                "date": datetime.strptime(rec.get("Date"), "%d.%m.%Y").date(),
                "iso": iso,
                "nominal": int(nom) if nominal else None,
                "rate": rate,
                "rate_per_unit": per_unit,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "iso", "nominal", "rate", "rate_per_unit"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def detect_denominations(
    df: pd.DataFrame, ratio: float = DENOMINATION_RATIO
) -> list[pd.Timestamp]:
    """Find dates where ``rate_per_unit`` jumps by more than ``ratio`` in one day.

    Returns the break dates (the day the new segment starts).
    """
    if df.empty:
        return []
    pu = df["rate_per_unit"].astype(float)
    rel = pu / pu.shift(1)
    breaks = df[(rel > ratio) | (rel < 1 / ratio)]
    return list(breaks["date"])


def fetch_currency(
    iso: str,
    start: date,
    end: date,
    cfg: FetchConfig | None = None,
    trim_denominations: bool = True,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Collect one currency's rate series over ``[start, end]``.

    Returns ``(df, breaks)``.  When ``trim_denominations`` is True the series is
    trimmed at the last break, keeping only data *after* it (the current
    currency).
    """
    if iso not in CURRENCY_CODES:
        raise ValueError(f"Unknown currency {iso!r}. Available: {list(CURRENCY_CODES)}")
    code = CURRENCY_CODES[iso]
    cfg = cfg or FetchConfig()
    frames = []
    for lo, hi in _date_ranges(start, end):
        frames.append(_parse_xml(_fetch_chunk(code, lo, hi, cfg), iso))
        time.sleep(INTERCHUNK_SLEEP)
    if not frames:
        return (
            pd.DataFrame(columns=["date", "iso", "nominal", "rate", "rate_per_unit"]),
            [],
        )
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "iso"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    breaks = detect_denominations(df)
    if breaks and trim_denominations:
        last_break = max(breaks)
        print(
            f"    ! rate_per_unit break on {last_break.date()} - "
            "trimming series (denomination / currency change)"
        )
        df = df[df["date"] > last_break].reset_index(drop=True)
    return df, breaks


def fetch_all(
    start: date,
    end: date,
    cfg: FetchConfig | None = None,
    trim_denominations: bool = True,
) -> dict[str, pd.DataFrame]:
    """Collect rates for every currency in ``CURRENCY_CODES``. Returns ``{iso: df}``."""
    out: dict[str, pd.DataFrame] = {}
    for iso in CURRENCY_CODES:
        print(f"  -> {iso} ...", end=" ", flush=True)
        df, breaks = fetch_currency(iso, start, end, cfg, trim_denominations)
        note = f" (+{len(breaks)} break)" if breaks else ""
        print(f"{len(df)} records{note}")
        out[iso] = df
        time.sleep(INTERCURRENCY_SLEEP)
    return out


def _write(df: pd.DataFrame, out: str) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _default_end() -> date:
    return date.today()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect CBR exchange rates")
    parser.add_argument("--currency", "-c", help="ISO code (TJS/UZS/KGS/AMD/KZT/USD/EUR/CNY)")
    parser.add_argument(
        "--all", action="store_true", help="Collect every currency, one CSV each"
    )
    parser.add_argument(
        "--from", dest="date_from", default=None, help="Start (YYYY-MM-DD), default 2000-01-01"
    )
    parser.add_argument(
        "--to", dest="date_to", default=None, help="End (YYYY-MM-DD), default today"
    )
    parser.add_argument(
        "--output", "-o", default="currency_data", help="CSV directory (default currency_data/)"
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim the series at rate_per_unit breaks (keep denominations)",
    )
    args = parser.parse_args(argv)

    if not args.all and not args.currency:
        parser.error("specify --currency <ISO> or --all")

    start = _parse_date(args.date_from) if args.date_from else DEFAULT_START
    end = _parse_date(args.date_to) if args.date_to else _default_end()
    if end < start:
        parser.error("--to is earlier than --from")

    print(f"Collecting CBR rates for {start} .. {end}")
    trim = not args.no_trim

    def _save(iso: str, df: pd.DataFrame) -> str:
        return _write(df, os.path.join(args.output, f"rates_{iso}.csv"))

    if args.all:
        result = fetch_all(start, end, trim_denominations=trim)
        print("\nDone:")
        for iso, df in result.items():
            path = _save(iso, df)
            span = f"{df['date'].min().date()}..{df['date'].max().date()}" if len(df) else "-"
            print(f"  {iso}: {len(df):5} records | {span} | {path}")
    else:
        df, breaks = fetch_currency(args.currency, start, end, trim_denominations=trim)
        path = _save(args.currency, df)
        span = f"{df['date'].min().date()}..{df['date'].max().date()}" if len(df) else "-"
        print(f"\nDone: {len(df)} records | {span} | {path}")
        if len(df):
            print(df.head(3).to_string(index=False))
            print("  ...")
            print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
