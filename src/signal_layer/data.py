"""CBR effective-date data, without fabricated publication timestamps."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from .config import Config

CURRENCY_CODES = {
    "TJS": "R01670",
    "UZS": "R01717",
    "KGS": "R01370",
    "AMD": "R01060",
    "KZT": "R01335",
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
}


def normalize(frame: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    required = {"date", "iso", "nominal", "rate"}
    if not required <= set(frame):
        raise ValueError(f"Missing data columns: {sorted(required - set(frame))}")
    if lag_days < 0:
        raise ValueError("Availability lag cannot be negative")
    f = frame.copy()
    f["effective_date"] = pd.to_datetime(f["date"], errors="raise").dt.normalize()
    if f["effective_date"].isna().any():
        raise ValueError("Missing effective date")
    f["iso"] = f.iso.astype(str).str.strip().str.upper()
    if not f.iso.str.fullmatch("[A-Z]{3}").all():
        raise ValueError("Invalid ISO code")
    for col in ("nominal", "rate"):
        f[col] = pd.to_numeric(f[col].astype(str).str.replace(",", "."), errors="raise")
        if not (np.isfinite(f[col]) & (f[col] > 0)).all():
            raise ValueError(f"{col} must be finite and positive")
    f["rub_per_unit"] = f.rate / f.nominal
    if "rate_per_unit" in f and not np.allclose(
        pd.to_numeric(f.rate_per_unit), f.rub_per_unit, rtol=2e-5, atol=1e-9
    ):
        raise ValueError("rate_per_unit disagrees with rate / nominal")
    if f.duplicated(["iso", "effective_date"]).any():
        raise ValueError("Duplicate currency / effective date observations")
    # XML Record.Date is the EFFECTIVE date, not the release date.
    f["date"] = f.effective_date + pd.Timedelta(days=lag_days)
    return (
        f[["date", "effective_date", "iso", "nominal", "rate", "rub_per_unit"]]
        .sort_values(["date", "iso"], kind="stable")
        .reset_index(drop=True)
    )


def load_rates(config: Config, as_of=None) -> pd.DataFrame:
    frames = []
    for iso in sorted(set(config.data.corridors + config.data.context)):
        path = Path(config.data.directory) / f"rates_{iso}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; use fx-signals fetch or copy m0/currency_data"
            )
        f = normalize(pd.read_csv(path), config.data.availability_lag_days)
        if set(f.iso) != {iso}:
            raise ValueError(f"Unexpected currencies in {path}")
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.loc[panel.date >= pd.Timestamp(config.data.start)]
    for end in (config.data.end, as_of):
        if end is not None:
            panel = panel.loc[panel.date <= pd.Timestamp(end).normalize()]
    if panel.empty:
        raise ValueError("No rates in requested interval")
    return panel.sort_values(["date", "iso"]).reset_index(drop=True)


def daily_rates(group: pd.DataFrame, max_stale_days: int, end=None) -> pd.DataFrame:
    """Calendar carry-forward is allowed for labels; never backfill, never extrapolate tail."""
    g = group.sort_values("date").set_index("date")
    last = g.index.max() if end is None else min(pd.Timestamp(end), g.index.max())
    idx = pd.date_range(g.index.min(), last, freq="D")
    out = g[["rub_per_unit"]].reindex(idx).ffill()
    source_dates = pd.Series(g.index, index=g.index).reindex(idx).ffill()
    out["age_days"] = (pd.Series(idx, index=idx) - source_dates).dt.days
    out.loc[out.age_days > max_stale_days, "rub_per_unit"] = np.nan
    out.index.name = "date"
    return out


def data_manifest(config: Config) -> dict:
    result = {}
    for iso in sorted(set(config.data.corridors + config.data.context)):
        p = Path(config.data.directory) / f"rates_{iso}.csv"
        f = pd.read_csv(p)
        result[p.name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "rows": len(f),
            "start": str(f.date.min()),
            "end": str(f.date.max()),
        }
    if config.data.holidays_file:
        p = Path(config.data.holidays_file)
        result["holidays"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return result


def fetch_rates(directory, currencies, start, end):
    """Explicit network operation; training/backtests themselves are fully offline."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for iso in currencies:
        if iso not in CURRENCY_CODES:
            raise ValueError(f"Unsupported CBR currency: {iso}")
        rows = []
        lo, end_date = pd.Timestamp(start), pd.Timestamp(end)
        if lo > end_date:
            raise ValueError("start must not be after end")
        while lo <= end_date:
            hi = min(lo + pd.Timedelta(days=365), end_date)
            params = {
                "date_req1": lo.strftime("%d/%m/%Y"),
                "date_req2": hi.strftime("%d/%m/%Y"),
                "VAL_NM_RQ": CURRENCY_CODES[iso],
            }
            for attempt in range(4):
                try:
                    r = requests.get(
                        "https://www.cbr.ru/scripts/XML_dynamic.asp",
                        params=params,
                        timeout=45,
                        headers={"User-Agent": "FXSignals/0.2"},
                    )
                    r.raise_for_status()
                    root = ElementTree.fromstring(r.content)
                    break
                except (requests.RequestException, ElementTree.ParseError):
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            for rec in root.findall("Record"):

                def number(text):
                    return float(text.replace(" ", "").replace(",", "."))

                nominal, rate = number(rec.findtext("Nominal")), number(rec.findtext("Value"))
                rows.append(
                    {
                        "date": pd.to_datetime(rec.attrib["Date"], dayfirst=True).date(),
                        "iso": iso,
                        "nominal": nominal,
                        "rate": rate,
                        "rate_per_unit": rate / nominal,
                    }
                )
            lo = hi + pd.Timedelta(days=1)
            time.sleep(0.2)
        if not rows:
            raise ValueError(f"CBR returned no observations for {iso}")
        f = pd.DataFrame(rows).drop_duplicates(["date", "iso"]).sort_values("date")
        normalize(f)
        temp = out / f"rates_{iso}.csv.tmp"
        f.to_csv(temp, index=False)
        temp.replace(out / f"rates_{iso}.csv")
