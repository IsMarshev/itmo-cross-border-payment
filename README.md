# Cross-border payment signal layer

Backend foundation for an explainable signal layer that finds potentially
favourable days for cross-border RUB transfers.  The model layer is intentionally
not included yet: the first milestone is a canonical, validated and
leakage-safe representation of public exchange-rate observations.

## Setup

```bash
poetry install
poetry run pytest
poetry run ruff check .
```

## Minimal API

```bash
poetry run uvicorn signal_layer.api.app:app --reload
```

The service exposes `GET /health`, `GET /v1/rates/{ISO}/latest` and
`GET /v1/signals/{ISO}/evaluate`. The signal endpoint is a deterministic,
fact-only baseline: it returns a *candidate* for communication when the current
available rate is low relative to its trailing history. It does not deliver a
push, retain customer data or claim to forecast a future exchange rate.

## Rate data contract

Input files are named `currency_data/rates_<ISO>.csv` and must contain:

- `date` — quote date;
- `iso` — a three-letter currency code;
- `nominal` — number of currency units in the official quote;
- `rate` — RUB for that nominal.

`signal_layer.data.read_rate_directory()` returns the canonical panel.  Its
`rub_per_unit` column is always RUB for one unit of foreign currency, so rates
with different official nominals can safely be compared. `available_on` applies
a one-calendar-day publication lag by default; feature code must use this date
instead of the quote date to remain as-of safe.
