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
