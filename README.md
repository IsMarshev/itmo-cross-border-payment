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

The service exposes:

- `GET /health`;
- `GET /v1/rates/{ISO}/latest?as_of=YYYY-MM-DD`;
- `GET /v1/signals/{ISO}/evaluate?as_of=YYYY-MM-DD&strategy=baseline|ridge`;
- `POST /v1/backtests/run`.

The signal endpoint returns a non-stateful *candidate*, not a delivered push.
Its client-facing message contains only a trailing-rate fact. The optional
Ridge strategy is trained only on targets whose full future horizon was already
available by the requested as-of date.

## CBSB-1 benchmark and the utility/risk model

`BENCHMARK.md` defines how a signal layer is judged: every strategy — random,
rule, model, oracle — gets the same push budget, is evaluated only out of time,
and is tested against a matched random schedule rather than against zero.

```bash
uv run python -m signal_layer.run_benchmark --out reports/benchmark
```

A run writes `reports/benchmark/dashboard.html` — a standalone page (inline SVG,
no external dependencies, light and dark) that leads with the run's conclusions
— alongside `scorecard.md` and the raw CSVs.

`signal_layer.signals` is what ships, and the only entry point that serves: a
z-score whose window is chosen walk-forward, which refuses any day whose push
would have no true favourable fact to state. 81.7bp of client money per transfer
against 23.4 for the same rule with a fixed window, significant on all five
corridors. It takes no strategy parameter — the benchmark makes that choice.

```python
from signal_layer.signals import signal_table, signals_asof
signal_table(panel, ["TJS", "UZS"])       # the brief's signal table
signals_asof(panel, ["TJS"], "2026-06-15")  # what we would have sent on that date
```

`signal_layer.utility_risk` is the MVP scored by it: three walk-forward heads
(P(local minimum), expected advantage, P(bad push)) combined into a mean-risk
score in basis points, `score = [u - lambda*risk] - [baseline]`, where `lambda`
is the price of an asymmetric error. Read `BENCHMARK.md` before the numbers.

## Stage-4 backtest

Run the canonical baseline backtest and write an exhaustive decision journal,
matched random schedules and a summary report:

```bash
poetry run python -m signal_layer.run_backtest \
  --corridors TJS UZS KGS AMD KZT \
  --score-source baseline \
  --horizon 20 \
  --out reports/backtest
```

`decision_log.jsonl` includes every evaluated day, the historical threshold,
remaining communication slots, decision reason and realised outcome. Outcomes
are joined only after the chronological policy has completed. The random
baseline selects the same number of dates inside every corridor and
communication window. Confidence intervals use moving-block bootstrap.

Each run also produces a standalone `dashboard.html` with KPI, confidence
intervals, matched-random comparison, risk, signal frequency and outcome
distribution. Regenerate it independently with:

```bash
poetry run python -m signal_layer.dashboard --report-dir reports/backtest
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
