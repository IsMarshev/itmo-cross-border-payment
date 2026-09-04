"""CBSB-1: the benchmark definition, as data.

One question, asked the same way for every strategy: *if the client transfers on
the days we would have written to them, how much better off are they than a
client who transfers on a random day — and how often was the statement in the
push actually true?*

The spec below is the whole contract. Everything else in this package only
executes it. Keeping thresholds here (rather than inline in the runner) means a
reviewer can read the pass/fail bar in one screen and a team can tighten it
without touching evaluation code.

Design choices worth knowing before reading the numbers
-------------------------------------------------------
* **Same push budget for everybody.** Every strategy — random, rule, model,
  oracle — is filtered through the *same* communication policy (<= 2 signals per
  week, cooldown, pacing quantile). The benchmark therefore measures only the
  quality of *day selection*, never the ability to spam.
* **Out-of-time, not out-of-sample.** Folds are consecutive half-year windows.
  A strategy sees a fold only after every earlier fold has been used for fitting
  and calibration. Averages across the whole history are not reported as the
  headline; the per-fold spread is.
* **The client acts one observation late.** A signal computed from the quote of
  day ``T`` is acted on at the next published quote (``execution_offset``). This
  removes the last ambiguity around the CBR publication lag, at the cost of
  looking slightly worse than a same-day-execution benchmark would.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class Gate:
    """One pass/fail condition from the case brief's "обязательные условия"."""

    name: str
    question: str  # what a reader should understand this gate to mean
    metric: str  # column in the summary frame
    op: str  # ">=", ">", "<=", "<", "between"
    bound: float | tuple[float, float]
    target: float | None = None  # the brief's stretch value, if any
    mandatory: bool = True

    def check(self, value: float) -> bool | None:
        """``True``/``False``, or ``None`` when the metric could not be computed."""
        if value is None or pd.isna(value):
            return None
        if self.op == ">=":
            return bool(value >= self.bound)
        if self.op == ">":
            return bool(value > self.bound)
        if self.op == "<=":
            return bool(value <= self.bound)
        if self.op == "<":
            return bool(value < self.bound)
        if self.op == "between":
            low, high = self.bound  # type: ignore[misc]
            return bool(low <= value <= high)
        raise ValueError(f"Unknown gate operator {self.op!r}")

    def describe(self) -> str:
        if self.op == "between":
            low, high = self.bound  # type: ignore[misc]
            return f"{low:g} <= {self.metric} <= {high:g}"
        return f"{self.metric} {self.op} {self.bound:g}"


# The brief's "обязательные условия работоспособности", one gate each.
GATES: tuple[Gate, ...] = (
    Gate(
        name="G1_lift",
        question="Сигнал информативнее случайного дня?",
        metric="hit_lift",
        op=">=",
        bound=1.0,
        target=1.3,
    ),
    Gate(
        name="G2_advantage",
        question="Выгода момента значимо больше нуля?",
        metric="window_advantage_ci_low",
        op=">",
        bound=0.0,
    ),
    Gate(
        name="G3_significance",
        question="Выигрыш в валюте отличим от случайного расписания?",
        metric="p_value",
        op="<=",
        bound=0.05,
    ),
    Gate(
        name="G4_frequency",
        question="Темп укладывается в коммуникационную политику?",
        metric="per_week",
        op="between",
        bound=(0.5, 2.0),
    ),
    Gate(
        name="G5_evenness",
        question="Сигналы идут ровно, а не пачками?",
        metric="interval_cv",
        op="<=",
        bound=1.0,
    ),
    Gate(
        name="G6_gap",
        question="Нет кварталов молчания?",
        metric="max_gap_days",
        op="<=",
        bound=75.0,
    ),
    Gate(
        name="G7_risk",
        question="Плохих пушей не больше, чем у случайного дня?",
        metric="bad_push_delta",
        op="<=",
        bound=0.0,
    ),
)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Everything a run needs, fixed before any strategy is evaluated."""

    # --- corridors ---
    corridors: tuple[str, ...] = ("TJS", "UZS", "KGS", "AMD", "KZT")
    # Loaded for cross-currency context features, never scored as a corridor.
    context_corridors: tuple[str, ...] = ("USD",)

    # --- outcome definition ---
    horizon: int = 10
    """Days over which a push's statement must hold. The brief asks for
    h in {1,3,5,10,20}; 10 is the headline, the rest are reported alongside."""
    reported_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    execution_offset: int = 1
    """The client transfers at the next published quote after the signal."""
    epsilon_bps: float = 0.0
    """Tolerance when checking "the rate stayed no worse"."""
    bad_push_bps: float = 100.0
    """A signal is a *bad push* if the average rate over the next ``horizon``
    days turned out more than this much better than what the client got: we told
    them to transfer and the moment improved without them. This is the expensive
    error the brief singles out. 100 bps = 1%, against a monthly range on these
    corridors of roughly 6%; measuring it against the horizon's single luckiest
    day instead would flag ~72% of all days and carry no information."""
    local_min_tolerance_bps: float = 10.0
    """Slack when labelling a day a local minimum of the +-h window."""

    # --- evaluation window ---
    eval_start: pd.Timestamp = pd.Timestamp("2021-09-01")
    fold_months: int = 6

    # --- communication policy, identical for every strategy ---
    max_signals_per_week: int = 2
    cooldown_observations: int = 3
    threshold_lookback: int = 250
    minimum_threshold_history: int = 20

    # --- statistics ---
    random_trials: int = 500
    bootstrap_trials: int = 1_000
    seed: int = 0
    fdr_alpha: float = 0.05

    gates: tuple[Gate, ...] = field(default=GATES)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.execution_offset < 0:
            raise ValueError("execution_offset must be non-negative")
        if self.fold_months <= 0:
            raise ValueError("fold_months must be positive")
        if not self.corridors:
            raise ValueError("at least one corridor must be scored")

    @property
    def all_currencies(self) -> tuple[str, ...]:
        """Corridors plus the context series needed by the feature layer."""
        return tuple(dict.fromkeys(self.corridors + self.context_corridors))

    def folds(self, last_date: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        """Consecutive out-of-time windows covering ``[eval_start, last_date]``."""
        bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        start = pd.Timestamp(self.eval_start)
        step = pd.DateOffset(months=self.fold_months)
        while start <= last_date:
            end = min(pd.Timestamp(last_date), start + step - pd.Timedelta(days=1))
            bounds.append((f"{start:%Y-%m}..{end:%Y-%m}", start, end))
            start = start + step
        return bounds
