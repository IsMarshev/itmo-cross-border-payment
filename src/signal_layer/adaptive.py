"""Walk-forward calibration: let each corridor pick its own parameter.

The brief asks for windows and thresholds calibrated per corridor, because the
corridors do not share a volatility. It also forbids look-ahead, so the choice
has to be made from the past alone and re-made as history accumulates.

The mechanism here is deliberately thinner than a model. It never fits
coefficients; it *selects* among candidate score series that are each already a
valid indicator. That matters given what the benchmark found: fitting weights to
this target loses to a single robust rule, while a selection among robust rules
keeps their ranking intact. What is estimated is one discrete choice per refit,
which is about the least a calibration step can spend.

Selection criterion is the trailing Spearman correlation between a candidate's
score and the client money the day actually delivered, computed only over rows
whose outcome had already matured at decision time. Rank correlation rather than
mean gain: the policy selects days by rank, so the criterion should score the
ranking, and a rank statistic is not dragged around by one violent month.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import build_labels
from .rules import RULE_NAMES, rule_score
from .utility_risk import _training_prefix_lengths

# Candidate windows offered to the calibration. The grid reaches well below the
# 60-day span baked into `features.py` because the selection kept pinning its
# lower edge, and an optimum at a grid boundary is not an optimum.
ZSCORE_SPANS: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 250)
PERCENTILE_WINDOWS: tuple[int, ...] = (20, 30, 60, 90, 180, 250)

SCORE_SCHEMA: tuple[str, ...] = (
    "quote_date",
    "available_on",
    "iso",
    "rub_per_unit",
    "score",
    "chosen",
)


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Parameters of the calibration itself, fixed before any run."""

    horizon: int = 10
    execution_offset: int = 1
    refit_every: int = 21
    min_train: int = 500
    lookback: int = 750
    """Trailing matured rows a candidate is judged on. Long enough to span
    several regimes, short enough that a decade-old regime cannot pin the
    choice."""


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, the way Spearman is defined.

    Ordinal ranks from a double ``argsort`` would break ties arbitrarily, which
    on a constant candidate manufactures a perfectly ordered rank vector out of
    no information at all.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    ordered = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, NaN-safe. NumPy only — this project has no SciPy.

    A candidate with no variation carries no ranking and must score NaN rather
    than a number: it is checked on the values, not on their ranks, because
    ties resolved arbitrarily would hide exactly that case.
    """
    usable = np.isfinite(a) & np.isfinite(b)
    if usable.sum() < 30:
        return float("nan")
    x, y = a[usable], b[usable]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    rank_x, rank_y = _rankdata(x), _rankdata(y)
    if rank_x.std() == 0 or rank_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def zscore_candidates(spans: tuple[int, ...]) -> dict[str, Callable[[pd.Series], pd.Series]]:
    """Deviation below an EWMA, in rolling-sigma units, at several spans."""

    def make(span: int) -> Callable[[pd.Series], pd.Series]:
        def build(values: pd.Series) -> pd.Series:
            mean = values.ewm(span=span, adjust=False).mean()
            alpha = 2.0 / (span + 1.0)
            variance = ((values - mean) ** 2).ewm(alpha=alpha, adjust=False).mean()
            deviation = np.sqrt(variance).replace(0, np.nan)
            return -(values - mean) / deviation

        return build

    return {f"span={span}": make(span) for span in spans}


def percentile_candidates(
    windows: tuple[int, ...],
) -> dict[str, Callable[[pd.Series], pd.Series]]:
    """How low the rate sits in its own trailing range, at several windows."""

    def make(window: int) -> Callable[[pd.Series], pd.Series]:
        def build(values: pd.Series) -> pd.Series:
            return 1.0 - values.rolling(window, min_periods=max(20, window // 4)).rank(
                pct=True
            )

        return build

    return {f"window={window}": make(window) for window in windows}


def walk_forward_tuned(
    panel: pd.DataFrame,
    iso: str,
    candidates: dict[str, Callable[[pd.Series], pd.Series]] | None = None,
    config: TuningConfig | None = None,
    *,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score one corridor, re-choosing the candidate as history accumulates.

    ``candidates`` maps a label to a builder over the corridor's rate series.
    Pass ``features`` instead to select among the rule indicators themselves,
    which is the "which indicator suits this corridor" question the brief poses.
    """
    resolved = config or TuningConfig()
    corridor = panel[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    if "available_on" not in corridor.columns:
        corridor["available_on"] = corridor["quote_date"]
    if corridor.empty:
        return pd.DataFrame(columns=SCORE_SCHEMA)

    values = corridor["rub_per_unit"].astype(float)
    if features is not None:
        rows = features[features["iso"] == iso].sort_values("quote_date")
        matrix = pd.DataFrame(
            {name: rule_score(name, rows).to_numpy(dtype=float) for name in RULE_NAMES}
        )
        matrix.index = corridor.index[: len(matrix)]
        matrix = matrix.reindex(corridor.index)
    else:
        built = candidates or zscore_candidates((20, 60, 120, 250))
        matrix = pd.DataFrame({name: fn(values) for name, fn in built.items()})

    labels = build_labels(
        panel,
        horizon=resolved.horizon,
        execution_offset=resolved.execution_offset,
    )
    labels = labels[labels["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    joined = corridor.merge(
        labels[["quote_date", "currency_gain_bps", "label_available_on"]],
        on="quote_date",
        how="left",
    )
    gain = joined["currency_gain_bps"].to_numpy(dtype=float)
    train_lengths = _training_prefix_lengths(
        joined["label_available_on"].to_numpy(dtype="datetime64[ns]"),
        joined["available_on"].to_numpy(dtype="datetime64[ns]"),
    )

    names = list(matrix.columns)
    scores = matrix.to_numpy(dtype=float)
    n = len(corridor)
    chosen = np.full(n, "", dtype=object)
    out = np.full(n, np.nan)
    current: int | None = None
    last_fit_at = -(10**9)

    for i in range(n):
        k = int(train_lengths[i])
        if k < resolved.min_train:
            continue
        if current is None or (i - last_fit_at) >= resolved.refit_every:
            start = max(0, k - resolved.lookback)
            best, best_rho = None, -np.inf
            for index in range(len(names)):
                rho = _spearman(scores[start:k, index], gain[start:k])
                if np.isfinite(rho) and rho > best_rho:
                    best, best_rho = index, rho
            if best is not None:
                current = best
                last_fit_at = i
        if current is None or not np.isfinite(scores[i, current]):
            continue
        out[i] = scores[i, current]
        chosen[i] = names[current]

    result = pd.DataFrame(
        {
            "quote_date": corridor["quote_date"],
            "available_on": corridor["available_on"],
            "iso": iso,
            "rub_per_unit": corridor["rub_per_unit"],
            "score": out,
            "chosen": chosen,
        }
    )
    return result.dropna(subset=["score"]).reset_index(drop=True)
