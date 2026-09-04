"""The field of contenders the benchmark scores.

Every strategy answers the same narrow question — *how favourable is today?* —
by producing one number per corridor per trading day. The runner then pushes all
of them through the identical communication policy, so nothing in this file can
buy an advantage by signalling more often.

Only the ordering of a score matters, never its scale: the policy selects by
trailing quantile. That is what lets a percentile rule, a z-score and a model
denominated in basis points sit in the same table.

The field
---------
``random``            matched random days at the same cadence — the brief's
                      reference. Produced by the runner's sampler, not here.
``percentile``        the brief's *level* indicator: rate low in its 90-day range.
``zscore``            textbook mean reversion against an EWMA(60).
``momentum``          the brief's *momentum* indicator: N falling days in a row.
``drawdown``          distance below the 60-day high.
``seasonal``          the brief's *seasonality* indicator: cheap for this
                      calendar month by multi-year standards.
``reversal``          the brief's *"окно закрывается"*: near the 90-day minimum
                      and turning up.
``utility_only``      the MVP's utility head alone (lambda = 0). The ablation
                      that isolates what the risk head contributes.
``utility_risk``      the MVP: utility minus lambda-weighted risk, silent when
                      the expected saving does not cover the risk.
``utility_risk_paced``the same score without the "stay silent" floor, so it
                      spends exactly the same push budget as everyone else.
``oracle``            perfect knowledge of each day's value, but still decided
                      online through the shared policy. The ceiling on *scoring*
                      with the policy held fixed.
``oracle_topk``       perfect knowledge and free choice of the best days each
                      week. The ceiling on the whole problem, and the
                      denominator of the CBSB score. The gap between the two
                      oracles is what the greedy online policy costs.

Message scenarios
-----------------
A strategy is judged by the hit rule of the message it would actually send. The
two rules are not interchangeable and, on this data, point in opposite
directions: "курс останется не хуже" is satisfied precisely when the rate keeps
falling, which is when the client should have waited. Strategies that aim at a
local minimum therefore carry the "окно закрывается" message, and both hit rates
are reported for every strategy so the tension stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..rules import BLOCKED as _BLOCKED_SENTINEL
from ..rules import rule_score
from ..utility_risk import UtilityRiskConfig, rescore, walk_forward_scores
from .spec import BenchmarkSpec

SCORE_SCHEMA: tuple[str, ...] = ("quote_date", "available_on", "iso", "rub_per_unit", "score")

_BLOCKED = _BLOCKED_SENTINEL


@dataclass(frozen=True, slots=True)
class Strategy:
    """One contender: how it scores a day, and how its message is judged."""

    name: str
    kind: str  # "rule" | "model" | "oracle" — where the score comes from
    scenario: str  # "favourable_now" | "window_closing"
    description: str
    minimum_score: float | None = None
    lam: float | None = None  # price of error, for the utility-risk family
    rule: str | None = None
    """Which rule expression to score with, when it is not the strategy's own
    name — a diagnostic twin reuses its contender's rule."""
    selection: str = "policy"
    """How scores become signals. ``policy`` is the shared online communication
    policy every contender must live with. ``weekly_best`` picks the highest
    scoring days of each week with hindsight over the week — impossible live,
    and included only to separate a bad *score* from a bad *policy*."""


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "percentile", "rule", "favourable_now",
        "Курс в нижней части 90-дневного диапазона",
    ),
    Strategy(
        "zscore", "rule", "favourable_now",
        "Курс ниже EWMA(60) в единицах скользящей сигмы",
    ),
    Strategy(
        "momentum", "rule", "favourable_now",
        "Курс снижается N дней подряд",
    ),
    Strategy(
        "drawdown", "rule", "favourable_now",
        "Курс просел от 60-дневного максимума",
    ),
    Strategy(
        "seasonal", "rule", "favourable_now",
        "Курс дешёвый для этого календарного месяца по многолетней норме",
    ),
    Strategy(
        "reversal", "rule", "window_closing",
        "Курс был у 90-дневного минимума и пошёл вверх",
        minimum_score=-1e5,
    ),
    Strategy(
        "utility_only", "model", "window_closing",
        "MVP без модели риска (lambda = 0): чистое ожидание выгоды",
        lam=0.0,
    ),
    Strategy(
        "utility_risk", "model", "window_closing",
        "MVP: полезность минус lambda x риск, молчит при отрицательном счёте",
        minimum_score=0.0,
        lam=2.0,
    ),
    Strategy(
        "utility_risk_paced", "model", "window_closing",
        "MVP без порога молчания — тот же бюджет пушей, что у остальных",
        lam=2.0,
    ),
    Strategy(
        "oracle", "oracle", "window_closing",
        "Идеальный счёт при той же политике отправки — потолок для качества счёта",
    ),
    Strategy(
        "oracle_topk", "oracle", "window_closing",
        "Идеальный выбор лучших дней недели — потолок задачи, знаменатель CBSB",
        selection="weekly_best",
    ),
    Strategy(
        "utility_risk_weekly", "model", "window_closing",
        "Диагностика: счёт MVP, но выбор лучших дней недели вместо жадной политики",
        lam=2.0,
        selection="weekly_best",
    ),
    Strategy(
        "percentile_weekly", "rule", "favourable_now",
        "Диагностика: правило процентиля с недельным выбором вместо жадной политики",
        rule="percentile",
        selection="weekly_best",
    ),
)

DEFAULT_STRATEGY_NAMES: tuple[str, ...] = tuple(s.name for s in STRATEGIES)

_BY_NAME = {s.name: s for s in STRATEGIES}


def get_strategy(name: str) -> Strategy:
    try:
        return _BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_BY_NAME))
        raise KeyError(f"Unknown strategy {name!r}; known: {known}") from None


def _rule_score(name: str, features: pd.DataFrame) -> pd.Series:
    """One rule's score. Definitions live in ``signal_layer.rules`` because the
    utility/risk model consumes the same expressions as its feature set."""
    return rule_score(name, features)


class ModelScoreCache:
    """Fits the walk-forward heads once per corridor and shares them.

    ``utility_only``, ``utility_risk`` and ``utility_risk_paced`` differ only in
    the price of error and the silence floor, so they must not pay for three
    separate walk-forward runs.
    """

    def __init__(self, config: UtilityRiskConfig):
        self.config = config
        self._scores: dict[str, pd.DataFrame] = {}
        self._coefficients: list[pd.DataFrame] = []

    def get(self, panel: pd.DataFrame, iso: str) -> pd.DataFrame:
        if iso not in self._scores:
            scores, coefficients = walk_forward_scores(panel, iso, self.config)
            self._scores[iso] = scores
            if len(coefficients):
                self._coefficients.append(coefficients)
        return self._scores[iso]

    @property
    def coefficients(self) -> pd.DataFrame:
        if not self._coefficients:
            return pd.DataFrame()
        return pd.concat(self._coefficients, ignore_index=True)


def build_scores(
    strategy: Strategy,
    spec: BenchmarkSpec,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cache: ModelScoreCache,
) -> pd.DataFrame:
    """Score every trading day of every scored corridor for one strategy."""
    if strategy.kind == "rule":
        rows = features[features["iso"].isin(spec.corridors)].copy()
        rows["score"] = _rule_score(strategy.rule or strategy.name, rows)
        return rows.dropna(subset=["score"])[list(SCORE_SCHEMA)].reset_index(drop=True)

    if strategy.kind == "oracle":
        rows = labels[
            labels["iso"].isin(spec.corridors) & labels["outcome_complete"]
        ].copy()
        rows["score"] = rows["currency_gain_bps"]
        return rows.dropna(subset=["score"])[list(SCORE_SCHEMA)].reset_index(drop=True)

    if strategy.kind == "model":
        parts = []
        for iso in spec.corridors:
            scores = cache.get(panel, iso)
            if scores.empty:
                continue
            lam = cache.config.lam if strategy.lam is None else strategy.lam
            parts.append(rescore(scores, lam)[list(SCORE_SCHEMA)])
        if not parts:
            return pd.DataFrame(columns=list(SCORE_SCHEMA))
        return pd.concat(parts, ignore_index=True)

    raise ValueError(f"Unknown strategy kind {strategy.kind!r}")
