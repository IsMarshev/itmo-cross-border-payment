"""MVP utility-and-risk model for the signal layer.

The brief asks two things that the earlier Ridge value-head does not answer.
First, a classifier for "did this day turn out to be a local minimum of the +-h
window". Second, a threshold that respects an *asymmetric* price of error:
telling a client to transfer and then watching the rate fall further is more
expensive than staying quiet through a good day, because it costs the client
money **and** burns a scarce push slot.

This module answers both with three small, inspectable heads fitted
walk-forward per corridor:

``p_min``   logistic — P(today is a local minimum of the +-h window)
            The brief's classification target. Reported as the confidence a
            push can carry, never as the decision itself.
``u_bps``   ridge — E[advantage in bps] of transferring now versus a typical day
            in the next ``h``. This is the *utility*: money the client saves.
``p_bad``   logistic — P(the average rate over the next ``h`` days turns out
            more than ``bad_push_bps`` better than what the client just got).
            Multiplied by the historical size of such a miss it becomes
            ``risk_bps``: money the client loses by acting on our word.

Utility and risk are deliberately two halves of the *same* quantity. Risk is the
lower partial moment of the advantage, not a different statistic such as the
distance to the horizon's luckiest day — that one is positive nearly always on a
random walk and would drown the utility term without carrying information.

The decision score is a mean-risk objective in basis points, stated relative to
an ordinary day in the same corridor::

    score = [u_bps - lambda * risk_bps] - [base_u_bps - lambda * base_risk_bps]

The bracketed baseline is the same objective evaluated unconditionally on the
training window, so ``score >= 0`` reads as *"this day is worth a push, an
ordinary day is not"*. Without that centring the threshold would be vacuous: the
unconditional expected saving on these corridors is about +15 bps while the
unconditional downside is about 155 bps, so an absolute ``score >= 0`` rule
would silence the model on every day of every corridor.

``lambda`` is the only judgement call, and it is a product parameter rather than
a fitted one: it states how many roubles of missed opportunity the business will
trade for one rouble of client loss after a push. ``lambda = 0`` is risk-neutral
expected value; ``lambda = 2`` (the default) reflects that a bad push consumes
the next contact as well as the current one. The benchmark reports the whole
sensitivity curve instead of pretending one value is optimal.

Sending on ``score >= 0`` *is* the asymmetric threshold the brief asks for. The
communication policy then thins whatever survives down to the 1-2 pushes per
week the channel allows.

Leakage contract
----------------
At decision time ``T`` (the quote's ``available_on``) a head may train only on
rows whose label had already matured by ``T``. Because label maturity is
monotone in quote order, that training set is a prefix and is found with a
binary search. Heads are refitted every ``refit_every`` observations and reused
in between; the reused model is always an *older* one, so the cadence can only
make the evaluation more conservative, never leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, compute_features
from .labels import build_labels
from .models import RidgeModel
from .rules import RULE_FEATURE_COLUMNS, rule_matrix

SCORE_COLUMNS: tuple[str, ...] = (
    "quote_date",
    "available_on",
    "iso",
    "rub_per_unit",
    "p_min",
    "p_bad",
    "u_bps",
    "risk_bps",
    "base_u_bps",
    "base_risk_bps",
    "score",
    "training_observations",
)


@dataclass(frozen=True, slots=True)
class UtilityRiskConfig:
    """Parameters fixed before any fit happens."""

    horizon: int = 10
    execution_offset: int = 1
    bad_push_bps: float = 100.0
    """A push is bad if the average rate over the next ``horizon`` days was more
    than this much better than what the client got. 100 bps = 1%, against a
    monthly range on these corridors of roughly 6%."""
    local_min_tolerance_bps: float = 10.0
    lam: float = 2.0
    """Price of error: how much more a rouble lost after a push costs than a
    rouble of missed opportunity."""
    min_train: int = 750
    refit_every: int = 21
    ridge_alpha: float = 1.0
    logit_l2: float = 1.0
    feature_set: str = "raw"
    """Which design matrix the heads see.

    ``raw``    the 22 backward-looking features from ``features.py``. The
               default, on evidence rather than taste.
    ``rules``  the indicator scores from ``signal_layer.rules``. Trying the
               brief's own combination step — weight the indicators instead of
               re-deriving them from raw features — looked like the obvious fix
               for the low signal-to-noise ratio, and it *lost badly*: -7.0 bps
               against +14.0 for ``raw``. The rules are strongly collinear with
               each other, and a linear blend fitted to a noisy target destroys
               the individually robust ranking each one has on its own. Kept
               selectable so the result stays reproducible.
    ``both``   the union: +12.2 bps, behind ``raw``. Seven extra collinear
               columns buy nothing.
    """

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("lam must be non-negative")
        if self.refit_every <= 0 or self.min_train <= 0:
            raise ValueError("refit_every and min_train must be positive")
        if self.feature_set not in ("rules", "raw", "both"):
            raise ValueError(f"Unknown feature_set {self.feature_set!r}")

    @property
    def columns(self) -> tuple[str, ...]:
        """The design-matrix columns implied by ``feature_set``."""
        if self.feature_set == "rules":
            return RULE_FEATURE_COLUMNS
        if self.feature_set == "raw":
            return FEATURE_COLUMNS
        return (*RULE_FEATURE_COLUMNS, *FEATURE_COLUMNS)


class LogisticIRLS:
    """L2-penalised logistic regression fitted by Newton/IRLS on standardised inputs.

    Small enough to read, which matters: the brief bars generative models from
    the signal loop precisely so that every push can be traced to coefficients
    somebody can look at.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 30, tol: float = 1e-7):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.constant_: float | None = None  # set when the target is degenerate

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticIRLS:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0

        rate = float(y.mean()) if len(y) else 0.0
        if len(y) == 0 or rate <= 0.0 or rate >= 1.0:
            # Nothing to separate; fall back to the base rate.
            self.constant_ = min(max(rate, 1e-6), 1 - 1e-6)
            self.coef_ = np.zeros(X.shape[1])
            return self
        self.constant_ = None

        Xs = (X - self.mean_) / self.scale_
        n, k = Xs.shape
        design = np.hstack([Xs, np.ones((n, 1))])
        penalty = self.l2 * np.eye(k + 1)
        penalty[-1, -1] = 0.0  # never penalise the intercept
        weights = np.zeros(k + 1)
        weights[-1] = np.log(rate / (1.0 - rate))

        for _ in range(self.max_iter):
            eta = np.clip(design @ weights, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-eta))
            variance = np.clip(probability * (1.0 - probability), 1e-8, None)
            working = eta + (y - probability) / variance
            lhs = design.T @ (variance[:, None] * design) + penalty
            rhs = design.T @ (variance * working)
            try:
                updated = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(updated)):
                break
            step = float(np.max(np.abs(updated - weights)))
            weights = updated
            if step < self.tol:
                break

        self.coef_ = weights[:-1]
        self.intercept_ = float(weights[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(np.atleast_2d(X)), self.constant_)
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        eta = np.clip(Xs @ self.coef_ + self.intercept_, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-eta))


_FAR_FUTURE = np.datetime64("2262-01-01", "ns")


def _training_prefix_lengths(
    label_available_on: np.ndarray, decision_time: np.ndarray
) -> np.ndarray:
    """How many rows have a matured label by each row's decision time.

    ``label_available_on`` is non-decreasing in quote order (it is the
    availability date of an observation a fixed number of steps ahead), so the
    admissible training set is always a prefix and one binary search per row
    suffices. Rows whose label has not matured at all are pushed to the far
    future so they never enter a training set.
    """
    matured = np.where(np.isnat(label_available_on), _FAR_FUTURE, label_available_on)
    matured = np.maximum.accumulate(matured)
    return np.searchsorted(matured, decision_time, side="right")


def _prepare(panel: pd.DataFrame, iso: str, config: UtilityRiskConfig) -> pd.DataFrame:
    """Feature/label matrix for one corridor, warm-up rows dropped."""
    features = compute_features(panel)
    features = pd.concat([features, rule_matrix(features)], axis=1)
    labels = build_labels(
        panel,
        horizon=config.horizon,
        execution_offset=config.execution_offset,
        bad_push_bps=config.bad_push_bps,
        local_min_tolerance_bps=config.local_min_tolerance_bps,
    )
    label_columns = [
        "quote_date",
        "iso",
        "fwd_advantage_bps",
        "adverse_bps",
        "bad_push",
        "is_local_min",
        "label_available_on",
        "outcome_complete",
    ]
    joined = features[features["iso"] == iso].merge(
        labels.loc[labels["iso"] == iso, label_columns],
        on=["quote_date", "iso"],
        how="inner",
        validate="one_to_one",
    )
    joined = joined.dropna(subset=list(config.columns))
    return joined.sort_values("quote_date").reset_index(drop=True)


def walk_forward_scores(
    panel: pd.DataFrame,
    iso: str,
    config: UtilityRiskConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every day of one corridor as it would have been scored live.

    Returns ``(scores, coefficients)``. ``scores`` carries the three heads, the
    risk in basis points and the net decision score; ``coefficients`` records
    one row per refit so the model's reasoning can be inspected over time.
    """
    resolved = config or UtilityRiskConfig()
    data = _prepare(panel, iso, resolved)
    if data.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS), pd.DataFrame()

    columns = list(resolved.columns)
    X = data[columns].to_numpy(dtype=float)
    decision_time = data["available_on"].to_numpy(dtype="datetime64[ns]")
    label_time = data["label_available_on"].to_numpy(dtype="datetime64[ns]")
    y_utility = data["fwd_advantage_bps"].to_numpy(dtype=float)
    y_bad = data["bad_push"].to_numpy(dtype=float)
    y_min = data["is_local_min"].to_numpy(dtype=float)
    adverse = data["adverse_bps"].to_numpy(dtype=float)

    train_lengths = _training_prefix_lengths(label_time, decision_time)

    n = len(data)
    p_min = np.full(n, np.nan)
    p_bad = np.full(n, np.nan)
    u_bps = np.full(n, np.nan)
    risk_bps = np.full(n, np.nan)
    base_u = np.full(n, np.nan)
    base_risk = np.full(n, np.nan)
    training_observations = train_lengths.copy()

    utility_head: RidgeModel | None = None
    min_head: LogisticIRLS | None = None
    bad_head: LogisticIRLS | None = None
    mean_bad_move = float("nan")
    baseline_utility = float("nan")
    baseline_risk = float("nan")
    last_fit_at = -(10**9)
    coefficient_rows: list[dict[str, object]] = []

    for i in range(n):
        k = int(train_lengths[i])
        if k < resolved.min_train:
            continue
        if utility_head is None or (i - last_fit_at) >= resolved.refit_every:
            X_train = X[:k]
            utility_head = RidgeModel(alpha=resolved.ridge_alpha).fit(X_train, y_utility[:k])
            min_head = LogisticIRLS(l2=resolved.logit_l2).fit(X_train, y_min[:k])
            bad_head = LogisticIRLS(l2=resolved.logit_l2).fit(X_train, y_bad[:k])
            is_bad = y_bad[:k] > 0.5
            mean_bad_move = float(adverse[:k][is_bad].mean()) if is_bad.any() else 0.0
            # What the same objective is worth on an ordinary day of this
            # corridor, measured on the training window only.
            baseline_utility = float(y_utility[:k].mean())
            baseline_risk = float((adverse[:k] * is_bad).mean())
            last_fit_at = i
            coefficient_rows.append(
                {
                    "iso": iso,
                    "fitted_on": data.loc[i, "quote_date"],
                    "training_observations": k,
                    "mean_bad_move_bps": mean_bad_move,
                    "base_utility_bps": baseline_utility,
                    "base_risk_bps": baseline_risk,
                    "base_rate_local_min": float(y_min[:k].mean()),
                    "base_rate_bad_push": float(y_bad[:k].mean()),
                    **{
                        f"u_{name}": float(coefficient)
                        for name, coefficient in zip(
                            columns, utility_head.coef_, strict=True
                        )
                    },
                    **{
                        f"bad_{name}": float(coefficient)
                        for name, coefficient in zip(
                            columns, bad_head.coef_, strict=True
                        )
                    },
                }
            )

        row = X[i : i + 1]
        u_bps[i] = float(utility_head.predict(row)[0])
        p_min[i] = float(min_head.predict_proba(row)[0])
        p_bad[i] = float(bad_head.predict_proba(row)[0])
        risk_bps[i] = p_bad[i] * mean_bad_move
        base_u[i] = baseline_utility
        base_risk[i] = baseline_risk

    scores = pd.DataFrame(
        {
            "quote_date": data["quote_date"],
            "available_on": data["available_on"],
            "iso": iso,
            "rub_per_unit": data["rub_per_unit"],
            "p_min": p_min,
            "p_bad": p_bad,
            "u_bps": u_bps,
            "risk_bps": risk_bps,
            "base_u_bps": base_u,
            "base_risk_bps": base_risk,
            "training_observations": training_observations,
        }
    )
    scores = scores.dropna(subset=["u_bps"]).reset_index(drop=True)
    return rescore(scores, resolved.lam), pd.DataFrame(coefficient_rows)


def rescore(scores: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Re-derive the net score at a different price of error.

    The heads do not depend on ``lambda``, so the whole sensitivity curve is
    column arithmetic away and needs no refitting.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    updated = scores.copy()
    objective = updated["u_bps"] - lam * updated["risk_bps"]
    baseline = updated["base_u_bps"] - lam * updated["base_risk_bps"]
    updated["score"] = objective - baseline
    return updated


def scores_asof(
    panel: pd.DataFrame,
    iso: str,
    asof: pd.Timestamp,
    config: UtilityRiskConfig | None = None,
) -> pd.DataFrame:
    """Scores exactly as they would have looked at ``asof`` — the audit entry point.

    The brief disqualifies any look-ahead, so it must be possible to ask "what
    did the model say on date T" from a truncated panel and get the same answer
    as the historical run. Truncating the panel first is the strongest form of
    that check: data after ``asof`` is not merely masked, it is absent.
    """
    asof = pd.Timestamp(asof)
    truncated = panel[panel["quote_date"] <= asof]
    if truncated.empty:
        raise ValueError(f"No observations on or before {asof:%Y-%m-%d}")
    scores, _ = walk_forward_scores(truncated, iso, config)
    return scores
