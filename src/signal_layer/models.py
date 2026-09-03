"""Model layer: a regularised linear model for expected transfer advantage.

This is the Stage-5 "value head" from implementation_plan.md: a Ridge regression
predicting the client's advantage (in basis points) of transferring today versus
the median rate over the next ``H`` observations. Ridge (not OLS) is chosen on
purpose: the effective sample is small and features are correlated, so
regularisation buys stability.

The model is trained per corridor with walk-forward (expanding window): fit on
all data available up to ``T``, predict at ``T``. Features come from
``signal_layer.features`` and are strictly backward-looking, so the only
look-ahead lives in the target ``advantage``, which is known only in the past
during training and unknown at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, compute_features

# Target: client advantage of transferring today vs the median of the next H obs.
# advantage = (median(p_{t+1..t+H}) - p_t) / p_t * 10_000  [basis points]
DEFAULT_H = 20


@dataclass
class RidgeConfig:
    alpha: float = 1.0
    h: int = DEFAULT_H


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """One as-of model prediction made from an already available quote."""

    currency: str
    quote_date: date
    available_on: date
    predicted_advantage_bps: float
    training_observations: int
    model: Literal["ridge", "catboost"]


class RidgeModel:
    """Closed-form Ridge regression fit on standardised features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        Xs = (X - self.mean_) / self.scale_
        n, k = Xs.shape
        # Closed form: w = (X'X + alpha I)^-1 X'y, intercept via centred y.
        Xb = np.hstack([Xs, np.ones((n, 1))])
        reg = self.alpha * np.eye(k + 1)
        reg[-1, -1] = 0.0  # do not penalise the intercept
        w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.coef_ = w[:-1]
        self.intercept_ = float(w[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.scale_
        return Xs @ self.coef_ + self.intercept_


def make_target(panel: pd.DataFrame, iso: str, h: int = DEFAULT_H) -> pd.DataFrame:
    """Build the advantage target for one corridor.

    ``advantage = (median(p_{t+1..t+H}) - p_t) / p_t * 10_000``. The last ``h``
    observations have no full future window and get NaN (dropped at fit time).
    """
    if h <= 0:
        raise ValueError("h must be positive")
    s = panel.loc[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    if "available_on" not in s.columns:
        s["available_on"] = s["quote_date"]
    v = s["rub_per_unit"].astype(float).to_numpy()
    adv = np.full(len(v), np.nan)
    target_available_on = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for i in range(len(v) - h):
        future = v[i + 1 : i + 1 + h]
        adv[i] = (np.median(future) - v[i]) / v[i] * 10_000.0
        target_available_on.iloc[i] = s.loc[i + h, "available_on"]
    s = s.assign(advantage=adv, target_available_on=target_available_on)
    return s[
        [
            "quote_date",
            "available_on",
            "iso",
            "rub_per_unit",
            "advantage",
            "target_available_on",
        ]
    ]


def build_dataset(
    panel: pd.DataFrame,
    iso: str,
    h: int = DEFAULT_H,
    *,
    include_unlabelled: bool = False,
) -> pd.DataFrame:
    """Join features and targets while retaining target maturity timestamps."""
    feats = compute_features(panel)
    feats = feats[feats["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    tgt = make_target(panel, iso, h=h)
    df = feats.merge(
        tgt[["quote_date", "advantage", "target_available_on"]],
        on="quote_date",
        how="inner",
    )
    required = list(FEATURE_COLUMNS)
    if not include_unlabelled:
        required.append("advantage")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def _walk_forward(
    dataset: pd.DataFrame,
    iso: str,
    min_train: int,
    fit_fn,
) -> pd.DataFrame:
    """Expanding walk-forward using only targets matured by decision time."""
    X = dataset[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    predictions = np.full(len(dataset), np.nan)
    training_sizes = np.zeros(len(dataset), dtype=int)
    for i, row in dataset.iterrows():
        decision_time = row["available_on"]
        train_mask = (
            dataset["advantage"].notna()
            & dataset["target_available_on"].notna()
            & (dataset["target_available_on"] <= decision_time)
            & (dataset["quote_date"] < row["quote_date"])
        )
        train_indices = np.flatnonzero(train_mask.to_numpy())
        training_sizes[i] = len(train_indices)
        if len(train_indices) < min_train:
            continue
        X_train = X[train_indices]
        y_train = dataset.iloc[train_indices]["advantage"].to_numpy(dtype=float)
        predict_fn = fit_fn(X_train, y_train)
        predictions[i] = float(predict_fn(X[i : i + 1])[0])

    result = dataset[
        ["quote_date", "available_on", "rub_per_unit", "advantage", "target_available_on"]
    ].copy()
    result.insert(2, "iso", iso)
    result["pred_advantage"] = predictions
    result["training_observations"] = training_sizes
    return result.dropna(subset=["pred_advantage"]).reset_index(drop=True)


def walk_forward_predict(
    panel: pd.DataFrame,
    iso: str,
    *,
    min_train: int = 500,
    h: int = DEFAULT_H,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Expanding-window walk-forward prediction of advantage for one corridor.

    For each date ``T`` from the ``min_train``-th observation onward, fit on all
    data *before* ``T`` (target uses future up to ``T-1``'s horizon, which is in
    the past relative to ``T``) and predict the advantage at ``T``.

    Returns a frame with ``quote_date, iso, rub_per_unit, advantage (actual),
    pred_advantage`` — the raw model output. Thresholding into signals happens
    in the backtester / policy, not here.
    """
    df = build_dataset(panel, iso, h=h, include_unlabelled=True)

    def fit_fn(X_tr, y_tr):
        m = RidgeModel(alpha=alpha).fit(X_tr, y_tr)
        return m.predict

    return _walk_forward(X, y, dates, iso, rates, min_train, fit_fn)
