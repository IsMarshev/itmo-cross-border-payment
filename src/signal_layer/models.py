"""Model layer: a regularised linear model for expected transfer advantage.

This is the Stage-5 "value head" from implementation_plan.md: a Ridge regression
predicting the client's advantage (in basis points) of transferring today versus
the median rate over the next ``H`` observations. Ridge (not OLS) is chosen on
purpose: the effective sample is small and features are correlated, so
regularisation buys stability. A GBDT challenger is deferred to a later stage.

The model is trained per corridor with walk-forward (expanding window): fit on
all data available up to ``T``, predict at ``T``. Features come from
``signal_layer.features`` and are strictly backward-looking, so the only
look-ahead lives in the target ``advantage``, which is known only in the past
during training and unknown at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    s = panel.loc[panel["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    v = s["rub_per_unit"].astype(float).to_numpy()
    adv = np.full(len(v), np.nan)
    for i in range(len(v) - h):
        future = v[i + 1 : i + 1 + h]
        adv[i] = (np.median(future) - v[i]) / v[i] * 10_000.0
    s = s.assign(advantage=adv)
    return s[["quote_date", "iso", "rub_per_unit", "advantage"]]


def build_dataset(
    panel: pd.DataFrame, iso: str, h: int = DEFAULT_H
) -> pd.DataFrame:
    """Join features and target for one corridor, dropping rows with NaNs."""
    feats = compute_features(panel)
    feats = feats[feats["iso"] == iso].sort_values("quote_date").reset_index(drop=True)
    tgt = make_target(panel, iso, h=h)
    df = feats.merge(tgt[["quote_date", "advantage"]], on="quote_date", how="inner")
    df = df.dropna(subset=list(FEATURE_COLUMNS) + ["advantage"]).reset_index(drop=True)
    return df


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
    df = build_dataset(panel, iso, h=h)
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = df["advantage"].to_numpy(dtype=float)
    preds = np.full(len(df), np.nan)
    for i in range(min_train, len(df)):
        model = RidgeModel(alpha=alpha).fit(X[:i], y[:i])
        preds[i] = float(model.predict(X[i : i + 1])[0])
    df = df.assign(pred_advantage=preds)
    return df[
        ["quote_date", "iso", "rub_per_unit", "advantage", "pred_advantage"]
    ].dropna(subset=["pred_advantage"])
