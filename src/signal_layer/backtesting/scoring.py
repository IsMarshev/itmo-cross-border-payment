"""Leakage-safe score sources consumed by the same policy engine."""

from __future__ import annotations

import pandas as pd

from signal_layer.features import compute_features
from signal_layer.models import walk_forward_predict


def build_baseline_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Combine two factual trailing percentiles into an interpretable score.

    Both component percentiles are in ``[0, 1]`` and are computed using only
    current and previous observations. A larger score means a cheaper current
    quote relative to its recent regime.
    """
    features = compute_features(panel)
    if "available_on" not in features.columns:
        features["available_on"] = features["quote_date"]
    features["score"] = (
        (1.0 - features["residual_pct"]) + (1.0 - features["pct_rank_90"])
    ) / 2.0
    features["score_source"] = "baseline"
    return features[
        ["quote_date", "available_on", "iso", "rub_per_unit", "score", "score_source"]
    ].dropna(subset=["score"])


def build_live_scores(
    panel: pd.DataFrame,
    corridors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """The score that actually ships: the calibrated, truth-gated z-score.

    Delegates to :mod:`signal_layer.signals` rather than re-deriving anything,
    so this backtest and the serving path cannot drift apart.
    """
    from signal_layer.signals import SignalLayerConfig, score

    config = SignalLayerConfig()
    parts: list[pd.DataFrame] = []
    for corridor in corridors:
        scored = score(panel, corridor, config)
        if len(scored):
            scored = scored.assign(score_source="live")
            parts.append(
                scored[
                    [
                        "quote_date",
                        "available_on",
                        "iso",
                        "rub_per_unit",
                        "score",
                        "score_source",
                    ]
                ]
            )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"])


def build_model_scores(
    panel: pd.DataFrame,
    corridors: list[str] | tuple[str, ...],
    *,
    horizon: int = 20,
    min_train: int = 500,
    alpha: float = 1.0,
    model: str = "ridge",
) -> pd.DataFrame:
    """Return purged walk-forward predictions in the policy score schema."""
    parts: list[pd.DataFrame] = []
    for corridor in corridors:
        predictions = walk_forward_predict(
            panel,
            corridor,
            h=horizon,
            min_train=min_train,
            alpha=alpha,
            model=model,
        )
        scored = predictions.rename(columns={"pred_advantage": "score"}).copy()
        scored["score_source"] = model
        parts.append(
            scored[
                [
                    "quote_date",
                    "available_on",
                    "iso",
                    "rub_per_unit",
                    "score",
                    "score_source",
                ]
            ]
        )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["iso", "quote_date"])
