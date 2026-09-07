"""Application service for reproducible Stage-4 backtests."""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from signal_layer.backtesting import (
    BacktestConfig,
    BacktestResult,
    PolicyConfig,
    build_baseline_scores,
    build_model_scores,
    run_backtest,
)
from signal_layer.services.rates import RateService

ScoreSource = Literal["baseline", "ridge"]


class BacktestService:
    """Load an as-of panel, build scores and execute the shared policy engine."""

    def __init__(self, rate_service: RateService) -> None:
        self._rate_service = rate_service

    def run(
        self,
        corridors: list[str],
        *,
        score_source: ScoreSource = "baseline",
        as_of: date | None = None,
        horizon: int = 20,
        epsilon_bps: float = 30.0,
        window: Literal["week", "month"] = "week",
        max_signals_per_window: int = 2,
        cooldown_observations: int = 3,
        min_train: int = 500,
        random_trials: int = 200,
        bootstrap_trials: int = 1_000,
        seed: int = 0,
    ) -> BacktestResult:
        if not corridors:
            raise ValueError("At least one corridor is required")
        requested = list(dict.fromkeys(currency.upper() for currency in corridors))
        context = [*requested]
        if "USD" not in context:
            context.append("USD")
        panel = self._rate_service.panel(context)
        if as_of is not None:
            panel = panel.loc[panel["available_on"] <= pd.Timestamp(as_of)].reset_index(drop=True)

        if score_source == "baseline":
            scores = build_baseline_scores(panel)
            scores = scores.loc[scores["iso"].isin(requested)].reset_index(drop=True)
            minimum_score = 0.5
        elif score_source == "ridge":
            scores = build_model_scores(
                panel,
                requested,
                horizon=horizon,
                min_train=min_train,
                model="ridge",
            )
            minimum_score = 0.0
        else:
            raise ValueError(f"Unsupported score source: {score_source}")

        policy = PolicyConfig(
            window=window,
            max_signals_per_window=max_signals_per_window,
            cooldown_observations=cooldown_observations,
            minimum_score=minimum_score,
        )
        config = BacktestConfig(
            horizon=horizon,
            epsilon_bps=epsilon_bps,
            random_trials=random_trials,
            bootstrap_trials=bootstrap_trials,
            seed=seed,
            policy=policy,
        )
        evaluated_panel = panel.loc[panel["iso"].isin(requested)].reset_index(drop=True)
        return run_backtest(evaluated_panel, scores, config)
