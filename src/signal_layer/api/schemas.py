"""HTTP response schemas; domain services remain framework-independent."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    observation_count: int = Field(ge=0)
    latest_available_on: date | None = None
    detail: str | None = None


class RateQuoteResponse(BaseModel):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    quote_date: date
    available_on: date
    rub_per_unit: float = Field(gt=0)


class SignalEvaluationResponse(BaseModel):
    """One corridor's answer for one date, as the live layer produced it.

    The fields after ``decision`` carry the brief's signal-table contract:
    indicator, direction, strength, indicator speed and recommended scenario.
    They are absent on a hold, where there is nothing to describe.
    """

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    as_of: date
    quote: RateQuoteResponse
    indicator: str
    decision: Literal["candidate", "hold"]
    reason: str
    message: str | None = None
    direction: Literal["down", "up"] | None = None
    speed: Literal["fast", "medium", "slow", "unknown"] | None = None
    scenario: Literal["favourable_now", "window_closing"] | None = None
    window: str | None = None
    strength: float | None = None
    strength_pct: float | None = Field(default=None, ge=0, le=1)
    deviation_pct: float | None = None
    level_percentile: float | None = Field(default=None, ge=0, le=100)


class BacktestRequest(BaseModel):
    corridors: list[str] = Field(min_length=1, max_length=5)
    score_source: Literal["baseline", "ridge"] = "baseline"
    as_of: date | None = None
    horizon: int = Field(default=20, ge=1, le=60)
    epsilon_bps: float = Field(default=30.0, ge=0, le=1_000)
    window: Literal["week", "month"] = "week"
    max_signals_per_window: int = Field(default=2, ge=1, le=10)
    cooldown_observations: int = Field(default=3, ge=0, le=60)
    min_train: int = Field(default=500, ge=50)
    random_trials: int = Field(default=200, ge=10, le=1_000)
    bootstrap_trials: int = Field(default=1_000, ge=100, le=5_000)
    seed: int = 0

    @field_validator("corridors")
    @classmethod
    def validate_corridors(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().upper() for value in values]
        if any(len(value) != 3 or not value.isalpha() for value in normalized):
            raise ValueError("corridors must contain three-letter ISO codes")
        if len(set(normalized)) != len(normalized):
            raise ValueError("corridors must not contain duplicates")
        return normalized


class BacktestSummaryResponse(BaseModel):
    iso: str
    n_signals: int = Field(ge=0)
    per_week: float | None
    series_share: float | None
    mean_advantage_bps: float | None
    median_advantage_bps: float | None
    p10_advantage_bps: float | None
    hit_rate: float | None
    negative_share: float | None
    early_send_rate: float | None
    p90_regret_bps: float | None
    advantage_ci_low: float | None
    advantage_ci_high: float | None
    random_mean_advantage_bps: float | None
    random_hit_rate: float | None
    advantage_delta_bps: float | None
    advantage_lift: float | None
    hit_rate_lift: float | None


class BacktestResponse(BaseModel):
    score_source: Literal["baseline", "ridge"]
    horizon: int
    decision_count: int = Field(ge=0)
    signal_count: int = Field(ge=0)
    summary: list[BacktestSummaryResponse]
