"""HTTP response schemas; domain services remain framework-independent."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


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
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    as_of: date
    quote: RateQuoteResponse
    reference_observations: int = Field(gt=0)
    favourable_percentile: float = Field(ge=0, le=100)
    decision: Literal["candidate", "hold"]
    reason: str
    message: str | None = None
