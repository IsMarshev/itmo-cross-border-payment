from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI

from signal_layer.api.app import create_app
from signal_layer.config import Settings


def _write_usd_history(data_dir: Path, *, observations: int = 61) -> None:
    data_dir.mkdir()
    dates = pd.date_range("2026-01-01", periods=observations, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "iso": "USD",
            "nominal": 1,
            "rate": [100 * (0.999**index) for index in range(observations)],
        }
    )
    frame.to_csv(data_dir / "rates_USD.csv", index=False)


def _app_with_history(tmp_path: Path, *, observations: int = 61) -> FastAPI:
    data_dir = tmp_path / "currency_data"
    _write_usd_history(data_dir, observations=observations)
    return create_app(Settings(data_dir=data_dir))


def _get(app: FastAPI, path: str, params: dict[str, str] | None = None) -> httpx.Response:
    async def make_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, params=params)

    return asyncio.run(make_request())


def test_health_reports_normalized_source_readiness(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path)

    response = _get(app, "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "observation_count": 61,
        "latest_available_on": "2026-03-03",
        "detail": None,
    }


def test_latest_rate_uses_only_quote_available_as_of_requested_date(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path, observations=2)

    response = _get(app, "/v1/rates/USD/latest", params={"as_of": "2026-01-02"})

    assert response.status_code == 200
    assert response.json() == {
        "currency": "USD",
        "quote_date": "2026-01-01",
        "available_on": "2026-01-02",
        "rub_per_unit": 100.0,
    }


def test_signal_evaluation_returns_only_a_factual_candidate(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path)

    response = _get(app, "/v1/signals/USD/evaluate", params={"as_of": "2026-03-03"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["currency"] == "USD"
    assert payload["as_of"] == date(2026, 3, 3).isoformat()
    assert payload["quote"]["quote_date"] == "2026-03-02"
    assert payload["reference_observations"] == 60
    assert payload["favourable_percentile"] == 100.0
    assert payload["strategy"] == "baseline"
    assert payload["predicted_advantage_bps"] is None
    assert payload["decision"] == "candidate"
    assert payload["message"] == (
        "Курс USD сейчас ниже, чем в 100% из последних 60 доступных наблюдений."
    )


def test_signal_evaluation_requires_history_instead_of_guessing(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path, observations=10)

    response = _get(app, "/v1/signals/USD/evaluate", params={"as_of": "2026-01-11"})

    assert response.status_code == 422
    assert "60 are required" in response.json()["detail"]


def test_signal_evaluation_can_use_the_ridge_model(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path, observations=620)
    as_of = (pd.Timestamp("2026-01-01") + pd.offsets.Day(620)).date().isoformat()

    response = _get(
        app,
        "/v1/signals/USD/evaluate",
        params={"as_of": as_of, "strategy": "ridge"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "ridge"
    assert payload["training_observations"] >= 500
    assert isinstance(payload["predicted_advantage_bps"], float)


def test_backtest_endpoint_returns_stage_four_summary(tmp_path: Path) -> None:
    app = _app_with_history(tmp_path, observations=140)

    async def post_backtest() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/backtests/run",
                json={
                    "corridors": ["USD"],
                    "score_source": "baseline",
                    "horizon": 5,
                    "window": "month",
                    "max_signals_per_window": 2,
                    "cooldown_observations": 1,
                    "random_trials": 10,
                    "bootstrap_trials": 100,
                },
            )

    response = asyncio.run(post_backtest())

    assert response.status_code == 200
    payload = response.json()
    assert payload["score_source"] == "baseline"
    assert payload["horizon"] == 5
    assert payload["decision_count"] > 0
    assert payload["signal_count"] > 0
    assert {row["iso"] for row in payload["summary"]} == {"USD", "ALL"}
    assert "random_mean_advantage_bps" in payload["summary"][0]
