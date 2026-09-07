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


def test_signal_evaluation_holds_when_the_layer_has_no_case(tmp_path: Path) -> None:
    """Short history cannot calibrate a window, so the layer must hold, not guess."""
    app = _app_with_history(tmp_path)

    response = _get(app, "/v1/signals/USD/evaluate", params={"as_of": "2026-03-03"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["currency"] == "USD"
    assert payload["as_of"] == date(2026, 3, 3).isoformat()
    assert payload["quote"]["quote_date"] == "2026-03-02"
    assert payload["indicator"] == "zscore_tuned"
    assert payload["decision"] == "hold"
    assert payload["message"] is None


def test_signal_evaluation_carries_the_briefs_signal_table_fields(tmp_path: Path) -> None:
    """When the layer does fire, the response is the brief's signal row."""
    app = _app_with_history(tmp_path, observations=900)
    as_of = (pd.Timestamp("2026-01-01") + pd.offsets.Day(899)).date().isoformat()

    response = _get(app, "/v1/signals/USD/evaluate", params={"as_of": as_of})

    assert response.status_code == 200
    payload = response.json()
    assert payload["indicator"] == "zscore_tuned"
    assert payload["decision"] in {"candidate", "hold"}
    if payload["decision"] == "candidate":
        assert payload["direction"] in {"down", "up"}
        assert payload["speed"] in {"fast", "medium", "slow"}
        assert payload["scenario"] == "favourable_now"
        assert payload["window"].startswith("span=")
        assert 0.0 <= payload["strength_pct"] <= 1.0
        # A push may only claim a fact that holds, so a fired signal always
        # sits below the trend its own window measured.
        assert payload["deviation_pct"] < 0
        assert payload["message"]


def test_signal_evaluation_never_offers_a_strategy_switch(tmp_path: Path) -> None:
    """The benchmark picks what ships; the endpoint must not reopen that."""
    app = _app_with_history(tmp_path)

    response = _get(
        app, "/v1/signals/USD/evaluate", params={"as_of": "2026-03-03", "strategy": "ridge"}
    )

    # An unknown query parameter is ignored rather than selecting anything.
    assert response.status_code == 200
    assert "strategy" not in response.json()


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
