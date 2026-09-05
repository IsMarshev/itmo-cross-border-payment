"""HTTP backend for the simulation stand.

Thin on purpose. Every rule the stand demonstrates — which day earns a push,
what that push may claim, and whether the claim still holds when the client
finally opens it — lives in :mod:`signal_layer.services.simulation`. This module
resolves a corridor and a date to that service's answer and serves the page.

    uv run python demo/sim/server.py       # → http://127.0.0.1:8100
    docker compose up --build              # то же самое в контейнере

The static case stand rides along on ``/stand/`` so one process serves both.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi import Path as PathParam
from fastapi.staticfiles import StaticFiles

from signal_layer.config import Settings
from signal_layer.services import RateService, SimulationService

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_DIR = Path(__file__).resolve().parents[1]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the stand's app, loading the panel and the walk-forward run once."""
    app = FastAPI(
        title="Signal layer · simulation stand",
        version="0.1.0",
        description="Calendar-day simulation of the signal layer and the push it sends.",
    )
    resolved = settings or Settings.from_environment()
    rate_service = RateService(resolved.data_dir)
    # The chronological run is computed here, at startup, so a day step is a
    # lookup rather than half a second of walk-forward tuning.
    app.state.simulation = SimulationService(rate_service)

    @app.get("/api/corridors", tags=["simulation"])
    def corridors(request: Request) -> list[dict]:
        """Every corridor the stand can play, with its own disclosure threshold."""
        return [
            {
                "iso": meta.iso,
                "name": meta.name,
                "country": meta.country,
                "first_date": meta.first_date.isoformat(),
                "last_date": meta.last_date.isoformat(),
                "sim_start": meta.sim_start.isoformat(),
                "threshold_bps": meta.threshold_bps,
                "trend_span": meta.trend_span,
            }
            for meta in request.app.state.simulation.corridors()
        ]

    @app.get("/api/series/{iso}", tags=["simulation"])
    def series(
        request: Request, iso: str = PathParam(pattern=r"^[A-Za-z]{3}$")
    ) -> dict:
        """The drawn window in full; the page reveals it as the playhead moves."""
        simulation = request.app.state.simulation
        return {"iso": iso.upper(), "series": _or_404(simulation.series, iso)}

    @app.get("/api/day/{iso}", tags=["simulation"])
    def day(
        request: Request,
        iso: str = PathParam(pattern=r"^[A-Za-z]{3}$"),
        on: date = Query(description="The simulated calendar day."),
    ) -> dict:
        """One step of the simulation: is today worth a push in this corridor?"""
        simulation = request.app.state.simulation
        decision = _or_404(simulation.day, iso, on)
        return {
            "iso": decision.iso,
            "day": decision.day.isoformat(),
            "has_quote": decision.has_quote,
            "rate": decision.rate,
            "decision": decision.decision,
            "signal": decision.signal,
        }

    @app.get("/api/freshness/{iso}", tags=["simulation"])
    def freshness(
        request: Request,
        iso: str = PathParam(pattern=r"^[A-Za-z]{3}$"),
        signal_date: date = Query(description="The day the push was sent."),
        as_of: date = Query(description="The day the client opened it."),
    ) -> dict:
        """Whether the push's claim still holds, and how loudly to say it does not."""
        simulation = request.app.state.simulation
        result = _or_404(simulation.freshness, iso, signal_date, as_of)
        return {
            "level": result.level,
            "delta_bps": result.delta_bps,
            "threshold_bps": result.threshold_bps,
            "signal_date": result.signal_date.isoformat(),
            "signal_rate": result.signal_rate,
            "current_date": result.current_date.isoformat(),
            "current_rate": result.current_rate,
        }

    # The static case stand rides along so one container serves both demos.
    # Mounted before the root, because the root mount swallows every path.
    app.mount("/stand", StaticFiles(directory=DEMO_DIR, html=True), name="case-stand")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="simulation")
    return app


def _or_404(call, *args):
    """Turn a corridor or date the service does not know into a 404."""
    try:
        return call(*args)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error


app = create_app()


if __name__ == "__main__":
    import os

    import uvicorn

    # Loopback locally, every interface inside a container: a bind to 127.0.0.1
    # there is reachable only from within the container itself.
    uvicorn.run(
        app,
        host=os.getenv("SIM_HOST", "127.0.0.1"),
        port=int(os.getenv("SIM_PORT", "8100")),
        log_level="warning",
    )
