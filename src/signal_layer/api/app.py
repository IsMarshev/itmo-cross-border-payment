"""FastAPI application factory for the minimal signal-layer backend."""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse

from signal_layer.api.schemas import HealthResponse, RateQuoteResponse, SignalEvaluationResponse
from signal_layer.config import Settings
from signal_layer.services import (
    InsufficientHistoryError,
    RateDataUnavailableError,
    RateNotFoundError,
    RateService,
    SignalService,
)
from signal_layer.services.rates import RateQuote


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an app that can be configured independently in tests and deployment."""
    app = FastAPI(
        title="Cross-border payment signal layer",
        version="0.1.0",
        description="As-of exchange-rate facts and deterministic signal candidates.",
    )
    resolved_settings = settings or Settings.from_environment()
    rate_service = RateService(resolved_settings.data_dir)
    app.state.rate_service = rate_service
    app.state.signal_service = SignalService(rate_service)

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse | JSONResponse:
        readiness = request.app.state.rate_service.readiness()
        response = HealthResponse(
            status="ok" if readiness.ready else "degraded",
            observation_count=readiness.observation_count,
            latest_available_on=readiness.latest_available_on,
            detail=readiness.detail,
        )
        if readiness.ready:
            return response
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    @app.get("/v1/rates/{currency}/latest", response_model=RateQuoteResponse, tags=["rates"])
    def latest_rate(
        request: Request,
        currency: str = Path(pattern=r"^[A-Za-z]{3}$"),
        as_of: date = Query(description="Decision date; future quotes are excluded."),
    ) -> RateQuoteResponse:
        quote = _latest_quote_or_http_error(request.app.state.rate_service, currency, as_of)
        return _rate_quote_response(quote)

    @app.get(
        "/v1/signals/{currency}/evaluate",
        response_model=SignalEvaluationResponse,
        tags=["signals"],
    )
    def evaluate_signal(
        request: Request,
        currency: str = Path(pattern=r"^[A-Za-z]{3}$"),
        as_of: date = Query(description="Decision date; future quotes are excluded."),
    ) -> SignalEvaluationResponse:
        try:
            evaluation = request.app.state.signal_service.evaluate(currency, as_of)
        except RateNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        except InsufficientHistoryError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        except RateDataUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error

        return SignalEvaluationResponse(
            currency=evaluation.currency,
            as_of=evaluation.as_of,
            quote=_rate_quote_response(evaluation.quote),
            reference_observations=evaluation.reference_observations,
            favourable_percentile=evaluation.favourable_percentile,
            decision=evaluation.decision,
            reason=evaluation.reason,
            message=evaluation.message,
        )

    return app


def _latest_quote_or_http_error(
    rate_service: RateService, currency: str, as_of: date
) -> RateQuote:
    try:
        return rate_service.latest_quote(currency, as_of)
    except RateNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RateDataUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def _rate_quote_response(quote: RateQuote) -> RateQuoteResponse:
    return RateQuoteResponse(
        currency=quote.currency,
        quote_date=quote.quote_date,
        available_on=quote.available_on,
        rub_per_unit=quote.rub_per_unit,
    )


app = create_app()
