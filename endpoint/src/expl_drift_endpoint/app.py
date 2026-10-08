"""HTTP service for remote Laya score extraction."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from expl_drift_endpoint.contract import ExtractionRequest, ExtractionResponse, Readiness


def _load_adapter() -> Any:
    """Load optional GPU dependencies only in the deployed model service."""
    from expl_drift_endpoint.laya_adapter import LayaAdapter

    return LayaAdapter.from_environment()


def create_app(adapter_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Create the service; tests can supply an adapter without model dependencies."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.adapter = adapter_factory() if adapter_factory is not None else None
        yield

    application = FastAPI(
        title="expl_drift extraction endpoint",
        version="0.2.0",
        lifespan=lifespan,
    )

    @application.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health", response_model=Readiness)
    def health() -> Readiness | JSONResponse:
        adapter = application.state.adapter
        if adapter is None:
            return JSONResponse(
                status_code=503,
                content=Readiness().model_dump(mode="json"),
            )
        return Readiness(
            status="ready",
            reason=None,
            extraction_available=True,
            capabilities=["decision_scores"],
            model=adapter.model_revision,
        )

    @application.post("/extract", response_model=ExtractionResponse)
    def extract(request: ExtractionRequest) -> ExtractionResponse:
        adapter = application.state.adapter
        if adapter is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "request_id": request.request_id,
                    "reason": Readiness().reason,
                },
            )
        try:
            return adapter.extract(request)
        except Exception as error:
            from expl_drift_endpoint.laya_adapter import (
                IncompatibleRevision,
                InvalidModelOutput,
                UnsupportedCapability,
            )

            if isinstance(error, IncompatibleRevision):
                raise HTTPException(status_code=409, detail=str(error)) from error
            if isinstance(error, UnsupportedCapability):
                raise HTTPException(status_code=501, detail=str(error)) from error
            if isinstance(error, InvalidModelOutput):
                raise HTTPException(status_code=502, detail=str(error)) from error
            raise

    return application


app = create_app()
model_app = create_app(adapter_factory=_load_adapter)
