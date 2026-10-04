"""HTTP scaffold; no model calls or drift calculations run in this increment."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from expl_drift_endpoint.contract import ExtractionRequest, Readiness


def create_app() -> FastAPI:
    """Expose transport liveness, honest readiness, and validated input."""
    application = FastAPI(title="expl_drift extraction endpoint", version="0.1.0")

    @application.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health", response_model=Readiness, status_code=503)
    def health() -> Readiness:
        return Readiness()

    @application.post("/extract", response_class=JSONResponse, status_code=503)
    def extract(request: ExtractionRequest) -> JSONResponse:
        # Validation is active; model execution deliberately remains unavailable.
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "request_id": request.request_id,
                "reason": Readiness().reason,
            },
        )

    return application


app = create_app()
