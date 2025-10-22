"""FastAPI application exposing sunrise and sunset computations."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta, timezone
from typing import List, Optional

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.astro import EphemerisError, compute_sun_times, load_ephemeris
from core.ephemeris import EphemerisAcquisitionError, resolve_ephemeris_source
from models import ErrorResponse, HealthResponse, SunQueryParams, SunResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("sunrise-api")

APP_DESCRIPTION = (
    "High-precision sunrise and sunset calculations based on JPL DE ephemerides"
)

@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - exercised in integration tests
    try:
        source_path = resolve_ephemeris_source()
    except EphemerisAcquisitionError as exc:
        LOGGER.error(json.dumps({"event": "ephemeris_acquire_failed", "error": str(exc)}))
        raise
    LOGGER.info(
        json.dumps({"event": "startup", "ephemeris_source": str(source_path)})
    )
    global EPHEMERIS_FILES
    try:
        EPHEMERIS_FILES = load_ephemeris(str(source_path))
    except EphemerisError as exc:
        LOGGER.error(json.dumps({"event": "ephemeris_load_failed", "error": str(exc)}))
        raise
    yield


app = FastAPI(
    title="Riseset API",
    description=APP_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://risesetol.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EPHEMERIS_FILES: List[str] = []


def _format_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _format_local(dt: Optional[datetime], offset_hours: Optional[float]) -> Optional[str]:
    if dt is None or offset_hours is None:
        return None
    offset = timezone(timedelta(hours=offset_hours))
    return dt.astimezone(offset).isoformat()


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(code=code, error=message)
    LOGGER.error(json.dumps({"event": "error", "code": code, "message": message}))
    if hasattr(payload, "model_dump"):
        content = payload.model_dump()
    else:  # pragma: no cover - pydantic v1 fallback
        content = payload.dict()
    return JSONResponse(status_code=status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = ", ".join(error["msg"] for error in exc.errors())
    return _error_response(422, "validation_error", messages)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("error") or detail.get("message") or str(detail)
    elif isinstance(detail, list):
        message = ", ".join(str(item) for item in detail)
    else:
        message = str(detail)
    return _error_response(exc.status_code, f"http_{exc.status_code}", message)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception", exc_info=exc)
    return _error_response(500, "internal_error", "Unhandled server error")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        ephemeris_loaded=bool(EPHEMERIS_FILES),
        files=EPHEMERIS_FILES,
    )


@app.get(
    "/sun",
    response_model=SunResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def sun_endpoint(params: SunQueryParams = Depends()) -> SunResponse:
    start_time = time.perf_counter()
    try:
        result = compute_sun_times(
            date_utc=params.date_utc,
            lat=params.lat,
            lon=params.lon,
            elev_m=params.elev_m,
            twilight=params.twilight.value,
            pressure_hpa=params.pressure_hpa,
            temperature_c=params.temperature_c,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EphemerisError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response = SunResponse(
        status=result["status"],
        date_utc=params.date_utc,
        latitude=params.lat,
        longitude=params.lon,
        elevation_m=params.elev_m,
        twilight=params.twilight,
        sunrise_utc=_format_utc(result.get("sunrise")),
        sunset_utc=_format_utc(result.get("sunset")),
        offset_hours=params.offset_hours,
        sunrise_local=_format_local(result.get("sunrise"), params.offset_hours),
        sunset_local=_format_local(result.get("sunset"), params.offset_hours),
    )

    LOGGER.info(
        json.dumps(
            {
                "event": "sun",
                "lat": params.lat,
                "lon": params.lon,
                "date": params.date_utc.isoformat(),
                "twilight": params.twilight.value,
                "status": response.status,
                "duration_ms": round(duration_ms, 3),
            }
        )
    )
    return response
