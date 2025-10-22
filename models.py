"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Literal, Optional

try:  # pragma: no cover - exercised indirectly in tests
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback for environments with pydantic v1
    from pydantic import BaseModel, Field, validator  # type: ignore

    _PYDANTIC_V2 = False

    def field_validator(field_name: str, *args, **kwargs):  # type: ignore[override]
        return validator(field_name, *args, **kwargs)


class Twilight(str, Enum):
    """Enumeration of supported twilight definitions."""

    official = "official"
    civil = "civil"
    nautical = "nautical"
    astronomical = "astronomical"


class SunQueryParams(BaseModel):
    """Validated query parameters for the ``/sun`` endpoint."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True)  # type: ignore[assignment]
    else:  # pragma: no cover - pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    date_utc: date = Field(..., alias="date", description="UTC calendar date (YYYY-MM-DD)")
    elev_m: float = Field(0.0, ge=-500.0, description="Observer elevation in meters")
    pressure_hpa: float = Field(
        1013.25,
        ge=300.0,
        le=1100.0,
        description="Surface atmospheric pressure in hectopascals",
    )
    temperature_c: float = Field(
        10.0,
        ge=-80.0,
        le=60.0,
        description="Surface air temperature in degrees Celsius",
    )
    offset_hours: Optional[float] = Field(
        None,
        description="Optional fixed offset in hours applied to derive local times",
    )
    twilight: Twilight = Field(Twilight.official, description="Twilight definition")

    @field_validator("offset_hours")
    def validate_offset_hours(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not -24.0 <= value <= 24.0:
            raise ValueError("offset_hours must be within ±24 hours")
        return value


class SunResponse(BaseModel):
    """Successful sunrise/sunset response payload."""

    ok: bool = True
    status: str = Field(..., description="Computation status")
    date_utc: date = Field(..., description="Requested UTC date")
    latitude: float = Field(..., description="Latitude in degrees")
    longitude: float = Field(..., description="Longitude in degrees")
    elevation_m: float = Field(..., description="Elevation above mean sea level")
    twilight: Twilight = Field(..., description="Applied twilight definition")
    sunrise_utc: Optional[str] = Field(
        None, description="Sunrise time in UTC (ISO-8601)"
    )
    sunset_utc: Optional[str] = Field(
        None, description="Sunset time in UTC (ISO-8601)"
    )
    offset_hours: Optional[float] = Field(
        None, description="User-specified offset in hours"
    )
    sunrise_local: Optional[str] = Field(
        None, description="Sunrise expressed in local time when offset provided"
    )
    sunset_local: Optional[str] = Field(
        None, description="Sunset expressed in local time when offset provided"
    )
    source: Literal["CSPICE-DE"] = Field(
        "CSPICE-DE", description="Ephemeris source identifier"
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    ok: bool = True
    ephemeris_loaded: bool
    files: List[str]


class ErrorResponse(BaseModel):
    """Error payload."""

    ok: bool = False
    code: str
    error: str
