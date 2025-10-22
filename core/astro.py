"""Astronomical computations for sunrise and sunset times."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import erfa
import numpy as np
import spiceypy as spice

__all__ = ["load_ephemeris", "compute_sun_times", "TWILIGHT_ANGLES"]

LOGGER = logging.getLogger(__name__)

TWILIGHT_ANGLES: Dict[str, float] = {
    "official": -0.833,
    "civil": -6.0,
    "nautical": -12.0,
    "astronomical": -18.0,
}

EARTH_EQUATORIAL_RADIUS_KM = 6378.137  # WGS84 equatorial radius in kilometers.
EARTH_EQUATORIAL_RADIUS_M = EARTH_EQUATORIAL_RADIUS_KM * 1000.0
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84 flattening.

_LOADED_FILES: Optional[List[str]] = None
_LOAD_LOCK = Lock()


class EphemerisError(RuntimeError):
    """Raised when ephemeris loading or computation fails."""


@dataclass(frozen=True)
class _TimeScales:
    """Container for time-scale representations of a UTC instant."""

    utc: Tuple[float, float]
    ut1: Tuple[float, float]
    tt: Tuple[float, float]
    et: float


def load_ephemeris(bsp_dir: str) -> List[str]:
    """Load all SPK kernels from *bsp_dir* using :mod:`spiceypy`.

    Parameters
    ----------
    bsp_dir:
        Directory containing one or more ``.bsp`` files.

    Returns
    -------
    list[str]
        Sorted list of loaded kernel file names.

    Raises
    ------
    EphemerisError
        If the directory is missing or contains no ``.bsp`` files.
    """

    global _LOADED_FILES

    if _LOADED_FILES is not None:
        return _LOADED_FILES

    path = Path(bsp_dir).expanduser()
    if not path.is_dir():
        raise EphemerisError(f"Ephemeris directory not found: {path}")

    with _LOAD_LOCK:
        if _LOADED_FILES is not None:
            return _LOADED_FILES

        bsp_files = sorted(
            file for file in path.iterdir() if file.is_file() and file.suffix.lower() == ".bsp"
        )
        if not bsp_files:
            raise EphemerisError(
                f"No .bsp ephemeris files found in directory: {path}"
            )

        loaded: List[str] = []
        try:
            for bsp_file in bsp_files:
                spice.furnsh(str(bsp_file))
                loaded.append(bsp_file.name)
        except Exception as exc:  # pragma: no cover - defensive path.
            spice.reset()
            raise EphemerisError(f"Failed to load ephemeris file '{bsp_file}': {exc}") from exc

        _LOADED_FILES = loaded
        LOGGER.info(json.dumps({"event": "ephemeris_loaded", "files": loaded}))
        return loaded


def _datetime_to_timescales(dt: datetime) -> _TimeScales:
    """Convert a timezone-aware UTC datetime into multiple time scales."""

    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    dt_utc = dt.astimezone(UTC)
    utc1, utc2 = erfa.dtf2d(
        "UTC",
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour,
        dt_utc.minute,
        dt_utc.second + dt_utc.microsecond / 1_000_000,
    )
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    ut11, ut12 = erfa.utcut1(utc1, utc2, 0.0)
    et = (tt1 - erfa.DJ00) * erfa.DAYSEC + tt2 * erfa.DAYSEC
    return _TimeScales(utc=(utc1, utc2), ut1=(ut11, ut12), tt=(tt1, tt2), et=et)


def _site_vector(lat_rad: float, lon_rad: float, elev_m: float) -> np.ndarray:
    """Return the geocentric position vector for the observer in ITRF (km)."""

    altitude_km = elev_m / 1000.0
    vector = np.array(
        spice.georec(lon_rad, lat_rad, altitude_km, EARTH_EQUATORIAL_RADIUS_KM, EARTH_FLATTENING),
        dtype=float,
    )
    return vector


def _horizon_dip_degrees(elev_m: float) -> float:
    """Approximate depression of the horizon due to observer height."""

    if elev_m <= 0:
        return 0.0
    # Small-angle approximation valid for h << R.
    dip_rad = math.sqrt(2.0 * elev_m / EARTH_EQUATORIAL_RADIUS_M)
    return math.degrees(dip_rad)


def _twilight_altitude_degrees(twilight: str, elev_m: float) -> float:
    try:
        base_altitude = TWILIGHT_ANGLES[twilight]
    except KeyError as exc:
        raise ValueError(f"Unsupported twilight selector: {twilight}") from exc
    return base_altitude - _horizon_dip_degrees(elev_m)


def _sun_altitude_degrees(
    dt: datetime,
    site_vector: np.ndarray,
    site_up: np.ndarray,
) -> float:
    """Compute the apparent altitude of the sun in degrees above the geometric horizon."""

    times = _datetime_to_timescales(dt)
    sun_vector, _ = spice.spkpos("SUN", times.et, "J2000", "LT+S", "EARTH")
    rotation = np.array(erfa.c2t06a(*times.tt, *times.ut1, 0.0, 0.0), dtype=float)
    sun_itrf = rotation @ np.array(sun_vector, dtype=float)
    topocentric = sun_itrf - site_vector
    norm = np.linalg.norm(topocentric)
    if norm == 0:
        raise EphemerisError("Degenerate topocentric vector encountered")
    altitude = math.degrees(
        math.asin(float(np.clip(np.dot(topocentric / norm, site_up), -1.0, 1.0)))
    )
    return altitude


def _refine_crossing(
    start_dt: datetime,
    end_dt: datetime,
    site_vector: np.ndarray,
    site_up: np.ndarray,
    threshold: float,
    max_iterations: int = 24,
) -> datetime:
    """Refine the crossing between *start_dt* and *end_dt* via binary search."""

    value_start = _sun_altitude_degrees(start_dt, site_vector, site_up) - threshold
    value_end = _sun_altitude_degrees(end_dt, site_vector, site_up) - threshold
    if value_start == 0:
        return start_dt
    if value_end == 0:
        return end_dt
    low_dt, low_val = start_dt, value_start
    high_dt, high_val = end_dt, value_end
    for _ in range(max_iterations):
        mid_dt = low_dt + (high_dt - low_dt) / 2
        mid_val = _sun_altitude_degrees(mid_dt, site_vector, site_up) - threshold
        if abs(mid_val) < 1e-4 or (high_dt - low_dt) <= timedelta(seconds=1):
            return mid_dt
        if low_val * mid_val <= 0:
            high_dt, high_val = mid_dt, mid_val
        else:
            low_dt, low_val = mid_dt, mid_val
    return low_dt + (high_dt - low_dt) / 2


def compute_sun_times(
    date_utc: date,
    lat: float,
    lon: float,
    elev_m: float,
    twilight: str,
) -> Dict[str, object]:
    """Compute sunrise and sunset for the given UTC date and location.

    Parameters
    ----------
    date_utc:
        Date expressed in UTC.
    lat, lon:
        Geographic coordinates in degrees (east-positive longitude).
    elev_m:
        Observer elevation above mean sea level in meters.
    twilight:
        Twilight definition key.

    Returns
    -------
    dict
        Dictionary containing ``sunrise``, ``sunset``, and ``status`` keys.
    """

    if _LOADED_FILES is None:
        raise EphemerisError("Ephemeris kernels have not been loaded")

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    threshold = _twilight_altitude_degrees(twilight, elev_m)
    site_vector = _site_vector(lat_rad, lon_rad, elev_m)
    site_up = site_vector / np.linalg.norm(site_vector)

    window_start = datetime.combine(date_utc, datetime.min.time(), tzinfo=UTC)
    window_end = window_start + timedelta(days=1)
    step = timedelta(minutes=5)

    samples: List[float] = []
    times: List[datetime] = []

    current = window_start
    while current <= window_end:
        altitude = _sun_altitude_degrees(current, site_vector, site_up)
        samples.append(altitude - threshold)
        times.append(current)
        current += step

    min_val = min(samples)
    max_val = max(samples)

    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None

    for idx in range(1, len(times)):
        prev_val, curr_val = samples[idx - 1], samples[idx]
        if sunrise is None and prev_val < 0 <= curr_val:
            sunrise = _refine_crossing(
                times[idx - 1], times[idx], site_vector, site_up, threshold
            )
        if sunset is None and prev_val >= 0 > curr_val:
            sunset = _refine_crossing(
                times[idx - 1], times[idx], site_vector, site_up, threshold
            )

    if sunrise is not None or sunset is not None:
        status = "ok"
    elif max_val < 0:
        status = "polar_night"
    elif min_val > 0:
        status = "polar_day"
    else:  # pragma: no cover - fallback for unexpected geometries.
        status = "indeterminate"

    return {"sunrise": sunrise, "sunset": sunset, "status": status}
