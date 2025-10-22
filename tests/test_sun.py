from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import erfa
import numpy as np
import pytest
import spiceypy as spice
from fastapi.testclient import TestClient

import core.astro as astro
from core.astro import compute_sun_times

AU_KM = 149597870.700
STEP_HOURS = 6


def _datetime_to_tt(dt: datetime) -> tuple[float, float]:
    utc1, utc2 = erfa.dtf2d(
        "UTC",
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond / 1_000_000,
    )
    tai1, tai2 = erfa.utctai(utc1, utc2)
    return erfa.taitt(tai1, tai2)


def _datetime_to_et(dt: datetime) -> float:
    tt1, tt2 = _datetime_to_tt(dt)
    return (tt1 - erfa.DJ00) * erfa.DAYSEC + tt2 * erfa.DAYSEC


def _sun_and_earth_states(dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    tt1, tt2 = _datetime_to_tt(dt)
    pvh, pvb = erfa.epv00(tt1, tt2)
    pos_au = -np.array(pvh[0])
    vel_au_day = -np.array(pvh[1])
    pos_km = pos_au * AU_KM
    vel_km_s = vel_au_day * (AU_KM / erfa.DAYSEC)
    sun_state = np.concatenate([pos_km, vel_km_s])

    earth_pos_au = np.array(pvb[0])
    earth_vel_au_day = np.array(pvb[1])
    earth_state = np.concatenate(
        [earth_pos_au * AU_KM, earth_vel_au_day * (AU_KM / erfa.DAYSEC)]
    )
    return sun_state, earth_state


def _generate_test_kernel(output: Path) -> None:
    if output.exists():
        return
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    step = timedelta(hours=STEP_HOURS)
    sun_states: list[np.ndarray] = []
    earth_states: list[np.ndarray] = []
    ets: list[float] = []
    current = start
    while current <= end:
        sun_state, earth_state = _sun_and_earth_states(current)
        sun_states.append(sun_state)
        earth_states.append(earth_state)
        ets.append(_datetime_to_et(current))
        current += step
    sun_states_array = np.array(sun_states, dtype=float)
    earth_states_array = np.array(earth_states, dtype=float)
    step_seconds = ets[1] - ets[0]
    handle = spice.spkopn(str(output), "SUNTEST", 0)
    try:
        spice.spkw08(
            handle,
            10,
            399,
            "J2000",
            ets[0],
            ets[-1],
            "SUNTEST",
            7,
            len(ets),
            sun_states_array,
            ets[0],
            step_seconds,
        )
        spice.spkw08(
            handle,
            399,
            0,
            "J2000",
            ets[0],
            ets[-1],
            "EARTHTEST",
            7,
            len(ets),
            earth_states_array,
            ets[0],
            step_seconds,
        )
    finally:
        spice.spkcls(handle)


@pytest.fixture(scope="session")
def kernel_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("kernels")
    kernel_path = directory / "sun_2025.bsp"
    _generate_test_kernel(kernel_path)
    return directory


@pytest.fixture(scope="session", autouse=True)
def configure_ephemeris(kernel_dir: Path) -> Iterable[None]:
    spice.kclear()
    astro._LOADED_FILES = None  # type: ignore[attr-defined]
    astro.load_ephemeris(str(kernel_dir))
    yield
    spice.kclear()


@pytest.fixture(scope="session")
def api_client(kernel_dir: Path, configure_ephemeris: Iterable[None]) -> Iterable[TestClient]:
    import os

    os.environ["DE_BSP"] = str(kernel_dir)
    from sunrise_api import app

    with TestClient(app) as client:
        yield client


def _day_length_seconds(sunrise: datetime, sunset: datetime) -> float:
    delta = (sunset - sunrise).total_seconds()
    if delta < 0:
        delta += 86400.0
    return delta


def test_beijing_sunrise_sunset():
    result = compute_sun_times(
        date_utc=date(2025, 10, 21),
        lat=39.9042,
        lon=116.4074,
        elev_m=43.5,
        twilight="official",
    )
    assert result["status"] == "ok"
    sunrise = result["sunrise"]
    sunset = result["sunset"]
    assert sunrise is not None and sunset is not None
    day_length = _day_length_seconds(sunrise, sunset)
    assert (10 * 3600 - 300) <= day_length <= (16 * 3600 + 300)


def test_polar_day_svalbard():
    result = compute_sun_times(
        date_utc=date(2025, 6, 21),
        lat=78.2232,
        lon=15.6469,
        elev_m=0.0,
        twilight="civil",
    )
    assert result["status"] == "polar_day"
    assert result["sunrise"] is None
    assert result["sunset"] is None


def test_polar_night_svalbard():
    result = compute_sun_times(
        date_utc=date(2025, 12, 21),
        lat=78.2232,
        lon=15.6469,
        elev_m=0.0,
        twilight="civil",
    )
    assert result["status"] == "polar_night"
    assert result["sunrise"] is None
    assert result["sunset"] is None


def test_validation_error(api_client: TestClient) -> None:
    response = api_client.get(
        "/sun",
        params={
            "lat": 95,  # invalid latitude
            "lon": 0,
            "date": "2025-10-21",
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["ok"] is False


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["ephemeris_loaded"] is True
    assert payload["files"]
