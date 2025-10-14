
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from datetime import date
import os
from functools import lru_cache

# Import from the patched module placed alongside this file or adjust PYTHONPATH.
from lunar_calendar6_sunrise import DE441Reader, SunriseSunsetCalculator


@lru_cache(maxsize=1)
def _get_calculator() -> SunriseSunsetCalculator:
    de_bsp = os.environ.get("DE_BSP")
    if not de_bsp:
        raise RuntimeError(
            "环境变量 DE_BSP 未设置：请提供一个或多个 DE440/DE441 SPK 路径，"
            "可使用操作系统路径分隔符连接，或指定包含 .bsp 文件的目录。"
        )
    eph = DE441Reader(de_bsp)
    return SunriseSunsetCalculator(eph)


app = FastAPI(title="Sunrise/Sunset DE440/DE441 API")

class SunResponse(BaseModel):
    sunrise_utc: Optional[str]
    sunrise_local: Optional[str]
    sunset_utc: Optional[str]
    sunset_local: Optional[str]
    civil_dawn_utc: Optional[str]
    civil_dawn_local: Optional[str]
    civil_dusk_utc: Optional[str]
    civil_dusk_local: Optional[str]
    nautical_dawn_utc: Optional[str]
    nautical_dawn_local: Optional[str]
    nautical_dusk_utc: Optional[str]
    nautical_dusk_local: Optional[str]
    astronomical_dawn_utc: Optional[str]
    astronomical_dawn_local: Optional[str]
    astronomical_dusk_utc: Optional[str]
    astronomical_dusk_local: Optional[str]
    status: str
    tz_offset_hours: float

@app.get("/sun", response_model=SunResponse)
def sun(
    lat: float = Query(..., description="Latitude in degrees"),
    lon: float = Query(..., description="Longitude in degrees (east positive)"),
    elev: float = Query(0.0, description="Elevation in meters"),
    day: Optional[date] = Query(None, description="UTC date (YYYY-MM-DD). Default: today (UTC)"),
    pressure: float = Query(1013.25, description="Surface pressure in hPa"),
    temp: float = Query(10.0, description="Air temperature in °C"),
    dut1: float = Query(0.0, description="UT1-UTC seconds (IERS)"),
    xp: float = Query(0.0, description="Polar motion X (arcsec)"),
    yp: float = Query(0.0, description="Polar motion Y (arcsec)"),
):
    from datetime import datetime, timezone
    if day is None:
        today = datetime.utcnow().date()
    else:
        today = day
    calc = _get_calculator()
    res = calc.sunrise_sunset(today.year, today.month, today.day, lon, lat, elev, pressure, temp, dut1, xp, yp)
    return res
