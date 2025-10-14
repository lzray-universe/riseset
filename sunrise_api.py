
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from datetime import date
import os

# Import from the patched module placed alongside this file or adjust PYTHONPATH.
from lunar_calendar6_sunrise import DE441Reader, SunriseSunsetCalculator

DE_BSP = os.environ.get("DE_BSP", "/path/to/de441.bsp")
eph = DE441Reader(DE_BSP)
calc = SunriseSunsetCalculator(eph)

app = FastAPI(title="Sunrise/Sunset DE440/DE441 API")

class SunResponse(BaseModel):
    sunrise_utc: Optional[str]
    sunset_utc: Optional[str]
    status: str

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
    res = calc.sunrise_sunset(today.year, today.month, today.day, lon, lat, elev, pressure, temp, dut1, xp, yp)
    return res
