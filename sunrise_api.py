
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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


class SunRequest(BaseModel):
    """Parameters for a single sunrise/sunset computation."""

    lat: float = Field(description="Latitude in degrees")
    lon: float = Field(description="Longitude in degrees (east positive)")
    elev: float = Field(0.0, description="Elevation in meters")
    day: Optional[date] = Field(
        None, description="UTC date (YYYY-MM-DD). Default: today (UTC)"
    )
    pressure: float = Field(1013.25, description="Surface pressure in hPa")
    temp: float = Field(10.0, description="Air temperature in °C")
    dut1: float = Field(0.0, description="UT1-UTC seconds (IERS)")
    xp: float = Field(0.0, description="Polar motion X (arcsec)")
    yp: float = Field(0.0, description="Polar motion Y (arcsec)")
    tz_offset_hours: Optional[float] = Field(
        None, description="Manual time-zone offset in hours from UTC"
    )


class SunBatchRequest(BaseModel):
    requests: List[SunRequest]

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


class SunBatchResponse(BaseModel):
    results: List[SunResponse]


def _calculate_sunrise_sunset(params: SunRequest) -> Dict[str, object]:
    """Run the heavy sunrise/sunset calculation synchronously."""

    calc = _get_calculator()
    day = params.day or datetime.utcnow().date()
    result = calc.sunrise_sunset(
        day.year,
        day.month,
        day.day,
        params.lon,
        params.lat,
        params.elev,
        params.pressure,
        params.temp,
        params.dut1,
        params.xp,
        params.yp,
        params.tz_offset_hours,
    )
    return result


async def _calculate_async(params: SunRequest) -> Dict[str, object]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _calculate_sunrise_sunset, params)

@app.get("/sun", response_model=SunResponse)
def sun(
    lat: float = Query(..., description="Latitude in degrees"),
    lon: float = Query(..., description="Longitude in degrees (east positive)"),
    elev: float = Query(0.0, description="Elevation in meters"),
    day: Optional[date] = Query(
        None, description="UTC date (YYYY-MM-DD). Default: today (UTC)"
    ),
    pressure: float = Query(1013.25, description="Surface pressure in hPa"),
    temp: float = Query(10.0, description="Air temperature in °C"),
    dut1: float = Query(0.0, description="UT1-UTC seconds (IERS)"),
    xp: float = Query(0.0, description="Polar motion X (arcsec)"),
    yp: float = Query(0.0, description="Polar motion Y (arcsec)"),
    tz_offset_hours: Optional[float] = Query(
        None, description="Manual time-zone offset in hours from UTC"
    ),
):
    params = SunRequest(
        lat=lat,
        lon=lon,
        elev=elev,
        day=day,
        pressure=pressure,
        temp=temp,
        dut1=dut1,
        xp=xp,
        yp=yp,
        tz_offset_hours=tz_offset_hours,
    )
    return _calculate_sunrise_sunset(params)


@app.post("/sun", response_model=SunResponse)
async def sun_post(payload: SunRequest):
    return await _calculate_async(payload)


@app.post("/sun/batch", response_model=SunBatchResponse)
async def sun_batch(payload: SunBatchRequest):
    if not payload.requests:
        return {"results": []}
    results = await asyncio.gather(
        *(_calculate_async(request) for request in payload.requests)
    )
    return {"results": results}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve a lightweight HTML client for manual exploration."""

    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <title>Sunrise &amp; Sunset Calculator</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 2rem; background: #f6f8fa; color: #1f2328; }
        h1 { margin-bottom: 0.5rem; }
        p { max-width: 60rem; }
        table { border-collapse: collapse; margin-top: 1rem; width: 100%; max-width: 60rem; background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,0.15); }
        th, td { border: 1px solid #d0d7de; padding: 0.5rem; text-align: left; }
        th { background: #eef2ff; }
        input[type='number'], input[type='text'] { width: 100%; box-sizing: border-box; padding: 0.25rem 0.35rem; }
        input[type='date'] { width: 100%; box-sizing: border-box; padding: 0.2rem 0.35rem; }
        button { padding: 0.6rem 1.2rem; margin-top: 1rem; border: none; border-radius: 6px; background: #0969da; color: #fff; font-size: 1rem; cursor: pointer; }
        button.secondary { background: #6e7781; margin-left: 0.5rem; }
        button:disabled { background: #8c959f; cursor: wait; }
        pre { background: #fff; border: 1px solid #d0d7de; padding: 1rem; max-width: 60rem; overflow: auto; box-shadow: inset 0 1px 2px rgba(15,23,42,0.08); }
        .controls { margin-top: 0.5rem; }
    </style>
</head>
<body>
    <h1>Sunrise &amp; Sunset Calculator</h1>
    <p>Fill in one or more rows with the geographic and atmospheric parameters. Use the “Add row” button to compute multiple locations in parallel. All computations are performed via the FastAPI backend exposed by this service.</p>
    <form id=\"task-form\">
        <table>
            <thead>
                <tr>
                    <th>Latitude (°)</th>
                    <th>Longitude (°)</th>
                    <th>Elevation (m)</th>
                    <th>Date (UTC)</th>
                    <th>Pressure (hPa)</th>
                    <th>Temperature (°C)</th>
                    <th>DUT1 (s)</th>
                    <th>Polar X (arcsec)</th>
                    <th>Polar Y (arcsec)</th>
                    <th>TZ offset (h)</th>
                    <th>Remove</th>
                </tr>
            </thead>
            <tbody id=\"task-rows\"></tbody>
        </table>
        <div class=\"controls\">
            <button type=\"button\" id=\"add-row\">Add row</button>
            <button type=\"submit\" id=\"submit-btn\">Compute</button>
        </div>
    </form>
    <h2>Results</h2>
    <pre id=\"results\">Results will appear here…</pre>
    <script>
        const defaults = {
            lat: 39.9042,
            lon: 116.4074,
            elev: 0,
            pressure: 1013.25,
            temp: 10,
            dut1: 0,
            xp: 0,
            yp: 0
        };

        const rowsContainer = document.getElementById('task-rows');
        const addRowButton = document.getElementById('add-row');
        const submitBtn = document.getElementById('submit-btn');
        const resultsPre = document.getElementById('results');

        function createInput(type, name, step, value = '') {
            const input = document.createElement('input');
            input.type = type;
            input.name = name;
            if (step) input.step = step;
            if (value !== '') input.value = value;
            return input;
        }

        function addRow(data = {}) {
            const tr = document.createElement('tr');
            const fields = [
                { name: 'lat', type: 'number', step: '0.0001', value: data.lat ?? defaults.lat },
                { name: 'lon', type: 'number', step: '0.0001', value: data.lon ?? defaults.lon },
                { name: 'elev', type: 'number', step: '1', value: data.elev ?? defaults.elev },
                { name: 'day', type: 'date', step: null, value: data.day ?? '' },
                { name: 'pressure', type: 'number', step: '0.01', value: data.pressure ?? defaults.pressure },
                { name: 'temp', type: 'number', step: '0.01', value: data.temp ?? defaults.temp },
                { name: 'dut1', type: 'number', step: '0.0001', value: data.dut1 ?? defaults.dut1 },
                { name: 'xp', type: 'number', step: '0.0001', value: data.xp ?? defaults.xp },
                { name: 'yp', type: 'number', step: '0.0001', value: data.yp ?? defaults.yp },
                { name: 'tz_offset_hours', type: 'number', step: '0.1', value: data.tz_offset_hours ?? '' }
            ];

            fields.forEach(field => {
                const td = document.createElement('td');
                td.appendChild(createInput(field.type, field.name, field.step, field.value));
                tr.appendChild(td);
            });

            const removeTd = document.createElement('td');
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.textContent = '✕';
            removeBtn.className = 'secondary';
            removeBtn.addEventListener('click', () => {
                if (rowsContainer.children.length > 1) {
                    tr.remove();
                }
            });
            removeTd.appendChild(removeBtn);
            tr.appendChild(removeTd);
            rowsContainer.appendChild(tr);
        }

        function readRow(tr) {
            const data = {};
            for (const input of tr.querySelectorAll('input')) {
                const name = input.name;
                const raw = input.value.trim();
                if (!raw) continue;
                if (name === 'day') {
                    data.day = raw;
                    continue;
                }
                const num = Number(raw);
                if (!Number.isFinite(num)) {
                    throw new Error(`Invalid numeric value for ${name}`);
                }
                data[name] = num;
            }
            if (data.lat === undefined || data.lon === undefined) {
                throw new Error('Latitude and longitude are required for each row.');
            }
            return data;
        }

        addRow();

        addRowButton.addEventListener('click', () => addRow());

        document.getElementById('task-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            try {
                const requests = Array.from(rowsContainer.children).map(readRow);
                if (requests.length === 0) {
                    throw new Error('Please add at least one computation row.');
                }
                submitBtn.disabled = true;
                resultsPre.textContent = 'Computing…';
                const response = await fetch('/sun/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ requests })
                });
                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`API error: ${response.status} ${text}`);
                }
                const data = await response.json();
                resultsPre.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                resultsPre.textContent = error.message;
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("sunrise_api:app", host="0.0.0.0", port=7000)
