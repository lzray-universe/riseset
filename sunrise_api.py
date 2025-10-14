
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
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Sunrise &amp; Sunset Calculator</title>
    <style>
        :root {
            color-scheme: light;
        }
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f6f8fa; color: #1f2328; }
        h1 { margin-bottom: 0.25rem; }
        p.description { max-width: 60rem; margin-top: 0; }
        table { border-collapse: collapse; width: 100%; min-width: 60rem; background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,0.15); }
        th, td { border: 1px solid #d0d7de; padding: 0.55rem 0.5rem; text-align: left; white-space: nowrap; }
        th { background: #dbeafe; position: sticky; top: 0; z-index: 1; }
        input[type='number'], input[type='text'], input[type='date'] { width: 100%; box-sizing: border-box; padding: 0.35rem 0.4rem; border: 1px solid #afb8c1; border-radius: 4px; background: #fff; }
        input[type='number']:focus, input[type='text']:focus, input[type='date']:focus { outline: 2px solid #1f6feb33; border-color: #1f6feb; }
        button { padding: 0.55rem 1.2rem; border: none; border-radius: 6px; background: #0969da; color: #fff; font-size: 0.95rem; cursor: pointer; transition: background 0.2s ease; }
        button.secondary { background: #6e7781; }
        button.neutral { background: #4f5d75; }
        button:disabled { background: #8c959f; cursor: wait; }
        button + button { margin-left: 0.5rem; }
        form { margin-top: 1.5rem; }
        .controls { margin-top: 0.75rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; }
        .range-controls { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
        .range-controls label { font-weight: 600; display: flex; align-items: center; gap: 0.35rem; }
        .status-strip { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-top: 0.75rem; background: #fff; border: 1px solid #d0d7de; padding: 0.75rem 1rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,0.08); }
        .status-strip strong { color: #0969da; }
        .table-wrapper { overflow-x: auto; max-width: 100%; border-radius: 10px; border: 1px solid #d0d7de; background: #fff; box-shadow: 0 2px 4px rgba(15,23,42,0.12); }
        .view-controls { margin-top: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }
        .view-controls button[data-view].active { background: #0d6efd; }
        .view-controls button[data-view] { background: #6c757d; }
        .view-controls .spacer { flex: 1 1 auto; }
        .hidden { display: none !important; }
        #json-view { background: #0d1117; color: #f0f6fc; padding: 1rem; border-radius: 10px; border: 1px solid #30363d; max-height: 32rem; overflow: auto; box-shadow: inset 0 1px 2px rgba(15,23,42,0.2); }
        #results-message { margin-top: 1rem; font-weight: 600; color: #1f2328; }
        canvas#sun-track { display: block; margin-top: 1.5rem; width: 100%; max-width: 720px; height: auto; border: 1px solid #d0d7de; border-radius: 10px; background: linear-gradient(180deg, #fef3c7 0%, #dbeafe 100%); }
        .sun-track-note { font-size: 0.9rem; color: #57606a; margin-top: 0.5rem; max-width: 50rem; }
        @media (max-width: 960px) {
            body { margin: 1.25rem; }
            table { min-width: 48rem; }
        }
    </style>
</head>
<body>
    <h1>Sunrise &amp; Sunset Calculator</h1>
    <p class="description">自动填入位置后即可快速计算。支持时间段批量查询、表格/JSON 切换、CSV 导出和太阳轨迹可视化。</p>

    <div class="status-strip">
        <button type="button" id="locate-btn">📍 使用当前位置</button>
        <div id="location-status">尚未获取定位。</div>
        <div id="time-status"></div>
    </div>

    <form id="task-form">
        <div class="table-wrapper">
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
                <tbody id="task-rows"></tbody>
            </table>
        </div>
        <div class="controls">
            <div class="range-controls">
                <label>起始日期(UTC)<input type="date" id="range-start"></label>
                <label>结束日期(UTC)<input type="date" id="range-end"></label>
            </div>
            <div class="controls-buttons">
                <button type="button" id="add-row" class="neutral">添加行</button>
                <button type="submit" id="submit-btn">计算</button>
            </div>
        </div>
    </form>

    <section id="results-section" class="hidden">
        <div class="view-controls">
            <button type="button" data-view="table" class="active">表格视图</button>
            <button type="button" data-view="json">JSON 视图</button>
            <span class="spacer"></span>
            <button type="button" id="export-csv" class="secondary" disabled>导出 CSV</button>
        </div>
        <div id="results-message"></div>
        <div id="table-view">
            <div class="table-wrapper">
                <table id="results-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>日期</th>
                            <th>纬度</th>
                            <th>经度</th>
                            <th>Sunrise (Local)</th>
                            <th>Sunset (Local)</th>
                            <th>Civil Dawn</th>
                            <th>Civil Dusk</th>
                            <th>Nautical Dawn</th>
                            <th>Nautical Dusk</th>
                            <th>Astronomical Dawn</th>
                            <th>Astronomical Dusk</th>
                            <th>状态</th>
                            <th>TZ Offset (h)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        <pre id="json-view" class="hidden"></pre>
        <canvas id="sun-track" width="720" height="260"></canvas>
        <p class="sun-track-note">轨迹基于第一条结果，展示本地日照进度。若太阳极昼/极夜或缺少数据，则不显示定位点。</p>
    </section>

    <script>
        const defaults = {
            lat: 39.9042,
            lon: 116.4074,
            elev: 0,
            pressure: 1013.25,
            temp: 10,
            dut1: 0,
            xp: 0,
            yp: 0,
            tz_offset_hours: Number((-(new Date().getTimezoneOffset()) / 60).toFixed(2))
        };

        const rowsContainer = document.getElementById('task-rows');
        const addRowButton = document.getElementById('add-row');
        const submitBtn = document.getElementById('submit-btn');
        const rangeStartInput = document.getElementById('range-start');
        const rangeEndInput = document.getElementById('range-end');
        const locateBtn = document.getElementById('locate-btn');
        const locationStatus = document.getElementById('location-status');
        const timeStatus = document.getElementById('time-status');
        const resultsSection = document.getElementById('results-section');
        const resultsMessage = document.getElementById('results-message');
        const viewButtons = Array.from(document.querySelectorAll('[data-view]'));
        const tableView = document.getElementById('table-view');
        const jsonView = document.getElementById('json-view');
        const exportCsvBtn = document.getElementById('export-csv');
        const sunTrackCanvas = document.getElementById('sun-track');
        const tableBody = document.querySelector('#results-table tbody');

        let latestEntries = [];

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
                { name: 'tz_offset_hours', type: 'number', step: '0.01', value: data.tz_offset_hours ?? defaults.tz_offset_hours }
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

        function parseNumber(value, name) {
            const num = Number(value);
            if (!Number.isFinite(num)) {
                throw new Error(`Invalid numeric value for ${name}`);
            }
            return num;
        }

        function readRow(tr) {
            const data = {};
            for (const input of tr.querySelectorAll('input')) {
                const name = input.name;
                const raw = input.value.trim();
                if (name === 'day') {
                    if (raw) data.day = raw;
                    continue;
                }
                if (name === 'tz_offset_hours') {
                    const effective = raw === '' ? defaults.tz_offset_hours : parseNumber(raw, name);
                    data[name] = effective;
                    continue;
                }
                if (!raw) continue;
                data[name] = parseNumber(raw, name);
            }
            if (data.lat === undefined || data.lon === undefined) {
                throw new Error('Latitude and longitude are required for each row.');
            }
            if (data.tz_offset_hours === undefined) {
                data.tz_offset_hours = defaults.tz_offset_hours;
            }
            return data;
        }

        function updateTimeStatus() {
            const now = new Date();
            const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
            timeStatus.textContent = `本地时间：${now.toLocaleString()} (${tz})`;
        }

        updateTimeStatus();
        setInterval(updateTimeStatus, 1000);

        function setView(view) {
            if (view === 'json') {
                tableView.classList.add('hidden');
                jsonView.classList.remove('hidden');
            } else {
                tableView.classList.remove('hidden');
                jsonView.classList.add('hidden');
            }
            viewButtons.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === view);
            });
        }

        viewButtons.forEach(btn => {
            btn.addEventListener('click', () => setView(btn.dataset.view));
        });

        function ensureRowExists() {
            if (rowsContainer.children.length === 0) {
                addRow();
            }
        }

        locateBtn.addEventListener('click', () => {
            if (!navigator.geolocation) {
                locationStatus.textContent = '浏览器不支持定位。';
                return;
            }
            locationStatus.textContent = '正在获取定位…';
            navigator.geolocation.getCurrentPosition(position => {
                const { latitude, longitude } = position.coords;
                defaults.lat = Number(latitude.toFixed(4));
                defaults.lon = Number(longitude.toFixed(4));
                locationStatus.innerHTML = `当前位置：<strong>${defaults.lat}</strong>°, <strong>${defaults.lon}</strong>°`;
                ensureRowExists();
                const firstRow = rowsContainer.children[0];
                firstRow.querySelector('input[name="lat"]').value = defaults.lat;
                firstRow.querySelector('input[name="lon"]').value = defaults.lon;
                const tz = Number((-(new Date().getTimezoneOffset()) / 60).toFixed(2));
                defaults.tz_offset_hours = tz;
                for (const row of rowsContainer.children) {
                    const tzInput = row.querySelector('input[name="tz_offset_hours"]');
                    if (tzInput) tzInput.value = tz;
                }
            }, error => {
                locationStatus.textContent = `定位失败：${error.message}`;
            }, { enableHighAccuracy: true, timeout: 10000 });
        });

        function expandDateRange(startStr, endStr) {
            if (!startStr && !endStr) return null;
            const start = startStr ? new Date(`${startStr}T00:00:00Z`) : null;
            const end = endStr ? new Date(`${endStr}T00:00:00Z`) : start;
            if (!start || !end || Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf())) {
                throw new Error('时间段格式不正确，请使用 YYYY-MM-DD。');
            }
            if (end < start) {
                throw new Error('结束日期必须不早于起始日期。');
            }
            const days = [];
            const cursor = new Date(start.getTime());
            while (cursor <= end) {
                days.push(cursor.toISOString().slice(0, 10));
                cursor.setUTCDate(cursor.getUTCDate() + 1);
            }
            return days;
        }

        function buildRequestPayloads() {
            const rows = Array.from(rowsContainer.children).map(readRow);
            if (rows.length === 0) {
                throw new Error('请至少保留一行参数。');
            }
            const days = expandDateRange(rangeStartInput.value.trim(), rangeEndInput.value.trim());
            const requests = [];
            const metas = [];
            rows.forEach((row, rowIndex) => {
                if (days) {
                    days.forEach(day => {
                        const payload = { ...row, day };
                        requests.push(payload);
                        metas.push({ row: rowIndex + 1, day, lat: row.lat, lon: row.lon, tz_offset_hours: row.tz_offset_hours });
                    });
                } else {
                    const payload = { ...row };
                    if (row.day) {
                        payload.day = row.day;
                    } else {
                        delete payload.day;
                    }
                    requests.push(payload);
                    metas.push({ row: rowIndex + 1, day: row.day ?? null, lat: row.lat, lon: row.lon, tz_offset_hours: row.tz_offset_hours });
                }
            });
            return { requests, metas };
        }

        function formatDateTime(value) {
            if (!value) return '—';
            const dt = new Date(value);
            if (Number.isNaN(dt.valueOf())) return value;
            return dt.toLocaleString();
        }

        function resolveDisplayDate(entry) {
            if (entry.meta.day) return entry.meta.day;
            const candidates = [
                entry.data.sunrise_local,
                entry.data.sunset_local,
                entry.data.sunrise_utc,
                entry.data.sunset_utc,
                entry.data.civil_dawn_local,
                entry.data.civil_dawn_utc
            ];
            for (const value of candidates) {
                if (!value) continue;
                const dt = new Date(value);
                if (!Number.isNaN(dt.valueOf())) {
                    return dt.toISOString().slice(0, 10);
                }
            }
            return '未知';
        }

        function renderTable(entries) {
            tableBody.innerHTML = '';
            const fragment = document.createDocumentFragment();
            entries.forEach((entry, index) => {
                const tr = document.createElement('tr');
                const cells = [
                    index + 1,
                    resolveDisplayDate(entry),
                    entry.meta.lat.toFixed(4),
                    entry.meta.lon.toFixed(4),
                    formatDateTime(entry.data.sunrise_local),
                    formatDateTime(entry.data.sunset_local),
                    formatDateTime(entry.data.civil_dawn_local),
                    formatDateTime(entry.data.civil_dusk_local),
                    formatDateTime(entry.data.nautical_dawn_local),
                    formatDateTime(entry.data.nautical_dusk_local),
                    formatDateTime(entry.data.astronomical_dawn_local),
                    formatDateTime(entry.data.astronomical_dusk_local),
                    entry.data.status ?? '—',
                    typeof entry.data.tz_offset_hours === 'number' ? entry.data.tz_offset_hours.toFixed(2) : (typeof entry.meta.tz_offset_hours === 'number' ? entry.meta.tz_offset_hours.toFixed(2) : '—')
                ];
                cells.forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value;
                    tr.appendChild(td);
                });
                fragment.appendChild(tr);
            });
            tableBody.appendChild(fragment);
        }

        function renderJson(entries) {
            const payload = entries.map(entry => ({ meta: entry.meta, data: entry.data }));
            jsonView.textContent = JSON.stringify({ results: payload }, null, 2);
        }

        function csvEscape(value) {
            const str = value == null ? '' : String(value);
            if (/[",\n]/.test(str)) {
                return '"' + str.replace(/"/g, '""') + '"';
            }
            return str;
        }

        function exportCsv(entries) {
            if (!entries.length) return;
            const header = ['index', 'date', 'latitude', 'longitude', 'sunrise_local', 'sunset_local', 'civil_dawn_local', 'civil_dusk_local', 'nautical_dawn_local', 'nautical_dusk_local', 'astronomical_dawn_local', 'astronomical_dusk_local', 'status', 'tz_offset_hours'];
            const lines = [header.join(',')];
            entries.forEach((entry, index) => {
                const row = [
                    index + 1,
                    resolveDisplayDate(entry),
                    entry.meta.lat,
                    entry.meta.lon,
                    entry.data.sunrise_local ?? '',
                    entry.data.sunset_local ?? '',
                    entry.data.civil_dawn_local ?? '',
                    entry.data.civil_dusk_local ?? '',
                    entry.data.nautical_dawn_local ?? '',
                    entry.data.nautical_dusk_local ?? '',
                    entry.data.astronomical_dawn_local ?? '',
                    entry.data.astronomical_dusk_local ?? '',
                    entry.data.status ?? '',
                    (typeof entry.data.tz_offset_hours === 'number' ? entry.data.tz_offset_hours : entry.meta.tz_offset_hours) ?? ''
                ].map(csvEscape);
                lines.push(row.join(','));
            });
            const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            const now = new Date();
            const stamp = now.toISOString().replace(/[:T]/g, '-').split('.')[0];
            a.href = url;
            a.download = `sunrise_results_${stamp}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        exportCsvBtn.addEventListener('click', () => exportCsv(latestEntries));

        function drawSunTrack(entry) {
            const ctx = sunTrackCanvas.getContext('2d');
            ctx.clearRect(0, 0, sunTrackCanvas.width, sunTrackCanvas.height);
            if (!entry || entry.data.status !== 'ok' || !entry.data.sunrise_local || !entry.data.sunset_local) {
                ctx.fillStyle = '#475569';
                ctx.font = '16px system-ui';
                ctx.fillText('暂无可绘制的太阳轨迹，请检查输入或选择非极昼/极夜地点。', 24, 140);
                return;
            }
            const sunrise = new Date(entry.data.sunrise_local);
            const sunset = new Date(entry.data.sunset_local);
            if (Number.isNaN(sunrise.valueOf()) || Number.isNaN(sunset.valueOf()) || sunrise >= sunset) {
                ctx.fillStyle = '#475569';
                ctx.font = '16px system-ui';
                ctx.fillText('无法解析结果时间，暂不显示轨迹。', 24, 140);
                return;
            }
            const now = new Date();
            const total = sunset - sunrise;
            const progress = Math.min(Math.max((now - sunrise) / total, 0), 1);
            const width = sunTrackCanvas.width;
            const height = sunTrackCanvas.height;
            const marginX = 48;
            const baseY = height - 40;
            const amplitude = height - 120;

            ctx.lineWidth = 3;
            ctx.strokeStyle = '#1f6feb';
            ctx.beginPath();
            for (let i = 0; i <= 200; i++) {
                const ratio = i / 200;
                const x = marginX + ratio * (width - marginX * 2);
                const y = baseY - Math.sin(Math.PI * ratio) * amplitude;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            ctx.lineWidth = 1;
            ctx.strokeStyle = '#64748b';
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            ctx.moveTo(marginX, baseY);
            ctx.lineTo(width - marginX, baseY);
            ctx.stroke();
            ctx.setLineDash([]);

            const sunX = marginX + progress * (width - marginX * 2);
            const sunY = baseY - Math.sin(Math.PI * progress) * amplitude;
            const gradient = ctx.createRadialGradient(sunX, sunY, 5, sunX, sunY, 24);
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(1, '#fde68a00');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(sunX, sunY, 24, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#1f2937';
            ctx.font = '14px system-ui';
            ctx.fillText(`日出：${sunrise.toLocaleTimeString()}`, marginX, baseY + 28);
            ctx.textAlign = 'right';
            ctx.fillText(`日落：${sunset.toLocaleTimeString()}`, width - marginX, baseY + 28);
            ctx.textAlign = 'center';
            ctx.fillText(`当前进度：${Math.round(progress * 100)}%`, sunX, sunY - 18);
        }

        function updateResults(entries) {
            resultsSection.classList.remove('hidden');
            latestEntries = entries;
            if (!entries.length) {
                resultsMessage.textContent = '没有可显示的结果。';
                tableBody.innerHTML = '';
                jsonView.textContent = '';
                drawSunTrack(null);
                exportCsvBtn.disabled = true;
                return;
            }
            renderTable(entries);
            renderJson(entries);
            drawSunTrack(entries[0]);
            resultsMessage.textContent = `共获得 ${entries.length} 条结果。`;
            exportCsvBtn.disabled = false;
        }

        addRow();

        addRowButton.addEventListener('click', () => addRow());

        document.getElementById('task-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            try {
                const { requests, metas } = buildRequestPayloads();
                if (!requests.length) {
                    throw new Error('没有需要计算的任务。');
                }
                submitBtn.disabled = true;
                resultsSection.classList.remove('hidden');
                resultsMessage.textContent = '正在计算，请稍候…';
                tableBody.innerHTML = '';
                jsonView.textContent = '';
                drawSunTrack(null);
                exportCsvBtn.disabled = true;

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
                const results = (data.results ?? []).map((entry, idx) => ({
                    meta: metas[idx],
                    data: entry
                }));
                updateResults(results);
                setView('table');
            } catch (error) {
                resultsSection.classList.remove('hidden');
                resultsMessage.textContent = error instanceof Error ? error.message : String(error);
                tableBody.innerHTML = '';
                jsonView.textContent = '';
                drawSunTrack(null);
                exportCsvBtn.disabled = true;
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("sunrise_api:app", host="0.0.0.0", port=7000)
