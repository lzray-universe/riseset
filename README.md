# Riseset API

High-precision sunrise and sunset calculations backed by CSPICE ephemerides and
served through FastAPI. The service loads JPL DE-series SPK kernels (e.g.
`de442.bsp`) from a configurable directory, optionally downloading the kernel on
first boot.

## Features

- FastAPI service with `/health` and `/sun` endpoints.
- Structured JSON logging for observability.
- CSPICE-based computation supporting official, civil, nautical, and
  astronomical twilight definitions with horizon dip correction.
- Docker image ready for Render deployment with persistent disk for ephemerides.
- Pytest suite generating a compact ephemeris kernel for offline testing.

## Prerequisites

- Python 3.11+
- Network access on first run to download `de442.bsp`, or a locally cached
  ephemeris directory referenced by the optional `DE_BSP` environment variable.

## Local development

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. (Optional) Download an ephemeris manually and point the app at it:

   ```bash
   mkdir -p data/spk
   curl -fsSL https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp -o data/spk/de442.bsp
   export DE_BSP=$PWD/data/spk
   ```

   If you skip this step the application downloads `de442.bsp` into
   `~/.riseset/kernels` on first startup.

3. Start the API locally:

   ```bash
   uvicorn sunrise_api:app --reload --host 0.0.0.0 --port 8000
   ```

4. Query the service:

   ```bash
   curl "http://localhost:8000/sun?lat=35.6895&lon=139.6917&date=2025-10-21&elev_m=40&offset_hours=9&twilight=official"
   ```

   The response contains UTC and optional local times together with the
   computation status and data source identifier.

## Docker usage

Build and run the container locally:

```bash
docker build -t riseset .
docker run --rm -p 8000:8000 riseset
```

Mounting a host directory and exporting `DE_BSP` remains supported when you want
to reuse an existing kernel:

```bash
docker run --rm -p 8000:8000 -e DE_BSP=/data/spk -v "$PWD/data/spk:/data/spk" riseset
```

## Render deployment

The provided `render.yaml` defines a web service using the Docker image. Steps:

1. Provision a persistent disk (e.g. 5 GB) mounted at `/data` to cache the
   ephemeris across deployments (optional but recommended).
2. Deploy the service; the boot script validates any provided `DE_BSP`
   directory, falls back to the internal cache (`~/.riseset/kernels`), downloads
   `de442.bsp` if needed, and then starts Uvicorn on `$PORT`.

Render health checks should target `/health`, which reports whether the
ephemeris files were successfully loaded.

## Testing

The test suite synthesizes a compact SPK kernel (covering year 2025) using ERFA
and CSPICE routines, ensuring offline execution without large ephemeris files.
Run tests with:

```bash
pytest -q
```

Tests cover:

- Known sunrise/sunset bounds for representative locations.
- Polar day/night edge cases.
- Validation error handling.

## Project layout

```
riseset/
├─ sunrise_api.py         # FastAPI application
├─ core/astro.py          # Ephemeris loading and sun-time calculations
├─ models.py              # Pydantic request/response models
├─ scripts/boot.sh        # Render boot script (BSP check + Uvicorn)
├─ tests/test_sun.py      # Pytest suite
├─ Dockerfile             # Container definition
└─ render.yaml            # Render deployment descriptor
```
