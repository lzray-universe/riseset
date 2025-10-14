# Sunrise & Sunset Web API

This project packages a FastAPI service that exposes precise sunrise and sunset
computations backed by the high-accuracy `lunar_calendar6_sunrise.py`
implementation. The service provides both JSON APIs and a browser-friendly
interface that allows you to submit multiple computation tasks in parallel.

## Requirements

* Python 3.10+
* [FastAPI](https://fastapi.tiangolo.com/)
* [Uvicorn](https://www.uvicorn.org/)
* `pydantic`
* Access to a DE440/DE441 ephemeris SPK file. Set the environment variable
  `DE_BSP` to point to the file or a directory containing the `.bsp` kernels.

Install the Python dependencies (FastAPI, uvicorn, pydantic, etc.) with pip:

```bash
pip install -r requirements.txt  # if you maintain a requirements file
# or install packages manually:
pip install fastapi uvicorn pydantic
```

## Running the server

1. Download the required DE440/DE441 ephemeris kernels and set the environment
   variable `DE_BSP` before starting the service. For example:

   ```bash
   export DE_BSP="/data/de441.bsp"
   ```

2. Start the API with uvicorn (the application automatically binds to port
   `7000`):

   ```bash
   uvicorn sunrise_api:app --host 0.0.0.0 --port 7000
   ```

   Alternatively, you can run the module directly:

   ```bash
   python sunrise_api.py
   ```

3. Navigate to [http://localhost:7000](http://localhost:7000) to open the web
   UI. Use the “Add row” button to queue multiple computations and submit them
   as a batch. Each computation runs concurrently on the server. You can also
   call the JSON API endpoints directly:

   * `GET /sun` — query parameters corresponding to the computation inputs.
   * `POST /sun` — JSON body matching the computation inputs.
   * `POST /sun/batch` — JSON body containing an array of requests for
     concurrent evaluation.

## Response format

Both single and batch endpoints return a structure that includes sunrise,
sunset, civil/nautical/astronomical twilight times (UTC and local), the
computation status, and the time-zone offset used for local timestamps.

Example `POST /sun` payload:

```json
{
  "lat": 39.9042,
  "lon": 116.4074,
  "elev": 50,
  "day": "2024-06-01",
  "pressure": 1013.25,
  "temp": 20,
  "dut1": 0,
  "xp": 0,
  "yp": 0,
  "tz_offset_hours": 8
}
```

For more details on the underlying algorithm refer to
`lunar_calendar6_sunrise.py`.
