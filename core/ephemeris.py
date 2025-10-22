"""Utilities for acquiring CSPICE ephemeris kernels."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_EPHEMERIS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp"
)
DEFAULT_EPHEMERIS_FILENAME = "de442.bsp"
DEFAULT_CACHE_DIR = Path.home() / ".riseset" / "kernels"


class EphemerisAcquisitionError(RuntimeError):
    """Raised when the default ephemeris cannot be acquired."""


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, timeout=httpx.Timeout(120.0, connect=30.0)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", "0")) or None
            received = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
                    received += len(chunk)
        LOGGER.info(
            json.dumps(
                {
                    "event": "ephemeris_downloaded",
                    "url": url,
                    "destination": str(destination),
                    "bytes": received,
                    "total": total,
                }
            )
        )
    except Exception as exc:  # pragma: no cover - network/runtime errors are rare in tests.
        if destination.exists():
            destination.unlink()
        raise EphemerisAcquisitionError(f"Failed to download ephemeris from {url}: {exc}") from exc


def resolve_ephemeris_source() -> Path:
    """Return a path to a usable ephemeris kernel, downloading it if necessary."""

    override = os.environ.get("DE_BSP")
    if override:
        return Path(override).expanduser()

    cache_root = Path(
        os.environ.get("DE_BSP_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    ).expanduser()
    ephemeris_path = cache_root / DEFAULT_EPHEMERIS_FILENAME
    if not ephemeris_path.exists():
        LOGGER.info(
            json.dumps(
                {
                    "event": "ephemeris_downloading",
                    "url": DEFAULT_EPHEMERIS_URL,
                    "destination": str(ephemeris_path),
                }
            )
        )
        _download_file(DEFAULT_EPHEMERIS_URL, ephemeris_path)

    return ephemeris_path

