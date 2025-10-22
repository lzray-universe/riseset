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


def _ensure_ephemeris(path: Path) -> Path:
    """Ensure *path* references an existing BSP file or directory containing one."""

    if path.exists():
        if path.is_file():
            if path.suffix.lower() != ".bsp":
                raise EphemerisAcquisitionError(
                    f"Ephemeris file must have .bsp extension: {path}"
                )
            return path
        if path.is_dir():
            existing = sorted(path.glob("*.bsp"))
            if existing:
                return path
            destination = path / DEFAULT_EPHEMERIS_FILENAME
            LOGGER.info(
                json.dumps(
                    {
                        "event": "ephemeris_downloading",
                        "url": DEFAULT_EPHEMERIS_URL,
                        "destination": str(destination),
                    }
                )
            )
            _download_file(DEFAULT_EPHEMERIS_URL, destination)
            return path
        raise EphemerisAcquisitionError(
            f"Ephemeris path is not a file or directory: {path}"
        )

    if path.suffix.lower() == ".bsp":
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            json.dumps(
                {
                    "event": "ephemeris_downloading",
                    "url": DEFAULT_EPHEMERIS_URL,
                    "destination": str(path),
                }
            )
        )
        _download_file(DEFAULT_EPHEMERIS_URL, path)
        return path

    path.mkdir(parents=True, exist_ok=True)
    destination = path / DEFAULT_EPHEMERIS_FILENAME
    LOGGER.info(
        json.dumps(
            {
                "event": "ephemeris_downloading",
                "url": DEFAULT_EPHEMERIS_URL,
                "destination": str(destination),
            }
        )
    )
    _download_file(DEFAULT_EPHEMERIS_URL, destination)
    return path


def resolve_ephemeris_source() -> Path:
    """Return a path to a usable ephemeris kernel, downloading it if necessary."""

    override = os.environ.get("DE_BSP")
    if override:
        return _ensure_ephemeris(Path(override).expanduser())

    cache_root = Path(
        os.environ.get("DE_BSP_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    ).expanduser()
    ephemeris_path = cache_root / DEFAULT_EPHEMERIS_FILENAME
    return _ensure_ephemeris(ephemeris_path)

