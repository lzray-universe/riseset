#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${DE_BSP:-}" ]]; then
  mkdir -p "${DE_BSP}"
  if [[ -n "${EPHEMERIS_URL:-}" ]]; then
    has_bsp=$(find "${DE_BSP}" -maxdepth 1 -type f -name '*.bsp' -print -quit || true)
    if [[ -z "${has_bsp}" ]]; then
      echo "Downloading ephemeris from ${EPHEMERIS_URL}"
      tmpfile=$(mktemp)
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${EPHEMERIS_URL}" -o "${tmpfile}"
      elif command -v wget >/dev/null 2>&1; then
        wget -O "${tmpfile}" "${EPHEMERIS_URL}"
      else
        echo "Neither curl nor wget is available to download ephemeris" >&2
        exit 1
      fi
      filename=$(basename "${EPHEMERIS_URL}")
      mv "${tmpfile}" "${DE_BSP}/${filename}"
    fi
  fi
fi

python - <<'PY'
from core.ephemeris import EphemerisAcquisitionError, resolve_ephemeris_source
import sys

try:
    path = resolve_ephemeris_source()
except EphemerisAcquisitionError as exc:  # pragma: no cover - executed at runtime
    print(f"Failed to prepare ephemeris: {exc}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"Using ephemeris source: {path}", file=sys.stderr)
PY

exec uvicorn sunrise_api:app --host 0.0.0.0 --port "${PORT:-8000}"
