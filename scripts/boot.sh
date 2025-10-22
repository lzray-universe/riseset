#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DE_BSP:-}" ]]; then
  echo "DE_BSP environment variable must be set" >&2
  exit 1
fi

mkdir -p "${DE_BSP}"

has_bsp=$(find "${DE_BSP}" -maxdepth 1 -type f -name '*.bsp' -print -quit || true)
if [[ -z "${has_bsp}" ]]; then
  if [[ -n "${EPHEMERIS_URL:-}" ]]; then
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
  else
    echo "Warning: no BSP files found in ${DE_BSP} and EPHEMERIS_URL not set" >&2
  fi
fi

echo "Ephemeris directory contents:" >&2
find "${DE_BSP}" -maxdepth 1 -type f -name '*.bsp' -printf '  %f\n' || true

exec uvicorn sunrise_api:app --host 0.0.0.0 --port "${PORT:-8000}"
