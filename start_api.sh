#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
EPHEMERIS_FILE="${DATA_DIR}/de442s.bsp"
EPHEMERIS_URL="https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de442s.bsp"

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install --upgrade -r "${ROOT_DIR}/requirements.txt"
fi

ensure_ephemeris() {
  mkdir -p "${DATA_DIR}"
  if [[ ! -s "${EPHEMERIS_FILE}" ]]; then
    echo "Downloading JPL DE442 ephemeris (this may take a while)..."
    tmp_file="${EPHEMERIS_FILE}.part"
    if command -v curl >/dev/null 2>&1; then
      curl -fL --progress-bar "${EPHEMERIS_URL}" -o "${tmp_file}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${tmp_file}" "${EPHEMERIS_URL}"
    else
      echo "Error: Neither curl nor wget is available to download ${EPHEMERIS_URL}" >&2
      exit 1
    fi
    mv "${tmp_file}" "${EPHEMERIS_FILE}"
  else
    echo "Using cached ephemeris at ${EPHEMERIS_FILE}"
  fi
}

set_de_bsp_default() {
  ensure_ephemeris
  export DE_BSP="${EPHEMERIS_FILE}"
  echo "DE_BSP 环境变量未设置，已自动指向 ${DE_BSP}"
}

validate_de_bsp() {
  IFS=":" read -ra paths <<< "${DE_BSP}"
  for path in "${paths[@]}"; do
    if [[ -f "${path}" ]]; then
      continue
    elif [[ -d "${path}" ]]; then
      if ls "${path}"/*.bsp >/dev/null 2>&1; then
        continue
      fi
    fi
    echo "警告：DE_BSP 中的路径无效或不包含 .bsp 文件：${path}" >&2
  done
}

if [[ -z "${DE_BSP:-}" ]]; then
  set_de_bsp_default
else
  validate_de_bsp
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec uvicorn sunrise_api:app --host 0.0.0.0 --port 7000 "$@"
