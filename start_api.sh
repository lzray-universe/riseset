#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
EPHEMERIS_FILE="${DATA_DIR}/de442s.bsp"
EPHEMERIS_URL="https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de442s.bsp"

install_dependencies() {
  local requirements_file="${ROOT_DIR}/requirements.txt"
  [[ -f "${requirements_file}" ]] || return 0

  echo "Installing Python dependencies from requirements.txt..."

  local -A apt_package_map=(
    [fastapi]="python3-fastapi"
    [uvicorn]="python3-uvicorn"
    [pydantic]="python3-pydantic"
    [numpy]="python3-numpy"
    [astropy]="python3-astropy"
    [joblib]="python3-joblib"
    [spiceypy]="python3-spiceypy"
    [pyerfa]="python3-pyerfa"
  )

  local -a apt_packages=()
  local -a apt_fallback_requirements=()
  local -a pip_requirements=()
  while IFS= read -r requirement || [[ -n "${requirement}" ]]; do
    requirement="${requirement%%#*}"
    requirement="${requirement//[$'\t\r\n ']}"
    [[ -n "${requirement}" ]] || continue

    local key="${requirement}"
    key="${key%%[<>=!~]*}"         # strip version specifiers
    key="${key%%[*}"               # strip extras
    key="${key,,}"                 # normalize to lowercase

    if [[ -n "${apt_package_map[${key}]:-}" ]]; then
      apt_packages+=("${apt_package_map[${key}]}")
      apt_fallback_requirements+=("${requirement}")
    else
      pip_requirements+=("${requirement}")
    fi
  done < "${requirements_file}"

  if [[ ${#apt_packages[@]} -gt 0 ]] && command -v apt-get >/dev/null 2>&1; then
    local -a unique_apt_packages
    mapfile -t unique_apt_packages < <(printf '%s\n' "${apt_packages[@]}" | sort -u)
    local apt_cmd=(apt-get)
    if [[ ${EUID} -ne 0 ]]; then
      if command -v sudo >/dev/null 2>&1; then
        apt_cmd=(sudo apt-get)
      else
        echo "Warning: apt-get requires root privileges. Falling back to pip for all packages." >&2
        pip_requirements+=("${apt_fallback_requirements[@]}")
        unique_apt_packages=()
      fi
    fi

    if [[ ${#unique_apt_packages[@]} -gt 0 ]]; then
      echo "Using apt-get to install available system packages..."
      if "${apt_cmd[@]}" update && "${apt_cmd[@]}" install -y "${unique_apt_packages[@]}"; then
        :
      else
        echo "Warning: apt-get installation failed; falling back to pip for unmet requirements." >&2
        pip_requirements+=("${apt_fallback_requirements[@]}")
      fi
    fi
  else
    pip_requirements+=("${apt_fallback_requirements[@]}")
  fi

  if [[ ${#pip_requirements[@]} -gt 0 ]]; then
    echo "Installing remaining dependencies with pip..."
    python3 -m pip install --upgrade "${pip_requirements[@]}"
  fi
}

install_dependencies

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
