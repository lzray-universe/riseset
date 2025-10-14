#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_BIN="${VENV_DIR}/bin"
PYTHON_BIN="${VENV_BIN}/python"

ensure_virtualenv() {
  if [[ -x "${PYTHON_BIN}" ]]; then
    return
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required but was not found in PATH." >&2
    exit 1
  fi

  echo "Creating Python virtual environment at ${VENV_DIR}..."
  python3 -m venv --system-site-packages "${VENV_DIR}"

  echo "Upgrading pip and build tooling in the virtual environment..."
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
}

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
    echo "Installing remaining dependencies with pip in the virtual environment..."
    "${PYTHON_BIN}" -m pip install --upgrade "${pip_requirements[@]}"
  fi
}

ensure_virtualenv
install_dependencies

find_local_bsp_files() {
  local -n _result=$1
  local search_dir="${2:-${DATA_DIR}}"

  _result=()
  [[ -d "${search_dir}" ]] || return 0

  while IFS= read -r -d '' file; do
    _result+=("${file}")
  done < <(find "${search_dir}" -maxdepth 1 -type f -name '*.bsp' -print0 | sort -z)
}

set_de_bsp_default() {
  mkdir -p "${DATA_DIR}"

  local -a bsp_files
  find_local_bsp_files bsp_files

  if [[ ${#bsp_files[@]} -eq 0 ]]; then
    echo "错误：在 ${DATA_DIR} 下未找到任何 .bsp 文件，请手动下载并设置 DE_BSP。" >&2
    exit 1
  fi

  local joined
  joined=$(IFS=:; printf '%s' "${bsp_files[*]}")
  export DE_BSP="${joined}"

  if [[ ${#bsp_files[@]} -eq 1 ]]; then
    echo "DE_BSP 环境变量未设置，已自动指向 ${DE_BSP}"
  else
    echo "DE_BSP 环境变量未设置，检测到多个 BSP 文件并已全部加入：${DE_BSP}"
  fi
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

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_BIN}:${PATH}"

exec "${PYTHON_BIN}" -m uvicorn sunrise_api:app --host 0.0.0.0 --port 7000 "$@"
