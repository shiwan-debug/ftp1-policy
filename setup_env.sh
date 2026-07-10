#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup for:
# - uv install + mirror
# - python venv (.venv) + basic python deps
# - build/install RoboMeshCat (local)
# - build/install Eigen3 + glog from LOCAL sources (no download; works for any Ubuntu version)
# - build/install IK (ikfk_lib)
#
# Usage:
#   bash setup_kinematic_env.sh
#   MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple" bash setup_kinematic_env.sh
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KINEMATIC_DIR="${ROOT_DIR}/data_processing/parse_data_module/sharpa_toolkit/kinematic"
ROBOMESHCAT_DIR="${KINEMATIC_DIR}/RoboMeshCat"
IK_DIR="${KINEMATIC_DIR}/ikfk_lib"

MIRROR_URL="${MIRROR_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MIRROR_HOST="${MIRROR_HOST:-}"
INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-1}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://bgithub.xyz/}"

log() { echo "[setup] $*"; }
die() { echo "[setup][ERR] $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

detect_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    die "python is required (pip alone is not enough)"
  fi
}

ensure_uv() {
  local py="$1"
  log "Installing uv via pip (python: $py) ..."
  # Do NOT write any global/user config; use mirror flags only for this invocation.
  if [ -n "${MIRROR_URL:-}" ]; then
    if [ -n "${MIRROR_HOST:-}" ]; then
      "$py" -m pip install -U -i "${MIRROR_URL}" --trusted-host "${MIRROR_HOST}" uv
    else
      "$py" -m pip install -U -i "${MIRROR_URL}" uv
    fi
  else
    "$py" -m pip install -U uv
  fi
  need_cmd uv
}

configure_mirrors() {
  log "Using mirror for this script session only: ${MIRROR_URL}"
  export UV_INDEX_URL="${MIRROR_URL}"
  export PIP_INDEX_URL="${MIRROR_URL}"


}

create_venv() {
  # Non-interactive + resumable:
  # - If venv looks healthy, reuse it.
  # - If it's missing critical bits (e.g. interrupted creation), clear+recreate without prompts.
  if [ -f "${ROOT_DIR}/.venv/bin/activate" ] && [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    log "Reusing existing venv: ${ROOT_DIR}/.venv"
  else
    log "Creating venv at: ${ROOT_DIR}/.venv (auto-clear if exists)"
    # Hint from uv: UV_VENV_CLEAR=1 skips the interactive "replace it?" prompt.
    UV_VENV_CLEAR=1 uv venv "${ROOT_DIR}/.venv"
  fi

  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
  python -V
}

patch_transformers() {
  local src="${ROOT_DIR}/src/openpi/models_pytorch/transformers_replace"
  local dst="${ROOT_DIR}/.venv/lib/python3.11/site-packages/transformers"
  [ -d "${src}" ] || { log "transformers_replace dir not found, skipping patch."; return 0; }
  [ -d "${dst}" ] || { log "transformers package not found in venv, skipping patch."; return 0; }
  log "Patching transformers with custom replacements ..."
  cp -r "${src}"/* "${dst}"/
}

install_project_deps() {
  # Install repo root dependencies from pyproject.toml into the active venv.
  # This can be very large (e.g. torch/jax). You can skip with:
  #   INSTALL_PROJECT_DEPS=0 bash setup_kinematic_env.sh
  if [ "${INSTALL_PROJECT_DEPS}" = "0" ]; then
    log "INSTALL_PROJECT_DEPS=0: skipping repo pyproject.toml dependencies."
    return 0
  fi

  log "Installing repo dependencies from ${ROOT_DIR}/pyproject.toml into .venv (editable) ..."
  if [ -n "${MIRROR_HOST:-}" ]; then
    export PIP_TRUSTED_HOST="${MIRROR_HOST}"
  fi
  uv pip install --index-url "${MIRROR_URL}" -e "${ROOT_DIR}"

  patch_transformers
}

install_python_deps() {
  log "Installing basic python deps into .venv (using uv pip) ..."
  if [ -n "${MIRROR_HOST:-}" ]; then
    export PIP_TRUSTED_HOST="${MIRROR_HOST}"
  fi

  uv pip install --index-url "${MIRROR_URL}" -U pip setuptools wheel


  # Mentioned in setup_env_guide.md
  uv pip install --index-url "${MIRROR_URL}" -U transforms3d kinpy

  # RoboMeshCat deps are listed in its pyproject.toml; this helps pre-resolve them with the mirror.
  uv pip install --index-url "${MIRROR_URL}" -U pin meshcat trimesh imageio imageio-ffmpeg
}

install_robomeshcat() {
  [ -d "${ROBOMESHCAT_DIR}" ] || die "RoboMeshCat dir not found: ${ROBOMESHCAT_DIR}"
  log "Installing RoboMeshCat from local path: ${ROBOMESHCAT_DIR}"
  uv pip install --index-url "${MIRROR_URL}" "${ROBOMESHCAT_DIR}"
}

ensure_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    return 0
  fi
  die "sudo is required to install system deps (glog/eigen). Please install sudo or run as root."
}

install_system_deps_apt_ubuntu() {
  ensure_sudo
  log "Installing system deps via apt (Eigen3 + glog + gflags + build tools) ..."
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential \
    cmake \
    make \
    g++ \
    pkg-config \
    wget \
    ca-certificates \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev
}

have_eigen_340_local() {
  local macros="/usr/local/include/eigen3/Eigen/src/Core/util/Macros.h"
  [ -f "${macros}" ] || return 1

  # Parse:
  #   #define EIGEN_WORLD_VERSION 3
  #   #define EIGEN_MAJOR_VERSION 4
  #   #define EIGEN_MINOR_VERSION 0
  local w m n
  w="$(grep -E '^[[:space:]]*#define[[:space:]]+EIGEN_WORLD_VERSION[[:space:]]+' "${macros}" | awk '{print $3}' | tail -n 1)"
  m="$(grep -E '^[[:space:]]*#define[[:space:]]+EIGEN_MAJOR_VERSION[[:space:]]+' "${macros}" | awk '{print $3}' | tail -n 1)"
  n="$(grep -E '^[[:space:]]*#define[[:space:]]+EIGEN_MINOR_VERSION[[:space:]]+' "${macros}" | awk '{print $3}' | tail -n 1)"
  [ "${w}.${m}.${n}" = "3.4.0" ]
}

have_glog_060_local() {
  # Prefer pkg-config if available, since it is stable across distros.
  local v=""
  if command -v pkg-config >/dev/null 2>&1; then
    v="$(pkg-config --modversion libglog 2>/dev/null || true)"
    if [ -z "${v}" ]; then
      v="$(pkg-config --modversion libgoogle-glog 2>/dev/null || true)"
    fi
  fi

  if [ -n "${v}" ]; then
    [ "${v}" = "0.6.0" ]
    return $?
  fi

  # Fallback: check lib presence under /usr/local.
  [ -f /usr/local/lib/libglog.so ] || [ -f /usr/local/lib64/libglog.so ]
}

clean_stale_cmake_cache() {
  local build_dir="$1"
  local src_dir="$2"
  local cache_file="${build_dir}/CMakeCache.txt"
  [ -f "${cache_file}" ] || return 0

  local cached_src
  cached_src="$(grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' "${cache_file}" 2>/dev/null | cut -d= -f2-)"
  if [ -n "${cached_src}" ] && [ "${cached_src}" != "${src_dir}" ]; then
    log "CMake cache path mismatch (cached: ${cached_src}, current: ${src_dir}); clearing ${build_dir}"
    rm -rf "${build_dir}"
  fi
}

install_eigen_from_local_source() {
  ensure_sudo
  local build_root="$1"
  local src_dir="${KINEMATIC_DIR}/eigen-3.4.0"
  [ -d "${src_dir}" ] || die "Eigen source dir not found (expected local checkout): ${src_dir}"

  log "Building Eigen 3.4.0 from LOCAL source into /usr/local ..."
  mkdir -p "${build_root}"
  clean_stale_cmake_cache "${build_root}/eigen-build" "${src_dir}"
  cmake -S "${src_dir}" -B "${build_root}/eigen-build" -DCMAKE_INSTALL_PREFIX=/usr/local
  sudo cmake --build "${build_root}/eigen-build" -j"$(nproc || echo 4)"
  sudo cmake --install "${build_root}/eigen-build"
}

install_glog_from_local_source() {
  ensure_sudo
  local build_root="$1"
  local src_dir="${KINEMATIC_DIR}/glog-0.6.0"
  [ -d "${src_dir}" ] || die "glog source dir not found (expected local checkout): ${src_dir}"

  log "Building glog v0.6.0 from LOCAL source into /usr/local ..."
  mkdir -p "${build_root}"
  clean_stale_cmake_cache "${build_root}/glog-build" "${src_dir}"
  cmake -S "${src_dir}" -B "${build_root}/glog-build" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  sudo cmake --build "${build_root}/glog-build" -j"$(nproc || echo 4)"
  sudo cmake --install "${build_root}/glog-build"
  sudo ldconfig || true
}

install_system_deps() {
  # Always build Eigen + glog from local sources under ${KINEMATIC_DIR}, regardless of Ubuntu version.
  # (This avoids version mismatches from distro packages and skips any download step.)
  local build_root="${KINEMATIC_DIR}/.build"
  if [ "${CLEAN_BUILD:-0}" = "1" ]; then
    log "CLEAN_BUILD=1: removing local build dir: ${build_root}"
    rm -rf "${build_root}"
  fi

  # Build toolchain + glog dependency headers best-effort.
  if command -v apt-get >/dev/null 2>&1; then
    ensure_sudo
    sudo apt-get update -y
    sudo apt-get install -y build-essential cmake make g++ pkg-config ca-certificates libgflags-dev || true
  fi

  if have_eigen_340_local; then
    log "Eigen 3.4.0 already present in /usr/local; skipping build."
  else
    log "Eigen 3.4.0 not found (or version mismatch); building from local source."
    install_eigen_from_local_source "${build_root}"
  fi

  if have_glog_060_local; then
    log "glog 0.6.0 already present; skipping build."
  else
    log "glog 0.6.0 not found (or version mismatch); building from local source."
    install_glog_from_local_source "${build_root}"
  fi
}

build_install_ik() {
  [ -d "${IK_DIR}" ] || die "IK dir not found: ${IK_DIR}"

  log "Building/installing IK from: ${IK_DIR}"
  pushd "${IK_DIR}" >/dev/null

  # If we built libs into /usr/local, the fix script forces proper include/lib resolution.
  local use_fix=0
  if [ -f /usr/local/include/eigen3/Eigen/Core ] || [ -f /usr/local/lib/libglog.so ]; then
    use_fix=1
  fi

  if [ "${use_fix}" -eq 1 ] && [ -f "./build-fix-ldpath.sh" ]; then
    log "Using ./build-fix-ldpath.sh (non-interactive: choose option 1)"
    printf "1\n" | bash ./build-fix-ldpath.sh
  else
    log "Using ./build.sh (non-interactive: choose option 1)"
    printf "1\n" | bash ./build.sh
  fi

  popd >/dev/null
}

main() {
  need_cmd pip || die "pip is required"
  local py
  py="$(detect_python)"

  ensure_uv "${py}"
  configure_mirrors
  create_venv
  install_project_deps
  install_python_deps
  install_robomeshcat
  install_system_deps
  build_install_ik

  log "Done. To activate: source \"${ROOT_DIR}/.venv/bin/activate\""
}

main "$@"


