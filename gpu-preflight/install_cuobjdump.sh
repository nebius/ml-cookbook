#!/usr/bin/env bash

set -Eeuo pipefail

readonly CUDA_KEYRING_VERSION="${CUDA_KEYRING_VERSION:-1.1-1}"
readonly CUDA_CUOBJDUMP_FALLBACK_PACKAGE="${CUDA_CUOBJDUMP_FALLBACK_PACKAGE:-cuda-cuobjdump-13-0}"

usage() {
  cat <<'EOF'
Install the newest NVIDIA cuda-cuobjdump package available for this host, plus the
CUDA 13.0 cuobjdump that GPU Preflight retries when the newest one rejects an older
fat binary ("Invalid ELF" from cuobjdump 13.1+ on some CUDA 12.x libraries).

Usage:
  ./install_cuobjdump.sh

Optional environment variables:
  CUDA_CUOBJDUMP_PACKAGE  Pin a package, for example cuda-cuobjdump-13-4.
  CUDA_CUOBJDUMP_FALLBACK_PACKAGE
                          Fallback analyzer package (default: cuda-cuobjdump-13-0;
                          set to "none" to skip it).
  CUDA_REPO_DISTRO        Override repository distribution, for example ubuntu2404.
  CUDA_REPO_ARCH          Override repository architecture: x86_64 or sbsa.
  CUDA_KEYRING_VERSION    Override the NVIDIA keyring package version.
EOF
}

fail() {
  echo "install_cuobjdump.sh: $*" >&2
  exit 1
}

case "${1:-}" in
  "") ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

command -v apt-get >/dev/null 2>&1 || fail "apt-get is required"
command -v dpkg >/dev/null 2>&1 || fail "dpkg is required"
command -v apt-cache >/dev/null 2>&1 || fail "apt-cache is required"
[[ -r /etc/os-release ]] || fail "/etc/os-release is required"

# shellcheck disable=SC1091
. /etc/os-release
case "${ID:-}" in
  ubuntu|debian) ;;
  *) fail "supported distributions are Ubuntu and Debian; detected ${ID:-unknown}" ;;
esac
[[ -n "${VERSION_ID:-}" ]] || fail "VERSION_ID is missing from /etc/os-release"

detected_distro="${ID}${VERSION_ID//./}"
case "$(dpkg --print-architecture)" in
  amd64) detected_arch="x86_64" ;;
  arm64) detected_arch="sbsa" ;;
  *) fail "supported CPU architectures are amd64 and arm64" ;;
esac

repo_distro="${CUDA_REPO_DISTRO:-${detected_distro}}"
repo_arch="${CUDA_REPO_ARCH:-${detected_arch}}"
case "${repo_arch}" in
  x86_64|sbsa) ;;
  *) fail "CUDA_REPO_ARCH must be x86_64 or sbsa" ;;
esac
repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_distro}/${repo_arch}"

if ((EUID == 0)); then
  sudo_command=()
else
  command -v sudo >/dev/null 2>&1 || fail "sudo is required when not running as root"
  sudo_command=(sudo)
fi

temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "${temporary_directory}"' EXIT
keyring_path="${temporary_directory}/cuda-keyring.deb"
keyring_url="${repo_url}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb"

echo "Configuring NVIDIA CUDA repository: ${repo_url}"
"${sudo_command[@]}" apt-get update
"${sudo_command[@]}" apt-get install -y --no-install-recommends ca-certificates curl
curl --fail --show-error --silent --location --output "${keyring_path}" "${keyring_url}" \
  || fail "could not download ${keyring_url}; verify the detected distribution and architecture"
"${sudo_command[@]}" dpkg -i "${keyring_path}"
"${sudo_command[@]}" apt-get update

cuobjdump_package="${CUDA_CUOBJDUMP_PACKAGE:-}"
if [[ -n "${cuobjdump_package}" ]]; then
  [[ "${cuobjdump_package}" =~ ^cuda-cuobjdump-[0-9]+-[0-9]+$ ]] \
    || fail "CUDA_CUOBJDUMP_PACKAGE must look like cuda-cuobjdump-13-4"
else
  cuobjdump_package="$({
    apt-cache pkgnames \
      | grep -E '^cuda-cuobjdump-[0-9]+-[0-9]+$' \
      | LC_ALL=C sort -Vu \
      | tail -n 1
  } || true)"
fi
[[ -n "${cuobjdump_package}" ]] \
  || fail "the configured NVIDIA repository contains no cuda-cuobjdump packages"

echo "Installing ${cuobjdump_package}"
"${sudo_command[@]}" apt-get install -y "${cuobjdump_package}"

fallback_package=""
if [[ "${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}" != "none" \
  && "${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}" != "${cuobjdump_package}" ]]; then
  [[ "${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}" =~ ^cuda-cuobjdump-[0-9]+-[0-9]+$ ]] \
    || fail "CUDA_CUOBJDUMP_FALLBACK_PACKAGE must look like cuda-cuobjdump-13-0 or be none"
  if apt-cache show "${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}" >/dev/null 2>&1; then
    echo "Installing fallback analyzer ${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}"
    "${sudo_command[@]}" apt-get install -y "${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}"
    fallback_package="${CUDA_CUOBJDUMP_FALLBACK_PACKAGE}"
  else
    echo "Fallback analyzer ${CUDA_CUOBJDUMP_FALLBACK_PACKAGE} is not in this repository; skipping." >&2
  fi
fi

cuobjdump_path="$({
  dpkg-query -L "${cuobjdump_package}" \
    | grep -E '/bin/cuobjdump$' \
    | head -n 1
} || true)"
[[ -n "${cuobjdump_path}" && -x "${cuobjdump_path}" ]] \
  || fail "${cuobjdump_package} installed, but its cuobjdump executable was not found"

command_link="/usr/local/bin/cuobjdump"
"${sudo_command[@]}" install -d -m 0755 "$(dirname "${command_link}")"
if [[ -L "${command_link}" ]]; then
  existing_target="$(readlink "${command_link}")"
  case "${existing_target}" in
    /usr/local/cuda-*/bin/cuobjdump)
      "${sudo_command[@]}" ln -sfn "${cuobjdump_path}" "${command_link}"
      ;;
    *)
      echo "Not replacing existing ${command_link} -> ${existing_target}; use ${cuobjdump_path} explicitly." >&2
      ;;
  esac
elif [[ -e "${command_link}" ]]; then
  echo "Not replacing existing ${command_link}; use ${cuobjdump_path} explicitly." >&2
else
  "${sudo_command[@]}" ln -s "${cuobjdump_path}" "${command_link}"
fi

echo
echo "Installed analyzer:"
"${cuobjdump_path}" --version
echo
echo "Executable: ${cuobjdump_path}"
echo "Package: ${cuobjdump_package}"
if [[ -n "${fallback_package}" ]]; then
  fallback_path="$(dpkg-query -L "${fallback_package}" | grep -E '/bin/cuobjdump$' | head -n 1 || true)"
  echo "Fallback analyzer: ${fallback_path:-not found} (${fallback_package})"
fi
if [[ -L "${command_link}" ]]; then
  echo "Command link: ${command_link} -> $(readlink "${command_link}")"
fi
