"""Detection helpers and normalized facts derived from a container image."""

from __future__ import annotations

import re
from dataclasses import dataclass

CUDA_TARGET_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<kind>sm|compute)[_-]?(?P<number>\d{2,3})"
    r"(?P<suffix>[af]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TORCH_ARCH_RE = re.compile(
    r"(?<![0-9])(?P<major>\d{1,2})\.(?P<minor>\d)(?P<suffix>[af]?)"
    r"(?P<ptx>\+PTX)?(?![0-9])",
    re.IGNORECASE,
)
CUDA_VERSION_RE = re.compile(
    r"\bCUDA(?:_TOOLKIT)?_VERSION\s*[:=]\s*[\"']?"
    r"v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
CUDA_PACKAGE_VERSION_RE = re.compile(
    r"\bcuda(?:-[a-z0-9]+)*[-_](?P<major>11|12|13)[-_.](?P<minor>\d+)\b",
    re.IGNORECASE,
)
CUDA_IMAGE_VERSION_RE = re.compile(
    r"\bcuda:(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
NCCL_VERSION_RE = re.compile(
    r"\b(?:NVIDIA_)?NCCL_VERSION\s*[:=]\s*[\"']?"
    r"(?:\d+:)?v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
NCCL_PACKAGE_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:libnccl(?:2|-dev)?|nccl|nvidia[-_]nccl(?:[-_]cu\d+)?)"
    r"\s*(?:==|=|[-_])\s*(?:\d+:)?v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

UCX_PACKAGE_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:libucx\d*(?:-dev)?|openucx|ucx(?:-tools)?)"
    r"\s*(?:==|=|[-_])\s*(?:\d+:)?v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
HPCX_BUNDLE_RE = re.compile(
    r"\bhpc[-_]?x[-_\s]v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
HPCX_PACKAGE_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])hpcx(?:-[a-z]+)?"
    r"\s*(?:==|=)\s*(?:\d+:)?v?(?P<version>\d+\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

PACKAGE_PATTERNS = {
    "cuda": re.compile(r"\bcuda(?:-toolkit)?\b|\bcu(?:12|13)\d\b", re.IGNORECASE),
    "deepep": re.compile(r"\bdeep[-_]?ep\b", re.IGNORECASE),
    "flashinfer": re.compile(r"\bflashinfer(?:-python)?\b", re.IGNORECASE),
    "gdrcopy": re.compile(r"\bgdrcopy\b|\blibgdrapi\b", re.IGNORECASE),
    "hpcx": re.compile(r"\bhpcx\b|\bhpc-x\b", re.IGNORECASE),
    "hybridep": re.compile(r"\bhybrid[-_]?ep\b", re.IGNORECASE),
    "jax": re.compile(r"\bjax(?:lib)?\b", re.IGNORECASE),
    "megatron-lm": re.compile(r"\bmegatron[-_]?lm\b", re.IGNORECASE),
    "mooncake": re.compile(r"\bmooncake(?:-transfer-engine)?\b", re.IGNORECASE),
    "nccl": re.compile(r"\bnccl\b|\blibnccl\d*\b|\bnvidia-nccl\b", re.IGNORECASE),
    "nixl": re.compile(r"\bnixl(?:-cu\d+)?\b", re.IGNORECASE),
    "nvshmem": re.compile(r"\bnvshmem\b|\bnvidia-nvshmem\b", re.IGNORECASE),
    "pytorch": re.compile(r"\bpytorch\b|\btorch(?:==|~=|>=|<=|\[|\s)", re.IGNORECASE),
    "sglang": re.compile(r"\bsglang\b", re.IGNORECASE),
    "transformer-engine": re.compile(r"\btransformer[-_]?engine\b", re.IGNORECASE),
    "ucx": re.compile(r"\bucx\b|\blibucx\d*\b|\bopenucx\b", re.IGNORECASE),
    "vllm": re.compile(r"\bvllm\b", re.IGNORECASE),
    "xla": re.compile(r"\bxla\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class UcxModuleGroup:
    """One UCX transport-module directory and the modules installed in it.

    UCX can be built with its transports linked into ``libuct`` instead of loaded
    as modules, so an absent CUDA module only means something when the directory
    proves the build is modular.
    """

    directory: str
    modules: tuple[str, ...]

    @property
    def has_cuda_transport(self) -> bool:
        return any(module.startswith("libuct_") and "cuda" in module for module in self.modules)


@dataclass(frozen=True)
class ImageFacts:
    path: str
    reference: str
    final_platform: str | None
    cuda_targets: tuple[str, ...]
    cuda_versions: tuple[str, ...]
    nccl_versions: tuple[str, ...]
    detected_packages: tuple[str, ...]
    environment: dict[str, str]
    ucx_versions: tuple[str, ...] = ()
    hpcx_versions: tuple[str, ...] = ()
    ucx_module_groups: tuple[UcxModuleGroup, ...] = ()

    @property
    def image_references(self) -> tuple[str, ...]:
        return (self.reference,)


def detect_cuda_targets(text: str, environment: dict[str, str] | None = None) -> set[str]:
    """Detect CUDA code-generation targets in text and known environment values."""
    targets = {
        f"{match.group('kind').lower()}_{int(match.group('number'))}{match.group('suffix').lower()}"
        for match in CUDA_TARGET_RE.finditer(text)
    }
    value = (environment or {}).get("TORCH_CUDA_ARCH_LIST", "")
    for match in TORCH_ARCH_RE.finditer(value):
        number = f"{match.group('major')}{match.group('minor')}"
        suffix = match.group("suffix").lower()
        targets.add(f"sm_{number}{suffix}")
        if match.group("ptx"):
            targets.add(f"compute_{number}{suffix}")
    return targets


def detect_packages(text: str) -> set[str]:
    """Detect known frameworks and communication libraries in arbitrary text."""
    return {name for name, pattern in PACKAGE_PATTERNS.items() if pattern.search(text)}


def detect_cuda_versions(text: str, environment: dict[str, str] | None = None) -> set[str]:
    """Detect explicit CUDA toolkit/runtime versions."""
    versions = {match.group("version") for match in CUDA_VERSION_RE.finditer(text)}
    versions.update(match.group("version") for match in CUDA_IMAGE_VERSION_RE.finditer(text))
    for match in CUDA_PACKAGE_VERSION_RE.finditer(text):
        versions.add(f"{match.group('major')}.{match.group('minor')}")
    for name, value in (environment or {}).items():
        if name.upper() == "CUDA_VERSION" and (match := re.search(r"\d+\.\d+(?:\.\d+)*", value)):
            versions.add(match.group())
    return versions


def detect_ucx_versions(text: str, environment: dict[str, str] | None = None) -> set[str]:
    """Detect UCX releases from package records and explicit environment values.

    A bare ``libucp.so.0`` soname is deliberately not a signal: it identifies the
    ABI, not the UCX release, and reporting it as a version would invent evidence.
    """
    versions = {match.group("version") for match in UCX_PACKAGE_VERSION_RE.finditer(text)}
    for name, value in (environment or {}).items():
        if name.upper() in {"UCX_VERSION", "OPENUCX_VERSION"} and (
            match := re.search(r"(?:\d+:)?v?(\d+\.\d+(?:\.\d+)*)", value)
        ):
            versions.add(match.group(1))
    return versions


def detect_hpcx_versions(text: str, environment: dict[str, str] | None = None) -> set[str]:
    """Detect HPC-X releases from bundle directory names, packages, and environment."""
    versions = {match.group("version") for match in HPCX_BUNDLE_RE.finditer(text)}
    versions.update(match.group("version") for match in HPCX_PACKAGE_VERSION_RE.finditer(text))
    for name, value in (environment or {}).items():
        if name.upper() not in {"HPCX_VERSION", "HPCX_DIR", "HPCX_HOME"}:
            continue
        versions.update(match.group("version") for match in HPCX_BUNDLE_RE.finditer(value))
        if name.upper() == "HPCX_VERSION" and (
            match := re.search(r"(?:\d+:)?v?(\d+\.\d+(?:\.\d+)*)", value)
        ):
            versions.add(match.group(1))
    return versions


def detect_nccl_versions(text: str, environment: dict[str, str] | None = None) -> set[str]:
    """Detect NCCL runtime versions from package records and explicit environment values."""
    versions = {match.group("version") for match in NCCL_VERSION_RE.finditer(text)}
    versions.update(match.group("version") for match in NCCL_PACKAGE_VERSION_RE.finditer(text))
    for name, value in (environment or {}).items():
        if name.upper() in {"NCCL_VERSION", "NVIDIA_NCCL_VERSION"} and (
            match := re.search(r"(?:\d+:)?v?(\d+\.\d+(?:\.\d+)*)", value)
        ):
            versions.add(match.group(1))
    return versions
