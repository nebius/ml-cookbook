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

PACKAGE_PATTERNS = {
    "cuda": re.compile(r"\bcuda(?:-toolkit)?\b|\bcu(?:12|13)\d\b", re.IGNORECASE),
    "deepep": re.compile(r"\bdeep[-_]?ep\b", re.IGNORECASE),
    "flashinfer": re.compile(r"\bflashinfer(?:-python)?\b", re.IGNORECASE),
    "gdrcopy": re.compile(r"\bgdrcopy\b|\blibgdrapi\b", re.IGNORECASE),
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
    "vllm": re.compile(r"\bvllm\b", re.IGNORECASE),
    "xla": re.compile(r"\bxla\b", re.IGNORECASE),
}


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
