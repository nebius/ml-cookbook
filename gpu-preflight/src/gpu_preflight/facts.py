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
    "cutlass-dsl": re.compile(r"\bnvidia[-_]?cutlass[-_]?dsl\b|\bcute[-_]?dsl\b", re.IGNORECASE),
    "deepep": re.compile(r"\bdeep[-_]?ep\b", re.IGNORECASE),
    "flash-attn": re.compile(r"\bflash[-_]?attn\b|\bflash[-_]?attention\b", re.IGNORECASE),
    "flashinfer": re.compile(r"\bflashinfer(?:-python)?\b", re.IGNORECASE),
    "gdrcopy": re.compile(r"\bgdrcopy\b|\blibgdrapi\b", re.IGNORECASE),
    "hpcx": re.compile(r"\bhpcx\b|\bhpc-x\b", re.IGNORECASE),
    "hybridep": re.compile(r"\bhybrid[-_]?ep\b", re.IGNORECASE),
    "jax": re.compile(r"\bjax(?:lib)?\b", re.IGNORECASE),
    "liger-kernel": re.compile(r"\bliger[-_]?kernel\b", re.IGNORECASE),
    "megatron-lm": re.compile(r"\bmegatron[-_]?lm\b", re.IGNORECASE),
    "mooncake": re.compile(r"\bmooncake(?:-transfer-engine)?\b", re.IGNORECASE),
    "nccl": re.compile(r"\bnccl\b|\blibnccl\d*\b|\bnvidia-nccl\b", re.IGNORECASE),
    "nixl": re.compile(r"\bnixl(?:[-_]cu\d+)?\b", re.IGNORECASE),
    "numba": re.compile(r"\bnumba(?:[-_]cuda)?\b", re.IGNORECASE),
    "nvshmem": re.compile(r"\bnvshmem\b|\bnvidia-nvshmem\b", re.IGNORECASE),
    "pytorch": re.compile(r"\bpytorch\b|\btorch(?:==|~=|>=|<=|\[|\s)", re.IGNORECASE),
    "sglang": re.compile(r"\bsglang\b", re.IGNORECASE),
    "transformer-engine": re.compile(r"\btransformer[-_]?engine\b", re.IGNORECASE),
    "triton": re.compile(r"\b(?:pytorch-)?triton(?:-lang)?\b", re.IGNORECASE),
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


# ELF e_machine values, mapped to the same vocabulary as a profile's
# hardware.cpu_architectures. Only the machines a container image plausibly ships are
# named; anything else is reported by its raw e_machine so an unexpected build still
# shows up instead of being silently treated as native.
# EM_CUDA. A .cubin is an ELF object whose machine is CUDA because it holds device
# code, so it matches no CPU architecture by design and the dynamic loader never sees
# it -- the driver does, so it is not evidence about the host architecture.
ELF_MACHINE_CUDA = 0xBE
ELF_MACHINES = {
    0x03: "i386",
    0x28: "arm",
    0x3E: "amd64",
    0xB7: "arm64",
    0xF3: "riscv64",
    0x15: "ppc64le",
    0x16: "s390x",
}


def elf_architecture(header: bytes) -> str | None:
    """Read the CPU architecture out of an ELF header, or None if this is not an ELF."""
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    byteorder = "little" if header[5] == 1 else "big"
    machine = int.from_bytes(header[18:20], byteorder)
    if machine == ELF_MACHINE_CUDA:
        return None
    return ELF_MACHINES.get(machine, f"elf-machine-0x{machine:x}")


@dataclass(frozen=True)
class LibraryArchitectureEvidence:
    """One CPU architecture observed among an image's binaries, with example paths."""

    architecture: str
    count: int
    sample_paths: tuple[str, ...]
    device_code_count: int = 0


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
    installed_packages: tuple[str, ...] = ()
    library_architectures: tuple[LibraryArchitectureEvidence, ...] = ()
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


PACKAGE_RECORD_RE = re.compile(r"^(?P<name>[^=]+)==(?P<version>\d+(?:\.\d+)*)")
PACKAGE_NAME_SEPARATORS = re.compile(r"[-_.]+")


def normalize_package_name(name: str) -> str:
    """Normalize a distribution name to align with PyPI (i.e. cuTe-DSL == cutlass_dsl)."""
    return PACKAGE_NAME_SEPARATORS.sub("-", name.strip().lower())


def package_versions(records: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Index ``name==version`` inventory records by normalized distribution name.

    One name can carry several versions: an image may install the same package into
    both the system and the Python prefix. Every version is kept, because which one a
    process actually loads is a runtime property this scan cannot observe.
    """
    versions: dict[str, set[str]] = {}
    for record in records:
        match = PACKAGE_RECORD_RE.match(record)
        if match:
            versions.setdefault(normalize_package_name(match.group("name")), set()).add(
                match.group("version")
            )
    return {name: tuple(sorted(found)) for name, found in versions.items()}


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
