"""Container registry and local image inspection without executing the image."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from gpu_preflight.compatibility import (
    best_cuda_code_path,
    classify_cuda_code_path,
    summarize_compatibility,
)
from gpu_preflight.cuda_inspection import (
    CudaArtifactEvidence,
    CuobjdumpTool,
    classify_artifact_role,
    is_core_framework_library,
    is_required_runtime_library,
    kernel_catalog_declared_targets,
    kernel_catalog_group,
    kernel_catalog_kind,
    resolve_cuobjdump,
)
from gpu_preflight.facts import (
    ImageFacts,
    detect_cuda_targets,
    detect_cuda_versions,
    detect_nccl_versions,
    detect_packages,
)
from gpu_preflight.inspection import (
    Finding,
    FindingDetail,
    InspectionReport,
    ScanMetadata,
    TargetReport,
    inspect,
    refresh_target,
)
from gpu_preflight.profiles import HardwareProfile

VALID_SCAN_MODES = {"none", "manifest", "pull", "filesystem"}
CUDA_TARGET_BYTES_RE = re.compile(
    rb"(?<![A-Za-z0-9])(?P<kind>sm|compute)[_-]?(?P<number>\d{2,3})"
    rb"(?P<suffix>[af]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PACKAGE_METADATA_NAMES = {"METADATA", "PKG-INFO"}
BINARY_SUFFIXES = (".a", ".bin", ".cubin", ".fatbin", ".ptx", ".so")
VERSIONED_SO_RE = re.compile(r"\.so(?:\.\d+)*$", re.IGNORECASE)
DEFAULT_MAX_SCAN_BYTES = 16384 * 1024 * 1024
DEFAULT_MAX_CUOBJDUMP_FILES = 1024
DEFAULT_MAX_ARTIFACT_BYTES = 2048 * 1024 * 1024
SKIPPED_CANDIDATE_RECORD_LIMIT = 200
METADATA_SCAN_LIMIT_BYTES = 64 * 1024 * 1024
METADATA_FILE_LIMIT_BYTES = 8 * 1024 * 1024
BINARY_FILE_LIMIT_BYTES = 64 * 1024 * 1024


class ImageInspectionError(ValueError):
    """Raised when registry or local Docker image inspection cannot complete."""


def _redact_sensitive_diagnostic(value: str) -> str:
    value = re.sub(
        r"(?i)(authorization\s*:\s*bearer|password|passwd|token)(\s*[:=]?\s*)\S+",
        r"\1\2[REDACTED]",
        value,
    )
    return re.sub(r"(?i)(https?://[^\s/@:]+:)[^\s/@]+@", r"\1[REDACTED]@", value)


@dataclass(frozen=True)
class PlatformManifest:
    platform: str
    digest: str
    config_digest: str | None
    compressed_size_bytes: int


@dataclass(frozen=True)
class ImageManifestSet:
    reference: str
    platforms: tuple[PlatformManifest, ...]


@dataclass(frozen=True)
class SkippedCudaCandidate:
    path: str
    size_bytes: int
    reason: str


@dataclass(frozen=True)
class KernelCatalogEvidence:
    """Aggregated evidence for a runtime-selectable CUDA artifact catalog."""

    group: str
    kind: str
    member_count: int
    target_variants: int
    declared_targets: tuple[str, ...]
    samples_inspected: int
    samples_incomplete: int
    verified_targets: tuple[str, ...]
    member_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class NcclVersionEvidence:
    """A version signal and the image record that produced it."""

    version: str
    source: str
    name: str
    path: str | None = None


@dataclass
class _KernelCatalogAccumulator:
    kind: str = "precompiled-cubin"
    member_count: int = 0
    variants: set[tuple[str, ...]] = field(default_factory=set)
    declared_targets: set[str] = field(default_factory=set)
    samples_inspected: int = 0
    samples_incomplete: int = 0
    verified_targets: set[str] = field(default_factory=set)
    member_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RootfsScan:
    known_packages: tuple[str, ...]
    installed_packages: tuple[str, ...]
    cuda_targets: tuple[str, ...]
    cuda_versions: tuple[str, ...]
    nccl_versions: tuple[str, ...]
    files_seen: int
    bytes_scanned: int
    metadata_bytes_scanned: int
    truncated: bool
    metadata_truncated: bool
    cuda_candidate_files: int
    cuda_artifacts: tuple[CudaArtifactEvidence, ...]
    cuobjdump_available: bool
    cuobjdump_version: str | None
    cuobjdump_files_analyzed: int
    cuobjdump_targets_found: int
    cuobjdump_no_device_code: int
    cuobjdump_incomplete: int
    marker_only_files: int
    skipped_no_cuobjdump: int
    skipped_non_runtime: int
    skipped_scan_budget: int
    skipped_artifact_limit: int
    skipped_file_limit: int
    skipped_cuda_candidates: tuple[SkippedCudaCandidate, ...] = ()
    kernel_catalogs: tuple[KernelCatalogEvidence, ...] = ()
    kernel_catalog_members: int = 0
    kernel_catalog_grouped_members: int = 0
    dispatch_module_members: int = 0
    dispatch_module_groups: int = 0
    nccl_evidence: tuple[NcclVersionEvidence, ...] = ()


@dataclass(frozen=True)
class PulledImageFacts:
    platform: str
    digest: str
    environment: dict[str, str]
    known_packages: tuple[str, ...]
    installed_packages: tuple[str, ...]
    cuda_targets: tuple[str, ...]
    cuda_versions: tuple[str, ...]
    nccl_versions: tuple[str, ...]
    user: str | None
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    rootfs: RootfsScan | None = None
    nccl_evidence: tuple[NcclVersionEvidence, ...] = ()


def _platform_name(os_name: str | None, architecture: str | None) -> str | None:
    if not os_name or not architecture or architecture == "unknown":
        return None
    aliases = {"aarch64": "arm64", "x86_64": "amd64"}
    return f"{os_name.lower()}/{aliases.get(architecture.lower(), architecture.lower())}"


def _manifest_record(record: dict[str, object]) -> PlatformManifest | None:
    descriptor = record.get("Descriptor") or record.get("descriptor") or record
    if not isinstance(descriptor, dict):
        return None
    platform_data = descriptor.get("platform")
    if not isinstance(platform_data, dict):
        return None
    platform = _platform_name(
        str(platform_data.get("os") or ""),
        str(platform_data.get("architecture") or ""),
    )
    if not platform:
        return None
    digest = str(descriptor.get("digest") or "")
    schema_manifest = (
        record.get("SchemaV2Manifest")
        or record.get("schemaV2Manifest")
        or record.get("OCIManifest")
        or record.get("ociManifest")
    )
    config_digest: str | None = None
    layer_size = 0
    if isinstance(schema_manifest, dict):
        config = schema_manifest.get("config")
        if isinstance(config, dict) and config.get("digest"):
            config_digest = str(config["digest"])
        layers = schema_manifest.get("layers")
        if isinstance(layers, list):
            layer_size = sum(
                int(layer.get("size", 0)) for layer in layers if isinstance(layer, dict)
            )
    return PlatformManifest(
        platform=platform,
        digest=digest,
        config_digest=config_digest,
        compressed_size_bytes=layer_size,
    )


def parse_manifest_output(reference: str, text: str) -> ImageManifestSet:
    """Parse `docker manifest inspect --verbose` JSON output."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ImageInspectionError("Docker returned invalid manifest JSON") from error

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("manifests"), list):
        records = payload["manifests"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ImageInspectionError("Docker returned an unsupported manifest document")

    platforms = tuple(
        manifest
        for item in records
        if isinstance(item, dict) and (manifest := _manifest_record(item)) is not None
    )
    if not platforms:
        raise ImageInspectionError("No Linux platform manifests were found")
    return ImageManifestSet(reference=reference, platforms=platforms)


def _docker_error(arguments: tuple[str, ...], stderr: str) -> ImageInspectionError:
    detail = _redact_sensitive_diagnostic(" ".join(stderr.strip().split()))
    # Docker prints rejected manifests as Go byte slices ("[123 10 32 32 ...]"); a long run
    # of byte values carries no diagnostic value and would otherwise fill the whole message.
    detail = re.sub(r"(?:\b\d{1,3}\b[ ,]+){32,}\d{1,3}\b", "<raw bytes omitted>", detail)
    detail = detail[-1500:] or "no diagnostic was returned"
    operation = " ".join(arguments[:3])
    return ImageInspectionError(f"docker {operation} failed: {detail}")


def _repository(reference: str) -> str:
    """Return the repository part of ``name[:tag][@digest]``, keeping registry ports."""
    name = reference.rsplit("@", maxsplit=1)[0]
    registry, slash, last = name.rpartition("/")
    return f"{registry}{slash}{last.split(':', maxsplit=1)[0]}"


def _digest_reference(reference: str) -> str:
    """Return ``name@digest`` for ``name:tag@digest``; other references are unchanged.

    The digest alone identifies the content. Docker rejects ``manifest inspect`` of a
    tag-plus-digest reference that resolves to a single-platform manifest ("unsupported
    manifest format"), while ``name@digest`` works for both single- and multi-platform images.
    """
    if "@" not in reference:
        return reference
    return f"{_repository(reference)}@{reference.rsplit('@', maxsplit=1)[1]}"


def _exact_reference(reference: str, digest: str) -> str:
    return f"{_repository(reference)}@{digest}" if digest else _digest_reference(reference)


def _environment(items: object) -> dict[str, str]:
    values: dict[str, str] = {}
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, str):
            name, separator, value = item.partition("=")
            if separator:
                values[name] = value
    return values


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        return (value,)
    return ()


def _metadata_package(data: bytes) -> str | None:
    name: str | None = None
    version: str | None = None
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        if key.lower() == "name":
            name = value.strip()
        elif key.lower() == "version":
            version = value.strip()
        if name and version:
            return f"{name}=={version}"
    return name


def _dpkg_packages(data: bytes) -> set[str]:
    packages: set[str] = set()
    for block in data.decode("utf-8", errors="replace").split("\n\n"):
        name: str | None = None
        version: str | None = None
        for line in block.splitlines():
            if line.startswith("Package: "):
                name = line.removeprefix("Package: ").strip()
            elif line.startswith("Version: "):
                version = line.removeprefix("Version: ").strip()
        if name:
            packages.add(f"{name}=={version}" if version else name)
    return packages


def _apk_packages(data: bytes) -> set[str]:
    packages: set[str] = set()
    for block in data.decode("utf-8", errors="replace").split("\n\n"):
        name: str | None = None
        version: str | None = None
        for line in block.splitlines():
            if line.startswith("P:"):
                name = line[2:].strip()
            elif line.startswith("V:"):
                version = line[2:].strip()
        if name:
            packages.add(f"{name}=={version}" if version else name)
    return packages


def _cuda_targets_bytes(data: bytes) -> set[str]:
    return {
        f"{match.group('kind').decode().lower()}_"
        f"{int(match.group('number').decode())}{match.group('suffix').decode().lower()}"
        for match in CUDA_TARGET_BYTES_RE.finditer(data)
    }


def _version_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _cuda_version_file(data: bytes) -> set[str]:
    text = data.decode("utf-8", errors="replace")
    versions = detect_cuda_versions(text)
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict):
        cuda = document.get("cuda")
        if isinstance(cuda, dict) and cuda.get("version"):
            versions.add(str(cuda["version"]))
    for match in re.finditer(
        r"\bCUDA(?: Toolkit)?(?: Version)?\s+v?(?P<version>\d+\.\d+(?:\.\d+)*)",
        text,
        re.IGNORECASE,
    ):
        versions.add(match.group("version"))
    return versions


def _nccl_header_version(data: bytes) -> set[str]:
    """Read the public NCCL version macros when a development header is packaged."""
    text = data.decode("utf-8", errors="replace")
    components: dict[str, str] = {}
    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^\s*#\s*define\s+NCCL_{name}\s+(\d+)\b", text, re.MULTILINE)
        if match:
            components[name] = match.group(1)
    if len(components) != 3:
        return set()
    return {f"{components['MAJOR']}.{components['MINOR']}.{components['PATCH']}"}


def _nccl_package_evidence(
    package: str,
    *,
    source: str,
    path: str,
) -> set[NcclVersionEvidence]:
    """Return NCCL evidence for one normalized package inventory record."""
    name = package.partition("==")[0]
    return {
        NcclVersionEvidence(
            version=version,
            source=source,
            name=name,
            path=path,
        )
        for version in detect_nccl_versions(package)
    }


def _nccl_config_evidence(
    environment: dict[str, str],
    detected_versions: set[str],
) -> tuple[NcclVersionEvidence, ...]:
    """Attribute image-config NCCL versions to explicit environment variables."""
    evidence: set[NcclVersionEvidence] = set()
    for name, value in environment.items():
        if name.upper() not in {"NCCL_VERSION", "NVIDIA_NCCL_VERSION"}:
            continue
        for version in detect_nccl_versions(f"{name}={value}", {name: value}):
            evidence.add(
                NcclVersionEvidence(
                    version=version,
                    source="image-config",
                    name=name,
                )
            )
    attributed = {item.version for item in evidence}
    evidence.update(
        NcclVersionEvidence(
            version=version,
            source="image-config",
            name="image configuration",
        )
        for version in detected_versions - attributed
    )
    return tuple(
        sorted(evidence, key=lambda item: (_version_sort_key(item.version), item.source, item.name))
    )


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    allowance: int,
) -> bytes:
    if allowance <= 0:
        return b""
    file_object = archive.extractfile(member)
    if file_object is None:
        return b""
    return file_object.read(min(member.size, allowance))


def _stream_cuda_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    allowance: int,
    destination: BinaryIO | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Stream bounded CUDA candidate bytes, optionally to a cuobjdump input file."""
    file_object = archive.extractfile(member)
    if file_object is None or allowance <= 0:
        return 0, ()
    remaining = min(member.size, allowance)
    bytes_read = 0
    carry = b""
    targets: set[str] = set()
    while remaining:
        chunk = file_object.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        if destination:
            destination.write(chunk)
        targets.update(_cuda_targets_bytes(carry + chunk))
        carry = (carry + chunk)[-64:]
        bytes_read += len(chunk)
        remaining -= len(chunk)
    return bytes_read, tuple(sorted(targets))


def scan_rootfs_tar(
    stream: BinaryIO,
    max_scan_bytes: int,
    *,
    cuobjdump: CuobjdumpTool | None = None,
    max_cuobjdump_files: int | None = DEFAULT_MAX_CUOBJDUMP_FILES,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    progress: Callable[[int, int, int], None] | None = None,
) -> RootfsScan:
    """Inspect a merged rootfs tar stream without extracting it."""
    known_packages: set[str] = set()
    installed_packages: set[str] = set()
    cuda_targets: set[str] = set()
    cuda_versions: set[str] = set()
    nccl_versions: set[str] = set()
    nccl_evidence: set[NcclVersionEvidence] = set()
    files_seen = 0
    bytes_scanned = 0
    metadata_bytes_scanned = 0
    truncated = False
    metadata_truncated = False
    cuda_candidate_files = 0
    cuda_artifacts: list[CudaArtifactEvidence] = []
    skipped_cuda_candidates: list[SkippedCudaCandidate] = []
    cuobjdump_files_analyzed = 0
    cuobjdump_targets_found = 0
    cuobjdump_no_device_code = 0
    cuobjdump_incomplete = 0
    marker_only_files = 0
    skipped_no_cuobjdump = 0
    skipped_non_runtime = 0
    skipped_scan_budget = 0
    skipped_artifact_limit = 0
    skipped_file_limit = 0
    kernel_catalogs: dict[str, _KernelCatalogAccumulator] = {}
    kernel_catalog_sample_keys: set[tuple[str, tuple[str, ...]]] = set()
    kernel_catalog_members = 0
    kernel_catalog_grouped_members = 0
    last_progress_at = time.monotonic()

    try:
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                if member.issym() or member.islnk():
                    cuda_versions.update(detect_cuda_versions(member.linkname))
                if not member.isfile():
                    continue
                files_seen += 1
                if progress and time.monotonic() - last_progress_at >= 30:
                    progress(files_seen, bytes_scanned, cuobjdump_files_analyzed)
                    last_progress_at = time.monotonic()
                path_parts = PurePosixPath(member.name).parts
                if member.name.startswith("/") or ".." in path_parts:
                    continue
                name = member.name.removeprefix("./")
                lower_name = name.lower()
                known_packages.update(detect_packages(name))
                cuda_targets.update(detect_cuda_targets(name))

                basename = PurePosixPath(name).name
                is_python_metadata = basename in PACKAGE_METADATA_NAMES and (
                    ".dist-info/" in lower_name or ".egg-info/" in lower_name
                )
                is_system_inventory = lower_name in {
                    "lib/apk/db/installed",
                    "var/lib/dpkg/status",
                }
                is_cuda_version_file = lower_name in {
                    "usr/local/cuda/version.json",
                    "usr/local/cuda/version.txt",
                }
                is_nccl_header = basename.lower() == "nccl.h" and any(
                    marker in lower_name for marker in ("/include/", "usr/include/", "opt/nccl/")
                )
                is_conda_metadata = "/conda-meta/" in f"/{lower_name}" and lower_name.endswith(
                    ".json"
                )
                is_cuda_candidate = (
                    lower_name.endswith(BINARY_SUFFIXES) or VERSIONED_SO_RE.search(lower_name)
                ) and any(
                    marker in lower_name
                    for marker in (
                        "cuda",
                        "flash",
                        "jax",
                        "nccl",
                        "nvidia",
                        "nvshmem",
                        "torch",
                    )
                )
                if not (
                    is_python_metadata
                    or is_system_inventory
                    or is_conda_metadata
                    or is_cuda_candidate
                    or is_cuda_version_file
                    or is_nccl_header
                ):
                    continue

                if is_cuda_candidate:
                    cuda_candidate_files += 1
                    artifact_role = classify_artifact_role(f"/{name}")
                    catalog_group = kernel_catalog_group(f"/{name}")
                    catalog_kind = kernel_catalog_kind(f"/{name}")
                    catalog_targets = tuple(
                        sorted(
                            {
                                *detect_cuda_targets(name),
                                *kernel_catalog_declared_targets(f"/{name}"),
                            }
                        )
                    )
                    catalog_representative = True
                    catalog: _KernelCatalogAccumulator | None = None
                    if catalog_group:
                        if catalog_kind == "precompiled-cubin":
                            kernel_catalog_members += 1
                        catalog = kernel_catalogs.setdefault(
                            catalog_group,
                            _KernelCatalogAccumulator(kind=catalog_kind or "precompiled-cubin"),
                        )
                        catalog.member_count += 1
                        catalog.member_paths.append(f"/{name}")
                        catalog.declared_targets.update(catalog_targets)
                        catalog_key = (catalog_group, catalog_targets or ("unknown",))
                        catalog_representative = catalog_key not in kernel_catalog_sample_keys
                        if catalog_representative:
                            kernel_catalog_sample_keys.add(catalog_key)
                            catalog.variants.add(catalog_key[1])
                        else:
                            if catalog.kind == "precompiled-cubin":
                                kernel_catalog_grouped_members += 1
                            # Shape-specific catalog members with the same declared target are
                            # selectable alternatives. The representative for this catalog/target
                            # pair carries the binary evidence; duplicates are not coverage gaps.
                            continue
                    allowance = max_scan_bytes - bytes_scanned
                    artifact: CudaArtifactEvidence | None = None
                    skip_reason: str | None = None
                    # The rootfs is a single tar stream, so files cannot be reordered; core
                    # framework libraries instead bypass the shared budget and file cap.
                    core_library = artifact_role == "runtime" and is_core_framework_library(name)
                    if cuobjdump is None:
                        skip_reason = "no-cuobjdump"
                        skipped_no_cuobjdump += 1
                    elif artifact_role not in {"runtime", "kernel-catalog"}:
                        skip_reason = "non-runtime"
                        skipped_non_runtime += 1
                    elif member.size > max_artifact_bytes:
                        skip_reason = "artifact-limit"
                        skipped_artifact_limit += 1
                    elif (
                        not core_library
                        and max_cuobjdump_files is not None
                        and cuobjdump_files_analyzed >= max_cuobjdump_files
                    ):
                        skip_reason = "file-limit"
                        skipped_file_limit += 1
                    elif not core_library and (allowance <= 0 or member.size > allowance):
                        skip_reason = "scan-budget"
                        skipped_scan_budget += 1
                    if (
                        skip_reason
                        and skip_reason != "non-runtime"
                        and artifact_role in {"runtime", "kernel-catalog"}
                        and (
                            core_library
                            or len(skipped_cuda_candidates) < SKIPPED_CANDIDATE_RECORD_LIMIT
                        )
                    ):
                        skipped_cuda_candidates.append(
                            SkippedCudaCandidate(
                                path=f"/{name}",
                                size_bytes=member.size,
                                reason=skip_reason,
                            )
                        )
                    planned_cuobjdump = cuobjdump is not None and skip_reason is None
                    if allowance <= 0 and not planned_cuobjdump:
                        truncated = True
                        continue
                    temporary_path: Path | None = None
                    try:
                        if planned_cuobjdump:
                            with tempfile.NamedTemporaryFile(
                                prefix="gpu-preflight-",
                                suffix=PurePosixPath(name).suffix,
                                delete=False,
                            ) as temporary:
                                temporary_path = Path(temporary.name)
                                read_size, marker_targets = _stream_cuda_member(
                                    archive,
                                    member,
                                    member.size,
                                    temporary,
                                )
                        else:
                            read_size, marker_targets = _stream_cuda_member(
                                archive,
                                member,
                                min(allowance, BINARY_FILE_LIMIT_BYTES),
                            )
                        bytes_scanned += read_size
                        complete_file = read_size == member.size
                        if not complete_file and bytes_scanned >= max_scan_bytes:
                            truncated = True

                        if planned_cuobjdump and complete_file and temporary_path:
                            artifact = cuobjdump.inspect(temporary_path, f"/{name}", member.size)
                            cuobjdump_files_analyzed += 1
                            if catalog:
                                catalog.samples_inspected += 1
                                if artifact.complete:
                                    catalog.verified_targets.update(artifact.targets)
                                else:
                                    catalog.samples_incomplete += 1
                            if not artifact.complete:
                                cuobjdump_incomplete += 1
                            elif artifact.targets:
                                cuobjdump_targets_found += 1
                            else:
                                cuobjdump_no_device_code += 1
                        elif marker_targets:
                            diagnostic = None
                            if cuobjdump is None:
                                diagnostic = (
                                    "cuobjdump was unavailable; targets are byte-marker evidence"
                                )
                            elif artifact_role != "runtime":
                                diagnostic = (
                                    f"classified as {artifact_role}; not selected for runtime "
                                    "fatbin analysis"
                                )
                            elif not complete_file:
                                diagnostic = "file exceeded the candidate-content scan budget"
                            elif member.size > max_artifact_bytes:
                                diagnostic = "file exceeded the per-artifact cuobjdump limit"
                            else:
                                diagnostic = "maximum cuobjdump file count was reached"
                            artifact = CudaArtifactEvidence(
                                path=f"/{name}",
                                size_bytes=member.size,
                                targets=marker_targets,
                                method="marker-scan",
                                complete=False,
                                role=artifact_role,
                                diagnostic=diagnostic,
                            )
                            marker_only_files += 1
                    finally:
                        if temporary_path:
                            temporary_path.unlink(missing_ok=True)
                    # Retain enumerated CUDA targets and incomplete inspections that require review.
                    # Successful negative results are aggregate counters to keep reports small.
                    if artifact is not None and (artifact.targets or not artifact.complete):
                        cuda_artifacts.append(artifact)
                    if artifact_role in {"runtime", "kernel-catalog"}:
                        if artifact and artifact.method == "cuobjdump" and artifact.complete:
                            cuda_targets.update(artifact.targets)
                        else:
                            cuda_targets.update(marker_targets)
                    # Binary names and cuobjdump/marker output carry the useful facts. Decoding
                    # hundreds of MiB of ELF data and running package regexes over it is both
                    # expensive and redundant.
                    continue
                else:
                    allowance = METADATA_SCAN_LIMIT_BYTES - metadata_bytes_scanned
                    if allowance <= 0:
                        metadata_truncated = True
                        continue
                    data = _read_member(
                        archive,
                        member,
                        min(allowance, METADATA_FILE_LIMIT_BYTES),
                    )
                    metadata_bytes_scanned += len(data)
                    if len(data) < member.size:
                        metadata_truncated = True

                decoded = data.decode("latin-1", errors="ignore")
                known_packages.update(detect_packages(decoded))
                if is_cuda_version_file:
                    cuda_versions.update(_cuda_version_file(data))
                if is_nccl_header:
                    header_versions = _nccl_header_version(data)
                    nccl_versions.update(header_versions)
                    nccl_evidence.update(
                        NcclVersionEvidence(
                            version=version,
                            source="header",
                            name="NCCL version macros",
                            path=f"/{name}",
                        )
                        for version in header_versions
                    )
                new_inventory: set[str] = set()
                inventory_source: str | None = None
                if is_python_metadata:
                    if package := _metadata_package(data):
                        new_inventory.add(package)
                        inventory_source = "python-package"
                elif lower_name == "var/lib/dpkg/status":
                    new_inventory.update(_dpkg_packages(data))
                    inventory_source = "system-package"
                elif lower_name == "lib/apk/db/installed":
                    new_inventory.update(_apk_packages(data))
                    inventory_source = "system-package"
                elif is_conda_metadata:
                    try:
                        conda = json.loads(data)
                    except json.JSONDecodeError:
                        conda = {}
                    if isinstance(conda, dict) and conda.get("name"):
                        package = str(conda["name"])
                        if conda.get("version"):
                            package += f"=={conda['version']}"
                        new_inventory.add(package)
                        inventory_source = "conda-package"
                installed_packages.update(new_inventory)
                known_packages.update(detect_packages(" ".join(new_inventory)))
                cuda_versions.update(detect_cuda_versions(" ".join(new_inventory)))
                nccl_versions.update(detect_nccl_versions(" ".join(new_inventory)))
                if inventory_source:
                    for package in new_inventory:
                        nccl_evidence.update(
                            _nccl_package_evidence(
                                package,
                                source=inventory_source,
                                path=f"/{name}",
                            )
                        )
    except tarfile.TarError as error:
        raise ImageInspectionError(f"Docker exported an invalid rootfs tar: {error}") from error

    # A lone FlashInfer cache module is an ordinary runtime artifact. Only retain
    # architecture-module catalogs when multiple selectable variants establish an
    # actual dispatch family.
    singleton_dispatch_groups = {
        group
        for group, catalog in kernel_catalogs.items()
        if catalog.kind == "architecture-module" and catalog.member_count < 2
    }
    if singleton_dispatch_groups:
        cuda_artifacts = [
            replace(artifact, role="runtime")
            if kernel_catalog_group(artifact.path) in singleton_dispatch_groups
            else artifact
            for artifact in cuda_artifacts
        ]
    retained_catalogs = {
        group: catalog
        for group, catalog in kernel_catalogs.items()
        if group not in singleton_dispatch_groups
    }
    dispatch_module_members = sum(
        catalog.member_count
        for catalog in retained_catalogs.values()
        if catalog.kind == "architecture-module"
    )
    dispatch_module_groups = sum(
        catalog.kind == "architecture-module" for catalog in retained_catalogs.values()
    )

    return RootfsScan(
        known_packages=tuple(sorted(known_packages)),
        installed_packages=tuple(sorted(installed_packages))[:1000],
        cuda_targets=tuple(sorted(cuda_targets)),
        cuda_versions=tuple(sorted(cuda_versions, key=_version_sort_key)),
        nccl_versions=tuple(sorted(nccl_versions, key=_version_sort_key)),
        files_seen=files_seen,
        bytes_scanned=bytes_scanned,
        metadata_bytes_scanned=metadata_bytes_scanned,
        truncated=truncated,
        metadata_truncated=metadata_truncated,
        cuda_candidate_files=cuda_candidate_files,
        cuda_artifacts=tuple(cuda_artifacts),
        cuobjdump_available=cuobjdump is not None,
        cuobjdump_version=cuobjdump.version if cuobjdump else None,
        cuobjdump_files_analyzed=cuobjdump_files_analyzed,
        cuobjdump_targets_found=cuobjdump_targets_found,
        cuobjdump_no_device_code=cuobjdump_no_device_code,
        cuobjdump_incomplete=cuobjdump_incomplete,
        marker_only_files=marker_only_files,
        skipped_no_cuobjdump=skipped_no_cuobjdump,
        skipped_non_runtime=skipped_non_runtime,
        skipped_scan_budget=skipped_scan_budget,
        skipped_artifact_limit=skipped_artifact_limit,
        skipped_file_limit=skipped_file_limit,
        skipped_cuda_candidates=tuple(skipped_cuda_candidates),
        kernel_catalogs=tuple(
            KernelCatalogEvidence(
                group=group,
                kind=catalog.kind,
                member_count=catalog.member_count,
                target_variants=len(catalog.variants),
                declared_targets=tuple(sorted(catalog.declared_targets)),
                samples_inspected=catalog.samples_inspected,
                samples_incomplete=catalog.samples_incomplete,
                verified_targets=tuple(sorted(catalog.verified_targets)),
                member_paths=tuple(catalog.member_paths),
            )
            for group, catalog in sorted(retained_catalogs.items())
        ),
        kernel_catalog_members=kernel_catalog_members,
        kernel_catalog_grouped_members=kernel_catalog_grouped_members,
        dispatch_module_members=dispatch_module_members,
        dispatch_module_groups=dispatch_module_groups,
        nccl_evidence=tuple(
            sorted(
                nccl_evidence,
                key=lambda item: (
                    _version_sort_key(item.version),
                    item.source,
                    item.name,
                    item.path or "",
                ),
            )
        ),
    )


class DockerImageClient:
    """Read image metadata and files through the Docker CLI without running it."""

    def __init__(
        self,
        *,
        command_timeout_seconds: int = 3600,
        cuobjdump_mode: str = "auto",
        cuobjdump_timeout_seconds: int = 30,
        max_cuobjdump_files: int | None = DEFAULT_MAX_CUOBJDUMP_FILES,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        cuobjdump_fallback_mode: str = "auto",
    ) -> None:
        self.docker_binary = shutil.which("docker")
        self.command_timeout_seconds = command_timeout_seconds
        self.cuobjdump = resolve_cuobjdump(
            cuobjdump_mode,
            cuobjdump_timeout_seconds,
            cuobjdump_fallback_mode,
        )
        self.max_cuobjdump_files = max_cuobjdump_files
        self.max_artifact_bytes = max_artifact_bytes

    def preflight(self, *, require_daemon: bool, require_cuobjdump: bool) -> None:
        """Fail before registry access or image pulls when host prerequisites are missing."""
        if not self.docker_binary:
            raise ImageInspectionError(
                "docker CLI was not found; install Docker and ensure `docker` is on PATH"
            )
        if require_cuobjdump and self.cuobjdump is None:
            raise ImageInspectionError(
                "cuobjdump was not found; install a recent NVIDIA CUDA Toolkit, add cuobjdump "
                "to PATH, or pass --cuobjdump /path/to/cuobjdump"
            )
        if require_daemon:
            try:
                self._run(("info", "--format", "{{.ServerVersion}}"))
            except ImageInspectionError as error:
                raise ImageInspectionError(
                    "Docker daemon is not accessible; start Docker and confirm the current user "
                    f"can run `docker info`: {error}"
                ) from error

    def _run(self, arguments: tuple[str, ...]) -> str:
        if not self.docker_binary:
            raise ImageInspectionError("docker CLI was not found")
        try:
            result = subprocess.run(  # noqa: S603 - argv only; shell execution is disabled
                (self.docker_binary, *arguments),
                capture_output=True,
                check=False,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except FileNotFoundError as error:
            raise ImageInspectionError("docker CLI was not found") from error
        except subprocess.TimeoutExpired as error:
            operation = " ".join(arguments[:3])
            raise ImageInspectionError(
                f"docker {operation} timed out after {self.command_timeout_seconds} seconds"
            ) from error
        if result.returncode:
            raise _docker_error(arguments, result.stderr)
        return result.stdout

    def inspect_manifest(self, reference: str) -> ImageManifestSet:
        output = self._run(("manifest", "inspect", "--verbose", _digest_reference(reference)))
        return parse_manifest_output(reference, output)

    def pull_config(
        self,
        reference: str,
        manifest: PlatformManifest,
    ) -> PulledImageFacts:
        exact_reference = _exact_reference(reference, manifest.digest)
        free_gib = shutil.disk_usage(tempfile.gettempdir()).free / (1024**3)
        compressed_gib = manifest.compressed_size_bytes / (1024**3)
        print(
            f"gpu-preflight: pulling image {reference} for {manifest.platform}; "
            f"compressed layers are approximately {compressed_gib:.2f} GiB and "
            f"{free_gib:.1f} GiB is free in the local temporary filesystem...",
            file=sys.stderr,
            flush=True,
        )
        self._run(("pull", "--platform", manifest.platform, exact_reference))
        payload = json.loads(self._run(("image", "inspect", exact_reference)))
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            raise ImageInspectionError("docker image inspect returned an unsupported document")
        image_data = payload[0]
        config = image_data.get("Config")
        config = config if isinstance(config, dict) else {}
        environment = _environment(config.get("Env"))
        metadata_text = json.dumps(config, sort_keys=True)
        detected_nccl_versions = detect_nccl_versions(metadata_text, environment)
        return PulledImageFacts(
            platform=_platform_name(
                str(image_data.get("Os") or "linux"),
                str(image_data.get("Architecture") or ""),
            )
            or manifest.platform,
            digest=manifest.digest,
            environment=environment,
            known_packages=tuple(sorted(detect_packages(metadata_text))),
            installed_packages=(),
            cuda_targets=tuple(sorted(detect_cuda_targets(metadata_text, environment))),
            cuda_versions=tuple(
                sorted(
                    detect_cuda_versions("", environment),
                    key=_version_sort_key,
                )
            ),
            nccl_versions=tuple(
                sorted(
                    detected_nccl_versions,
                    key=_version_sort_key,
                )
            ),
            user=str(config.get("User")) if config.get("User") else None,
            entrypoint=_string_tuple(config.get("Entrypoint")),
            command=_string_tuple(config.get("Cmd")),
            nccl_evidence=_nccl_config_evidence(environment, detected_nccl_versions),
        )

    def scan_filesystem(
        self,
        reference: str,
        manifest: PlatformManifest,
        config: PulledImageFacts,
        max_scan_bytes: int,
    ) -> PulledImageFacts:
        exact_reference = _exact_reference(reference, manifest.digest)
        print(
            f"gpu-preflight: scanning merged filesystem for {reference} on {manifest.platform}...",
            file=sys.stderr,
            flush=True,
        )
        analyzer = (
            f"cuobjdump {self.cuobjdump.version}"
            if self.cuobjdump
            else "byte-marker fallback (cuobjdump unavailable)"
        )
        inspection_scope = (
            f"up to {self.max_cuobjdump_files} files"
            if self.max_cuobjdump_files is not None
            else "all eligible inspection units"
        )
        print(
            f"gpu-preflight: CUDA artifact analyzer: {analyzer}; {inspection_scope} and "
            f"{self.max_artifact_bytes / (1024**2):.0f} MiB per artifact...",
            file=sys.stderr,
            flush=True,
        )
        container_id = self._run(
            ("create", "--platform", manifest.platform, exact_reference)
        ).strip()
        if not container_id:
            raise ImageInspectionError("docker create did not return a container ID")
        process: subprocess.Popen[bytes] | None = None
        try:
            if not self.docker_binary:
                raise ImageInspectionError("docker CLI was not found")
            process = subprocess.Popen(  # noqa: S603 - fixed Docker export argv
                (self.docker_binary, "export", container_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None:
                raise ImageInspectionError("docker export did not provide a rootfs stream")
            rootfs = scan_rootfs_tar(
                process.stdout,
                max_scan_bytes,
                cuobjdump=self.cuobjdump,
                max_cuobjdump_files=self.max_cuobjdump_files,
                max_artifact_bytes=self.max_artifact_bytes,
                progress=lambda files, size, analyzed: print(
                    f"gpu-preflight: filesystem scan progress: {files} files observed, "
                    f"{size / (1024**2):.1f} MiB candidate content read, "
                    f"{analyzed} files passed to cuobjdump...",
                    file=sys.stderr,
                    flush=True,
                ),
            )
            print(
                f"gpu-preflight: filesystem scan complete: {rootfs.files_seen} files, "
                f"{len(rootfs.cuda_artifacts)} CUDA artifact records, "
                f"{rootfs.cuobjdump_files_analyzed} cuobjdump inspections.",
                file=sys.stderr,
                flush=True,
            )
            stderr = (
                process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            )
            return_code = process.wait()
            if return_code:
                raise _docker_error(("export", container_id), stderr)
        finally:
            if process and process.poll() is None:
                process.kill()
                process.wait()
            with suppress(ImageInspectionError):
                self._run(("rm", "--volumes", container_id))
        return replace(
            config,
            known_packages=tuple(sorted({*config.known_packages, *rootfs.known_packages})),
            installed_packages=rootfs.installed_packages,
            cuda_targets=tuple(sorted({*config.cuda_targets, *rootfs.cuda_targets})),
            cuda_versions=tuple(
                sorted(
                    {*config.cuda_versions, *rootfs.cuda_versions},
                    key=_version_sort_key,
                )
            ),
            nccl_versions=tuple(
                sorted(
                    {*config.nccl_versions, *rootfs.nccl_versions},
                    key=_version_sort_key,
                )
            ),
            rootfs=rootfs,
            nccl_evidence=tuple(
                sorted(
                    {*config.nccl_evidence, *rootfs.nccl_evidence},
                    key=lambda item: (
                        _version_sort_key(item.version),
                        item.source,
                        item.name,
                        item.path or "",
                    ),
                )
            ),
        )


def _architecture(platform: str) -> str:
    return platform.rsplit("/", maxsplit=1)[-1]


def _major_minor(value: str) -> tuple[int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _cuobjdump_is_older(
    cuobjdump_version: str | None,
    image_cuda_versions: tuple[str, ...],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return analyzer/image releases when the analyzer predates image CUDA evidence."""
    analyzer = _major_minor(cuobjdump_version or "")
    image_releases = tuple(
        release for version in image_cuda_versions if (release := _major_minor(version))
    )
    if analyzer is None or not image_releases:
        return None
    newest_image = max(image_releases)
    return (analyzer, newest_image) if analyzer < newest_image else None


# CUDA Toolkit libraries are versioned with the toolkit release and support every GPU that
# release supports. Several of them keep kernels where cuobjdump cannot list them (cuFFT keeps
# most in a vendor-packaged .ldata section; cuFFTW and NPP core libraries carry one placeholder
# cubin), so the release, not the cuobjdump listing, decides their compatibility.
CUDA_TOOLKIT_LIBRARY_PREFIXES = (
    "libcublas",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libnpp",
    "libnvblas",
    "libnvjpeg",
)


def _is_cuda_toolkit_library(path: str) -> bool:
    return PurePosixPath(path.lower()).name.startswith(CUDA_TOOLKIT_LIBRARY_PREFIXES)


def _toolkit_release(path: str, image_cuda_versions: tuple[str, ...]) -> str | None:
    """Return the CUDA major.minor release that shipped a toolkit library, when knowable.

    Uses the versioned toolkit directory (``/usr/local/cuda-12.8/...``) first, then the
    CUDA-major wheel layout (``site-packages/nvidia/cu13/lib``) matched against the image's
    CUDA versions, then the image's CUDA version when it is unambiguous.
    """
    if match := re.search(r"/cuda-(\d+\.\d+)/", path):
        return match.group(1)
    releases = {release for version in image_cuda_versions if (release := _major_minor(version))}
    if match := re.search(r"/nvidia/cu(\d+)/", path):
        major = int(match.group(1))
        same_major = sorted(release for release in releases if release[0] == major)
        return f"{same_major[-1][0]}.{same_major[-1][1]}" if same_major else f"{major}.0"
    if len(releases) == 1:
        release = next(iter(releases))
        return f"{release[0]}.{release[1]}"
    return None


def _toolkit_release_supports(
    path: str,
    profile: HardwareProfile,
    image_cuda_versions: tuple[str, ...],
) -> bool:
    """Return whether a CUDA Toolkit library ships with a release that supports the GPU."""
    if not _is_cuda_toolkit_library(path) or not profile.minimum_cuda_version:
        return False
    release = _toolkit_release(path, image_cuda_versions)
    return bool(release) and _major_minor(release) >= _major_minor(profile.minimum_cuda_version)


def _declares_other_architecture_only(path: str, targets: tuple[str, ...]) -> bool:
    """Return whether a module holds only architecture-specific code its file name declares.

    For example torchao's ``_C_cutlass_90a.abi3.so`` carries only ``sm_90a`` cubins: it is a
    Hopper-only alternate implementation, the same pattern as FlashInfer's ``_sm90`` modules.
    """
    if not targets or not all(target.endswith("a") for target in targets):
        return False
    basename = PurePosixPath(path.lower()).name
    numbers = {re.sub(r"\D", "", target) for target in targets}
    return any(
        re.search(rf"(?:^|[_.-])(?:sm_?)?{number}a(?:[_.-]|$)", basename) for number in numbers
    )


def _cuda_exception_component(path: str) -> str:
    """Return a stable, customer-facing component name from an image path."""
    normalized = path.lower()
    if "flashinfer" in normalized:
        return "FlashInfer"
    if "/hpcx/hcoll/" in normalized or "hmca_gpu_cuda" in normalized:
        return "HPC-X HCOLL"
    if "vllm" in normalized and ("flash_attn" in normalized or "_vllm_fa" in normalized):
        return "vLLM FlashAttention"
    if "sgl_kernel" in normalized:
        return "SGLang kernels"
    if "xformers" in normalized:
        return "XFormers"
    if _is_cuda_toolkit_library(path):
        return "CUDA Toolkit libraries"
    if "nccl" in normalized or "nvshmem" in normalized:
        return "NVIDIA communication libraries"
    if "torch" in normalized:
        return "PyTorch extensions"
    if "jax" in normalized or "xla" in normalized:
        return "JAX/XLA extensions"
    return "Other CUDA extensions"


def _cuda_exception_assessment(
    path: str,
    kind: str,
    *,
    toolkit_release: str | None = None,
    minimum_cuda_version: str | None = None,
    targets: tuple[str, ...] = (),
) -> tuple[str, str, str]:
    """Classify an unresolved unit without claiming that the workload loads it."""
    component = _cuda_exception_component(path)
    if kind == "runtime-artifact" and _declares_other_architecture_only(path, targets):
        return component, "optional-alternate-gpu-implementation", "alternate-architecture-only"
    if (
        component == "CUDA Toolkit libraries"
        and kind == "runtime-artifact"
        and toolkit_release
        and minimum_cuda_version
    ):
        if _major_minor(toolkit_release) >= _major_minor(minimum_cuda_version):
            return (
                component,
                "vendor-library-supported-by-release",
                "toolkit-release-supports-target",
            )
        return component, "likely-runtime-compatibility-gap", "toolkit-release-predates-target"
    if kind == "architecture-module-group":
        return (
            component,
            "optional-alternate-gpu-implementation",
            "alternate-architecture-cache-without-compatible-variant",
        )
    if kind == "kernel-catalog":
        return (
            component,
            "optional-alternate-gpu-implementation",
            "kernel-catalog-without-compatible-member",
        )
    normalized = path.lower()
    if (
        component == "FlashInfer"
        and "flashinfer_jit_cache" in normalized
        and re.search(r"(?:^|[/_.-])(?:sm)?\d{2,3}[af]?(?:[/_.-]|$)", normalized)
    ):
        return (
            component,
            "optional-alternate-gpu-implementation",
            "alternate-architecture-only",
        )
    if component in {
        "CUDA Toolkit libraries",
        "HPC-X HCOLL",
        "NVIDIA communication libraries",
    }:
        return component, "likely-runtime-compatibility-gap", "no-compatible-target"
    return (
        component,
        "runtime-relevance-not-statically-established",
        "no-compatible-target",
    )


def _cuda_exception_reason_summary(reason: str) -> str:
    return {
        "alternate-architecture-cache-without-compatible-variant": (
            "Alternate-architecture cache has no target-compatible sibling or PTX/JIT fallback"
        ),
        "kernel-catalog-without-compatible-member": (
            "Kernel catalog has no target-compatible member or PTX/JIT fallback"
        ),
        "no-compatible-target": "Inspected binary has no compatible embedded CUDA target",
        "toolkit-release-supports-target": (
            "Shipped with a CUDA release that supports this GPU; NVIDIA packages these kernels "
            "in ways cuobjdump does not fully list"
        ),
        "toolkit-release-predates-target": (
            "Shipped with a CUDA release older than this GPU's minimum supported release"
        ),
        "alternate-architecture-only": (
            "Architecture-specific implementation has no identified compatible sibling or "
            "PTX/JIT fallback"
        ),
    }[reason]


def _cuda_compatibility_exception_groups(
    exceptions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate complete exception records for concise text and HTML summaries."""
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for item in exceptions:
        key = tuple(
            str(item[field])
            for field in (
                "target",
                "platform",
                "component",
                "classification",
                "reason",
                "reason_summary",
                "status_effect",
            )
        )
        group = grouped.setdefault(
            key,
            {
                "target": item["target"],
                "platform": item["platform"],
                "component": item["component"],
                "classification": item["classification"],
                "reason": item["reason"],
                "reason_summary": item["reason_summary"],
                "status_effect": item["status_effect"],
                "exception_count": 0,
                "file_count": 0,
                "artifact_count": 0,
                "catalog_count": 0,
                "representative_paths": [],
                "inventory_reference": "observations.cuda_compatibility_exceptions",
            },
        )
        group["exception_count"] = int(group["exception_count"]) + 1
        if item["kind"] == "runtime-artifact":
            group["artifact_count"] = int(group["artifact_count"]) + 1
            paths = [str(item["path"])]
        else:
            group["catalog_count"] = int(group["catalog_count"]) + 1
            member_paths = item.get("member_paths", [])
            paths = [str(value) for value in member_paths] or [str(item["path"])]
        group["file_count"] = int(group["file_count"]) + len(paths)
        representatives = group["representative_paths"]
        if isinstance(representatives, list):
            for path in paths:
                if path not in representatives and len(representatives) < 3:
                    representatives.append(path)
    return sorted(
        grouped.values(),
        key=lambda item: (
            str(item["target"]),
            str(item["classification"]),
            str(item["component"]),
            str(item["reason"]),
        ),
    )


def _cuda_compatibility_exceptions(
    rootfs: RootfsScan,
    profile: HardwareProfile,
    *,
    platform: str,
    analyzer_matches_image: bool,
    image_cuda_versions: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Return every verified runtime unit lacking a compatible target."""
    if not analyzer_matches_image:
        return []
    target = profile.compute_capability_number
    exceptions: list[dict[str, object]] = []
    for catalog in sorted(rootfs.kernel_catalogs, key=lambda item: item.group):
        result = summarize_compatibility(catalog.verified_targets, target)
        fully_inspected = (
            catalog.target_variants
            and catalog.samples_inspected >= catalog.target_variants
            and not catalog.samples_incomplete
        )
        if (not result or not result.compatible) and fully_inspected:
            kind = (
                "architecture-module-group"
                if catalog.kind == "architecture-module"
                else "kernel-catalog"
            )
            component, classification, reason = _cuda_exception_assessment(catalog.group, kind)
            member_paths = list(catalog.member_paths) or sorted(
                artifact.path
                for artifact in rootfs.cuda_artifacts
                if artifact.role == "kernel-catalog"
                and kernel_catalog_group(artifact.path) == catalog.group
            )
            exceptions.append(
                {
                    "target": profile.id,
                    "platform": platform,
                    "kind": kind,
                    "path": catalog.group,
                    "targets": list(catalog.verified_targets),
                    "size_bytes": None,
                    "member_count": catalog.member_count,
                    "member_paths": member_paths,
                    "component": component,
                    "classification": classification,
                    "reason": reason,
                    "reason_summary": _cuda_exception_reason_summary(reason),
                    "status_effect": "review-required",
                    "has_compatible_sibling": False,
                    "has_jit_fallback": False,
                }
            )
    for artifact in sorted(rootfs.cuda_artifacts, key=lambda item: item.path):
        if not (
            artifact.role == "runtime"
            and artifact.method == "cuobjdump"
            and artifact.complete
            and artifact.targets
        ):
            continue
        result = summarize_compatibility(artifact.targets, target)
        if not result or not result.compatible:
            component, classification, reason = _cuda_exception_assessment(
                artifact.path,
                "runtime-artifact",
                toolkit_release=_toolkit_release(artifact.path, image_cuda_versions),
                minimum_cuda_version=profile.minimum_cuda_version,
                targets=artifact.targets,
            )
            exceptions.append(
                {
                    "target": profile.id,
                    "platform": platform,
                    "kind": "runtime-artifact",
                    "path": artifact.path,
                    "targets": list(artifact.targets),
                    "size_bytes": artifact.size_bytes,
                    "member_count": None,
                    "member_paths": [],
                    "component": component,
                    "classification": classification,
                    "reason": reason,
                    "reason_summary": _cuda_exception_reason_summary(reason),
                    "status_effect": (
                        "informational"
                        if classification == "vendor-library-supported-by-release"
                        else "review-required"
                    ),
                    "has_compatible_sibling": False,
                    "has_jit_fallback": False,
                }
            )
    return exceptions


def _incomplete_reason_counts(rootfs: RootfsScan) -> dict[str, int]:
    """Aggregate incomplete cuobjdump inspections into customer-facing reason groups."""
    counts = {"timeout": 0, "analyzer-error": 0, "io-error": 0, "other": 0}
    for artifact in rootfs.cuda_artifacts:
        if artifact.method != "cuobjdump" or artifact.complete:
            continue
        reason = artifact.incomplete_reason or "other"
        counts[reason if reason in counts else "other"] += 1
    return counts


def _cuda_artifact_findings(
    rootfs: RootfsScan,
    profile: HardwareProfile,
    image_cuda_versions: tuple[str, ...],
    *,
    max_scan_bytes: int,
    max_cuobjdump_files: int | None,
    max_artifact_bytes: int,
    cuobjdump_timeout_seconds: int,
) -> tuple[Finding, ...]:
    """Evaluate each successfully enumerated CUDA artifact independently for a target GPU."""
    findings: list[Finding] = []
    analyzer_mismatch = _cuobjdump_is_older(rootfs.cuobjdump_version, image_cuda_versions)
    if analyzer_mismatch:
        analyzer, image = analyzer_mismatch
        findings.append(
            Finding(
                id="GPF-CUDA-015",
                severity="warning",
                category="evidence",
                title="CUDA artifact analyzer predates the image toolkit",
                evidence=(
                    f"cuobjdump CUDA {analyzer[0]}.{analyzer[1]} inspected artifacts from an "
                    f"image containing CUDA {image[0]}.{image[1]}. Older analyzers may reject or "
                    "misread newer fat-binary formats and GPU targets."
                ),
                remediation=(
                    f"Repeat the scan with cuobjdump CUDA {image[0]}.{image[1]} or newer before "
                    "using per-artifact compatibility findings."
                ),
            )
        )
    inspected_with_targets = tuple(
        artifact
        for artifact in rootfs.cuda_artifacts
        if artifact.method == "cuobjdump" and artifact.complete and artifact.targets
    )
    incompatible_runtime: list[CudaArtifactEvidence] = []
    incompatible_non_runtime: list[CudaArtifactEvidence] = []
    compatible_runtime: list[CudaArtifactEvidence] = []
    if not analyzer_mismatch:
        for artifact in inspected_with_targets:
            if artifact.role == "kernel-catalog":
                continue
            result = summarize_compatibility(artifact.targets, profile.compute_capability_number)
            if result and result.compatible:
                if artifact.role == "runtime":
                    compatible_runtime.append(artifact)
            elif _toolkit_release_supports(artifact.path, profile, image_cuda_versions):
                # Assessed by CUDA release in GPF-CUDA-020, not by its cuobjdump listing.
                continue
            elif artifact.role == "runtime":
                incompatible_runtime.append(artifact)
            else:
                incompatible_non_runtime.append(artifact)

    compatible_catalogs: list[KernelCatalogEvidence] = []
    incompatible_catalogs: list[KernelCatalogEvidence] = []
    if not analyzer_mismatch:
        for catalog in rootfs.kernel_catalogs:
            result = summarize_compatibility(
                catalog.verified_targets,
                profile.compute_capability_number,
            )
            if result and result.compatible:
                compatible_catalogs.append(catalog)
            elif (
                catalog.target_variants
                and catalog.samples_inspected >= catalog.target_variants
                and not catalog.samples_incomplete
            ):
                incompatible_catalogs.append(catalog)

    release_assessed = {
        artifact.path
        for artifact in inspected_with_targets
        if _toolkit_release_supports(artifact.path, profile, image_cuda_versions)
    }
    inspected_runtime_with_targets = tuple(
        artifact
        for artifact in inspected_with_targets
        if artifact.role == "runtime"
        and (artifact.path not in release_assessed or artifact in compatible_runtime)
    )

    precompiled_catalogs = [
        catalog for catalog in rootfs.kernel_catalogs if catalog.kind == "precompiled-cubin"
    ]
    dispatch_catalogs = [
        catalog for catalog in rootfs.kernel_catalogs if catalog.kind == "architecture-module"
    ]
    incompatible_precompiled_catalogs = [
        catalog for catalog in incompatible_catalogs if catalog.kind == "precompiled-cubin"
    ]
    incompatible_dispatch_catalogs = [
        catalog for catalog in incompatible_catalogs if catalog.kind == "architecture-module"
    ]
    compatible_precompiled_catalogs = [
        catalog for catalog in compatible_catalogs if catalog.kind == "precompiled-cubin"
    ]
    compatible_dispatch_catalogs = [
        catalog for catalog in compatible_catalogs if catalog.kind == "architecture-module"
    ]
    compatible_inspection_units = len(compatible_runtime) + len(compatible_catalogs)
    incompatible_inspection_units = len(incompatible_runtime) + len(incompatible_catalogs)
    total_inspection_units = compatible_inspection_units + incompatible_inspection_units
    compatibility_exceptions = _cuda_compatibility_exceptions(
        rootfs,
        profile,
        platform="",
        analyzer_matches_image=not bool(analyzer_mismatch),
        image_cuda_versions=image_cuda_versions,
    )
    review_exceptions = [
        item for item in compatibility_exceptions if item["status_effect"] == "review-required"
    ]
    exception_groups = _cuda_compatibility_exception_groups(review_exceptions)

    toolkit_by_release = [
        item
        for item in compatibility_exceptions
        if item["classification"] == "vendor-library-supported-by-release"
    ]
    if toolkit_by_release:
        releases = {
            str(item["path"]): _toolkit_release(str(item["path"]), image_cuda_versions) or "unknown"
            for item in toolkit_by_release
        }
        findings.append(
            Finding(
                id="GPF-CUDA-020",
                severity="info",
                category="compatibility",
                title="CUDA Toolkit libraries were assessed by CUDA release",
                evidence=(
                    f"{len(toolkit_by_release)} CUDA Toolkit library file(s) list no compatible "
                    f"target with cuobjdump but ship with CUDA "
                    f"{', '.join(sorted(set(releases.values())))}, which meets the "
                    f"{profile.display_name} minimum of CUDA {profile.minimum_cuda_version}. "
                    "NVIDIA packages kernels for these libraries in ways cuobjdump does not fully "
                    "list (for example cuFFT's vendor-packaged section and single placeholder "
                    "cubins in cuFFTW and NPP core libraries)."
                ),
                remediation=(
                    "No image change is needed for these libraries. Exercise them in the hardware "
                    "test plan if the workload uses them."
                ),
                details=tuple(
                    FindingDetail(label=path, value=f"CUDA {release}")
                    for path, release in sorted(releases.items())[:8]
                ),
            )
        )
    required_without_code = [
        artifact for artifact in incompatible_runtime if is_required_runtime_library(artifact.path)
    ]
    if required_without_code:
        findings.append(
            Finding(
                id="GPF-CUDA-021",
                severity="blocker",
                category="compatibility",
                title="A library every GPU job loads has no code for this GPU",
                evidence=(
                    "cuobjdump listed every embedded target in "
                    + "; ".join(
                        f"{artifact.path} ({', '.join(artifact.targets)})"
                        for artifact in required_without_code
                    )
                    + f". None is a cubin or PTX that runs on sm_{profile.compute_capability_number}. "
                    "PyTorch loads libtorch_cuda and libc10_cuda for any GPU work, and multi-GPU "
                    "jobs load NCCL for every collective."
                ),
                remediation=(
                    f"Use a framework and NCCL release built for "
                    f"{', '.join(profile.native_sm_targets)}, or one that retains PTX the driver "
                    "can compile for this GPU."
                ),
            )
        )
    uninspected_core = [
        (candidate.path, candidate.size_bytes, f"not inspected: {candidate.reason}")
        for candidate in rootfs.skipped_cuda_candidates
        if is_core_framework_library(candidate.path)
    ] + [
        (
            artifact.path,
            artifact.size_bytes,
            f"incomplete: {artifact.diagnostic or artifact.incomplete_reason or 'analyzer error'}",
        )
        for artifact in rootfs.cuda_artifacts
        if artifact.method == "cuobjdump"
        and not artifact.complete
        and is_core_framework_library(artifact.path)
    ]
    if uninspected_core:
        findings.append(
            Finding(
                id="GPF-CUDA-019",
                severity="info",
                category="evidence",
                title="Core framework libraries were not inspected",
                evidence=(
                    f"{len(uninspected_core)} framework library file(s) that GPU jobs normally "
                    "load (PyTorch CUDA, NCCL, FlashAttention, Transformer Engine) have no "
                    "cuobjdump target listing, so the image cannot be assessed as ready."
                ),
                remediation=(
                    "Raise --max-artifact-mb above the largest listed size or resolve the named "
                    "analyzer error, then repeat the scan."
                ),
                details=tuple(
                    FindingDetail(
                        label=path,
                        value=f"{size / (1024**2):.0f} MiB",
                        context=reason,
                    )
                    for path, size, reason in uninspected_core[:8]
                ),
            )
        )

    if compatible_inspection_units and not incompatible_inspection_units:
        findings.append(
            Finding(
                id="GPF-CUDA-009",
                severity="info",
                category="compatibility",
                title="Inspected runtime CUDA artifacts are compatible",
                evidence=(
                    f"cuobjdump enumerated targets in {total_inspection_units} runtime "
                    "inspection unit(s), including grouped dispatch catalogs; every unit "
                    f"contains a compatible target for sm_{profile.compute_capability_number}."
                ),
                remediation="Retain this per-file evidence with the validated image digest.",
            )
        )
    elif compatible_inspection_units:
        findings.append(
            Finding(
                id="GPF-CUDA-010",
                severity="info",
                category="compatibility",
                title="Some inspected CUDA artifacts are compatible",
                evidence=(
                    f"cuobjdump compatibility summary for sm_{profile.compute_capability_number}."
                ),
                remediation="Review the incompatible runtime artifacts listed in GPF-CUDA-016.",
                details=(
                    FindingDetail(
                        label="Runtime inspection units with enumerated targets",
                        value=total_inspection_units,
                    ),
                    FindingDetail(
                        label="Compatible inspection units",
                        value=compatible_inspection_units,
                    ),
                    FindingDetail(
                        label="Inspection units requiring review",
                        value=incompatible_inspection_units,
                    ),
                ),
            )
        )
    if rootfs.kernel_catalogs:
        catalog_details = [
            FindingDetail(
                label="Precompiled cubin catalog groups",
                value=len(precompiled_catalogs),
            ),
            FindingDetail(
                label="Shape-specific catalog cubins",
                value=rootfs.kernel_catalog_members,
            ),
            FindingDetail(
                label="Duplicate catalog alternatives collapsed",
                value=rootfs.kernel_catalog_grouped_members,
            ),
            FindingDetail(
                label="Architecture-variant module groups",
                value=len(dispatch_catalogs),
            ),
            FindingDetail(
                label="Modules grouped as dispatch alternatives",
                value=rootfs.dispatch_module_members,
            ),
            FindingDetail(
                label="Catalog target variants inspected",
                value=sum(catalog.samples_inspected for catalog in rootfs.kernel_catalogs),
            ),
            FindingDetail(
                label="Compatible precompiled cubin catalogs",
                value=len(compatible_precompiled_catalogs),
            ),
            FindingDetail(
                label="Precompiled cubin catalogs without compatible code",
                value=len(incompatible_precompiled_catalogs),
            ),
            FindingDetail(
                label="Compatible architecture-module groups",
                value=len(compatible_dispatch_catalogs),
            ),
            FindingDetail(
                label="Architecture-module groups without compatible code",
                value=len(incompatible_dispatch_catalogs),
            ),
        ]
        findings.append(
            Finding(
                id="GPF-CUDA-018",
                severity="info",
                category="compatibility",
                title="Precompiled CUDA dispatch catalogs were grouped",
                evidence=(
                    f"Grouped {rootfs.kernel_catalog_members} selectable cubin member(s) into "
                    f"{len(precompiled_catalogs)} cubin catalog(s), and "
                    f"{rootfs.dispatch_module_members} architecture-selected module(s) into "
                    f"{len(dispatch_catalogs)} dispatch group(s). Compatibility is evaluated "
                    "across the selectable targets in each group."
                ),
                remediation=(
                    "Use catalog-level results to assess packaged dispatch coverage. This does "
                    "not prove that every operation, shape, or data type has an optimized path; "
                    "validate representative workload operations on hardware."
                ),
                details=tuple(catalog_details),
            )
        )

    if review_exceptions:
        classification_labels = {
            "likely-runtime-compatibility-gap": "Likely runtime compatibility gap",
            "optional-alternate-gpu-implementation": "Optional alternate-GPU implementation",
            "runtime-relevance-not-statically-established": (
                "Runtime relevance not statically established"
            ),
        }
        incompatible_details = tuple(
            FindingDetail(
                label=str(group["component"]),
                value=int(group["file_count"]),
                context=(
                    f"{classification_labels[str(group['classification'])]}; "
                    f"{group['reason_summary']}; {group['exception_count']} review unit(s)"
                ),
            )
            for group in exception_groups
        ) + (
            FindingDetail(
                label="Full inventory",
                value="JSON or HTML appendix",
                context="JSON: observations.cuda_compatibility_exceptions",
            ),
        )
        findings.append(
            Finding(
                id="GPF-CUDA-016",
                severity="warning",
                category="compatibility",
                title="Potentially incompatible CUDA artifacts require review",
                evidence=(
                    f"cuobjdump identified {len(review_exceptions)} binary review unit(s), "
                    f"representing {sum(int(group['file_count']) for group in exception_groups)} "
                    f"file(s), without a compatible target for "
                    f"sm_{profile.compute_capability_number}. Results are grouped by component "
                    "and reason; static inspection cannot prove that the workload loads an "
                    "artifact or selects a catalog path."
                ),
                remediation=(
                    "Prioritize likely runtime gaps, then establish whether workload-specific "
                    "extensions are loaded. Use the complete JSON inventory or expandable HTML "
                    "appendix for paths. Select a target-supported vendor release or rebuild "
                    "application-owned extensions with a native cubin or compatible retained PTX."
                ),
                details=incompatible_details,
            )
        )
    runtime_with_targets = inspected_runtime_with_targets
    coverage_target_sets = [artifact.targets for artifact in runtime_with_targets]
    coverage_target_sets.extend(catalog.verified_targets for catalog in rootfs.kernel_catalogs)
    if coverage_target_sets and not analyzer_mismatch:
        code_path_counts = {
            "exact-architecture-cubin": 0,
            "exact-family-cubin": 0,
            "exact-base-cubin": 0,
            "compatible-family-cubin": 0,
            "compatible-base-cubin": 0,
            "ptx-jit": 0,
            "incompatible": 0,
        }
        code_path_targets = {code_path: set() for code_path in code_path_counts}
        hardware = profile.compute_capability_number
        for targets in coverage_target_sets:
            best_code_path = best_cuda_code_path(targets, hardware)
            code_path_counts[best_code_path] += 1
            code_path_targets[best_code_path].update(
                target
                for target in targets
                if classify_cuda_code_path(target, hardware) == best_code_path
            )

        def target_context(code_path: str) -> str | None:
            targets = ", ".join(sorted(code_path_targets[code_path]))
            return targets or None

        coverage_details: list[FindingDetail] = []
        if f"sm_{hardware}a" in profile.native_sm_targets:
            coverage_details.append(
                FindingDetail(
                    label="Architecture-specific cubin",
                    value=code_path_counts["exact-architecture-cubin"],
                    context=f"sm_{hardware}a",
                )
            )
        if f"sm_{hardware}f" in profile.native_sm_targets:
            coverage_details.append(
                FindingDetail(
                    label="Family-specific cubin",
                    value=code_path_counts["exact-family-cubin"],
                    context=f"sm_{hardware}f",
                )
            )
        coverage_details.extend(
            (
                FindingDetail(
                    label="Exact base cubin",
                    value=code_path_counts["exact-base-cubin"],
                    context=f"sm_{hardware}",
                ),
                FindingDetail(
                    label="Compatible older family cubin",
                    value=code_path_counts["compatible-family-cubin"],
                    context=target_context("compatible-family-cubin"),
                ),
                FindingDetail(
                    label="Compatible older base cubin",
                    value=code_path_counts["compatible-base-cubin"],
                    context=target_context("compatible-base-cubin"),
                ),
                FindingDetail(
                    label="PTX JIT fallback",
                    value=code_path_counts["ptx-jit"],
                    context=target_context("ptx-jit"),
                ),
                FindingDetail(
                    label="No compatible code",
                    value=code_path_counts["incompatible"],
                ),
            )
        )
        target_specialized = (
            code_path_counts["exact-architecture-cubin"] + code_path_counts["exact-family-cubin"]
        )
        if target_specialized:
            interpretation = (
                f"{target_specialized} binary unit(s) contain target-specialized cubins for "
                "this GPU. Binary units are files or grouped dispatch catalogs, not operations, "
                "features, or distinct workload code paths."
            )
        elif code_path_counts["exact-base-cubin"]:
            interpretation = (
                "An exact base cubin is present, but no exact family- or architecture-specific "
                "cubin was observed in the inspected runtime artifacts."
            )
        elif (
            code_path_counts["compatible-family-cubin"] or code_path_counts["compatible-base-cubin"]
        ):
            interpretation = (
                "The inspected compatible artifacts rely on cubins built for earlier targets; "
                "compatibility does not establish target-specific optimization."
            )
        elif code_path_counts["ptx-jit"]:
            interpretation = (
                "The inspected compatible artifacts rely on PTX JIT; compatibility does not "
                "establish target-specific optimization."
            )
        else:
            interpretation = (
                "No compatible embedded code was found in the inspected binary units; review "
                "the grouped exceptions before hardware testing."
            )
        findings.append(
            Finding(
                id="GPF-CUDA-017",
                severity="info",
                category="compatibility",
                title="CUDA code-path optimization coverage",
                evidence=(
                    f"Best available code path for each of {len(coverage_target_sets)} inspected "
                    f"binary unit(s) on sm_{hardware}; each file counts once and each grouped "
                    "dispatch catalog counts once. These counts do not represent operations, "
                    f"features, or distinct workload code paths. "
                    f"{interpretation}"
                ),
                remediation=(
                    "Use this breakdown to prioritize artifacts that the workload actually loads, "
                    "then benchmark representative operations on the target hardware; cubin "
                    "presence alone does not prove performance."
                ),
                details=tuple(coverage_details),
            )
        )
    if incompatible_non_runtime:
        examples = "; ".join(
            f"{artifact.path} ({artifact.role}): {', '.join(artifact.targets)}"
            for artifact in incompatible_non_runtime[:8]
        )
        findings.append(
            Finding(
                id="GPF-CUDA-013",
                severity="info",
                category="evidence",
                title="Non-runtime CUDA artifacts lack a compatible target",
                evidence=(
                    f"Found {len(incompatible_non_runtime)} test/tooling artifact(s) without a "
                    f"compatible sm_{profile.compute_capability_number} target: {examples}."
                ),
                remediation=(
                    "No runtime action is required unless these test or tooling artifacts are part "
                    "of the intended workload path."
                ),
            )
        )
    if not rootfs.cuobjdump_available or not inspected_with_targets:
        reason = (
            "cuobjdump was not available on the GPU Preflight host"
            if not rootfs.cuobjdump_available
            else "cuobjdump did not successfully enumerate targets in any CUDA artifacts"
        )
        findings.append(
            Finding(
                id="GPF-CUDA-011",
                severity="info",
                category="evidence",
                title="Per-artifact CUDA evidence is incomplete",
                evidence=(
                    f"{reason}; {rootfs.cuda_candidate_files} candidate file(s) were observed and "
                    f"{len(rootfs.cuda_artifacts)} artifact record(s) were retained."
                ),
                remediation=(
                    "Install NVIDIA cuobjdump or pass --cuobjdump PATH, increase bounded scan limits "
                    "when needed, and repeat the filesystem scan."
                ),
            )
        )

    marker_only = rootfs.marker_only_files
    if marker_only or rootfs.cuobjdump_incomplete:
        marker_records = tuple(
            artifact for artifact in rootfs.cuda_artifacts if artifact.method == "marker-scan"
        )
        marker_roles = {
            role: sum(artifact.role == role for artifact in marker_records)
            for role in ("runtime", "kernel-catalog", "test", "tooling", "host-runtime")
        }
        incomplete_artifacts = tuple(
            artifact
            for artifact in rootfs.cuda_artifacts
            if artifact.method == "cuobjdump" and not artifact.complete
        )
        reason_counts = _incomplete_reason_counts(rootfs)
        next_steps: list[str] = []
        if marker_only:
            next_steps.append(
                "Treat unverified target strings as discovery hints, not embedded device-code "
                "proof."
            )
        if reason_counts["timeout"]:
            suggested_timeout = max(cuobjdump_timeout_seconds * 2, 120)
            next_steps.append(
                f"Retry timed-out artifacts with --cuobjdump-timeout {suggested_timeout}."
            )
        if reason_counts["analyzer-error"] or reason_counts["io-error"] or reason_counts["other"]:
            next_steps.append(
                "Review the named analyzer diagnostic; changing scan limits will not fix a "
                "format or parser error."
            )
        evidence_details = [
            FindingDetail(label="Unverified target-string records", value=marker_only),
            FindingDetail(label="Runtime", value=marker_roles["runtime"]),
            FindingDetail(
                label="Kernel catalogs",
                value=marker_roles["kernel-catalog"],
            ),
            FindingDetail(label="Tests", value=marker_roles["test"]),
            FindingDetail(label="Tooling", value=marker_roles["tooling"]),
            FindingDetail(label="Host-side runtime", value=marker_roles["host-runtime"]),
            FindingDetail(
                label="Incomplete cuobjdump inspections",
                value=rootfs.cuobjdump_incomplete,
            ),
            FindingDetail(label="Timeouts", value=reason_counts["timeout"]),
            FindingDetail(label="Analyzer errors", value=reason_counts["analyzer-error"]),
            FindingDetail(label="I/O errors", value=reason_counts["io-error"]),
            FindingDetail(label="Other incomplete inspections", value=reason_counts["other"]),
        ]
        evidence_details.extend(
            FindingDetail(
                label=artifact.path,
                value=artifact.diagnostic or "cuobjdump did not complete",
                context="incomplete inspection",
            )
            for artifact in incomplete_artifacts[:3]
        )
        findings.append(
            Finding(
                id="GPF-CUDA-012",
                severity="info",
                category="evidence",
                title="Some CUDA target records are unverified",
                evidence=(
                    f"{marker_only} artifact(s) contain target-like strings that cuobjdump did "
                    f"not verify; {rootfs.cuobjdump_incomplete} cuobjdump inspection(s) were "
                    "incomplete."
                ),
                remediation=" ".join(next_steps),
                details=tuple(evidence_details),
            )
        )
    unresolved_runtime = (
        rootfs.skipped_no_cuobjdump
        + rootfs.skipped_scan_budget
        + rootfs.skipped_artifact_limit
        + rootfs.skipped_file_limit
    )
    if unresolved_runtime or rootfs.cuobjdump_incomplete:
        completed = max(0, rootfs.cuobjdump_files_analyzed - rootfs.cuobjdump_incomplete)
        coverage_denominator = completed + unresolved_runtime + rootfs.cuobjdump_incomplete
        coverage_percent = (
            round(completed / coverage_denominator * 100, 1) if coverage_denominator else None
        )
        rerun_flags: list[str] = []
        if coverage_percent is not None and coverage_percent >= 90:
            rerun_guidance = (
                f"Artifact inspection already covers {coverage_percent:.1f}% of loadable runtime "
                "inspection units. Review the remaining paths in the report appendix and increase only "
                "the relevant bound if the workload may load one of them."
            )
        elif rootfs.skipped_scan_budget:
            current_scan_mb = max(1, max_scan_bytes // (1024 * 1024))
            rerun_flags.append(f"--max-scan-mb {current_scan_mb * 4}")
            prospective_files = (
                rootfs.cuobjdump_files_analyzed
                + rootfs.skipped_scan_budget
                + rootfs.skipped_file_limit
            )
            suggested_file_limit = 1
            while suggested_file_limit < prospective_files:
                suggested_file_limit *= 2
            if max_cuobjdump_files is not None and suggested_file_limit > max_cuobjdump_files:
                rerun_flags.append(f"--max-cuobjdump-files {suggested_file_limit}")
        elif rootfs.skipped_file_limit:
            if max_cuobjdump_files is not None:
                rerun_flags.append(f"--max-cuobjdump-files {max_cuobjdump_files * 2}")
            else:
                rerun_flags.append("--exhaustive")
        if rootfs.skipped_artifact_limit:
            current_artifact_mb = max(1, max_artifact_bytes // (1024 * 1024))
            rerun_flags.append(f"--max-artifact-mb {current_artifact_mb * 2}")
        if rootfs.skipped_no_cuobjdump:
            rerun_flags.append("--cuobjdump /path/to/cuobjdump")
        if coverage_percent is not None and coverage_percent >= 90:
            pass
        elif rerun_flags:
            rerun_guidance = (
                f"For a deeper rerun, start with {' '.join(rerun_flags)}; this is a bounded "
                "starting point, so increase again if the new report still shows skipped "
                "runtime candidates. Limits with zero skips do not need adjustment."
            )
        else:
            rerun_guidance = (
                "Review the named incomplete inspection and its diagnostic before relying on a "
                "positive compatibility assessment."
            )
        findings.append(
            Finding(
                id="GPF-CUDA-014",
                severity="info",
                category="evidence",
                title="Runtime CUDA artifact inspection has bounded gaps",
                evidence=(
                    f"Coverage is bounded: {unresolved_runtime} eligible runtime inspection "
                    "unit(s) "
                    "were not inspected with cuobjdump."
                ),
                remediation=rerun_guidance,
                details=(
                    FindingDetail(
                        label="Observed CUDA candidates",
                        value=rootfs.cuda_candidate_files,
                    ),
                    FindingDetail(
                        label="Inspected with cuobjdump",
                        value=rootfs.cuobjdump_files_analyzed,
                    ),
                    FindingDetail(
                        label="Role-excluded",
                        value=rootfs.skipped_non_runtime,
                        context="tests, tooling, or host-side runtime",
                    ),
                    FindingDetail(
                        label="Not inspected: scan budget",
                        value=rootfs.skipped_scan_budget,
                    ),
                    FindingDetail(
                        label="Not inspected: artifact-size limit",
                        value=rootfs.skipped_artifact_limit,
                    ),
                    FindingDetail(
                        label="Not inspected: file-count limit",
                        value=rootfs.skipped_file_limit,
                    ),
                    FindingDetail(
                        label="Not inspected: analyzer unavailable",
                        value=rootfs.skipped_no_cuobjdump,
                    ),
                    FindingDetail(
                        label="Incomplete cuobjdump inspections",
                        value=rootfs.cuobjdump_incomplete,
                    ),
                ),
            )
        )
    return tuple(findings)


def _select_manifest(
    manifests: ImageManifestSet,
    facts: ImageFacts,
    profile: HardwareProfile,
) -> PlatformManifest | None:
    if facts.final_platform:
        return next(
            (item for item in manifests.platforms if item.platform == facts.final_platform),
            None,
        )
    return next(
        (
            item
            for item in manifests.platforms
            if _architecture(item.platform) in profile.cpu_architectures
        ),
        None,
    )


def _append_global_finding(
    report: InspectionReport,
    finding: Finding,
) -> InspectionReport:
    global_findings = (*report.global_findings, finding)
    targets = tuple(
        refresh_target(target, target.findings, global_findings) for target in report.targets
    )
    return replace(report, global_findings=global_findings, targets=targets)


def _finish_scan_report(
    report: InspectionReport,
    *,
    started_at: float,
    mode: str,
    settings: dict[str, object],
    capabilities: dict[str, object],
) -> InspectionReport:
    duration = round(time.monotonic() - started_at, 3)
    return replace(
        report,
        duration_seconds=duration,
        scan=ScanMetadata(
            mode=mode,
            duration_seconds=duration,
            settings=settings,
            capabilities=capabilities,
        ),
    )


def inspect_with_image(
    facts: ImageFacts,
    profiles: tuple[HardwareProfile, ...],
    scan_mode: str = "filesystem",
    max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    cuobjdump_mode: str = "auto",
    cuobjdump_timeout_seconds: int = 30,
    cuobjdump_fallback_mode: str = "auto",
    max_cuobjdump_files: int | None = DEFAULT_MAX_CUOBJDUMP_FILES,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    command_timeout_seconds: int = 3600,
    client: DockerImageClient | None = None,
) -> InspectionReport:
    """Enrich image-reference facts with registry, configuration, and rootfs evidence."""
    started_at = time.monotonic()
    if scan_mode not in VALID_SCAN_MODES:
        raise ValueError(f"Unsupported image scan mode: {scan_mode}")
    settings: dict[str, object] = {
        "max_candidate_bytes": max_scan_bytes,
        "max_artifact_bytes": max_artifact_bytes,
        "max_cuobjdump_files": max_cuobjdump_files,
        "cuobjdump_mode": cuobjdump_mode,
        "cuobjdump_fallback_mode": cuobjdump_fallback_mode,
        "cuobjdump_timeout_seconds": cuobjdump_timeout_seconds,
        "command_timeout_seconds": command_timeout_seconds,
    }
    capabilities: dict[str, object] = {
        "registry_manifest_inspected": False,
        "image_config_inspected": False,
        "filesystem_inspected": False,
        "cuobjdump_available": False,
        "cuobjdump_version": None,
    }
    base_report = inspect(facts, profiles)
    if scan_mode == "none":
        report = _append_global_finding(
            base_report,
            Finding(
                id="GPF-IMAGE-007",
                severity="info",
                category="evidence",
                title="Image contents were not inspected",
                evidence="Scan mode none used image-reference text only.",
                remediation=(
                    "Run a filesystem image inspection for configuration, package inventory, and "
                    "per-artifact CUDA evidence."
                ),
            ),
        )
        return _finish_scan_report(
            report,
            started_at=started_at,
            mode="none",
            settings=settings,
            capabilities=capabilities,
        )
    image_reference = facts.reference
    if client is None:
        client = DockerImageClient(
            command_timeout_seconds=command_timeout_seconds,
            cuobjdump_mode=cuobjdump_mode,
            cuobjdump_timeout_seconds=cuobjdump_timeout_seconds,
            max_cuobjdump_files=max_cuobjdump_files,
            max_artifact_bytes=max_artifact_bytes,
            cuobjdump_fallback_mode=cuobjdump_fallback_mode,
        )
        client.preflight(
            require_daemon=scan_mode in {"pull", "filesystem"},
            require_cuobjdump=scan_mode == "filesystem",
        )
    analyzer = getattr(client, "cuobjdump", None)
    if analyzer is not None and analyzer.fallback is not None:
        capabilities["cuobjdump_fallback_version"] = analyzer.fallback.version
    manifests = client.inspect_manifest(image_reference)

    capabilities["registry_manifest_inspected"] = True

    pulled: dict[str, PulledImageFacts] = {}
    if scan_mode in {"pull", "filesystem"}:
        selected_manifests = {
            selected.platform: selected
            for profile in profiles
            if (selected := _select_manifest(manifests, facts, profile)) is not None
        }
        for platform, selected in selected_manifests.items():
            content = client.pull_config(image_reference, selected)
            if scan_mode == "filesystem":
                content = client.scan_filesystem(
                    image_reference,
                    selected,
                    content,
                    max_scan_bytes,
                )
            pulled[platform] = content

    target_reports: list[TargetReport] = []
    selected_observations: dict[str, object] = {}
    image_packages: set[str] = set()
    installed_packages: set[str] = set()
    image_cuda_targets: set[str] = set()
    verified_image_cuda_targets: set[str] = set()
    unverified_image_cuda_target_strings: set[str] = set()
    image_cuda_versions: set[str] = set()
    image_nccl_versions: set[str] = set()
    nccl_evidence_observations: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    cuda_artifacts: list[dict[str, object]] = []
    cuda_compatibility_exceptions: list[dict[str, object]] = []
    skipped_cuda_candidates: list[dict[str, object]] = []
    kernel_catalog_observations: dict[tuple[str, str], dict[str, object]] = {}
    rootfs_statistics: dict[str, dict[str, object]] = {}
    cuda_target_summaries: dict[str, dict[str, object]] = {}
    for profile in profiles:
        selected = _select_manifest(manifests, facts, profile)
        image_findings: list[Finding] = []
        content = pulled.get(selected.platform) if selected else None
        analyzer_matches_image = not (
            content
            and content.rootfs
            and _cuobjdump_is_older(content.rootfs.cuobjdump_version, content.cuda_versions)
        )
        verified_content_targets = tuple(
            sorted(
                {
                    target
                    for artifact in (
                        content.rootfs.cuda_artifacts if content and content.rootfs else ()
                    )
                    if analyzer_matches_image
                    and artifact.method == "cuobjdump"
                    and artifact.complete
                    for target in artifact.targets
                }
            )
        )
        if selected and scan_mode in {"manifest", "pull"}:
            image_findings.append(
                Finding(
                    id="GPF-IMAGE-008",
                    severity="info",
                    category="evidence",
                    title="Image filesystem was not inspected",
                    evidence=(
                        "Manifest mode inspected registry metadata only."
                        if scan_mode == "manifest"
                        else "Pull mode inspected the image configuration but not its filesystem."
                    ),
                    remediation=(
                        "Run a filesystem image inspection to inventory packages and inspect "
                        "individual CUDA artifacts."
                    ),
                )
            )
        if selected:
            gibibytes = selected.compressed_size_bytes / (1024**3)
            image_findings.append(
                Finding(
                    id="GPF-IMAGE-001",
                    severity="info",
                    category="compatibility",
                    title="Compatible image manifest found",
                    evidence=(
                        f"{selected.platform} -> {selected.digest or 'digest unavailable'}; "
                        f"compressed layers approximately {gibibytes:.2f} GiB."
                    ),
                    remediation="Pin the validated platform digest for reproducible deployment.",
                )
            )
            selected_observations[profile.id] = {
                "platform": selected.platform,
                "digest": selected.digest,
                "compressed_size_bytes": selected.compressed_size_bytes,
            }
        else:
            available = ", ".join(item.platform for item in manifests.platforms)
            image_findings.append(
                Finding(
                    id="GPF-IMAGE-002",
                    severity="blocker",
                    category="compatibility",
                    title="No compatible image platform manifest",
                    evidence=(
                        f"Available platforms: {available}; {profile.display_name} expects "
                        f"{', '.join(profile.cpu_architectures)}."
                    ),
                    remediation="Publish a native image variant for the target CPU architecture.",
                )
            )

        target_facts = replace(
            facts,
            final_platform=selected.platform if selected else facts.final_platform,
            cuda_targets=tuple(
                sorted(
                    {
                        *facts.cuda_targets,
                        *(
                            verified_content_targets
                            if content and content.rootfs
                            else content.cuda_targets
                            if content
                            else ()
                        ),
                    }
                )
            ),
            cuda_versions=tuple(
                sorted(
                    {*facts.cuda_versions, *(content.cuda_versions if content else ())},
                    key=_version_sort_key,
                )
            ),
            nccl_versions=tuple(
                sorted(
                    {*facts.nccl_versions, *(content.nccl_versions if content else ())},
                    key=_version_sort_key,
                )
            ),
            detected_packages=tuple(
                sorted(
                    {
                        *facts.detected_packages,
                        *(content.known_packages if content else ()),
                    }
                )
            ),
            environment={
                **facts.environment,
                **(content.environment if content else {}),
            },
        )
        target_report = inspect(target_facts, (profile,)).targets[0]
        if selected and content:
            scan_detail = "image configuration"
            if content.rootfs:
                limit_notes = []
                if content.rootfs.truncated:
                    limit_notes.append("candidate-content scan limit reached")
                if content.rootfs.metadata_truncated:
                    limit_notes.append("metadata scan limit reached")
                limit_note = "; " + "; ".join(limit_notes) if limit_notes else ""
                scan_detail = (
                    f"merged rootfs: {content.rootfs.files_seen} files observed, "
                    f"{content.rootfs.bytes_scanned / (1024**2):.1f} MiB of candidate content and "
                    f"{content.rootfs.metadata_bytes_scanned / (1024**2):.1f} MiB of metadata read"
                    f"{limit_note}"
                )
                image_findings.extend(
                    _cuda_artifact_findings(
                        content.rootfs,
                        profile,
                        content.cuda_versions,
                        max_scan_bytes=max_scan_bytes,
                        max_cuobjdump_files=max_cuobjdump_files,
                        max_artifact_bytes=max_artifact_bytes,
                        cuobjdump_timeout_seconds=cuobjdump_timeout_seconds,
                    )
                )
                cuda_compatibility_exceptions.extend(
                    _cuda_compatibility_exceptions(
                        content.rootfs,
                        profile,
                        platform=content.platform,
                        analyzer_matches_image=analyzer_matches_image,
                        image_cuda_versions=content.cuda_versions,
                    )
                )
                rootfs_statistics[selected.platform] = {
                    "files_seen": content.rootfs.files_seen,
                    "candidate_bytes_scanned": content.rootfs.bytes_scanned,
                    "metadata_bytes_scanned": content.rootfs.metadata_bytes_scanned,
                    "candidate_scan_truncated": content.rootfs.truncated,
                    "metadata_scan_truncated": content.rootfs.metadata_truncated,
                    "cuda_candidate_files": content.rootfs.cuda_candidate_files,
                    "cuobjdump_version": content.rootfs.cuobjdump_version,
                    "cuobjdump_files_analyzed": content.rootfs.cuobjdump_files_analyzed,
                    "cuobjdump_targets_found": content.rootfs.cuobjdump_targets_found,
                    "cuobjdump_no_device_code": content.rootfs.cuobjdump_no_device_code,
                    "cuobjdump_incomplete": content.rootfs.cuobjdump_incomplete,
                    "cuobjdump_incomplete_reasons": _incomplete_reason_counts(content.rootfs),
                    "marker_only_files": content.rootfs.marker_only_files,
                    "kernel_catalog_groups": sum(
                        catalog.kind == "precompiled-cubin"
                        for catalog in content.rootfs.kernel_catalogs
                    ),
                    "kernel_catalog_members": content.rootfs.kernel_catalog_members,
                    "kernel_catalog_grouped_members": (
                        content.rootfs.kernel_catalog_grouped_members
                    ),
                    "dispatch_module_groups": content.rootfs.dispatch_module_groups,
                    "dispatch_module_members": content.rootfs.dispatch_module_members,
                    "kernel_catalog_target_variants": sum(
                        catalog.target_variants
                        for catalog in content.rootfs.kernel_catalogs
                        if catalog.kind == "precompiled-cubin"
                    ),
                    "dispatch_module_target_variants": sum(
                        catalog.target_variants
                        for catalog in content.rootfs.kernel_catalogs
                        if catalog.kind == "architecture-module"
                    ),
                    "skipped_no_cuobjdump": content.rootfs.skipped_no_cuobjdump,
                    "skipped_non_runtime": content.rootfs.skipped_non_runtime,
                    "skipped_scan_budget": content.rootfs.skipped_scan_budget,
                    "skipped_artifact_limit": content.rootfs.skipped_artifact_limit,
                    "skipped_file_limit": content.rootfs.skipped_file_limit,
                }
                cuda_artifacts.extend(
                    {
                        "platform": content.platform,
                        **asdict(artifact),
                    }
                    for artifact in content.rootfs.cuda_artifacts
                )
                skipped_cuda_candidates.extend(
                    {
                        "platform": content.platform,
                        **asdict(candidate),
                    }
                    for candidate in content.rootfs.skipped_cuda_candidates
                )
                for catalog in content.rootfs.kernel_catalogs:
                    kernel_catalog_observations[(content.platform, catalog.group)] = {
                        "platform": content.platform,
                        **asdict(catalog),
                    }
                verified_image_cuda_targets.update(verified_content_targets)
                unverified_image_cuda_target_strings.update(
                    target
                    for artifact in content.rootfs.cuda_artifacts
                    if artifact.method == "marker-scan"
                    for target in artifact.targets
                )
                capabilities["filesystem_inspected"] = True
                capabilities["cuobjdump_available"] = content.rootfs.cuobjdump_available
                capabilities["cuobjdump_version"] = content.rootfs.cuobjdump_version
                base_target = f"sm_{profile.compute_capability_number}"
                specialized_targets = tuple(
                    target for target in profile.native_sm_targets if target != base_target
                )
                cuda_target_summaries[profile.id] = {
                    "verified_target_count": len(verified_content_targets),
                    "base_target": base_target,
                    "base_target_present": base_target in verified_content_targets,
                    "specialized_targets_present": [
                        target
                        for target in specialized_targets
                        if target in verified_content_targets
                    ],
                    "specialized_targets_absent": [
                        target
                        for target in specialized_targets
                        if target not in verified_content_targets
                    ],
                }
            image_findings.append(
                Finding(
                    id="GPF-IMAGE-003",
                    severity="info",
                    category="evidence",
                    title="Pulled image content inspected without execution",
                    evidence=f"Inspected {scan_detail} for {selected.platform}.",
                    remediation=(
                        "Use hardware smoke tests to validate dynamic loading and runtime behavior."
                    ),
                )
            )
            image_packages.update(content.known_packages)
            installed_packages.update(content.installed_packages)
            image_cuda_targets.update(
                verified_content_targets if content.rootfs else content.cuda_targets
            )
            image_cuda_versions.update(content.cuda_versions)
            image_nccl_versions.update(content.nccl_versions)
            for evidence in content.nccl_evidence:
                path = evidence.path or ""
                key = (
                    content.platform,
                    evidence.version,
                    evidence.source,
                    evidence.name,
                    path,
                )
                nccl_evidence_observations[key] = {
                    "platform": content.platform,
                    **asdict(evidence),
                }
            capabilities["image_config_inspected"] = True
        findings = (*image_findings, *target_report.findings)
        artifact_counts: tuple[int, int] | None = None
        if content and content.rootfs and analyzer_matches_image:
            covered = max(
                0,
                content.rootfs.cuobjdump_files_analyzed - content.rootfs.cuobjdump_incomplete,
            )
            unresolved = (
                content.rootfs.cuobjdump_incomplete
                + content.rootfs.skipped_no_cuobjdump
                + content.rootfs.skipped_scan_budget
                + content.rootfs.skipped_artifact_limit
                + content.rootfs.skipped_file_limit
            )
            artifact_counts = (covered, covered + unresolved)
        target_reports.append(
            refresh_target(
                target_report,
                findings,
                base_report.global_findings,
                artifact_inspection_counts=artifact_counts,
            )
        )

    sorted_cuda_compatibility_exceptions = sorted(
        cuda_compatibility_exceptions,
        key=lambda item: (
            str(item["target"]),
            {
                "kernel-catalog": 0,
                "architecture-module-group": 1,
                "runtime-artifact": 2,
            }.get(str(item["kind"]), 3),
            str(item["path"]),
        ),
    )
    observations = {
        **base_report.observations,
        "image_scan_mode": scan_mode,
        "available_platforms": [item.platform for item in manifests.platforms],
        "selected_manifests": selected_observations,
        "image_packages": sorted(image_packages),
        "installed_packages": sorted(installed_packages)[:200],
        "image_cuda_targets": sorted(image_cuda_targets),
        "verified_image_cuda_targets": sorted(verified_image_cuda_targets),
        "unverified_image_cuda_target_strings": sorted(unverified_image_cuda_target_strings),
        "image_cuda_versions": sorted(image_cuda_versions, key=_version_sort_key),
        "image_nccl_versions": sorted(image_nccl_versions, key=_version_sort_key),
        "nccl_evidence": [
            nccl_evidence_observations[key]
            for key in sorted(
                nccl_evidence_observations,
                key=lambda item: (_version_sort_key(item[1]), item[0], *item[2:]),
            )
        ],
        "cuda_artifacts": cuda_artifacts,
        "cuda_compatibility_exceptions": sorted_cuda_compatibility_exceptions,
        "cuda_compatibility_exception_groups": _cuda_compatibility_exception_groups(
            sorted_cuda_compatibility_exceptions
        ),
        "skipped_cuda_candidates": skipped_cuda_candidates,
        "kernel_catalogs": [
            kernel_catalog_observations[key] for key in sorted(kernel_catalog_observations)
        ],
        "rootfs_statistics": rootfs_statistics,
        "cuda_target_summaries": cuda_target_summaries,
    }
    selected_platforms = {str(item["platform"]) for item in selected_observations.values()}
    if len(selected_platforms) == 1:
        observations["final_platform"] = next(iter(selected_platforms))
    elif selected_platforms:
        observations["final_platform"] = ", ".join(sorted(selected_platforms))
    report = replace(base_report, observations=observations, targets=tuple(target_reports))
    return _finish_scan_report(
        report,
        started_at=started_at,
        mode=scan_mode,
        settings=settings,
        capabilities=capabilities,
    )


def inspect_image_reference(
    reference: str,
    profiles: tuple[HardwareProfile, ...],
    platform: str | None = None,
    scan_mode: str = "filesystem",
    max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    cuobjdump_mode: str = "auto",
    cuobjdump_timeout_seconds: int = 30,
    cuobjdump_fallback_mode: str = "auto",
    max_cuobjdump_files: int | None = DEFAULT_MAX_CUOBJDUMP_FILES,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    command_timeout_seconds: int = 3600,
    client: DockerImageClient | None = None,
) -> InspectionReport:
    """Inspect a container reference directly, independently of host architecture."""
    if not reference.strip() or any(character.isspace() for character in reference):
        raise ValueError("Image reference must be a non-empty token without whitespace")
    if platform not in {None, "linux/amd64", "linux/arm64"}:
        raise ValueError("Platform must be linux/amd64 or linux/arm64")

    facts = ImageFacts(
        path=f"image://{reference}",
        reference=reference,
        final_platform=platform,
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        detected_packages=tuple(sorted(detect_packages(reference))),
        environment={},
    )
    return inspect_with_image(
        facts,
        profiles,
        scan_mode=scan_mode,
        max_scan_bytes=max_scan_bytes,
        cuobjdump_mode=cuobjdump_mode,
        cuobjdump_timeout_seconds=cuobjdump_timeout_seconds,
        cuobjdump_fallback_mode=cuobjdump_fallback_mode,
        max_cuobjdump_files=max_cuobjdump_files,
        max_artifact_bytes=max_artifact_bytes,
        command_timeout_seconds=command_timeout_seconds,
        client=client,
    )
