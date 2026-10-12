"""Bounded, non-executing inspection of CUDA objects with NVIDIA cuobjdump."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

CUDA_TARGET_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<kind>sm|compute)[_-]?(?P<number>\d{2,3})"
    r"(?P<suffix>[af]?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
FLASHINFER_JIT_CACHE_MARKER = "/flashinfer_jit_cache/jit_cache/"
FLASHINFER_SM_SELECTOR_RE = re.compile(
    r"_sm(?P<number>75|80|86|87|89|90|100|101|103|107|110|120|121)"
    r"(?P<suffix>[af]?)(?=_|$)",
    re.IGNORECASE,
)
FLASHINFER_TRAILING_SELECTOR_RE = re.compile(
    r"_(?P<number>75|80|86|87|89|90|100|101|103|107|110|120|121)"
    r"(?P<suffix>[af]?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CudaArtifactEvidence:
    """CUDA target evidence tied to one image filesystem artifact."""

    path: str
    size_bytes: int
    targets: tuple[str, ...]
    method: str
    complete: bool
    role: str
    diagnostic: str | None = None
    incomplete_reason: str | None = None


@dataclass(frozen=True)
class CuobjdumpTool:
    """Resolved host-side cuobjdump executable and bounded invocation settings."""

    executable: str
    version: str
    timeout_seconds: int

    def inspect(self, path: Path, image_path: str, size_bytes: int) -> CudaArtifactEvidence:
        try:
            result = subprocess.run(  # noqa: S603 - explicit argv; shell is disabled
                (self.executable, "--list-elf", "--list-ptx", str(path)),
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return CudaArtifactEvidence(
                path=image_path,
                size_bytes=size_bytes,
                targets=(),
                method="cuobjdump",
                complete=False,
                role=classify_artifact_role(image_path),
                diagnostic=f"cuobjdump timed out after {self.timeout_seconds} seconds",
                incomplete_reason="timeout",
            )
        except OSError as error:
            return CudaArtifactEvidence(
                path=image_path,
                size_bytes=size_bytes,
                targets=(),
                method="cuobjdump",
                complete=False,
                role=classify_artifact_role(image_path),
                diagnostic=f"cuobjdump could not inspect the artifact: {error}",
                incomplete_reason="io-error",
            )

        output = f"{result.stdout}\n{result.stderr}".replace(str(path), image_path)
        targets = cuda_targets_from_text(output)
        if result.returncode and "does not contain device code" in output.lower():
            return CudaArtifactEvidence(
                path=image_path,
                size_bytes=size_bytes,
                targets=(),
                method="cuobjdump",
                complete=True,
                role=classify_artifact_role(image_path),
                diagnostic="cuobjdump reported no embedded CUDA device code",
            )
        if result.returncode:
            detail = " ".join(output.split())[-500:] or f"exit status {result.returncode}"
            return CudaArtifactEvidence(
                path=image_path,
                size_bytes=size_bytes,
                targets=targets,
                method="cuobjdump",
                complete=False,
                role=classify_artifact_role(image_path),
                diagnostic=detail,
                incomplete_reason="analyzer-error",
            )
        return CudaArtifactEvidence(
            path=image_path,
            size_bytes=size_bytes,
            targets=targets,
            method="cuobjdump",
            complete=True,
            role=classify_artifact_role(image_path),
        )


def cuda_targets_from_text(text: str) -> tuple[str, ...]:
    """Normalize CUDA targets printed by cuobjdump."""
    return tuple(
        sorted(
            {
                f"{match.group('kind').lower()}_{int(match.group('number'))}"
                f"{match.group('suffix').lower()}"
                for match in CUDA_TARGET_TEXT_RE.finditer(text)
            }
        )
    )


def classify_artifact_role(image_path: str) -> str:
    """Estimate runtime relevance from well-known image path conventions."""
    normalized = f"/{image_path.lower().strip('/')}"
    basename = Path(normalized).name
    if any(
        marker in normalized
        for marker in (
            "/test/",
            "/tests/",
            "/testing/",
            "/test_data/",
            "/benchmarks/",
            "/perftest/",
            "/perftests/",
            "/perf_test/",
            "/perf-tests/",
        )
    ) or any(marker in basename for marker in ("perftest", "perf_test", "benchmark")):
        return "test"
    if any(
        marker in normalized
        for marker in (
            "/samples/",
            "/examples/",
            "/nsightsystems",
            "/compute-sanitizer/",
            "/nvvm/",
        )
    ):
        return "tooling"
    if (
        "/compat/" in normalized
        or basename.startswith("libcudart.so")
        or basename.startswith("libcufile")
        or basename.startswith("libcuda.so")
        or "libnvidia-ptxjitcompiler" in basename
        or "libnvidia-nvvm" in basename
        or "libnvvm" in basename
    ):
        return "host-runtime"
    if kernel_catalog_group(image_path):
        return "kernel-catalog"
    return "runtime"


def kernel_catalog_group(image_path: str) -> str | None:
    """Group runtime-selected CUDA artifacts shipped in precompiled catalogs.

    Framework images can contain tens of thousands of shape-specific cubins below a
    dispatch catalog.  Those files are alternatives selected by the runtime, not
    libraries that every workload necessarily loads.  The containing directory is a
    stable, deliberately conservative grouping boundary; ordinary standalone cubins
    outside an explicit ``cubins`` tree remain normal runtime artifacts.

    FlashInfer also ships architecture-selected modules in ``flashinfer_jit_cache``.
    A generic module and explicitly suffixed variants such as ``_sm90`` are one
    dispatch family. Plain numeric suffixes are accepted only in that known cache
    layout (for example ``fp4_quantization_100``), avoiding broad filename guesses.
    """
    normalized = f"/{image_path.lower().strip('/')}"
    if normalized.endswith(".cubin") and "/cubins/" in normalized:
        return str(Path(normalized).parent)
    dispatch = _flashinfer_dispatch_module(normalized)
    return dispatch[0] if dispatch else None


def kernel_catalog_kind(image_path: str) -> str | None:
    """Return the conservative catalog classification for an image artifact."""
    normalized = f"/{image_path.lower().strip('/')}"
    if normalized.endswith(".cubin") and "/cubins/" in normalized:
        return "precompiled-cubin"
    return "architecture-module" if _flashinfer_dispatch_module(normalized) else None


def kernel_catalog_declared_targets(image_path: str) -> tuple[str, ...]:
    """Read architecture selectors encoded in known catalog member paths."""
    normalized = f"/{image_path.lower().strip('/')}"
    targets = set(cuda_targets_from_text(normalized))
    dispatch = _flashinfer_dispatch_module(normalized)
    if dispatch:
        targets.update(dispatch[1])
    return tuple(sorted(targets))


def _flashinfer_dispatch_module(image_path: str) -> tuple[str, tuple[str, ...]] | None:
    """Return a normalized FlashInfer dispatch-family path and declared targets."""
    if FLASHINFER_JIT_CACHE_MARKER not in image_path or not image_path.endswith(".so"):
        return None
    path = Path(image_path)
    stem = path.stem
    # The cache layout is <module>/<module>.so. Requiring that shape keeps this
    # package-specific rule from absorbing unrelated shared libraries.
    if path.parent.name != stem:
        return None

    targets: set[str] = set()

    def remove_sm_selector(match: re.Match[str]) -> str:
        targets.add(f"sm_{int(match.group('number'))}{match.group('suffix').lower()}")
        return ""

    family = FLASHINFER_SM_SELECTOR_RE.sub(remove_sm_selector, stem)
    trailing = FLASHINFER_TRAILING_SELECTOR_RE.search(family)
    if trailing:
        targets.add(f"sm_{int(trailing.group('number'))}{trailing.group('suffix').lower()}")
        family = family[: trailing.start()]
    group = str(path.parent.parent / family)
    return group, tuple(sorted(targets))


def resolve_cuobjdump(mode: str, timeout_seconds: int) -> CuobjdumpTool | None:
    """Resolve cuobjdump from PATH, a well-known CUDA path, an explicit path, or off."""
    if mode == "off":
        return None
    if timeout_seconds <= 0:
        raise ValueError("cuobjdump timeout must be greater than zero")

    candidate: str | None
    if mode == "auto":
        candidate = shutil.which("cuobjdump")
        if candidate is None:
            well_known = Path("/usr/local/cuda/bin/cuobjdump")
            candidate = (
                str(well_known) if well_known.is_file() and os.access(well_known, os.X_OK) else None
            )
        if candidate is None:
            return None
    else:
        explicit = Path(mode).expanduser()
        if not explicit.is_file():
            raise ValueError(f"cuobjdump executable not found: {mode}")
        if not os.access(explicit, os.X_OK):
            raise ValueError(f"cuobjdump is not executable: {mode}")
        candidate = str(explicit)

    version = "unknown"
    try:
        result = subprocess.run(  # noqa: S603 - explicit argv; shell is disabled
            (candidate, "--version"),
            capture_output=True,
            check=False,
            text=True,
            timeout=min(timeout_seconds, 10),
        )
        version_text = " ".join(f"{result.stdout} {result.stderr}".split())
        if version_text:
            match = re.search(
                r"release\s+(?P<release>\d+(?:\.\d+)+),\s+V(?P<build>\d+(?:\.\d+)+)",
                version_text,
                re.IGNORECASE,
            )
            version = (
                f"CUDA {match.group('release')} (V{match.group('build')})"
                if match
                else version_text[-120:]
            )
    except OSError as error:
        raise ValueError(f"cuobjdump could not be executed: {candidate}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"cuobjdump --version timed out: {candidate}") from error
    if result.returncode:
        detail = " ".join(f"{result.stdout} {result.stderr}".split())[-500:]
        raise ValueError(
            f"cuobjdump --version failed for {candidate}: "
            f"{detail or f'exit status {result.returncode}'}"
        )
    return CuobjdumpTool(candidate, version, timeout_seconds)
