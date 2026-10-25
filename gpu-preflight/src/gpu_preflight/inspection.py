"""Static inspection rules and versioned report model."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache

from gpu_preflight import __version__
from gpu_preflight.compatibility import summarize_compatibility
from gpu_preflight.facts import ImageFacts
from gpu_preflight.profiles import FabricProfile, HardwareProfile, load_fabrics

UNRESOLVED_FINDINGS = {
    "GPF-CPU-003",
    "GPF-CUDA-001",
    "GPF-CUDA-006",
    "GPF-CUDA-011",
    "GPF-CUDA-014",
    "GPF-CUDA-015",
    "GPF-CUDA-019",
    "GPF-NCCL-001",
    "GPF-UCX-006",
}
PASSED_CHECK_FINDINGS = {
    "GPF-CPU-002",
    "GPF-CUDA-003",
    "GPF-CUDA-007",
    "GPF-CUDA-009",
    "GPF-IMAGE-001",
    "GPF-IMAGE-003",
    "GPF-NCCL-002",
    "GPF-UCX-005",
}

FINDING_DISPLAY_ORDER = {
    "GPF-PROFILE-001": 10,
    "GPF-IMAGE-001": 20,
    "GPF-IMAGE-002": 20,
    "GPF-IMAGE-003": 21,
    "GPF-IMAGE-007": 21,
    "GPF-IMAGE-008": 21,
    "GPF-CPU-001": 30,
    "GPF-CPU-002": 30,
    "GPF-CPU-003": 30,
    "GPF-CUDA-006": 40,
    "GPF-CUDA-007": 40,
    "GPF-CUDA-008": 40,
    "GPF-NCCL-001": 50,
    "GPF-NCCL-002": 50,
    "GPF-NCCL-003": 50,
    "GPF-UCX-001": 55,
    "GPF-UCX-002": 55,
    "GPF-UCX-003": 55,
    "GPF-UCX-004": 55,
    "GPF-UCX-005": 55,
    "GPF-UCX-006": 55,
    "GPF-CUDA-001": 60,
    "GPF-CUDA-002": 60,
    "GPF-CUDA-003": 60,
    "GPF-CUDA-004": 60,
    "GPF-CUDA-005": 61,
    "GPF-CUDA-015": 70,
    "GPF-CUDA-009": 71,
    "GPF-CUDA-010": 71,
    "GPF-CUDA-018": 72,
    "GPF-CUDA-017": 73,
    "GPF-CUDA-016": 74,
    "GPF-CUDA-013": 75,
    "GPF-CUDA-011": 80,
    "GPF-CUDA-012": 81,
    "GPF-CUDA-014": 82,
    "GPF-RUNTIME-001": 90,
}

STATIC_ANALYSIS_LIMITATIONS = (
    "The scan cannot determine which NCCL installation or dynamic library the workload will load.",
    "The scan cannot determine which optional native extension, dispatch catalog, or CUDA kernel variant the workload will select.",
    "Static inspection does not execute the workload or prove runtime correctness, stability, or performance.",
)


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    version: str


@dataclass(frozen=True)
class ScanMetadata:
    mode: str
    duration_seconds: float
    settings: dict[str, object]
    capabilities: dict[str, object]


@dataclass(frozen=True)
class FindingDetail:
    label: str
    value: str | int
    context: str | None = None


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    category: str
    title: str
    evidence: str
    remediation: str
    details: tuple[FindingDetail, ...] = ()


@dataclass(frozen=True)
class TargetReport:
    target: str
    display_name: str
    profile_maturity: str
    profile_schema_version: int
    profile_hash: str
    compatibility_status: str
    artifact_inspection_coverage_percent: float | None
    artifact_inspection_covered_candidates: int | None
    artifact_inspection_eligible_candidates: int | None
    reproducibility_status: str
    summary: str
    passed_checks: tuple[str, ...]
    evidence_limitations: tuple[str, ...]
    required_hardware_checks: tuple[str, ...]
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class InspectionReport:
    schema_version: int
    tool: ToolMetadata
    generated_at: str
    duration_seconds: float
    source: str
    scan: ScanMetadata
    disclaimer: str
    observations: dict[str, object]
    global_findings: tuple[Finding, ...]
    targets: tuple[TargetReport, ...]

    def to_dict(self) -> dict[str, object]:
        # Round-trip through the standard JSON encoder so tuples become schema-level arrays.
        return json.loads(json.dumps(asdict(self)))


def order_findings(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Order customer-facing findings from platform prerequisites to detailed evidence."""
    return tuple(
        sorted(
            findings,
            key=lambda finding: (FINDING_DISPLAY_ORDER.get(finding.id, 85), finding.id),
        )
    )


def _architecture_from_platform(platform: str | None) -> str | None:
    if not platform:
        return None
    architecture = platform.rsplit("/", maxsplit=1)[-1]
    return {"aarch64": "arm64", "x86_64": "amd64"}.get(architecture, architecture)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _compact_targets(values: tuple[str, ...], profile: HardwareProfile) -> str:
    native = tuple(
        value
        for value in values
        if value in profile.native_sm_targets
        or value == f"compute_{profile.compute_capability_number}"
    )
    if native:
        return f"native target(s) {', '.join(native)} among {len(values)} detected target(s)"
    preview = ", ".join(values[:12])
    suffix = f", plus {len(values) - 12} more" if len(values) > 12 else ""
    return f"{preview}{suffix}"


def _preferred_cuda_versions(values: tuple[str, ...]) -> tuple[str, ...]:
    """Prefer the toolkit release label over package-build versions in each family."""
    selected: dict[tuple[int, int], str] = {}
    for value in values:
        parts = _version_tuple(value)
        family = (parts[0], parts[1])
        existing = selected.get(family)
        if existing is None or len(parts) < len(_version_tuple(existing)):
            selected[family] = value
    return tuple(sorted(selected.values(), key=_version_tuple))


def _image_reference_findings(facts: ImageFacts) -> tuple[Finding, ...]:
    images = facts.image_references
    mutable = tuple(
        image
        for image in images
        if "@sha256:" not in image
        and (":" not in image.rsplit("/", maxsplit=1)[-1] or image.endswith(":latest"))
    )
    unpinned = tuple(image for image in images if "@sha256:" not in image)
    findings: list[Finding] = []
    if mutable:
        findings.append(
            Finding(
                id="GPF-REPRO-001",
                severity="warning",
                category="reproducibility",
                title="Mutable image reference",
                evidence=f"Mutable references: {', '.join(mutable)}",
                remediation="Use a versioned tag and record the resolved platform digest.",
            )
        )
    elif unpinned:
        findings.append(
            Finding(
                id="GPF-REPRO-002",
                severity="warning",
                category="reproducibility",
                title="Image reference is not digest-pinned",
                evidence=f"Unpinned references: {', '.join(unpinned)}",
                remediation=(
                    "Pin release artifacts by digest so validation evidence identifies exact layers."
                ),
            )
        )
    return tuple(findings)


@lru_cache(maxsize=1)
def _fabrics() -> dict[str, FabricProfile]:
    return load_fabrics()


def _ucx_installed(facts: ImageFacts) -> bool:
    """Return whether the image carries any UCX or HPC-X installation signal."""
    return bool(
        facts.ucx_versions
        or facts.hpcx_versions
        or facts.ucx_module_groups
        or {"ucx", "hpcx"} & set(facts.detected_packages)
    )


def _ucx_version_findings(
    facts: ImageFacts,
    profile: HardwareProfile,
    fabric: FabricProfile,
) -> tuple[Finding, ...]:
    """Compare detected UCX and HPC-X releases against the adapter support floor."""
    comparisons = (
        ("UCX", facts.ucx_versions, fabric.minimum_ucx_version),
        ("HPC-X", facts.hpcx_versions, fabric.minimum_hpcx_version),
    )
    evaluated = tuple(
        (label, detected, floor) for label, detected, floor in comparisons if detected and floor
    )
    if not evaluated:
        return ()

    below = tuple(
        (label, version, floor)
        for label, detected, floor in evaluated
        for version in detected
        if _version_tuple(version) < _version_tuple(floor)
    )
    if below:
        return (
            Finding(
                id="GPF-UCX-004",
                severity="warning",
                category="compatibility",
                title="Packaged UCX or HPC-X predates adapter support",
                evidence=(
                    f"{profile.display_name} uses {fabric.display_name}. "
                    + "; ".join(
                        f"detected {label} {version}, which predates the "
                        f"{label} {floor} support floor for this adapter"
                        for label, version, floor in below
                    )
                    + ". Static inspection cannot determine which installation the "
                    "workload loads."
                ),
                remediation=(
                    "Upgrade to an HPC-X or UCX release that lists this adapter as supported, "
                    "remove stale installations, and confirm the loaded transport on hardware "
                    "with ucx_info -d."
                ),
                details=tuple(
                    FindingDetail(label=label, value=version, context=f"floor {floor}")
                    for label, version, floor in below
                ),
            ),
        )
    return (
        Finding(
            id="GPF-UCX-005",
            severity="info",
            category="compatibility",
            title="UCX and HPC-X meet the adapter support floor",
            evidence=(
                f"{profile.display_name} uses {fabric.display_name}. "
                + "; ".join(
                    f"detected {label} {', '.join(detected)} against the {label} {floor} floor"
                    for label, detected, floor in evaluated
                )
                + ". This does not establish fabric correctness or performance."
            ),
            remediation=(
                "Run the required fabric tests on target hardware and confirm the transport "
                "actually selected."
            ),
        ),
    )


def _ucx_findings(facts: ImageFacts, profile: HardwareProfile) -> tuple[Finding, ...]:
    """Report on the UCX stack only when the image ships one.

    UCX is optional: NCCL reaches InfiniBand through its own plugin, so an image
    without UCX is ordinarily correct and produces no finding here. The exception
    is a package whose own transport backend is UCX.
    """
    findings: list[Finding] = []
    installed = _ucx_installed(facts)

    if "nixl" in facts.detected_packages and not installed:
        findings.append(
            Finding(
                id="GPF-UCX-003",
                severity="warning",
                category="compatibility",
                title="NIXL is installed without a UCX runtime",
                evidence=(
                    "NIXL was detected, but no UCX runtime library, transport module, or "
                    "package record was found in the image. UCX is NIXL's default transfer "
                    "backend."
                ),
                remediation=(
                    "Install UCX, through HPC-X or a distribution package, or confirm that "
                    "NIXL is configured for a backend that does not require it."
                ),
            )
        )

    if not installed:
        return tuple(findings)

    detected_versions = [
        *(f"UCX {version}" for version in facts.ucx_versions),
        *(f"HPC-X {version}" for version in facts.hpcx_versions),
    ]
    inventory_details = tuple(
        FindingDetail(
            label=group.directory,
            value=len(group.modules),
            context="modules, CUDA transport present"
            if group.has_cuda_transport
            else "modules, no CUDA transport",
        )
        for group in facts.ucx_module_groups
    )
    if detected_versions:
        findings.append(
            Finding(
                id="GPF-UCX-001",
                severity="info",
                category="evidence",
                title="UCX stack inventory",
                evidence=(
                    f"Detected {', '.join(detected_versions)}. Transport module directories "
                    f"found: {len(facts.ucx_module_groups)}."
                ),
                remediation=(
                    "Confirm the loaded build on hardware with ucx_info -v, since several "
                    "installations can coexist."
                ),
                details=inventory_details,
            )
        )
    else:
        findings.append(
            Finding(
                id="GPF-UCX-006",
                severity="info",
                category="evidence",
                title="UCX release was not established",
                evidence=(
                    "A UCX installation is present, but no release was determined: a bare "
                    "libucp soname identifies the ABI, not the UCX version, and no package "
                    "record or HPC-X bundle directory supplied one. Transport module "
                    f"directories found: {len(facts.ucx_module_groups)}."
                ),
                remediation=(
                    "Record the UCX release in package metadata or keep the HPC-X bundle "
                    "directory name, then confirm on hardware with ucx_info -v."
                ),
                details=inventory_details,
            )
        )

    # Only a directory that already holds transport modules proves the build is
    # modular. A UCX built with --enable-modules=no links CUDA support into
    # libuct itself and legitimately ships no libuct_cuda object.
    modular_without_cuda = tuple(
        group
        for group in facts.ucx_module_groups
        if any(module.startswith("libuct_") for module in group.modules)
        and not group.has_cuda_transport
    )
    if modular_without_cuda:
        findings.append(
            Finding(
                id="GPF-UCX-002",
                severity="warning",
                category="compatibility",
                title="UCX transport modules exclude CUDA",
                evidence=(
                    f"{len(modular_without_cuda)} modular UCX installation(s) ship transport "
                    "modules but no CUDA transport. Such a build moves GPU buffers through "
                    "host memory instead of using GPUDirect, which runs without error and "
                    "passes connectivity tests."
                ),
                remediation=(
                    "Install a UCX built with CUDA support, or the matching CUDA transport "
                    "package, and confirm with ucx_info -d that cuda_copy and cuda_ipc are "
                    "listed."
                ),
                details=tuple(
                    FindingDetail(
                        label=group.directory,
                        value=", ".join(group.modules[:6]),
                        context="transport modules found",
                    )
                    for group in modular_without_cuda
                ),
            )
        )

    if fabric := _fabrics().get(profile.compute_fabric or ""):
        findings.extend(_ucx_version_findings(facts, profile, fabric))
    return tuple(findings)


def _target_findings(facts: ImageFacts, profile: HardwareProfile) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    if profile.maturity == "provisional":
        findings.append(
            Finding(
                id="GPF-PROFILE-001",
                severity="warning",
                category="profile",
                title="Provisional hardware profile",
                evidence=f"{profile.display_name} is marked provisional.",
                remediation=(
                    "Treat static findings as guidance until the profile has hardware-backed evidence."
                ),
            )
        )

    declared_architecture = _architecture_from_platform(facts.final_platform)
    if declared_architecture and declared_architecture not in profile.cpu_architectures:
        findings.append(
            Finding(
                id="GPF-CPU-001",
                severity="blocker",
                category="compatibility",
                title="Final image platform conflicts with target CPU architecture",
                evidence=(
                    f"Resolved image platform is {facts.final_platform}; {profile.display_name} expects "
                    f"{', '.join(profile.cpu_architectures)}."
                ),
                remediation=(
                    "Publish a native image variant for the target architecture or a multi-platform "
                    "image index."
                ),
            )
        )
    elif declared_architecture:
        findings.append(
            Finding(
                id="GPF-CPU-002",
                severity="info",
                category="compatibility",
                title="Resolved CPU architecture matches",
                evidence=f"Resolved final platform is {facts.final_platform}.",
                remediation="Confirm native libraries inside the image match the resolved platform.",
            )
        )
    else:
        findings.append(
            Finding(
                id="GPF-CPU-003",
                severity="info",
                category="evidence",
                title="Final image architecture is not statically established",
                evidence="The image platform was not resolved from registry metadata.",
                remediation=(
                    "Inspect the built OCI manifest and confirm a native variant for "
                    f"{', '.join(profile.cpu_architectures)}."
                ),
            )
        )

    compatibility = summarize_compatibility(facts.cuda_targets, profile.compute_capability_number)
    if compatibility is None:
        findings.append(
            Finding(
                id="GPF-CUDA-001",
                severity="info",
                category="evidence",
                title="CUDA compilation targets were not found",
                evidence="No sm_* or compute_* targets were found in the available evidence.",
                remediation=(
                    "Review the inspected CUDA artifacts and scan limits; missing target evidence "
                    "does not prove that native extensions are absent."
                ),
            )
        )
    elif not compatibility.compatible:
        findings.append(
            Finding(
                id="GPF-CUDA-002",
                severity="blocker",
                category="compatibility",
                title="Declared CUDA targets are incompatible",
                evidence=(
                    f"Detected {_compact_targets(facts.cuda_targets, profile)}; "
                    "target compute capability is "
                    f"{profile.compute_capability}."
                ),
                remediation=(
                    f"Build for {', '.join(profile.native_sm_targets)} or retain a compatible PTX "
                    "target for JIT compilation."
                ),
            )
        )
    elif compatibility.classification == "native":
        findings.append(
            Finding(
                id="GPF-CUDA-003",
                severity="info",
                category="compatibility",
                title="Native CUDA target detected",
                evidence=(
                    f"Detected {_compact_targets(facts.cuda_targets, profile)} for compute capability "
                    f"{profile.compute_capability}."
                ),
                remediation="Confirm every CUDA-bearing library in the built image has equivalent coverage.",
            )
        )
    else:
        compatible_targets = tuple(
            item.target for item in compatibility.evaluations if item.compatible
        )
        compatible_preview = ", ".join(compatible_targets[:12])
        compatible_suffix = (
            f", plus {len(compatible_targets) - 12} more" if len(compatible_targets) > 12 else ""
        )
        findings.append(
            Finding(
                id="GPF-CUDA-004",
                severity="warning",
                category="compatibility",
                title="CUDA code is compatible but not native",
                evidence=(
                    f"Best match for sm_{profile.compute_capability_number} is "
                    f"{compatibility.classification}; compatible target evidence: "
                    f"{compatible_preview}{compatible_suffix}. "
                    f"{len(facts.cuda_targets)} total target(s) were detected."
                ),
                remediation=(
                    f"Add a native {profile.native_sm_targets[0]} build for predictable startup and "
                    "performance, then validate on hardware."
                ),
            )
        )

    architecture_specific_mismatches = (
        tuple(
            item.target
            for item in compatibility.evaluations
            if compatibility and item.target.endswith("a") and not item.compatible
        )
        if compatibility
        else ()
    )
    if (
        architecture_specific_mismatches
        and compatibility
        and compatibility.compatible
        and compatibility.classification != "native"
    ):
        findings.append(
            Finding(
                id="GPF-CUDA-005",
                severity="warning",
                category="compatibility",
                title="Architecture-specific CUDA targets do not carry forward",
                evidence=(
                    "Exact-only targets for another GPU were detected: "
                    + ", ".join(architecture_specific_mismatches)
                ),
                remediation=(
                    "Ensure affected kernels also contain a compatible baseline, family target, or "
                    "native target for this hardware."
                ),
            )
        )

    if profile.minimum_cuda_version:
        displayed_versions = _preferred_cuda_versions(facts.cuda_versions)
        if not facts.cuda_versions:
            findings.append(
                Finding(
                    id="GPF-CUDA-006",
                    severity="info",
                    category="evidence",
                    title="CUDA version was not established",
                    evidence=(
                        f"The target's compiler-support baseline is CUDA "
                        f"{profile.minimum_cuda_version}, but no CUDA version was detected."
                    ),
                    remediation=(
                        "Inspect image configuration and package metadata, or provide an explicit "
                        "CUDA_VERSION value."
                    ),
                )
            )
        elif any(
            _version_tuple(version) >= _version_tuple(profile.minimum_cuda_version)
            for version in facts.cuda_versions
        ):
            findings.append(
                Finding(
                    id="GPF-CUDA-007",
                    severity="info",
                    category="compatibility",
                    title="CUDA version meets the target baseline",
                    evidence=(
                        f"Detected CUDA {', '.join(displayed_versions)}; "
                        f"{profile.display_name} requires {profile.minimum_cuda_version} or newer "
                        "for native compiler-target support."
                    ),
                    remediation=(
                        "Confirm driver compatibility and validate the packaged CUDA libraries on "
                        "the target hardware."
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    id="GPF-CUDA-008",
                    severity="warning",
                    category="compatibility",
                    title="CUDA version predates native target support",
                    evidence=(
                        f"Detected CUDA {', '.join(displayed_versions)}; "
                        f"{profile.display_name} needs CUDA {profile.minimum_cuda_version} or newer "
                        "for native compiler-target support."
                    ),
                    remediation=(
                        f"Rebuild or rebase on CUDA {profile.minimum_cuda_version} or newer, then "
                        "reinspect every CUDA-bearing dependency."
                    ),
                )
            )

    if profile.minimum_nccl_version:
        minimum_nccl = _version_tuple(profile.minimum_nccl_version)
        detected_nccl = tuple(sorted(facts.nccl_versions, key=_version_tuple))
        if not detected_nccl:
            findings.append(
                Finding(
                    id="GPF-NCCL-001",
                    severity="info",
                    category="evidence",
                    title="NCCL version was not established",
                    evidence=(
                        f"{profile.display_name} has a minimum NCCL platform-support baseline of "
                        f"{profile.minimum_nccl_version}, but no NCCL version was detected in "
                        "image configuration, package metadata, or NCCL headers."
                    ),
                    remediation=(
                        "Run a filesystem inspection and package NCCL with version metadata. On "
                        "hardware, log the library actually loaded with NCCL_DEBUG=VERSION."
                    ),
                )
            )
        else:
            below_minimum = tuple(
                version for version in detected_nccl if _version_tuple(version) < minimum_nccl
            )
            if below_minimum:
                findings.append(
                    Finding(
                        id="GPF-NCCL-003",
                        severity="warning",
                        category="compatibility",
                        title="Packaged NCCL version predates target support",
                        evidence=(
                            f"Detected NCCL {', '.join(detected_nccl)}; "
                            f"{profile.display_name} requires NCCL "
                            f"{profile.minimum_nccl_version} or newer as its minimum "
                            "platform-support baseline. Older candidate(s): "
                            f"{', '.join(below_minimum)}. Static inspection cannot determine which "
                            "copy the runtime loader will select."
                        ),
                        remediation=(
                            f"Upgrade every NCCL runtime candidate to "
                            f"{profile.minimum_nccl_version} or newer, remove stale copies, and "
                            "confirm the loaded version with NCCL_DEBUG=VERSION on target hardware."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        id="GPF-NCCL-002",
                        severity="info",
                        category="compatibility",
                        title="NCCL version meets the target baseline",
                        evidence=(
                            f"Detected NCCL {', '.join(detected_nccl)}; "
                            f"{profile.display_name} requires {profile.minimum_nccl_version} or "
                            "newer as its minimum platform-support baseline. This does not establish "
                            "fabric correctness or performance."
                        ),
                        remediation=(
                            "Confirm the loaded NCCL version and run the required collective and "
                            "fabric tests on target hardware."
                        ),
                    )
                )

    findings.extend(_ucx_findings(facts, profile))

    findings.append(
        Finding(
            id="GPF-RUNTIME-001",
            severity="info",
            category="hardware-validation",
            title="Hardware validation is still required",
            evidence="Required checks: " + ", ".join(profile.runtime_checks),
            remediation=(
                "Run the generated test plan on the target hardware before claiming compatibility."
            ),
        )
    )
    return tuple(findings)


def compatibility_status_for(findings: tuple[Finding, ...]) -> str:
    """Return a compatibility result without conflating reproducibility warnings."""
    if any(
        finding.severity == "blocker" and finding.category == "compatibility"
        for finding in findings
    ):
        return "INCOMPATIBLE"
    compatibility_warnings = tuple(
        finding
        for finding in findings
        if finding.severity == "warning" and finding.category == "compatibility"
    )
    if any(finding.id != "GPF-CUDA-016" for finding in compatibility_warnings):
        return "ACTION_REQUIRED"
    if compatibility_warnings:
        return "REVIEW_REQUIRED"
    if any(
        finding.id in UNRESOLVED_FINDINGS or finding.category == "profile" for finding in findings
    ):
        return "INSUFFICIENT_EVIDENCE"
    return "READY_FOR_HARDWARE_TESTING"


def artifact_inspection_coverage_percent_for(
    findings: tuple[Finding, ...],
) -> float | None:
    """Return the percentage of loadable runtime candidates inspected with cuobjdump."""
    finding_ids = {finding.id for finding in findings}
    if finding_ids & {"GPF-CUDA-011", "GPF-CUDA-015", "GPF-IMAGE-007", "GPF-IMAGE-008"}:
        return None

    coverage_finding = next(
        (finding for finding in findings if finding.id == "GPF-CUDA-014"),
        None,
    )
    if coverage_finding:
        details = {detail.label: detail.value for detail in coverage_finding.details}

        def count(label: str) -> int:
            value = details.get(label, 0)
            return value if isinstance(value, int) else 0

        incomplete = count("Incomplete cuobjdump inspections")
        completed = max(0, count("Inspected with cuobjdump") - incomplete)
        unresolved = incomplete + sum(
            count(label)
            for label in (
                "Not inspected: scan budget",
                "Not inspected: artifact-size limit",
                "Not inspected: file-count limit",
                "Not inspected: analyzer unavailable",
            )
        )
        total = completed + unresolved
        if total:
            return round(completed / total * 100, 1)
        return None

    if "GPF-IMAGE-003" in finding_ids:
        return 100.0
    return None


def artifact_inspection_counts_for(
    findings: tuple[Finding, ...],
) -> tuple[int | None, int | None]:
    """Recover covered and eligible runtime candidate counts from bounded-scan evidence."""
    coverage_finding = next(
        (finding for finding in findings if finding.id == "GPF-CUDA-014"),
        None,
    )
    if not coverage_finding:
        return None, None
    details = {detail.label: detail.value for detail in coverage_finding.details}

    def count(label: str) -> int:
        value = details.get(label, 0)
        return value if isinstance(value, int) else 0

    incomplete = count("Incomplete cuobjdump inspections")
    covered = max(0, count("Inspected with cuobjdump") - incomplete)
    eligible = (
        covered
        + incomplete
        + sum(
            count(label)
            for label in (
                "Not inspected: scan budget",
                "Not inspected: artifact-size limit",
                "Not inspected: file-count limit",
                "Not inspected: analyzer unavailable",
            )
        )
    )
    return covered, eligible or None


def reproducibility_status_for(facts: ImageFacts) -> str:
    images = facts.image_references
    if not images:
        return "UNKNOWN"
    return "PINNED" if all("@sha256:" in image for image in images) else "UNPINNED"


def summary_for(status: str) -> str:
    return {
        "INCOMPATIBLE": "A static incompatibility must be fixed before hardware testing.",
        "ACTION_REQUIRED": "Address the listed compatibility risks before hardware testing.",
        "REVIEW_REQUIRED": (
            "Review artifacts whose workload relevance cannot be established by static inspection."
        ),
        "INSUFFICIENT_EVIDENCE": "More image evidence is required before compatibility can be assessed.",
        "READY_FOR_HARDWARE_TESTING": (
            "No static blocker was found; proceed to the required hardware tests."
        ),
    }[status]


def passed_checks_for(findings: tuple[Finding, ...]) -> tuple[str, ...]:
    return tuple(
        finding.title
        for finding in findings
        if finding.severity == "info" and finding.id in PASSED_CHECK_FINDINGS
    )


def evidence_limitations_for(findings: tuple[Finding, ...]) -> tuple[str, ...]:
    def material_limitation(finding: Finding) -> bool:
        if finding.id != "GPF-CUDA-012":
            return True
        details = {detail.label: detail.value for detail in finding.details}
        return bool(
            details.get("Runtime", 0)
            or details.get("Kernel catalogs", 0)
            or details.get("Incomplete cuobjdump inspections", 0)
        )

    return tuple(
        f"{finding.id}: {finding.title}"
        for finding in findings
        if finding.category in {"evidence", "profile"}
        and finding.id not in PASSED_CHECK_FINDINGS
        and material_limitation(finding)
    )


def refresh_target(
    target: TargetReport,
    findings: tuple[Finding, ...],
    global_findings: tuple[Finding, ...],
    artifact_inspection_counts: tuple[int, int] | None = None,
) -> TargetReport:
    findings = order_findings(findings)
    combined = (*global_findings, *findings)
    status = compatibility_status_for(combined)
    covered, eligible = (
        artifact_inspection_counts
        if artifact_inspection_counts is not None
        else artifact_inspection_counts_for(combined)
    )
    if artifact_inspection_counts is not None:
        coverage_percent = round(covered / eligible * 100, 1) if eligible else None
    else:
        coverage_percent = artifact_inspection_coverage_percent_for(combined)
    return TargetReport(
        target=target.target,
        display_name=target.display_name,
        profile_maturity=target.profile_maturity,
        profile_schema_version=target.profile_schema_version,
        profile_hash=target.profile_hash,
        compatibility_status=status,
        artifact_inspection_coverage_percent=coverage_percent,
        artifact_inspection_covered_candidates=covered,
        artifact_inspection_eligible_candidates=eligible,
        reproducibility_status=target.reproducibility_status,
        summary=summary_for(status),
        passed_checks=passed_checks_for(findings),
        evidence_limitations=evidence_limitations_for(combined),
        required_hardware_checks=target.required_hardware_checks,
        findings=findings,
    )


def inspect(facts: ImageFacts, profiles: tuple[HardwareProfile, ...]) -> InspectionReport:
    global_findings = _image_reference_findings(facts)
    reproducibility = reproducibility_status_for(facts)
    target_reports: list[TargetReport] = []
    for profile in profiles:
        findings = order_findings(_target_findings(facts, profile))
        combined = (*global_findings, *findings)
        status = compatibility_status_for(combined)
        covered, eligible = artifact_inspection_counts_for(combined)
        target_reports.append(
            TargetReport(
                target=profile.id,
                display_name=profile.display_name,
                profile_maturity=profile.maturity,
                profile_schema_version=profile.schema_version,
                profile_hash=profile.fingerprint,
                compatibility_status=status,
                artifact_inspection_coverage_percent=(
                    artifact_inspection_coverage_percent_for(combined)
                ),
                artifact_inspection_covered_candidates=covered,
                artifact_inspection_eligible_candidates=eligible,
                reproducibility_status=reproducibility,
                summary=summary_for(status),
                passed_checks=passed_checks_for(findings),
                evidence_limitations=evidence_limitations_for(combined),
                required_hardware_checks=profile.runtime_checks,
                findings=findings,
            )
        )
    return InspectionReport(
        schema_version=1,
        tool=ToolMetadata(name="nebius-gpu-preflight", version=__version__),
        generated_at=datetime.now(UTC).isoformat(),
        duration_seconds=0.0,
        source=facts.path,
        scan=ScanMetadata(
            mode="static",
            duration_seconds=0.0,
            settings={},
            capabilities={
                "registry_manifest_inspected": False,
                "image_config_inspected": False,
                "filesystem_inspected": False,
                "cuobjdump_available": False,
                "cuobjdump_version": None,
            },
        ),
        disclaimer=(
            "Static readiness is not hardware certification. Image contents, CUDA execution, "
            "fabric selection, correctness, and performance require additional validation."
        ),
        observations={
            "image_references": list(facts.image_references),
            "final_platform": facts.final_platform,
            "cuda_targets": list(facts.cuda_targets),
            "cuda_versions": list(facts.cuda_versions),
            "nccl_versions": list(facts.nccl_versions),
            "ucx_versions": list(facts.ucx_versions),
            "hpcx_versions": list(facts.hpcx_versions),
            "ucx_module_groups": [asdict(group) for group in facts.ucx_module_groups],
            "detected_packages": list(facts.detected_packages),
            "torch_cuda_arch_list": facts.environment.get("TORCH_CUDA_ARCH_LIST"),
            "static_analysis_limitations": list(STATIC_ANALYSIS_LIMITATIONS),
        },
        global_findings=global_findings,
        targets=tuple(target_reports),
    )


def has_threshold(report: InspectionReport, threshold: str) -> bool:
    findings = [
        *report.global_findings,
        *(finding for target in report.targets for finding in target.findings),
    ]
    if threshold == "never":
        return False
    if threshold == "warning":
        return any(finding.severity in {"blocker", "warning"} for finding in findings)
    return any(finding.severity == "blocker" for finding in findings)


def normalize_output_path(value: str) -> str:
    """Keep report source paths stable while avoiding accidental whitespace."""
    return re.sub(r"\s+", " ", value).strip()
