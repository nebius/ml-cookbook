"""Hardware profile loading and validation."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import resources
from pathlib import Path

SUPPORTED_MATURITY = {"supported", "provisional", "deprecated"}


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class RuntimePackageFloor:
    """The oldest release of a runtime compiler, DSL or kernel library that a vendor
    has published as supporting this hardware.

    These packages generate or select device code when the workload runs, so an image
    carries no compiled artifact to inspect for them. The installed version is the
    only static evidence, and a floor is recorded only where a release note states it:
    ``minimum_version`` is therefore optional, and a package listed without one is
    reported as present-but-unassessed rather than silently assumed compatible.

    A package belongs in this table only when artifact inspection cannot answer the
    question for it: JIT compilers and DSLs that emit no cubin at build time, and
    libraries whose architecture support is host-side rather than device code. A
    compiled extension does not belong, however prominent -- cuobjdump reads its sm_
    targets from the artifact, per build, and names the offending file in the report.
    Listing those would mean maintaining a roster of libraries that artifact
    inspection already discovers on its own, and the roster would be stale the moment
    a new one ships.

    Support does not always move forward along one version line, and a successor may
    ship under a different distribution name entirely. An entry marked
    ``applicability = "not-applicable"`` records that the package targets other
    hardware, which a version floor cannot express.
    """

    package: str
    role: str
    minimum_version: str | None
    note: str | None
    source: str | None
    applicability: str = "supported"
    # One library can reach an image under several distribution names: NVSHMEM ships
    # as nvidia-nvshmem-cu12/-cu13 from pip and libnvshmem3-cuda-12/-13 from apt, all
    # carrying the same upstream version. The lookup is an exact name match, so an
    # entry naming only one of them assesses only the images that chose that one.
    distributions: tuple[str, ...] = ()

    @property
    def lookup_names(self) -> tuple[str, ...]:
        return self.distributions or (self.package,)


SUPPORTED_PACKAGE_ROLES = {"compiler", "dsl", "kernel-library"}
SUPPORTED_PACKAGE_APPLICABILITY = {"supported", "not-applicable"}


@dataclass(frozen=True)
class FabricProfile:
    """Version floors and provenance for one compute-fabric network adapter.

    Adapter support is a property of the NIC generation, so one entry and one
    citation cover every GPU platform that ships the same adapter.
    """

    schema_version: int
    id: str
    display_name: str
    generation: int
    minimum_ucx_version: str | None
    minimum_hpcx_version: str | None
    notes: tuple[str, ...]
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class HardwareProfile:
    schema_version: int
    id: str
    aliases: tuple[str, ...]
    display_name: str
    maturity: str
    cpu_architectures: tuple[str, ...]
    gpu_architecture: str
    compute_capability: str
    native_sm_targets: tuple[str, ...]
    nebius_platforms: tuple[str, ...]
    compute_fabric: str | None
    minimum_cuda_version: str | None
    minimum_nccl_version: str | None
    runtime_packages: tuple[RuntimePackageFloor, ...]
    runtime_checks: tuple[str, ...]
    notes: tuple[str, ...]
    sources: tuple[Source, ...]

    @property
    def compute_capability_number(self) -> int:
        major, minor = self.compute_capability.split(".", maxsplit=1)
        return int(major) * 10 + int(minor)

    @property
    def fingerprint(self) -> str:
        """Stable hash identifying the exact profile inputs used by a report."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(payload.encode()).hexdigest()}"


def _required_table(document: dict[str, object], name: str, path: str) -> dict[str, object]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing [{name}] table")
    return value


def _strings(value: object, field: str, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {field} must be an array of strings")
    return tuple(value)


def _string(value: object, field: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, field, path)


VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")


def _optional_version(value: object, field: str, path: str) -> str | None:
    """Return a dotted numeric version floor, or None when the field is unset.

    Floors are compared component-wise against versions detected in an image, so a
    value such as "12" or "latest" would fail that comparison deep inside a scan.
    Rejecting it here names the file and the field instead.
    """
    version = _optional_string(value, field, path)
    if version is not None and not VERSION_RE.match(version):
        raise ValueError(
            f"{path}: {field} must be a dotted numeric version such as 12.9, got {version!r}"
        )
    return version


def _optional_lower(value: object, field: str, path: str) -> str | None:
    lowered = _optional_string(value, field, path)
    return lowered.lower() if lowered else None


def _sources(document: dict[str, object], path: str) -> tuple[Source, ...]:
    source_documents = document.get("sources", [])
    if not isinstance(source_documents, list):
        raise ValueError(f"{path}: sources must be an array of tables")
    sources = tuple(
        Source(
            title=_string(source.get("title"), "sources.title", path),
            url=_string(source.get("url"), "sources.url", path),
        )
        for source in source_documents
        if isinstance(source, dict)
    )
    if len(sources) != len(source_documents):
        raise ValueError(f"{path}: every source must be a table")
    return sources


def _runtime_packages(validation: dict[str, object], path: str) -> tuple[RuntimePackageFloor, ...]:
    entries = validation.get("runtime_packages", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: validation.runtime_packages must be an array of tables")
    floors: list[RuntimePackageFloor] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every validation.runtime_packages entry must be a table")
        package = _string(entry.get("package"), "validation.runtime_packages.package", path).lower()
        if package in seen:
            raise ValueError(f"{path}: validation.runtime_packages lists {package!r} twice")
        seen.add(package)
        role = _string(entry.get("role"), "validation.runtime_packages.role", path).lower()
        if role not in SUPPORTED_PACKAGE_ROLES:
            supported = ", ".join(sorted(SUPPORTED_PACKAGE_ROLES))
            raise ValueError(
                f"{path}: validation.runtime_packages.role must be one of {supported}, got {role!r}"
            )
        applicability = _string(
            entry.get("applicability", "supported"),
            "validation.runtime_packages.applicability",
            path,
        ).lower()
        if applicability not in SUPPORTED_PACKAGE_APPLICABILITY:
            supported = ", ".join(sorted(SUPPORTED_PACKAGE_APPLICABILITY))
            raise ValueError(
                f"{path}: validation.runtime_packages.applicability must be one of "
                f"{supported}, got {applicability!r}"
            )
        minimum_version = _optional_version(
            entry.get("minimum_version"),
            "validation.runtime_packages.minimum_version",
            path,
        )
        if applicability == "not-applicable" and minimum_version is not None:
            # A floor claims every later release works here; the marker claims none
            # does. Accepting both would leave the assessment undefined.
            raise ValueError(
                f"{path}: validation.runtime_packages entry {package!r} cannot set "
                "minimum_version while applicability is not-applicable"
            )
        raw_distributions = entry.get("distributions", [])
        if not isinstance(raw_distributions, list):
            raise ValueError(f"{path}: validation.runtime_packages.distributions must be an array")
        distributions = tuple(
            _string(item, "validation.runtime_packages.distributions", path).lower()
            for item in raw_distributions
        )
        floors.append(
            RuntimePackageFloor(
                distributions=distributions,
                package=package,
                role=role,
                minimum_version=minimum_version,
                note=_optional_string(entry.get("note"), "validation.runtime_packages.note", path),
                source=_optional_string(
                    entry.get("source"), "validation.runtime_packages.source", path
                ),
                applicability=applicability,
            )
        )
    return tuple(sorted(floors, key=lambda floor: floor.package))


def _optional_table(document: dict[str, object], name: str, path: str) -> dict[str, object]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {name} must be a table")
    return value


def parse_fabric(document: dict[str, object], path: str) -> FabricProfile:
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")

    validation = _optional_table(document, "validation", path)
    generation = document.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ValueError(f"{path}: generation must be a positive integer")

    return FabricProfile(
        schema_version=1,
        id=_string(document.get("id"), "id", path).lower(),
        display_name=_string(document.get("display_name"), "display_name", path),
        generation=generation,
        minimum_ucx_version=_optional_version(
            validation.get("minimum_ucx_version"), "validation.minimum_ucx_version", path
        ),
        minimum_hpcx_version=_optional_version(
            validation.get("minimum_hpcx_version"), "validation.minimum_hpcx_version", path
        ),
        notes=_strings(document.get("notes", []), "notes", path),
        sources=_sources(document, path),
    )


def parse_profile(document: dict[str, object], path: str) -> HardwareProfile:
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")

    hardware = _required_table(document, "hardware", path)
    validation = _required_table(document, "validation", path)
    maturity = _string(document.get("maturity"), "maturity", path)
    if maturity not in SUPPORTED_MATURITY:
        supported = ", ".join(sorted(SUPPORTED_MATURITY))
        raise ValueError(f"{path}: maturity must be one of {supported}")

    sources = _sources(document, path)

    profile = HardwareProfile(
        schema_version=1,
        id=_string(document.get("id"), "id", path).lower(),
        aliases=tuple(
            alias.lower() for alias in _strings(document.get("aliases"), "aliases", path)
        ),
        display_name=_string(document.get("display_name"), "display_name", path),
        maturity=maturity,
        cpu_architectures=_strings(
            hardware.get("cpu_architectures"), "hardware.cpu_architectures", path
        ),
        gpu_architecture=_string(
            hardware.get("gpu_architecture"), "hardware.gpu_architecture", path
        ),
        compute_capability=_string(
            hardware.get("compute_capability"), "hardware.compute_capability", path
        ),
        native_sm_targets=_strings(
            hardware.get("native_sm_targets"), "hardware.native_sm_targets", path
        ),
        nebius_platforms=_strings(
            hardware.get("nebius_platforms"), "hardware.nebius_platforms", path
        ),
        compute_fabric=_optional_lower(
            _optional_table(document, "fabric", path).get("compute_adapter"),
            "fabric.compute_adapter",
            path,
        ),
        minimum_cuda_version=_optional_version(
            validation.get("minimum_cuda_version"),
            "validation.minimum_cuda_version",
            path,
        ),
        minimum_nccl_version=_optional_version(
            validation.get("minimum_nccl_version"),
            "validation.minimum_nccl_version",
            path,
        ),
        runtime_packages=_runtime_packages(validation, path),
        runtime_checks=_strings(
            validation.get("runtime_checks"), "validation.runtime_checks", path
        ),
        notes=_strings(document.get("notes", []), "notes", path),
        sources=sources,
    )
    # Exercise parsing early so malformed values fail while loading profiles.
    _ = profile.compute_capability_number
    return profile


def _load_toml_path(path: Path) -> HardwareProfile:
    with path.open("rb") as handle:
        return parse_profile(tomllib.load(handle), str(path))


def load_fabrics() -> dict[str, FabricProfile]:
    loaded: dict[str, FabricProfile] = {}
    fabric_files = resources.files("gpu_preflight.data.fabrics")
    for resource in sorted(fabric_files.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".toml"):
            continue
        with resource.open("rb") as handle:
            fabric = parse_fabric(tomllib.load(handle), resource.name)
        if fabric.id in loaded:
            raise ValueError(f"{resource.name}: duplicate fabric id {fabric.id!r}")
        loaded[fabric.id] = fabric
    return loaded


def load_profiles(extra_directory: Path | None = None) -> dict[str, HardwareProfile]:
    loaded: dict[str, HardwareProfile] = {}
    profile_files = resources.files("gpu_preflight.data.profiles")
    for resource in sorted(profile_files.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".toml"):
            continue
        with resource.open("rb") as handle:
            profile = parse_profile(tomllib.load(handle), resource.name)
        loaded[profile.id] = profile

    if extra_directory:
        for path in sorted(extra_directory.glob("*.toml")):
            profile = _load_toml_path(path)
            if profile.id in loaded:
                raise ValueError(f"{path}: duplicate profile id {profile.id!r}")
            loaded[profile.id] = profile

    fabrics = load_fabrics()
    for profile in loaded.values():
        if profile.compute_fabric and profile.compute_fabric not in fabrics:
            choices = ", ".join(sorted(fabrics))
            raise ValueError(
                f"profile {profile.id!r} declares unknown compute fabric "
                f"{profile.compute_fabric!r}; available adapters: {choices}"
            )

    aliases: dict[str, str] = {}
    for profile in loaded.values():
        for alias in (profile.id, *profile.aliases):
            existing = aliases.get(alias)
            if existing and existing != profile.id:
                raise ValueError(
                    f"profile alias {alias!r} is shared by {existing!r} and {profile.id!r}"
                )
            aliases[alias] = profile.id
    return loaded


def resolve_profile(profiles: dict[str, HardwareProfile], requested: str) -> HardwareProfile:
    normalized = requested.lower()
    for profile in profiles.values():
        if normalized == profile.id or normalized in profile.aliases:
            return profile
    choices = ", ".join(sorted(profiles))
    raise KeyError(f"unknown target {requested!r}; available targets: {choices}")
