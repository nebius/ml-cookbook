"""Hardware profile loading and validation."""

from __future__ import annotations

import json
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
    minimum_cuda_version: str | None
    minimum_nccl_version: str | None
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


def parse_profile(document: dict[str, object], path: str) -> HardwareProfile:
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")

    hardware = _required_table(document, "hardware", path)
    validation = _required_table(document, "validation", path)
    maturity = _string(document.get("maturity"), "maturity", path)
    if maturity not in SUPPORTED_MATURITY:
        supported = ", ".join(sorted(SUPPORTED_MATURITY))
        raise ValueError(f"{path}: maturity must be one of {supported}")

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
        minimum_cuda_version=_optional_string(
            validation.get("minimum_cuda_version"),
            "validation.minimum_cuda_version",
            path,
        ),
        minimum_nccl_version=_optional_string(
            validation.get("minimum_nccl_version"),
            "validation.minimum_nccl_version",
            path,
        ),
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
