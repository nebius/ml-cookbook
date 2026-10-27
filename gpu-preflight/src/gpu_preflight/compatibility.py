"""CUDA target compatibility evaluation.

This module models documented CUDA compatibility semantics. It does not claim
that a compatible binary is performant or that a workload is hardware-validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_RE = re.compile(r"^(?P<kind>sm|compute)_(?P<number>\d+)(?P<suffix>[af]?)$")
FAMILY_COMPATIBILITY = {
    100: {100, 103, 107},
    103: {103, 107},
    107: {107},
    110: {110},
    120: {120, 121},
    121: {121},
}
CLASSIFICATION_RANK = {
    "incompatible": 0,
    "ptx-jit-compatible": 1,
    "binary-compatible": 2,
    "family-compatible": 3,
    "native": 4,
}
CODE_PATH_RANK = {
    "incompatible": 0,
    "ptx-jit": 1,
    "compatible-base-cubin": 2,
    "compatible-family-cubin": 3,
    "exact-base-cubin": 4,
    "exact-family-cubin": 5,
    "exact-architecture-cubin": 6,
}


@dataclass(frozen=True)
class CudaTarget:
    raw: str
    kind: str
    number: int
    suffix: str


@dataclass(frozen=True)
class TargetEvaluation:
    target: str
    classification: str
    compatible: bool
    reason: str


@dataclass(frozen=True)
class CompatibilitySummary:
    classification: str
    compatible: bool
    evaluations: tuple[TargetEvaluation, ...]


def parse_cuda_target(value: str) -> CudaTarget:
    normalized = value.lower().replace("-", "_")
    match = TARGET_RE.match(normalized)
    if not match:
        raise ValueError(f"unsupported CUDA target: {value!r}")
    return CudaTarget(
        raw=normalized,
        kind=match.group("kind"),
        number=int(match.group("number")),
        suffix=match.group("suffix"),
    )


def evaluate_cuda_target(value: str, hardware_number: int) -> TargetEvaluation:
    target = parse_cuda_target(value)
    if target.suffix == "a":
        compatible = target.number == hardware_number
        return TargetEvaluation(
            target=target.raw,
            classification="native" if compatible else "incompatible",
            compatible=compatible,
            reason=("architecture-specific targets require an exact compute-capability match"),
        )

    if target.suffix == "f":
        compatible = hardware_number in FAMILY_COMPATIBILITY.get(target.number, {target.number})
        return TargetEvaluation(
            target=target.raw,
            classification=(
                "native"
                if compatible and target.number == hardware_number
                else "family-compatible"
                if compatible
                else "incompatible"
            ),
            compatible=compatible,
            reason="family-specific target compatibility is defined by the CUDA family table",
        )

    if target.kind == "compute":
        compatible = hardware_number >= target.number
        return TargetEvaluation(
            target=target.raw,
            classification="ptx-jit-compatible" if compatible else "incompatible",
            compatible=compatible,
            reason="retained PTX can JIT on the same or a later compute capability",
        )

    target_major, hardware_major = target.number // 10, hardware_number // 10
    compatible = target_major == hardware_major and hardware_number >= target.number
    return TargetEvaluation(
        target=target.raw,
        classification=(
            "native"
            if target.number == hardware_number
            else "binary-compatible"
            if compatible
            else "incompatible"
        ),
        compatible=compatible,
        reason="cubin compatibility requires the same major and an equal or later minor capability",
    )


def summarize_compatibility(
    values: tuple[str, ...], hardware_number: int
) -> CompatibilitySummary | None:
    if not values:
        return None
    evaluations = tuple(evaluate_cuda_target(value, hardware_number) for value in values)
    compatible = tuple(item for item in evaluations if item.compatible)
    if not compatible:
        return CompatibilitySummary(
            classification="incompatible",
            compatible=False,
            evaluations=evaluations,
        )
    best = max(compatible, key=lambda item: CLASSIFICATION_RANK[item.classification])
    return CompatibilitySummary(
        classification=best.classification,
        compatible=True,
        evaluations=evaluations,
    )


def classify_cuda_code_path(value: str, hardware_number: int) -> str:
    """Classify one embedded CUDA target by its optimization level for a GPU."""
    target = parse_cuda_target(value)
    evaluation = evaluate_cuda_target(value, hardware_number)
    if not evaluation.compatible:
        return "incompatible"
    if target.kind == "compute":
        return "ptx-jit"
    if target.number == hardware_number:
        if target.suffix == "a":
            return "exact-architecture-cubin"
        if target.suffix == "f":
            return "exact-family-cubin"
        return "exact-base-cubin"
    if target.suffix == "f":
        return "compatible-family-cubin"
    return "compatible-base-cubin"


def best_cuda_code_path(values: tuple[str, ...], hardware_number: int) -> str:
    """Return the best compatible CUDA code path contained in one artifact."""
    if not values:
        return "incompatible"
    return max(
        (classify_cuda_code_path(value, hardware_number) for value in values),
        key=CODE_PATH_RANK.__getitem__,
    )
