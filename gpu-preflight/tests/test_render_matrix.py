"""Every combination of scan evidence must render and validate.

The report renderers walk evidence that varies widely between images: an image
with no CUDA at all, one whose artifacts were all skipped by a limit, one whose
catalogs were never inspected. This renders the cross product of those shapes
for several hardware profiles and checks that each one produces text, HTML and
schema-valid JSON rather than raising.
"""

from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path

from gpu_preflight.cuda_inspection import CudaArtifactEvidence
from gpu_preflight.facts import ImageFacts
from gpu_preflight.image import (
    ImageManifestSet,
    KernelCatalogEvidence,
    PlatformManifest,
    PulledImageFacts,
    RootfsScan,
    SkippedCudaCandidate,
    inspect_with_image,
)
from gpu_preflight.profiles import load_profiles
from gpu_preflight.render import render_html, render_json, render_text
from jsonschema import Draft202012Validator


class _Client:
    def __init__(self, manifest: PlatformManifest, pulled: PulledImageFacts) -> None:
        self.manifest, self.pulled = manifest, pulled

    def inspect_manifest(self, reference: str) -> ImageManifestSet:
        return ImageManifestSet(reference=reference, platforms=(self.manifest,))

    def pull_config(self, reference: str, selected: PlatformManifest) -> PulledImageFacts:
        return self.pulled

    def scan_filesystem(
        self,
        reference: str,
        selected: PlatformManifest,
        config: PulledImageFacts,
        max_scan_bytes: int,
    ) -> PulledImageFacts:
        return config


def _rootfs(**overrides: object) -> RootfsScan:
    base: dict[str, object] = {
        "known_packages": (),
        "installed_packages": (),
        "cuda_targets": (),
        "cuda_versions": (),
        "nccl_versions": (),
        "files_seen": 0,
        "bytes_scanned": 0,
        "metadata_bytes_scanned": 0,
        "truncated": False,
        "metadata_truncated": False,
        "cuda_candidate_files": 0,
        "cuda_artifacts": (),
        "cuobjdump_available": True,
        "cuobjdump_version": "CUDA 13.4 (V13.4.92)",
        "cuobjdump_files_analyzed": 0,
        "cuobjdump_targets_found": 0,
        "cuobjdump_no_device_code": 0,
        "cuobjdump_incomplete": 0,
        "marker_only_files": 0,
        "skipped_no_cuobjdump": 0,
        "skipped_non_runtime": 0,
        "skipped_scan_budget": 0,
        "skipped_artifact_limit": 0,
        "skipped_file_limit": 0,
        "skipped_cuda_candidates": (),
    }
    base.update(overrides)
    return RootfsScan(**base)  # type: ignore[arg-type]


ARTIFACTS = {
    "none": (),
    "no-device-code": (
        CudaArtifactEvidence(
            path="/opt/a.so",
            size_bytes=0,
            targets=(),
            method="cuobjdump",
            complete=True,
            role="runtime",
        ),
    ),
    "incompatible": (
        CudaArtifactEvidence(
            path="/opt/b.so",
            size_bytes=1024,
            targets=("sm_50",),
            method="cuobjdump",
            complete=True,
            role="runtime",
        ),
    ),
    "incomplete": (
        CudaArtifactEvidence(
            path="/opt/c.so",
            size_bytes=1024,
            targets=(),
            method="cuobjdump",
            complete=False,
            role="runtime",
            incomplete_reason="timeout",
        ),
    ),
    "marker-only": (
        CudaArtifactEvidence(
            path="/opt/d.so",
            size_bytes=1024,
            targets=("sm_90",),
            method="marker-scan",
            complete=False,
            role="runtime",
            diagnostic="byte-marker evidence",
        ),
    ),
    "role-excluded": (
        CudaArtifactEvidence(
            path="/opt/tests/e.so",
            size_bytes=1024,
            targets=("sm_50",),
            method="cuobjdump",
            complete=True,
            role="test",
        ),
    ),
    "many-targets": (
        CudaArtifactEvidence(
            path="/opt/f.so",
            size_bytes=10**12,
            targets=tuple(f"sm_{n}" for n in (50, 70, 80, 90, 100, 103, 107, 120)),
            method="cuobjdump",
            complete=True,
            role="runtime",
        ),
    ),
}

CATALOGS = {
    "none": (),
    "incompatible": (
        KernelCatalogEvidence(
            group="/opt/pkg/cubins/x",
            kind="precompiled-cubin",
            member_count=3,
            target_variants=1,
            declared_targets=("sm_80",),
            samples_inspected=1,
            samples_incomplete=0,
            verified_targets=("sm_80",),
            member_paths=("/opt/pkg/cubins/x/a.sm_80.cubin",),
        ),
    ),
    "uninspected": (
        KernelCatalogEvidence(
            group="/opt/pkg/cubins/y",
            kind="architecture-module",
            member_count=2,
            target_variants=2,
            declared_targets=(),
            samples_inspected=0,
            samples_incomplete=2,
            verified_targets=(),
            member_paths=(),
        ),
    ),
}

# "non-runtime" skips are counted, not recorded, so they are not a reason here.
SKIPPED = {
    "none": (),
    "every-reason": tuple(
        SkippedCudaCandidate(path=f"/opt/s{index}.so", size_bytes=index * 1024, reason=reason)
        for index, reason in enumerate(
            ("scan-budget", "artifact-limit", "file-limit", "no-cuobjdump")
        )
    ),
}


class RenderMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profiles()
        schema = Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json"
        cls.validator = Draft202012Validator(json.loads(schema.read_text(encoding="utf-8")))

    def test_every_evidence_shape_renders_and_validates(self) -> None:
        combinations = itertools.product(
            ("h100", "gb300", "vr", "rtx6000"), ARTIFACTS, CATALOGS, SKIPPED
        )
        checked = 0
        for profile_id, artifact_key, catalog_key, skipped_key in combinations:
            with self.subTest(
                profile=profile_id,
                artifacts=artifact_key,
                catalogs=catalog_key,
                skipped=skipped_key,
            ):
                profile = self.profiles[profile_id]
                architecture = "arm64" if "arm64" in profile.cpu_architectures else "amd64"
                artifacts = ARTIFACTS[artifact_key]
                rootfs = _rootfs(
                    cuda_artifacts=artifacts,
                    kernel_catalogs=CATALOGS[catalog_key],
                    skipped_cuda_candidates=SKIPPED[skipped_key],
                    cuda_versions=("12.9",),
                    nccl_versions=("2.27.5",),
                    files_seen=len(artifacts),
                    cuda_candidate_files=len(artifacts),
                    cuobjdump_files_analyzed=len(artifacts),
                    cuobjdump_incomplete=sum(1 for item in artifacts if not item.complete),
                    marker_only_files=sum(1 for item in artifacts if item.method == "marker-scan"),
                    kernel_catalog_members=sum(item.member_count for item in CATALOGS[catalog_key]),
                )
                manifest = PlatformManifest(
                    platform=f"linux/{architecture}",
                    digest="sha256:" + "a" * 64,
                    config_digest=None,
                    compressed_size_bytes=0,
                )
                pulled = PulledImageFacts(
                    platform=f"linux/{architecture}",
                    digest=manifest.digest,
                    environment={"TORCH_CUDA_ARCH_LIST": "9.0+PTX"},
                    known_packages=(),
                    installed_packages=(),
                    cuda_targets=(),
                    cuda_versions=("12.9",),
                    nccl_versions=("2.27.5",),
                    user=None,
                    entrypoint=(),
                    command=(),
                    rootfs=rootfs,
                )
                facts = ImageFacts(
                    path="image://example/image:1",
                    reference="example/image:1",
                    final_platform=f"linux/{architecture}",
                    cuda_targets=(),
                    cuda_versions=(),
                    nccl_versions=(),
                    detected_packages=(),
                    environment={},
                )
                report = inspect_with_image(
                    facts,
                    (profile,),
                    scan_mode="filesystem",
                    client=_Client(manifest, pulled),
                )
                self.assertTrue(render_text(report))
                self.assertIn("<html", render_html(report).lower())
                self.validator.validate(json.loads(render_json(report)))
                self.validator.validate(report.to_dict())
                checked += 1
        self.assertEqual(checked, 4 * len(ARTIFACTS) * len(CATALOGS) * len(SKIPPED))


if __name__ == "__main__":
    unittest.main()
