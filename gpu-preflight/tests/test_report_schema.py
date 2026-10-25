from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

from gpu_preflight.cuda_inspection import CudaArtifactEvidence
from gpu_preflight.facts import ImageFacts
from gpu_preflight.image import (
    ImageManifestSet,
    KernelCatalogEvidence,
    PlatformManifest,
    PulledImageFacts,
    RootfsScan,
    inspect_with_image,
)
from gpu_preflight.profiles import load_profiles
from gpu_preflight.render import render_html, render_json, render_text
from jsonschema import Draft202012Validator


class ReportSchemaTests(unittest.TestCase):
    def test_schema_is_valid_and_accepts_generated_static_report(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

        reference = "example/image@sha256:0123456789abcdef"
        facts = ImageFacts(
            path=f"image://{reference}",
            reference=reference,
            final_platform="linux/arm64",
            cuda_targets=("compute_103", "sm_103"),
            cuda_versions=("12.9",),
            nccl_versions=("2.27.5",),
            detected_packages=(),
            environment={},
        )
        report = inspect_with_image(facts, (load_profiles()["gb300"],), scan_mode="none")
        validator = Draft202012Validator(schema)
        validator.validate(report.to_dict())
        target_payload = report.to_dict()["targets"][0]
        self.assertNotIn("confidence", target_payload)
        self.assertNotIn("assessment_confidence", target_payload)
        self.assertNotIn("evidence_coverage", target_payload)
        self.assertIn("artifact_inspection_coverage_percent", target_payload)
        self.assertIn("artifact_inspection_covered_candidates", target_payload)
        self.assertIn("artifact_inspection_eligible_candidates", target_payload)

        image_payload = report.to_dict()
        image_payload["observations"].update(
            {
                "image_scan_mode": "filesystem",
                "available_platforms": ["linux/arm64"],
                "selected_manifests": {
                    "gb300": {
                        "platform": "linux/arm64",
                        "digest": "sha256:example",
                        "compressed_size_bytes": 1024,
                    }
                },
                "image_packages": ["pytorch"],
                "installed_packages": ["torch==2.8.0"],
                "image_cuda_targets": ["sm_103"],
                "verified_image_cuda_targets": ["sm_103"],
                "unverified_image_cuda_target_strings": [],
                "image_cuda_versions": ["12.9"],
                "image_nccl_versions": ["2.27.5"],
                "nccl_evidence": [
                    {
                        "platform": "linux/arm64",
                        "version": "2.27.5",
                        "source": "system-package",
                        "name": "libnccl2",
                        "path": "/var/lib/dpkg/status",
                    }
                ],
                "cuda_artifacts": [
                    {
                        "platform": "linux/arm64",
                        "path": "/usr/lib/libtorch_cuda.so",
                        "size_bytes": 1024,
                        "targets": ["sm_103"],
                        "method": "cuobjdump",
                        "complete": True,
                        "role": "runtime",
                        "diagnostic": None,
                        "incomplete_reason": None,
                    }
                ],
                "cuda_compatibility_exceptions": [
                    {
                        "target": "gb300",
                        "platform": "linux/arm64",
                        "kind": "runtime-artifact",
                        "path": "/usr/lib/liboptional_cuda.so",
                        "targets": ["sm_90"],
                        "size_bytes": 2048,
                        "member_count": None,
                        "member_paths": [],
                        "component": "Other CUDA extensions",
                        "classification": "runtime-relevance-not-statically-established",
                        "reason": "no-compatible-target",
                        "reason_summary": (
                            "Inspected binary has no compatible embedded CUDA target"
                        ),
                        "status_effect": "review-required",
                    }
                ],
                "cuda_compatibility_exception_groups": [
                    {
                        "target": "gb300",
                        "platform": "linux/arm64",
                        "component": "Other CUDA extensions",
                        "classification": "runtime-relevance-not-statically-established",
                        "reason": "no-compatible-target",
                        "reason_summary": (
                            "Inspected binary has no compatible embedded CUDA target"
                        ),
                        "status_effect": "review-required",
                        "exception_count": 1,
                        "file_count": 1,
                        "artifact_count": 1,
                        "catalog_count": 0,
                        "representative_paths": ["/usr/lib/liboptional_cuda.so"],
                        "inventory_reference": "observations.cuda_compatibility_exceptions",
                    }
                ],
                "skipped_cuda_candidates": [
                    {
                        "platform": "linux/arm64",
                        "path": "/usr/lib/liboptional_cuda.so",
                        "size_bytes": 2048,
                        "reason": "scan-budget",
                    }
                ],
                "kernel_catalogs": [
                    {
                        "platform": "linux/arm64",
                        "group": "/opt/flashinfer/cubins/example",
                        "kind": "precompiled-cubin",
                        "member_count": 2,
                        "target_variants": 1,
                        "declared_targets": ["sm_90"],
                        "samples_inspected": 1,
                        "samples_incomplete": 0,
                        "verified_targets": ["sm_90"],
                        "member_paths": [
                            "/opt/flashinfer/cubins/example/a_sm90.cubin",
                            "/opt/flashinfer/cubins/example/b_sm90.cubin",
                        ],
                    }
                ],
                "rootfs_statistics": {
                    "linux/arm64": {
                        "files_seen": 10,
                        "candidate_bytes_scanned": 1024,
                        "metadata_bytes_scanned": 512,
                        "candidate_scan_truncated": False,
                        "metadata_scan_truncated": False,
                        "cuda_candidate_files": 1,
                        "cuobjdump_version": "CUDA 13.0 (V13.0.88)",
                        "cuobjdump_files_analyzed": 1,
                        "cuobjdump_targets_found": 1,
                        "cuobjdump_no_device_code": 0,
                        "cuobjdump_incomplete": 0,
                        "cuobjdump_incomplete_reasons": {
                            "timeout": 0,
                            "analyzer-error": 0,
                            "io-error": 0,
                            "other": 0,
                        },
                        "marker_only_files": 0,
                        "skipped_no_cuobjdump": 0,
                        "skipped_non_runtime": 0,
                        "skipped_scan_budget": 0,
                        "skipped_artifact_limit": 0,
                        "skipped_file_limit": 0,
                    }
                },
                "cuda_target_summaries": {
                    "gb300": {
                        "verified_target_count": 1,
                        "base_target": "sm_103",
                        "base_target_present": True,
                        "specialized_targets_present": [],
                        "specialized_targets_absent": ["sm_103f", "sm_103a"],
                    }
                },
            }
        )
        image_payload["targets"][0]["findings"][0]["details"] = [
            {
                "label": "Compatible older base cubin",
                "value": 6,
                "context": "sm_100",
            }
        ]
        validator.validate(image_payload)


class _StaticImageClient:
    """Serve one fixed manifest and rootfs so a full filesystem report can be rendered."""

    def __init__(self, manifest: PlatformManifest, pulled: PulledImageFacts) -> None:
        self.manifest = manifest
        self.pulled = pulled

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


class FilesystemReportSchemaTests(unittest.TestCase):
    """Validate a report carrying the full exception inventory against the schema.

    The report built here has a review-required artifact, a toolkit library
    assessed by release, an incomplete inspection and a fully inspected kernel
    catalog, so every required field of a cudaCompatibilityException is filled
    from real rendering rather than by hand.
    """

    def test_schema_accepts_a_filesystem_report_with_exceptions(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)

        rootfs = RootfsScan(
            known_packages=("pytorch",),
            installed_packages=("torch 2.9.0",),
            cuda_targets=("sm_100", "sm_103"),
            cuda_versions=("13.4",),
            nccl_versions=("2.27.5",),
            files_seen=4,
            bytes_scanned=4096,
            metadata_bytes_scanned=512,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=4,
            cuda_artifacts=(
                CudaArtifactEvidence(
                    path="/opt/torch/lib/libtorch_cuda.so",
                    size_bytes=1024,
                    targets=("sm_100", "sm_103"),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                # No compatible target: drives a review-required exception record.
                CudaArtifactEvidence(
                    path="/usr/lib/libcustom_cuda.so",
                    size_bytes=2048,
                    targets=("sm_70",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                # Toolkit library judged by release: drives an informational record.
                CudaArtifactEvidence(
                    path="/usr/local/cuda-13.4/lib64/libnppc.so.13",
                    size_bytes=4096,
                    targets=("sm_70",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                CudaArtifactEvidence(
                    path="/usr/lib/libbroken_cuda.so",
                    size_bytes=8192,
                    targets=(),
                    method="cuobjdump",
                    complete=False,
                    role="runtime",
                    diagnostic="cuobjdump reported an analyzer error",
                    incomplete_reason="analyzer-error",
                ),
            ),
            cuobjdump_available=True,
            cuobjdump_version="CUDA 13.4 (V13.4.92)",
            cuobjdump_files_analyzed=4,
            cuobjdump_targets_found=3,
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=1,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
            skipped_cuda_candidates=(),
            # A fully inspected catalog with no compatible member exercises the
            # catalog branch of the exception inventory, which the artifact
            # records above do not reach.
            kernel_catalogs=(
                KernelCatalogEvidence(
                    group="/opt/pkg/cubins/attention",
                    kind="precompiled-cubin",
                    member_count=2,
                    target_variants=1,
                    declared_targets=("sm_80",),
                    samples_inspected=1,
                    samples_incomplete=0,
                    verified_targets=("sm_80",),
                    member_paths=("/opt/pkg/cubins/attention/a.sm_80.cubin",),
                ),
            ),
            kernel_catalog_members=2,
            kernel_catalog_grouped_members=1,
        )
        manifest = PlatformManifest(
            platform="linux/arm64",
            digest="sha256:" + "a" * 64,
            config_digest="sha256:" + "b" * 64,
            compressed_size_bytes=1024,
        )
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=manifest.digest,
            environment={"CUDA_VERSION": "13.4"},
            known_packages=("pytorch",),
            installed_packages=("torch 2.9.0",),
            cuda_targets=("sm_100", "sm_103"),
            cuda_versions=("13.4",),
            nccl_versions=("2.27.5",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        facts = ImageFacts(
            path="image://example/image:latest",
            reference="example/image:latest",
            final_platform="linux/arm64",
            cuda_targets=(),
            cuda_versions=(),
            nccl_versions=(),
            detected_packages=(),
            environment={},
        )
        report = inspect_with_image(
            facts,
            (load_profiles()["gb300"],),
            scan_mode="filesystem",
            client=_StaticImageClient(manifest, pulled),
        )

        exceptions = report.observations["cuda_compatibility_exceptions"]
        self.assertTrue(exceptions, "expected the report to carry exception records")
        self.assertIn("review-required", {str(item["status_effect"]) for item in exceptions})
        # Both construction sites must be represented, so both are schema-checked.
        self.assertEqual(
            {str(item["kind"]) for item in exceptions},
            {"runtime-artifact", "kernel-catalog"},
        )
        # Each record carries the platform its evidence was observed on.
        self.assertEqual({str(item["platform"]) for item in exceptions}, {"linux/arm64"})

        validator.validate(report.to_dict())
        validator.validate(json.loads(render_json(report)))

        # The other renderings must not raise or drop the evidence.
        text = render_text(report)
        self.assertIn("GPF-CUDA-016", text)
        page = render_html(report)
        self.assertIn("<html", page.lower())
        self.assertNotIn("Traceback", text + page)


if __name__ == "__main__":
    unittest.main()


class HostileImageContentRenderingTests(unittest.TestCase):
    """Paths, diagnostics and environment values come from the scanned image.

    A hostile image can put HTML in any of them, so the rendered report must
    escape them rather than emit live markup.
    """

    PAYLOADS = (
        "<script>alert(1)</script>",
        '"onload=alert(2)"',
        "<img src=x onerror=alert(3)>",
        "' onmouseover='alert(4)",
        "</pre></div><script>alert(5)</script>",
    )

    def _report(self):
        artifacts = tuple(
            CudaArtifactEvidence(
                path=f"/opt/{payload}/lib{index}_cuda.so",
                size_bytes=1024,
                targets=("sm_70",),
                method="cuobjdump",
                complete=True,
                role="runtime",
                diagnostic=f"analyzer said {payload}",
            )
            for index, payload in enumerate(self.PAYLOADS)
        )
        rootfs = RootfsScan(
            known_packages=(f"pkg{self.PAYLOADS[0]}",),
            installed_packages=(f"installed {self.PAYLOADS[1]}",),
            cuda_targets=("sm_70",),
            cuda_versions=("12.9",),
            nccl_versions=(),
            files_seen=len(artifacts),
            bytes_scanned=1024,
            metadata_bytes_scanned=0,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=len(artifacts),
            cuda_artifacts=artifacts,
            cuobjdump_available=True,
            cuobjdump_version="CUDA 13.4 (V13.4.92)",
            cuobjdump_files_analyzed=len(artifacts),
            cuobjdump_targets_found=len(artifacts),
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=0,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
            skipped_cuda_candidates=(),
        )
        manifest = PlatformManifest(
            platform="linux/amd64",
            digest="sha256:" + "c" * 64,
            config_digest="sha256:" + "d" * 64,
            compressed_size_bytes=1024,
        )
        pulled = PulledImageFacts(
            platform="linux/amd64",
            digest=manifest.digest,
            environment={"TORCH_CUDA_ARCH_LIST": f"9.0 {self.PAYLOADS[2]}"},
            known_packages=rootfs.known_packages,
            installed_packages=rootfs.installed_packages,
            cuda_targets=("sm_70",),
            cuda_versions=("12.9",),
            nccl_versions=(),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        facts = ImageFacts(
            path="image://example/hostile:latest",
            reference="example/hostile:latest",
            final_platform="linux/amd64",
            cuda_targets=(),
            cuda_versions=(),
            nccl_versions=(),
            detected_packages=(),
            environment={},
        )
        return inspect_with_image(
            facts,
            (load_profiles()["h100"],),
            scan_mode="filesystem",
            client=_StaticImageClient(manifest, pulled),
        )

    def test_html_report_escapes_image_supplied_strings(self) -> None:
        page = render_html(self._report())

        # Only parsing distinguishes a live element from the same characters
        # rendered harmlessly as escaped text.
        class Collector(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.tags: list[str] = []
                self.attributes: list[tuple[str, str]] = []

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                self.attributes.extend((name, value or "") for name, value in attrs)

        collector = Collector()
        collector.feed(page)
        collector.close()

        self.assertNotIn("script", collector.tags, "image content produced a <script> element")
        self.assertNotIn("img", collector.tags, "image content produced an <img> element")
        for name, value in collector.attributes:
            self.assertFalse(
                name.startswith("on"),
                f"image content produced an event-handler attribute: {name}={value}",
            )
            self.assertNotIn("alert(", value, f"payload reached attribute {name}")

        # The evidence must survive, escaped, so the report still names the path.
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert", page)

    def test_json_report_round_trips_image_supplied_strings(self) -> None:
        document = json.loads(render_json(self._report()))
        blob = json.dumps(document)
        self.assertIn("script", blob)
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json"
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(document)
