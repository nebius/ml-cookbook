from __future__ import annotations

import json
import unittest
from pathlib import Path

from gpu_preflight.facts import ImageFacts
from gpu_preflight.image import inspect_with_image
from gpu_preflight.profiles import load_profiles
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
                        "has_compatible_sibling": False,
                        "has_jit_fallback": False,
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


if __name__ == "__main__":
    unittest.main()
