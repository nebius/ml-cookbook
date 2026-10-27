from __future__ import annotations

import re
import unittest
from pathlib import Path

from gpu_preflight.facts import ImageFacts
from gpu_preflight.inspection import (
    FINDING_DISPLAY_ORDER,
    Finding,
    FindingDetail,
    artifact_inspection_coverage_percent_for,
    compatibility_status_for,
    evidence_limitations_for,
    inspect,
)
from gpu_preflight.profiles import load_profiles


def image_facts(
    *,
    reference: str = "example/image@sha256:0123456789abcdef",
    platform: str | None = None,
    cuda_targets: tuple[str, ...] = (),
    cuda_versions: tuple[str, ...] = (),
    nccl_versions: tuple[str, ...] = (),
) -> ImageFacts:
    return ImageFacts(
        path=f"image://{reference}",
        reference=reference,
        final_platform=platform,
        cuda_targets=cuda_targets,
        cuda_versions=cuda_versions,
        nccl_versions=nccl_versions,
        detected_packages=(),
        environment={},
    )


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profiles()

    def test_all_requested_profiles_are_packaged(self) -> None:
        self.assertEqual(
            set(self.profiles),
            {"h100", "h200", "b200", "b300", "gb300", "rtx6000", "vr"},
        )

    def test_artifact_coverage_is_numeric_and_potential_artifacts_require_review(self) -> None:
        coverage = Finding(
            id="GPF-CUDA-014",
            severity="info",
            category="evidence",
            title="Runtime CUDA artifact coverage is incomplete",
            evidence="Coverage is bounded.",
            remediation="Increase scan limits.",
            details=(
                FindingDetail("Inspected with cuobjdump", 76),
                FindingDetail("Incomplete cuobjdump inspections", 0),
                FindingDetail("Not inspected: scan budget", 324),
                FindingDetail("Not inspected: artifact-size limit", 1),
                FindingDetail("Not inspected: file-count limit", 0),
                FindingDetail("Not inspected: analyzer unavailable", 0),
            ),
        )
        actionable = Finding(
            id="GPF-CUDA-016",
            severity="warning",
            category="compatibility",
            title="Runtime CUDA artifacts may lack a compatible target",
            evidence="Two runtime artifacts lack a compatible target.",
            remediation="Review the artifacts.",
        )
        findings = (coverage, actionable)

        self.assertEqual(compatibility_status_for(findings), "REVIEW_REQUIRED")
        self.assertEqual(artifact_inspection_coverage_percent_for(findings), 19.0)

    def test_uninspected_core_library_outranks_a_compatibility_warning(self) -> None:
        # A warning-level label reads as "fix these and ship". When a library every GPU job loads
        # was never read, the listed findings describe only the artifacts that could be read, so
        # the assessment is unsupported rather than merely actionable.
        core_gap = Finding(
            id="GPF-CUDA-019",
            severity="warning",
            category="evidence",
            title="Core framework libraries were not inspected",
            evidence="libnccl has no cuobjdump target listing.",
            remediation="Repeat the scan with a matching analyzer.",
        )
        nccl_warning = Finding(
            id="GPF-NCCL-003",
            severity="warning",
            category="compatibility",
            title="Packaged NCCL version predates target support",
            evidence="Detected NCCL 2.15.5.",
            remediation="Upgrade NCCL.",
        )
        self.assertEqual(
            compatibility_status_for((core_gap, nccl_warning)),
            "INSUFFICIENT_EVIDENCE",
        )
        self.assertEqual(compatibility_status_for((nccl_warning,)), "ACTION_REQUIRED")

    def test_confirmed_blocker_outranks_an_uninspected_core_library(self) -> None:
        core_gap = Finding(
            id="GPF-CUDA-019",
            severity="warning",
            category="evidence",
            title="Core framework libraries were not inspected",
            evidence="libnccl has no cuobjdump target listing.",
            remediation="Repeat the scan with a matching analyzer.",
        )
        blocker = Finding(
            id="GPF-CUDA-021",
            severity="blocker",
            category="compatibility",
            title="A library every GPU job loads has no code for this GPU",
            evidence="No target runs on sm_103.",
            remediation="Rebuild for the target.",
        )
        self.assertEqual(compatibility_status_for((core_gap, blocker)), "INCOMPATIBLE")

    def test_role_excluded_marker_strings_are_not_an_executive_limitation(self) -> None:
        finding = Finding(
            id="GPF-CUDA-012",
            severity="info",
            category="evidence",
            title="Some CUDA target records are unverified",
            evidence="All unverified records were role-excluded.",
            remediation="Retain as discovery evidence.",
            details=(
                FindingDetail("Runtime", 0),
                FindingDetail("Incomplete cuobjdump inspections", 0),
            ),
        )
        self.assertEqual(evidence_limitations_for((finding,)), ())

    def test_arm64_sm103_is_native_for_gb300(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("compute_103", "sm_103"),
            cuda_versions=("12.9.0",),
            nccl_versions=("2.27.5",),
        )
        report = inspect(facts, (self.profiles["gb300"],))
        target = report.targets[0]
        self.assertEqual(target.compatibility_status, "READY_FOR_HARDWARE_TESTING")
        self.assertEqual(target.reproducibility_status, "PINNED")
        self.assertNotIn("hardware_validation_status", report.to_dict()["targets"][0])
        self.assertIn("GPF-CUDA-003", {finding.id for finding in target.findings})
        self.assertIn("GPF-CUDA-007", {finding.id for finding in target.findings})
        self.assertIn("GPF-NCCL-002", {finding.id for finding in target.findings})

    def test_cuda_12_8_warns_for_gb300(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("sm_103",),
            cuda_versions=("12.8.1",),
        )
        report = inspect(facts, (self.profiles["gb300"],))
        target = report.targets[0]
        self.assertEqual(target.compatibility_status, "ACTION_REQUIRED")
        self.assertIn("GPF-CUDA-008", {finding.id for finding in target.findings})

    def test_native_target_suppresses_unrelated_exact_only_warning(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("sm_90a", "sm_103"),
            cuda_versions=("12.9",),
        )
        report = inspect(facts, (self.profiles["gb300"],))
        finding_ids = {finding.id for finding in report.targets[0].findings}
        self.assertIn("GPF-CUDA-003", finding_ids)
        self.assertNotIn("GPF-CUDA-005", finding_ids)

    def test_unknown_cuda_version_needs_evidence(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("sm_103",),
        )
        report = inspect(facts, (self.profiles["gb300"],))
        target = report.targets[0]
        self.assertEqual(target.compatibility_status, "INSUFFICIENT_EVIDENCE")
        self.assertIn("GPF-CUDA-006", {finding.id for finding in target.findings})

    def test_nccl_below_target_minimum_requires_action(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("sm_107",),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
        )
        target = inspect(facts, (self.profiles["vr"],)).targets[0]
        self.assertEqual(target.compatibility_status, "ACTION_REQUIRED")
        finding = next(item for item in target.findings if item.id == "GPF-NCCL-003")
        self.assertIn("2.32.3", finding.evidence)

    def test_all_packaged_nccl_candidates_must_meet_minimum(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            cuda_targets=("sm_103",),
            cuda_versions=("12.9",),
            nccl_versions=("2.26.5", "2.28.3"),
        )
        target = inspect(facts, (self.profiles["gb300"],)).targets[0]
        self.assertEqual(target.compatibility_status, "ACTION_REQUIRED")
        self.assertIn("GPF-NCCL-003", {finding.id for finding in target.findings})

    def test_missing_nccl_version_needs_evidence(self) -> None:
        facts = image_facts(
            platform="linux/amd64",
            cuda_targets=("sm_90",),
            cuda_versions=("11.8",),
        )
        target = inspect(facts, (self.profiles["h100"],)).targets[0]
        self.assertEqual(target.compatibility_status, "INSUFFICIENT_EVIDENCE")
        self.assertIn("GPF-NCCL-001", {finding.id for finding in target.findings})

    def test_target_specific_nccl_minimums(self) -> None:
        self.assertEqual(
            {key: self.profiles[key].minimum_nccl_version for key in self.profiles},
            {
                "h100": "2.15.1",
                "h200": "2.22.3",
                "b200": "2.25.1",
                "b300": "2.27.7",
                "gb300": "2.27.5",
                # No NVIDIA release note states an NCCL baseline for compute
                # capability 12.0, so RTX PRO 6000 records none.
                "rtx6000": None,
                "vr": "2.32.3",
            },
        )

    def test_nccl_baseline_is_enforced_for_every_hardware_type(self) -> None:
        cases = {
            "h100": ("linux/amd64", "sm_90", "11.8", "2.15.0", "2.15.1"),
            "h200": ("linux/amd64", "sm_90", "11.8", "2.22.2", "2.22.3"),
            "b200": ("linux/amd64", "sm_100", "12.8", "2.25.0", "2.25.1"),
            "b300": ("linux/amd64", "sm_103", "12.9", "2.27.6", "2.27.7"),
            "gb300": ("linux/arm64", "sm_103", "12.9", "2.27.4", "2.27.5"),
            "vr": ("linux/arm64", "sm_107", "13.4", "2.32.2", "2.32.3"),
        }
        for target_name, (platform, sm, cuda, old_nccl, minimum_nccl) in cases.items():
            with self.subTest(target=target_name, outcome="below-minimum"):
                target = inspect(
                    image_facts(
                        platform=platform,
                        cuda_targets=(sm,),
                        cuda_versions=(cuda,),
                        nccl_versions=(old_nccl,),
                    ),
                    (self.profiles[target_name],),
                ).targets[0]
                self.assertIn("GPF-NCCL-003", {finding.id for finding in target.findings})
            with self.subTest(target=target_name, outcome="minimum-met"):
                target = inspect(
                    image_facts(
                        platform=platform,
                        cuda_targets=(sm,),
                        cuda_versions=(cuda,),
                        nccl_versions=(minimum_nccl,),
                    ),
                    (self.profiles[target_name],),
                ).targets[0]
                self.assertIn("GPF-NCCL-002", {finding.id for finding in target.findings})

    def test_arm64_image_is_blocked_for_x86_target(self) -> None:
        facts = image_facts(
            reference="example/image:1.0",
            platform="linux/arm64",
            cuda_targets=("sm_103",),
        )
        report = inspect(facts, (self.profiles["b300"],))
        target = report.targets[0]
        self.assertEqual(target.compatibility_status, "INCOMPATIBLE")
        self.assertIn("GPF-CPU-001", {finding.id for finding in target.findings})

    def test_sm103_is_non_native_for_vr(self) -> None:
        facts = image_facts(
            reference="example/image:1.0",
            platform="linux/arm64",
            cuda_targets=("sm_103",),
        )
        report = inspect(facts, (self.profiles["vr"],))
        target = report.targets[0]
        self.assertEqual(target.compatibility_status, "ACTION_REQUIRED")
        self.assertIn("GPF-CUDA-004", {finding.id for finding in target.findings})

    def test_missing_platform_and_targets_need_evidence(self) -> None:
        facts = image_facts()
        report = inspect(facts, (self.profiles["h100"],))
        self.assertEqual(
            report.targets[0].compatibility_status,
            "INSUFFICIENT_EVIDENCE",
        )


class FindingDisplayOrderTest(unittest.TestCase):
    def test_every_emitted_finding_has_a_display_rank(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src" / "gpu_preflight"
        emitted = {
            match.group(1)
            for path in source.rglob("*.py")
            for match in re.finditer(r'id="(GPF-[A-Z]+-\d+)"', path.read_text())
        }
        self.assertTrue(emitted)
        self.assertEqual(emitted - set(FINDING_DISPLAY_ORDER), set())
        self.assertEqual(set(FINDING_DISPLAY_ORDER) - emitted, set())


if __name__ == "__main__":
    unittest.main()
