from __future__ import annotations

import re
import unittest
from dataclasses import replace
from pathlib import Path

from gpu_preflight.facts import ImageFacts, LibraryArchitectureEvidence
from gpu_preflight.inspection import (
    FINDING_DISPLAY_ORDER,
    Finding,
    FindingDetail,
    artifact_inspection_coverage_percent_for,
    compatibility_status_for,
    evidence_limitations_for,
    inspect,
)
from gpu_preflight.profiles import RuntimePackageFloor, load_profiles


def image_facts(
    *,
    reference: str = "example/image@sha256:0123456789abcdef",
    platform: str | None = None,
    cuda_targets: tuple[str, ...] = (),
    cuda_versions: tuple[str, ...] = (),
    nccl_versions: tuple[str, ...] = (),
    installed_packages: tuple[str, ...] = (),
    library_architectures: tuple[LibraryArchitectureEvidence, ...] = (),
    environment: dict[str, str] | None = None,
) -> ImageFacts:
    return ImageFacts(
        path=f"image://{reference}",
        reference=reference,
        final_platform=platform,
        cuda_targets=cuda_targets,
        cuda_versions=cuda_versions,
        nccl_versions=nccl_versions,
        detected_packages=(),
        environment=environment or {},
        installed_packages=installed_packages,
        library_architectures=library_architectures,
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


class RuntimePackageFindingTests(unittest.TestCase):
    """Triton, CuTe DSL and friends generate device code when the workload runs. An
    image can inspect at full artifact coverage while the kernels that will actually
    execute do not exist yet, so their installed version is assessed on its own.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profiles()

    def _ids(self, installed: tuple[str, ...], target: str = "gb300") -> dict[str, str]:
        facts = image_facts(platform="linux/arm64", installed_packages=installed)
        report = inspect(facts, (self.profiles[target],)).targets[0]
        return {finding.id: finding.evidence for finding in report.findings}

    def test_a_version_below_the_floor_is_a_warning(self) -> None:
        # Triton 3.5.0 is the release whose sm_103 support regressed; the floor the
        # release notes support is 3.5.1, so the broken release is caught by it.
        found = self._ids(("triton==3.5.0",))
        self.assertIn("GPF-PKG-002", found)
        self.assertIn("3.5.1", found["GPF-PKG-002"])

    def test_a_version_at_the_floor_passes(self) -> None:
        self.assertIn("GPF-PKG-001", self._ids(("triton==3.5.1",)))

    def test_a_package_the_image_does_not_install_says_nothing(self) -> None:
        found = self._ids(("numpy==2.1.0",))
        self.assertEqual({key for key in found if key.startswith("GPF-PKG")}, set())

    def test_a_package_published_for_other_hardware_is_a_warning(self) -> None:
        # No shipped profile needs this today: the libraries whose support is bounded
        # by generation are compiled ones, and artifact inspection names those. The
        # mechanism is kept for a JIT package that is ever restricted the same way.
        profile = replace(
            self.profiles["gb300"],
            runtime_packages=(
                RuntimePackageFloor(
                    package="example-dsl",
                    role="dsl",
                    minimum_version=None,
                    note=None,
                    source=None,
                    applicability="not-applicable",
                ),
            ),
        )
        facts = image_facts(platform="linux/arm64", installed_packages=("example-dsl==1.0.0",))
        found = {f.id for f in inspect(facts, (profile,)).targets[0].findings}
        self.assertIn("GPF-PKG-005", found)
        self.assertNotIn("GPF-PKG-003", found)

    def test_an_unfloored_package_is_reported_as_unassessed(self) -> None:
        found = self._ids(("numba==0.62.0",))
        self.assertIn("GPF-PKG-003", found)
        self.assertIn("0.62.0", found["GPF-PKG-003"])
        self.assertNotIn("GPF-PKG-002", found)

    def test_a_compiled_library_is_left_to_artifact_inspection(self) -> None:
        # flash-attn and nixl ship cubins, so cuobjdump answers them and the package
        # layer stays silent rather than duplicating a roster it would have to curate.
        found = self._ids(("flash_attn==2.8.3", "nixl==1.5.2"))
        self.assertEqual({key for key in found if key.startswith("GPF-PKG")}, set())

    def test_a_distribution_name_is_normalized_before_matching(self) -> None:
        # dist-info and dpkg disagree about separators and case.
        for record in ("nvidia_cutlass_dsl==4.4.0", "NVIDIA-Cutlass-DSL==4.4.0"):
            with self.subTest(record=record):
                self.assertIn("GPF-PKG-001", self._ids((record,)))

    def test_two_installed_versions_are_reported_as_unresolved(self) -> None:
        # The MiMo vLLM image carried NCCL 2.28.9 in its Python prefix and 2.30.4 in
        # the system one. Which copy a process loads is decided at runtime.
        found = self._ids(("triton==3.5.1", "triton==3.4.0"))
        self.assertIn("GPF-PKG-004", found)
        self.assertIn("3.4.0", found["GPF-PKG-004"])
        self.assertIn("GPF-PKG-001", found)

    def test_the_same_image_is_judged_against_each_target_floor(self) -> None:
        # CuTe DSL 4.3.1 clears B300 and not GB300, from the same inventory.
        installed = ("nvidia-cutlass-dsl==4.3.1",)
        self.assertIn("GPF-PKG-001", self._ids(installed, target="b300"))
        self.assertIn("GPF-PKG-002", self._ids(installed, target="gb300"))

    def test_a_package_with_no_published_floor_reports_as_unassessed(self) -> None:
        # Hopper predates Triton's published release notes, so no floor can be cited.
        # The package is still named, because silence would read as "never looked".
        found = self._ids(("triton==3.5.1",), target="h100")
        self.assertIn("GPF-PKG-003", found)
        self.assertNotIn("GPF-PKG-002", found)

    def test_rubin_judges_triton_against_its_announced_release(self) -> None:
        self.assertIn("GPF-PKG-002", self._ids(("triton==3.7.1",), target="vr"))
        self.assertIn("GPF-PKG-001", self._ids(("triton==3.8.0",), target="vr"))


class LibraryArchitectureFindingTests(unittest.TestCase):
    """GPF-CPU-002 checks the manifest's declared architecture against the profile. It
    cannot see inside the image, which is why an ARM64 image carrying an x86-64
    extension used to pass every architecture check the scan performed.

    Foreign binaries alone are not the signal, though. Real images ship them on
    purpose: clang's i386 compiler-rt runtimes (278 of them in TensorRT-LLM), CUDA's
    32-bit interceptors, a wheel's cross-compilation stubs. What separates a packaging
    error from a toolchain payload is whether the foreign binary carries device code.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profiles()["gb300"]

    def _ids(self, *architectures: LibraryArchitectureEvidence) -> dict[str, str]:
        facts = image_facts(platform="linux/arm64", library_architectures=architectures)
        report = inspect(facts, (self.profile,)).targets[0]
        return {finding.id: finding.evidence for finding in report.findings}

    def test_a_foreign_binary_with_kernels_is_a_warning(self) -> None:
        found = self._ids(
            LibraryArchitectureEvidence("arm64", 2000, ("/usr/lib/libtorch_cuda.so",), 900),
            LibraryArchitectureEvidence("amd64", 2, ("/opt/app/_ext.so",), 2),
        )
        self.assertIn("GPF-CPU-004", found)
        self.assertIn("/opt/app/_ext.so", found["GPF-CPU-004"])
        self.assertIn("2 of 2002", found["GPF-CPU-004"])
        self.assertNotIn("GPF-CPU-005", found)
        self.assertNotIn("GPF-CPU-006", found)

    def test_foreign_binaries_without_kernels_are_only_evidence(self) -> None:
        # The TensorRT-LLM case: 278 i386 compiler-rt runtimes, no device code,
        # nothing for this GPU to be wrong about.
        found = self._ids(
            LibraryArchitectureEvidence("arm64", 2069, ("/usr/lib/libtorch_cuda.so",), 150),
            LibraryArchitectureEvidence(
                "i386", 278, ("/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-i386.so",)
            ),
        )
        self.assertIn("GPF-CPU-006", found)
        self.assertIn("278 of 2347", found["GPF-CPU-006"])
        self.assertNotIn("GPF-CPU-004", found)

    def test_one_bearing_group_outranks_a_harmless_one(self) -> None:
        found = self._ids(
            LibraryArchitectureEvidence("arm64", 100, ("/usr/lib/a.so",), 50),
            LibraryArchitectureEvidence("i386", 278, ("/usr/lib/rt-i386.so",)),
            LibraryArchitectureEvidence("amd64", 1, ("/opt/app/_ext.so",), 1),
        )
        self.assertIn("GPF-CPU-004", found)
        # The warning names the binary that matters, not the 278 that do not.
        self.assertIn("/opt/app/_ext.so", found["GPF-CPU-004"])
        self.assertNotIn("rt-i386", found["GPF-CPU-004"])

    def test_a_uniform_image_passes(self) -> None:
        found = self._ids(LibraryArchitectureEvidence("arm64", 2000, ("/usr/lib/a.so",), 80))
        self.assertIn("GPF-CPU-005", found)
        self.assertNotIn("GPF-CPU-004", found)
        self.assertNotIn("GPF-CPU-006", found)

    def test_the_declared_architecture_decides_what_is_foreign(self) -> None:
        # The same binaries are native on an amd64 image and foreign on an arm64 one.
        amd64 = (LibraryArchitectureEvidence("amd64", 5, ("/usr/lib/a.so",), 5),)
        facts = image_facts(platform="linux/amd64", library_architectures=amd64)
        ids = {f.id for f in inspect(facts, (load_profiles()["b300"],)).targets[0].findings}
        self.assertIn("GPF-CPU-005", ids)
        self.assertIn("GPF-CPU-004", self._ids(*amd64))

    def test_nothing_is_claimed_when_no_binary_was_inspected(self) -> None:
        found = self._ids()
        for identifier in ("GPF-CPU-004", "GPF-CPU-005", "GPF-CPU-006"):
            self.assertNotIn(identifier, found)

    def test_an_unresolved_platform_makes_no_architecture_claim(self) -> None:
        facts = image_facts(
            library_architectures=(LibraryArchitectureEvidence("amd64", 5, ("/a.so",), 5),)
        )
        ids = {f.id for f in inspect(facts, (self.profile,)).targets[0].findings}
        for identifier in ("GPF-CPU-004", "GPF-CPU-005", "GPF-CPU-006"):
            self.assertNotIn(identifier, ids)


class LibraryResolutionEvidenceTests(unittest.TestCase):
    """Which copy of a duplicated library a process loads is decided at start-up. The
    scan can read the image's own LD_LIBRARY_PATH; the rest belongs to the hardware run.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profiles()["gb300"]

    def test_the_declared_search_path_is_quoted(self) -> None:
        facts = image_facts(
            platform="linux/arm64",
            installed_packages=("triton==3.5.1", "triton==3.8.0"),
            environment={"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"},
        )
        report = inspect(facts, (self.profile,)).targets[0]
        evidence = {f.id: f.evidence for f in report.findings}
        self.assertIn("LD_LIBRARY_PATH=/usr/local/nvidia/lib64", evidence["GPF-PKG-004"])

    def test_an_unset_search_path_is_not_mentioned(self) -> None:
        facts = image_facts(
            platform="linux/arm64", installed_packages=("triton==3.5.1", "triton==3.8.0")
        )
        report = inspect(facts, (self.profile,)).targets[0]
        evidence = {f.id: f.evidence for f in report.findings}
        # The mechanism is still named in the generic sentence; only the quoted
        # value is absent, because there is none to quote.
        self.assertIn("LD_LIBRARY_PATH and sys.path", evidence["GPF-PKG-004"])
        self.assertNotIn("LD_LIBRARY_PATH=", evidence["GPF-PKG-004"])

    def test_every_profile_requires_capturing_resolved_libraries(self) -> None:
        for name, profile in load_profiles().items():
            with self.subTest(profile=name):
                self.assertIn("library-resolution", profile.runtime_checks)
