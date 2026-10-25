from __future__ import annotations

import unittest

from gpu_preflight.facts import (
    ImageFacts,
    UcxModuleGroup,
    detect_hpcx_versions,
    detect_packages,
    detect_ucx_versions,
)
from gpu_preflight.image import (
    _ucx_config_evidence,
    _ucx_package_evidence,
    _ucx_path_evidence,
    hpcx_versions,
    ucx_module_groups,
    ucx_versions,
)
from gpu_preflight.inspection import inspect
from gpu_preflight.profiles import load_profiles

HPCX_ROOT = "opt/hpcx-v2.21.1-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64"


def _facts(**overrides) -> ImageFacts:
    base = {
        "path": "image.tar",
        "reference": "registry.example/image@sha256:" + "0" * 64,
        "final_platform": "linux/amd64",
        "cuda_targets": ("sm_103",),
        "cuda_versions": ("12.9",),
        "nccl_versions": ("2.27.7",),
        "detected_packages": (),
        "environment": {},
    }
    base.update(overrides)
    return ImageFacts(**base)


def _ucx_findings(facts: ImageFacts, target: str = "b300"):
    profile = load_profiles()[target]
    return {
        finding.id: finding
        for finding in inspect(facts, (profile,)).targets[0].findings
        if finding.id.startswith("GPF-UCX")
    }


class DetectionTests(unittest.TestCase):
    def test_packages_are_detected(self) -> None:
        self.assertIn("ucx", detect_packages("/opt/hpcx/ucx/lib/libucp.so.0"))
        self.assertIn("hpcx", detect_packages("/opt/hpcx/ucx/lib/libucp.so.0"))
        self.assertIn("ucx", detect_packages("libucx0"))

    def test_hpcx_bundle_version(self) -> None:
        self.assertEqual(detect_hpcx_versions(f"/{HPCX_ROOT}/ucx/lib"), {"2.21.1"})
        self.assertEqual(detect_hpcx_versions("", {"HPCX_DIR": "/opt/hpcx-v2.23-gcc"}), {"2.23"})

    def test_ucx_package_version(self) -> None:
        self.assertEqual(detect_ucx_versions("libucx0==1.18.0"), {"1.18.0"})
        self.assertEqual(detect_ucx_versions("ucx-1.17.0-1.el8"), {"1.17.0"})
        self.assertEqual(detect_ucx_versions("", {"UCX_VERSION": "1.19.0"}), {"1.19.0"})

    def test_soname_is_not_a_version(self) -> None:
        # libucp.so.0.0.0 is an ABI soname, not a UCX release. Treating it as a
        # version would invent evidence and could satisfy an adapter floor falsely.
        self.assertEqual(detect_ucx_versions("libucp.so.0.0.0"), set())


class PathEvidenceTests(unittest.TestCase):
    def test_hpcx_bundle_is_recorded_once_per_root(self) -> None:
        evidence = set()
        for suffix in ("ucx/lib/libucp.so.0", "ucx/lib/ucx/libuct_cuda.so", "ompi/bin/mpirun"):
            evidence |= _ucx_path_evidence(f"{HPCX_ROOT}/{suffix}")
        bundles = [item for item in evidence if item.kind == "bundle"]
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].version, "2.21.1")
        self.assertEqual(bundles[0].path, f"/{HPCX_ROOT}")

    def test_runtime_libraries_carry_no_version(self) -> None:
        evidence = _ucx_path_evidence("usr/lib/x86_64-linux-gnu/libucp.so.0.0.0")
        self.assertEqual({item.kind for item in evidence}, {"runtime-library"})
        self.assertIsNone(next(iter(evidence)).version)

    def test_distribution_and_bundle_module_layouts(self) -> None:
        evidence = (
            *_ucx_path_evidence(f"{HPCX_ROOT}/ucx/lib/ucx/libuct_cuda.so"),
            *_ucx_path_evidence(f"{HPCX_ROOT}/ucx/lib/ucx/libuct_ib.so"),
            *_ucx_path_evidence("usr/lib/x86_64-linux-gnu/ucx/libuct_ib.so"),
        )
        groups = ucx_module_groups(tuple(evidence))
        self.assertEqual(
            [(group.directory, group.modules, group.has_cuda_transport) for group in groups],
            [
                (f"/{HPCX_ROOT}/ucx/lib/ucx", ("libuct_cuda.so", "libuct_ib.so"), True),
                ("/usr/lib/x86_64-linux-gnu/ucx", ("libuct_ib.so",), False),
            ],
        )

    def test_unrelated_paths_are_ignored(self) -> None:
        for path in (
            "usr/lib/python3/site-packages/torch/lib/libtorch.so",
            "usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "opt/flux/lib/libflux.so",
        ):
            with self.subTest(path=path):
                self.assertEqual(_ucx_path_evidence(path), set())

    def test_versions_are_split_by_family(self) -> None:
        evidence = (
            *_ucx_path_evidence(f"{HPCX_ROOT}/ucx/lib/libucp.so.0"),
            *_ucx_package_evidence(
                "libucx0==1.18.0", source="system-package", path="/var/lib/dpkg/status"
            ),
        )
        self.assertEqual(ucx_versions(tuple(evidence)), ("1.18.0",))
        self.assertEqual(hpcx_versions(tuple(evidence)), ("2.21.1",))

    def test_config_environment_evidence(self) -> None:
        evidence = _ucx_config_evidence({"HPCX_DIR": "/opt/hpcx-v2.23-gcc", "PATH": "/usr/bin"})
        self.assertEqual({item.version for item in evidence}, {"2.23"})


# Paths taken verbatim from `docker export` of two real images: NGC
# nvcr.io/nvidia/pytorch:25.04-py3, which installs HPC-X at an unversioned
# /opt/hpcx, and `FROM ubuntu:24.04` plus `apt-get install libucx0`.
NGC_HPCX_PATHS = (
    "opt/hpcx/VERSION",
    "opt/hpcx/ucx/lib/libucm.so.0.0.0",
    "opt/hpcx/ucx/lib/libucp.so.0.0.0",
    "opt/hpcx/ucx/lib/libuct.so.0.0.0",
    "opt/hpcx/ucx/lib/ucx/libucm_cuda.so.0.0.0",
    "opt/hpcx/ucx/lib/ucx/libuct_cuda.so.0.0.0",
    "opt/hpcx/ucx/lib/ucx/libuct_cuda_gdrcopy.so.0.0.0",
    "opt/hpcx/ucx/lib/ucx/libuct_ib.so.0.0.0",
    "opt/hpcx/ucx/lib/ucx/libucx_perftest_cuda.so.0.0.0",
    "opt/hpcx/ucc/lib/ucc/libucc_tl_cuda.so.1.0.0",
)
UBUNTU_LIBUCX_PATHS = (
    "usr/lib/x86_64-linux-gnu/libucp.so.0.0.0",
    "usr/lib/x86_64-linux-gnu/libuct.so.0.0.0",
    "usr/lib/x86_64-linux-gnu/ucx/libuct_cma.so.0",
    "usr/lib/x86_64-linux-gnu/ucx/libuct_ib.so.0.0.0",
    "usr/lib/x86_64-linux-gnu/ucx/libuct_rdmacm.so.0",
    "usr/lib/x86_64-linux-gnu/ucx/libucx_perftest_rocm.so.0.0.0",
)


def _groups_for(paths: tuple[str, ...]) -> tuple[UcxModuleGroup, ...]:
    evidence: set = set()
    for path in paths:
        evidence |= _ucx_path_evidence(path)
    return ucx_module_groups(tuple(evidence))


class ObservedLayoutTests(unittest.TestCase):
    def test_ngc_hpcx_layout(self) -> None:
        groups = _groups_for(NGC_HPCX_PATHS)
        self.assertEqual(len(groups), 1, "UCC modules must not form a UCX group")
        group = groups[0]
        self.assertEqual(group.directory, "/opt/hpcx/ucx/lib/ucx")
        self.assertTrue(group.has_cuda_transport)
        # Sonames collapse: three files on disk are one module here.
        self.assertEqual(group.modules.count("libuct_cuda.so"), 1)

    def test_ngc_version_comes_from_environment_not_the_path(self) -> None:
        # NGC installs the bundle at an unversioned /opt/hpcx, so the directory
        # name yields nothing and the environment has to supply the release.
        self.assertEqual(detect_hpcx_versions("/opt/hpcx/ucx/lib"), set())
        evidence = _ucx_config_evidence({"HPCX_VERSION": "2.16rc4", "OPENUCX_VERSION": "1.15.0"})
        self.assertEqual(hpcx_versions(tuple(evidence)), ("2.16",))
        self.assertEqual(ucx_versions(tuple(evidence)), ("1.15.0",))

    def test_hpcx_version_file_contents(self) -> None:
        self.assertEqual(detect_hpcx_versions("HPC-X v2.16"), {"2.16"})

    def test_perftest_module_is_not_a_cuda_transport(self) -> None:
        # libucx_perftest_cuda.so ships in the same directory but is a benchmark,
        # so it must not satisfy the CUDA transport check on its own.
        self.assertFalse(
            UcxModuleGroup("/d", ("libucx_perftest_cuda.so", "libuct_ib.so")).has_cuda_transport
        )

    def test_distribution_libucx_lacks_cuda_transport(self) -> None:
        groups = _groups_for(UBUNTU_LIBUCX_PATHS)
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0].has_cuda_transport)
        findings = _ucx_findings(
            _facts(
                detected_packages=("ucx",),
                ucx_versions=("1.16.0",),
                ucx_module_groups=groups,
            )
        )
        self.assertEqual(findings["GPF-UCX-002"].severity, "warning")


class FindingTests(unittest.TestCase):
    def test_absent_ucx_is_silent(self) -> None:
        # NCCL reaches InfiniBand through its own plugin, so a training image with
        # no UCX is ordinarily correct and must not be flagged.
        self.assertEqual(_ucx_findings(_facts(detected_packages=("pytorch", "nccl"))), {})

    def test_nixl_without_ucx_warns(self) -> None:
        findings = _ucx_findings(_facts(detected_packages=("nixl", "pytorch")))
        self.assertEqual(set(findings), {"GPF-UCX-003"})
        self.assertEqual(findings["GPF-UCX-003"].severity, "warning")

    def test_nixl_with_ucx_is_quiet(self) -> None:
        findings = _ucx_findings(
            _facts(detected_packages=("nixl", "ucx"), ucx_versions=("1.18.0",))
        )
        self.assertNotIn("GPF-UCX-003", findings)

    def test_modular_build_without_cuda_transport_warns(self) -> None:
        findings = _ucx_findings(
            _facts(
                detected_packages=("ucx",),
                ucx_versions=("1.18.0",),
                ucx_module_groups=(
                    UcxModuleGroup("/usr/lib/x86_64-linux-gnu/ucx", ("libuct_ib.so",)),
                ),
            )
        )
        self.assertEqual(findings["GPF-UCX-002"].severity, "warning")

    def test_cuda_transport_present_is_quiet(self) -> None:
        findings = _ucx_findings(
            _facts(
                detected_packages=("ucx",),
                ucx_versions=("1.18.0",),
                ucx_module_groups=(
                    UcxModuleGroup("/opt/ucx/lib/ucx", ("libuct_cuda.so", "libuct_ib.so")),
                ),
            )
        )
        self.assertNotIn("GPF-UCX-002", findings)

    def test_non_modular_build_is_not_flagged(self) -> None:
        # A UCX built with --enable-modules=no links CUDA support into libuct and
        # ships no libuct_* objects at all, so there is nothing to conclude.
        findings = _ucx_findings(
            _facts(
                detected_packages=("ucx",),
                ucx_versions=("1.18.0",),
                ucx_module_groups=(UcxModuleGroup("/opt/ucx/lib/ucx", ()),),
            )
        )
        self.assertNotIn("GPF-UCX-002", findings)

    def test_release_below_adapter_floor_warns(self) -> None:
        findings = _ucx_findings(_facts(detected_packages=("hpcx",), hpcx_versions=("2.16.2",)))
        self.assertEqual(findings["GPF-UCX-004"].severity, "warning")
        self.assertIn("ConnectX-8", findings["GPF-UCX-004"].evidence)

    def test_release_at_adapter_floor_passes(self) -> None:
        findings = _ucx_findings(_facts(detected_packages=("hpcx",), hpcx_versions=("2.21.1",)))
        self.assertIn("GPF-UCX-005", findings)
        self.assertNotIn("GPF-UCX-004", findings)

    def test_connectx_7_floor_is_lower(self) -> None:
        # HPC-X 2.16.2 predates ConnectX-8 but supports ConnectX-7, so the same
        # image passes on H100 and fails on B300.
        facts = _facts(detected_packages=("hpcx",), hpcx_versions=("2.16.2",))
        self.assertIn("GPF-UCX-005", _ucx_findings(facts, target="h100"))
        self.assertIn("GPF-UCX-004", _ucx_findings(facts, target="b300"))

    def test_adapter_without_floor_is_not_evaluated(self) -> None:
        # No HPC-X release announces ConnectX-9, so Vera Rubin records no floor.
        findings = _ucx_findings(
            _facts(detected_packages=("hpcx",), hpcx_versions=("2.16.2",)), target="vr"
        )
        self.assertNotIn("GPF-UCX-004", findings)
        self.assertNotIn("GPF-UCX-005", findings)

    def test_undetermined_release_is_an_evidence_gap(self) -> None:
        findings = _ucx_findings(
            _facts(
                detected_packages=("ucx",),
                ucx_module_groups=(
                    UcxModuleGroup("/usr/lib/ucx", ("libuct_cuda.so", "libuct_ib.so")),
                ),
            )
        )
        self.assertIn("GPF-UCX-006", findings)
        self.assertNotIn("GPF-UCX-001", findings)

    def test_evidence_limitation_is_reported(self) -> None:
        profile = load_profiles()["b300"]
        target = inspect(_facts(detected_packages=("ucx",)), (profile,)).targets[0]
        self.assertTrue(
            any("UCX" in item for item in target.evidence_limitations),
            target.evidence_limitations,
        )

    def test_ucx_transport_is_a_required_hardware_check(self) -> None:
        for identifier, profile in load_profiles().items():
            with self.subTest(profile=identifier):
                self.assertIn("ucx-transport", profile.runtime_checks)


if __name__ == "__main__":
    unittest.main()
