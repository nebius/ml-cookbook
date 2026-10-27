from __future__ import annotations

import unittest

from gpu_preflight.facts import (
    detect_cuda_targets,
    detect_cuda_versions,
    detect_nccl_versions,
    detect_packages,
)


class ImageFactsTests(unittest.TestCase):
    def test_environment_exposes_torch_cuda_targets(self) -> None:
        targets = detect_cuda_targets(
            "",
            {"TORCH_CUDA_ARCH_LIST": "9.0;10.3a+PTX"},
        )
        self.assertEqual(targets, {"sm_90", "sm_103a", "compute_103a"})

    def test_cuda_image_reference_exposes_toolkit_version(self) -> None:
        self.assertEqual(
            detect_cuda_versions("nvidia/cuda:12.9.1-runtime-ubuntu24.04"),
            {"12.9.1"},
        )

    def test_driver_requirement_is_not_an_installed_toolkit_version(self) -> None:
        self.assertEqual(
            detect_cuda_versions("", {"NVIDIA_REQUIRE_CUDA": "cuda>=12.0"}),
            set(),
        )

    def test_known_packages_are_detected_from_image_metadata(self) -> None:
        packages = detect_packages("torch==2.8.0 libnccl2 nixl-cu13 flashinfer-python")
        self.assertEqual(packages, {"flashinfer", "nccl", "nixl", "pytorch"})

    def test_nccl_versions_are_detected_from_common_package_records(self) -> None:
        versions = detect_nccl_versions(
            "libnccl2==2.27.5-1+cuda12.9 nvidia-nccl-cu13==2.32.3 nccl=1:2.28.3-1"
        )
        self.assertEqual(versions, {"2.27.5", "2.28.3", "2.32.3"})

    def test_nccl_version_is_detected_from_environment(self) -> None:
        self.assertEqual(
            detect_nccl_versions("", {"NCCL_VERSION": "2.27.7-1+cuda13.0"}),
            {"2.27.7"},
        )


if __name__ == "__main__":
    unittest.main()
