from __future__ import annotations

import unittest

from gpu_preflight.facts import (
    detect_cuda_targets,
    detect_cuda_versions,
    detect_nccl_versions,
    detect_packages,
    elf_architecture,
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


class ElfArchitectureTests(unittest.TestCase):
    """The image manifest declares one architecture for the whole image. The ELF header
    of each binary says what that binary was actually built for, and the two can
    disagree -- a wheel built for the wrong platform, or an extension copied between
    architectures, leaves a library the dynamic loader will refuse.
    """

    @staticmethod
    def _header(machine: int, *, elf_class: int = 2, data: int = 1) -> bytes:
        return (
            b"\x7fELF"
            + bytes([elf_class, data, 1, 0])
            + b"\x00" * 8
            + (3).to_bytes(2, "little")
            + machine.to_bytes(2, "little" if data == 1 else "big")
        )

    def test_known_machines_use_the_profile_vocabulary(self) -> None:
        # These must read the same way as a profile's hardware.cpu_architectures,
        # or every comparison against it reports a false mismatch.
        self.assertEqual(elf_architecture(self._header(0xB7)), "arm64")
        self.assertEqual(elf_architecture(self._header(0x3E)), "amd64")

    def test_a_big_endian_header_is_read_in_its_own_byte_order(self) -> None:
        self.assertEqual(elf_architecture(self._header(0x16, data=2)), "s390x")

    def test_an_unlisted_machine_is_named_rather_than_assumed_native(self) -> None:
        self.assertEqual(elf_architecture(self._header(0x999)), "elf-machine-0x999")

    def test_a_non_elf_file_has_no_architecture(self) -> None:
        for header in (b"!<arch>\n" + b"x" * 20, b"\x00" * 32, b""):
            with self.subTest(header=header[:8]):
                self.assertIsNone(elf_architecture(header))

    def test_a_header_shorter_than_e_machine_is_not_guessed_at(self) -> None:
        self.assertIsNone(elf_architecture(b"\x7fELF" + b"\x00" * 10))


class CudaObjectArchitectureTests(unittest.TestCase):
    """A .cubin is an ELF whose e_machine is EM_CUDA (190). It holds device code, runs
    on no CPU, and is loaded by the driver rather than by ld.so. Reporting one as a
    foreign-architecture host binary told us vLLM's FlashInfer cubins were unloadable
    on the very GPU they were built for.
    """

    def test_a_cuda_object_has_no_host_architecture(self) -> None:
        header = (
            b"\x7fELF"
            + bytes([2, 1, 1, 0])
            + b"\x00" * 8
            + (1).to_bytes(2, "little")
            + (0xBE).to_bytes(2, "little")
        )
        self.assertIsNone(elf_architecture(header))

    def test_a_host_binary_is_still_read(self) -> None:
        # The guard must not swallow ordinary objects: a .so holding an embedded
        # fatbin is an amd64 ELF and has to keep counting as one.
        header = (
            b"\x7fELF"
            + bytes([2, 1, 1, 0])
            + b"\x00" * 8
            + (3).to_bytes(2, "little")
            + (0x3E).to_bytes(2, "little")
        )
        self.assertEqual(elf_architecture(header), "amd64")
