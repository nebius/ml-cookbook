"""Regressions for the defects found by replaying the scanner over a public image corpus.

Each test names the image that exposed its defect. The scanner is judged on images nobody
curated for it, so the cases that matter are the ones a hand-written fixture would not
have thought to build: an executable with no extension, firmware for a processor inside
the GPU, a deliberately corrupt file a packaging tool ships on purpose.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import unittest
from unittest.mock import patch

from gpu_preflight.facts import elf_architecture, is_gpu_firmware
from gpu_preflight.image import (
    DockerImageClient,
    ImageInspectionError,
    PlatformManifest,
    PulledImageFacts,
    scan_rootfs_tar,
)


def commandless_facts() -> tuple[PlatformManifest, PulledImageFacts]:
    manifest = PlatformManifest(
        platform="linux/amd64",
        digest="sha256:" + "0" * 64,
        config_digest=None,
        compressed_size_bytes=0,
    )
    config = PulledImageFacts(
        platform="linux/amd64",
        digest=manifest.digest,
        environment={},
        known_packages=(),
        installed_packages=(),
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        user=None,
        entrypoint=(),
        command=(),
    )
    return manifest, config


class CommandlessImageTests(unittest.TestCase):
    """docker/docker-model-backend-llamacpp:v0.0.34-cuda declares no CMD and no ENTRYPOINT.

    docker create refuses such an image, so the scan died before reading a single file.
    The export container is never started, so any command satisfies it.
    """

    def test_export_supplies_an_entrypoint_that_cannot_run(self) -> None:
        client = DockerImageClient(cuobjdump_mode="off")
        manifest, config = commandless_facts()
        # An empty container ID aborts right after create, which is all this needs.
        with (
            patch.object(client, "_run", return_value="") as run,
            self.assertRaises(ImageInspectionError),
        ):
            client.scan_filesystem(
                "example/commandless:latest", manifest, config, max_scan_bytes=None
            )
        arguments = run.call_args.args[0]
        self.assertIn("--entrypoint", arguments)
        entrypoint = arguments[arguments.index("--entrypoint") + 1]
        self.assertTrue(entrypoint.startswith("/"))
        self.assertIn("never_executed", entrypoint)


class GpuFirmwareTests(unittest.TestCase):
    """nvcr.io/nvidia/driver ships RISC-V GSP firmware under /lib/firmware/nvidia.

    It runs on a processor inside the GPU, so counting it as a host CPU dependency
    put an architecture the host never executes into the report.
    """

    def test_driver_firmware_is_not_a_host_dependency(self) -> None:
        for path in (
            "/lib/firmware/nvidia/615.71.09/gsp_ga10x.bin",
            "usr/lib/firmware/nvidia/580.65.06/gsp_tu10x.bin",
        ):
            self.assertTrue(is_gpu_firmware(path, "riscv64"), path)

    def test_an_ordinary_riscv_binary_still_answers_for_itself(self) -> None:
        for path in (
            "/usr/lib/riscv64-linux-gnu/libfoo.so",
            "/opt/vendor/gsp_ga10x.bin",
            "/lib/firmware/nvidia/615.71.09/notgsp.bin",
        ):
            self.assertFalse(is_gpu_firmware(path, "riscv64"), path)

    def test_the_path_alone_does_not_excuse_another_architecture(self) -> None:
        self.assertFalse(is_gpu_firmware("/lib/firmware/nvidia/615.71.09/gsp_ga10x.bin", "amd64"))


def elf_bytes(machine: int = 0x3E, payload: bytes = b"") -> bytes:
    """Build a minimal 64-bit little-endian ELF header plus optional payload."""
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[6] = 1
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = machine.to_bytes(2, "little")
    return bytes(header) + payload


class GpuCodeObjectTests(unittest.TestCase):
    """nvcr.io/nvidia/pytorch:26.10 ships 5,623 AITER .co objects for AMD GPUs.

    They are ELF, but EM_AMDGPU, so the host CPU never loads them. Read as host
    libraries they drowned the two files that really were built for another CPU.
    """

    def test_amd_device_code_is_not_a_host_architecture(self) -> None:
        self.assertIsNone(elf_architecture(elf_bytes(machine=0xE0)))

    def test_cuda_device_code_is_not_a_host_architecture(self) -> None:
        self.assertIsNone(elf_architecture(elf_bytes(machine=0xBE)))

    def test_host_architectures_are_still_read(self) -> None:
        self.assertEqual(elf_architecture(elf_bytes(machine=0x3E)), "amd64")
        self.assertEqual(elf_architecture(elf_bytes(machine=0xB7)), "arm64")


def tar_of(members: tuple[tuple[str, bytes], ...]) -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    stream.seek(0)
    return stream


class CondaMetadataTests(unittest.TestCase):
    """dklabjan/keras-gpu stores gzip-compressed records under conda-meta.

    json.loads raises UnicodeDecodeError on those bytes, not JSONDecodeError, so the
    exception escaped the scan and the image produced no report at all.
    """

    def _packages(self, data: bytes) -> tuple[str, ...]:
        scan = scan_rootfs_tar(
            tar_of((("root/miniconda/conda-meta/keras-2.2.4-py36_0.json", data),)),
            None,
            cuobjdump=None,
        )
        return scan.installed_packages

    def test_a_gzipped_record_is_read_rather_than_crashing(self) -> None:
        record = json.dumps({"name": "keras", "version": "2.2.4"}).encode()
        self.assertEqual(self._packages(gzip.compress(record)), ("keras==2.2.4",))

    def test_a_plain_record_is_still_read(self) -> None:
        record = json.dumps({"name": "keras", "version": "2.2.4"}).encode()
        self.assertEqual(self._packages(record), ("keras==2.2.4",))

    def test_an_unreadable_record_is_skipped_without_aborting_the_scan(self) -> None:
        for data in (b"\x1f\x8bnot really gzip", b"\xa3\xff\x00 not utf-8", b"{"):
            self.assertEqual(self._packages(data), (), data)
