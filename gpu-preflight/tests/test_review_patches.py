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
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from gpu_preflight.compatibility import evaluate_cuda_target, summarize_compatibility
from gpu_preflight.cuda_inspection import read_ptx_declaration, resolve_cuobjdump
from gpu_preflight.facts import ImageFacts, elf_architecture, is_gpu_firmware
from gpu_preflight.image import (
    DockerImageClient,
    ImageInspectionError,
    PlatformManifest,
    PulledImageFacts,
    scan_rootfs_tar,
)
from gpu_preflight.inspection import _target_findings
from gpu_preflight.profiles import load_profiles


def unresolved_facts(final_platform: str | None = None) -> ImageFacts:
    return ImageFacts(
        path="",
        reference="example/image:latest",
        final_platform=final_platform,
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        detected_packages=(),
        environment={},
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
            client.scan_filesystem("example/commandless:latest", manifest, config)
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


def listing_cuobjdump(directory: str, capture: Path | None = None) -> object:
    """A cuobjdump stand-in reporting one sm_103 image, optionally saving its input."""
    executable = Path(directory) / "cuobjdump"
    # The analyzer is invoked with flags first and the file last.
    copy = (
        f'for a in "$@"; do last=$a; done; cp "$last" "{capture}"\n' if capture is not None else ""
    )
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then '
        "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
        f"{copy}"
        "echo 'ELF file    1: payload.1.sm_103.cubin'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return resolve_cuobjdump(str(executable), timeout_seconds=10, fallback_mode="off")


class ElfDiscoveryTests(unittest.TestCase):
    """tensorflow/serving keeps its whole workload in /usr/bin/tensorflow_model_server.

    An executable carries no suffix to recognize it by, so a name-only candidate gate
    never saw the one file in that image that actually runs.
    """

    def _scan(self, name: str, data: bytes, directory: str):
        return scan_rootfs_tar(tar_of(((name, data),)), cuobjdump=listing_cuobjdump(directory))

    def test_an_extensionless_executable_is_discovered_by_its_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan = self._scan("usr/bin/tensorflow_model_server", elf_bytes(), directory)
        self.assertEqual(scan.cuda_candidate_files, 1)
        self.assertEqual(
            [a.path for a in scan.cuda_artifacts], ["/usr/bin/tensorflow_model_server"]
        )

    def test_a_file_that_is_not_elf_is_still_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan = self._scan("usr/share/doc/readme", b"just text, no header at all", directory)
        self.assertEqual(scan.cuda_candidate_files, 0)

    def test_a_file_too_short_to_hold_a_header_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan = self._scan("usr/bin/tiny", b"\x7fELF", directory)
        self.assertEqual(scan.cuda_candidate_files, 0)

    def test_a_discovered_artifact_reaches_the_analyzer_whole(self) -> None:
        """The probe consumes bytes a tar stream cannot give back, so they must replay."""
        payload = bytes(range(256)) * 40
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / "captured.bin"
            tool = listing_cuobjdump(directory, capture=captured)
            scan_rootfs_tar(
                tar_of((("usr/bin/server", elf_bytes(payload=payload)),)), cuobjdump=tool
            )
            self.assertEqual(captured.read_bytes(), elf_bytes(payload=payload))


class StandalonePtxTests(unittest.TestCase):
    """chrstnhntschl/gpu_burn ships /root/compare.ptx, which it compiles at run time.

    cuobjdump finds no embedded device code in PTX source, which is the expected answer
    and establishes nothing. The module declares its own architecture at the top.
    """

    def _declare(self, text: str) -> tuple[tuple[str, ...], str | None]:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compare.ptx"
            source.write_text(text, encoding="utf-8")
            return read_ptx_declaration(source)

    def test_a_target_becomes_a_jit_compilable_virtual_architecture(self) -> None:
        targets, version = self._declare("//\n.version 6.5\n.target sm_52\n.address_size 64\n")
        self.assertEqual(targets, ("compute_52",))
        self.assertEqual(version, "6.5")

    def test_options_in_the_target_list_are_not_architectures(self) -> None:
        targets, _ = self._declare(".version 7.0\n.target sm_70, debug, map_f64_to_f32\n")
        self.assertEqual(targets, ("compute_70",))

    def test_an_architecture_conditional_target_keeps_its_suffix(self) -> None:
        targets, _ = self._declare(".version 8.3\n.target sm_90a\n")
        self.assertEqual(targets, ("compute_90a",))

    def test_several_architectures_are_all_reported(self) -> None:
        targets, _ = self._declare(".version 8.3\n.target sm_80, sm_90\n")
        self.assertEqual(targets, ("compute_80", "compute_90"))

    def test_a_module_without_a_target_declares_nothing(self) -> None:
        targets, version = self._declare(".version 6.5\n// no target directive here\n")
        self.assertEqual(targets, ())
        self.assertEqual(version, "6.5")

    def test_an_older_target_is_compatible_with_a_newer_gpu(self) -> None:
        """PTX is the one artifact where an older target is good news, not a shortfall."""
        profile = load_profiles()["b300"]
        summary = summarize_compatibility(("compute_52",), profile.compute_capability_number)
        self.assertTrue(summary.compatible)

    def test_an_architecture_conditional_target_does_not_carry_forward(self) -> None:
        profile = load_profiles()["b300"]
        summary = summarize_compatibility(("compute_90a",), profile.compute_capability_number)
        self.assertFalse(summary.compatible)

    def test_a_ptx_file_in_an_image_yields_verified_evidence(self) -> None:
        source = b".version 6.5\n.target sm_52\n.address_size 64\n.visible .entry burn()\n"
        with tempfile.TemporaryDirectory() as directory:
            scan = scan_rootfs_tar(
                tar_of((("root/compare.ptx", source),)),
                cuobjdump=listing_cuobjdump(directory),
            )
        self.assertEqual(len(scan.cuda_artifacts), 1)
        artifact = scan.cuda_artifacts[0]
        self.assertEqual(artifact.path, "/root/compare.ptx")
        self.assertEqual(artifact.method, "ptx-source")
        self.assertTrue(artifact.complete)
        self.assertEqual(artifact.targets, ("compute_52",))
        self.assertIn("compiles it at run time", artifact.diagnostic)

    def test_unreadable_ptx_is_an_evidence_gap_not_a_clean_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then '
                "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
                "echo 'does not contain device code' >&2; exit 1\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            tool = resolve_cuobjdump(str(executable), timeout_seconds=10, fallback_mode="off")
            scan = scan_rootfs_tar(
                tar_of((("root/opaque.ptx", b"// nothing a parser can use\n" * 4),)),
                cuobjdump=tool,
            )
        self.assertEqual(len(scan.cuda_artifacts), 1)
        artifact = scan.cuda_artifacts[0]
        self.assertFalse(artifact.complete)
        self.assertEqual(artifact.incomplete_reason, "analyzer-error")
        self.assertIn("uncompiled source", artifact.diagnostic)


class PtxPreambleReadTests(unittest.TestCase):
    """A file named .ptx is not always PTX, and the reader has to cope either way."""

    def _declaration(self, body: str) -> tuple[tuple[str, ...], str | None]:
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "kernel.ptx"
            module.write_text(body, encoding="utf-8")
            return read_ptx_declaration(module)

    def test_the_declaration_is_read_from_the_preamble(self) -> None:
        self.assertEqual(
            self._declaration(".version 8.5\n.target sm_90\n.address_size 64\n"),
            (("compute_90",), "8.5"),
        )

    def test_the_read_stops_once_the_module_body_starts(self) -> None:
        """Without a stop, a .ptx declaring no target is read to its last byte.

        berkeleydsep/datahub-scipy-ml-notebook ships eighteen of them.
        """
        body = ".version 8.5\n.visible .entry k()\n" + "add.s32 %r1, %r2, %r3;\n" * 50000
        targets, version = self._declaration(body)
        self.assertEqual(targets, ())
        self.assertEqual(version, "8.5")

    def test_a_file_that_is_not_ptx_at_all_declares_nothing(self) -> None:
        self.assertEqual(self._declaration("\x7fELF not really ptx\n" * 100), ((), None))

    def test_an_architecture_conditional_target_keeps_its_suffix(self) -> None:
        self.assertEqual(self._declaration(".target sm_90a\n")[0], ("compute_90a",))

    def test_several_targets_on_one_directive_are_all_read(self) -> None:
        self.assertEqual(
            self._declaration(".target sm_80, sm_90\n")[0], ("compute_80", "compute_90")
        )


class PtxJitFloorTests(unittest.TestCase):
    """gpu-burn ships /root/compare.ptx declaring .target sm_20.

    Reading it was the point of the standalone-PTX work, and having read it the scan
    called compute_20 JIT-compatible with sm_103 and asked only for a native build.
    No driver that runs a B300 can assemble compute_20: CUDA 12.8's Virtual
    Architecture Feature List begins at compute_50 and CUDA 13's at compute_75.
    """

    def test_fermi_ptx_cannot_jit_on_hardware_no_toolkit_pairs_it_with(self) -> None:
        for target in ("compute_20", "compute_21", "compute_30", "compute_35"):
            evaluation = evaluate_cuda_target(target, 103)
            self.assertFalse(evaluation.compatible, target)
            self.assertEqual(evaluation.classification, "incompatible", target)

    def test_the_band_only_cuda_12_assembles_is_not_a_clean_pass(self) -> None:
        for target in ("compute_50", "compute_61", "compute_70"):
            evaluation = evaluate_cuda_target(target, 103)
            self.assertTrue(evaluation.compatible, target)
            self.assertEqual(evaluation.classification, "ptx-jit-driver-dependent", target)

    def test_ptx_both_toolkits_assemble_is_unchanged(self) -> None:
        self.assertEqual(
            evaluate_cuda_target("compute_75", 103).classification, "ptx-jit-compatible"
        )
        self.assertEqual(
            evaluate_cuda_target("compute_90", 103).classification, "ptx-jit-compatible"
        )

    def test_a_newer_virtual_architecture_is_still_refused(self) -> None:
        self.assertFalse(evaluate_cuda_target("compute_120", 103).compatible)

    def test_an_image_whose_only_ptx_is_fermi_is_incompatible(self) -> None:
        summary = summarize_compatibility(("compute_20",), 103)
        assert summary is not None
        self.assertFalse(summary.compatible)
        self.assertEqual(summary.classification, "incompatible")

    def test_a_native_cubin_still_outranks_a_driver_dependent_jit(self) -> None:
        summary = summarize_compatibility(("compute_52", "sm_103"), 103)
        assert summary is not None
        self.assertEqual(summary.classification, "native")


class DriverDependentPtxFindingTests(unittest.TestCase):
    """The action for PTX a CUDA 13 driver refuses is not "add a native build"."""

    @staticmethod
    def _findings(targets: tuple[str, ...]) -> dict[str, object]:
        facts = unresolved_facts(final_platform="linux/amd64")
        facts = replace(facts, cuda_targets=targets)
        return {finding.id: finding for finding in _target_findings(facts, load_profiles()["b300"])}

    def test_the_driver_release_is_the_thing_to_establish(self) -> None:
        finding = self._findings(("compute_61",))["GPF-CUDA-004"]
        self.assertIn("CUDA 13", finding.remediation)
        self.assertIn("compute_75", finding.remediation)
        self.assertIn("older driver", finding.title)

    def test_an_ordinary_jit_still_asks_only_for_a_native_build(self) -> None:
        finding = self._findings(("compute_90",))["GPF-CUDA-004"]
        self.assertNotIn("CUDA 13", finding.remediation)
        self.assertIn("native", finding.remediation)
        self.assertEqual(finding.title, "CUDA code is compatible but not native")


class WheelToolkitProvenanceTests(unittest.TestCase):
    """jrottenberg/ffmpeg cleared four NPP copies with a CUDA release found elsewhere.

    A library's CUDA release has to come from something that names that library: its
    versioned install directory, or the wheel whose RECORD lists it.
    """

    SITE = "usr/local/lib/python3.12/site-packages"

    def _releases(self, members: tuple[tuple[str, bytes], ...]) -> dict[str, str]:
        scan = scan_rootfs_tar(tar_of(members), cuobjdump=None)
        return dict(scan.wheel_toolkit_releases)

    def _record(self, distribution: str, version: str, *paths: str) -> tuple[str, bytes]:
        body = "".join(f"{path},sha256=0,1\n" for path in paths)
        return (
            f"{self.SITE}/{distribution}-{version}.dist-info/RECORD",
            (body + "x/__init__.py,,\n").encode(),
        )

    def test_a_wheel_versioned_as_the_toolkit_names_the_release(self) -> None:
        releases = self._releases(
            (
                self._record(
                    "nvidia_cuda_runtime_cu12",
                    "12.4.127",
                    "nvidia/cuda_runtime/lib/libcudart.so.12",
                ),
                self._record("nvidia_cublas_cu12", "12.4.5.8", "nvidia/cublas/lib/libcublas.so.12"),
            )
        )
        self.assertEqual(releases.get(f"/{self.SITE}/nvidia/cublas/lib/libcublas.so.12"), "12.4")

    def test_a_component_version_is_not_read_as_a_cuda_release(self) -> None:
        """nvidia-cufft-cu12 is 11.2.1.3; reading 11.2 out of it invents a release."""
        releases = self._releases(
            (self._record("nvidia_cufft_cu12", "11.2.1.3", "nvidia/cufft/lib/libcufft.so.11"),)
        )
        self.assertEqual(releases, {})

    def test_cuda_13_wheels_dropped_the_suffix_and_are_still_authoritative(self) -> None:
        """runpod/pytorch installs nvidia-cuda-runtime 13.0.48, with no -cu13 on it.

        NVIDIA moved the release into the module path at CUDA 13, so a rule that
        demanded the suffix recognized no authoritative wheel at all. Nothing in
        site-packages got a release, and libcufft.so.12 -- byte-identical to the copy
        under /usr/local/cuda-13.0/ that the versioned path excused -- was sent to
        review on its own sm_75-only code.
        """
        releases = self._releases(
            (
                self._record("nvidia_cuda_runtime", "13.0.48", "nvidia/cu13/lib/libcudart.so.13"),
                self._record("nvidia_cufft", "12.0.0.15", "nvidia/cu13/lib/libcufft.so.12"),
            )
        )
        self.assertEqual(releases.get(f"/{self.SITE}/nvidia/cu13/lib/libcufft.so.12"), "13.0")

    def test_a_wheel_set_takes_the_release_its_runtime_wheel_pins(self) -> None:
        """A cu12 wheel set in an image that reports a newer 12.x elsewhere."""
        releases = self._releases(
            (
                self._record(
                    "nvidia_cuda_runtime_cu12",
                    "12.4.127",
                    "nvidia/cuda_runtime/lib/libcudart.so.12",
                ),
                self._record("nvidia_cufft_cu12", "11.2.1.3", "nvidia/cufft/lib/libcufft.so.11"),
                ("usr/local/cuda/version.txt", b"CUDA Version 12.9.41\n"),
            )
        )
        # 12.4, the release the wheel set was built against, not the 12.9 installed
        # elsewhere in the image.
        self.assertEqual(releases.get(f"/{self.SITE}/nvidia/cufft/lib/libcufft.so.11"), "12.4")

    def test_a_library_outside_any_wheel_is_unresolved(self) -> None:
        releases = self._releases((("usr/local/lib/libnppc.so.12", elf_bytes()),))
        self.assertEqual(releases, {})

    def test_a_record_entry_cannot_name_a_path_outside_its_tree(self) -> None:
        releases = self._releases(
            (
                self._record(
                    "nvidia_cuda_runtime_cu12",
                    "12.4.127",
                    "../../../../usr/local/lib/libcublas.so.12",
                    "/etc/libcublas.so.12",
                ),
            )
        )
        self.assertEqual(releases, {})


def silent_cuobjdump(directory: str) -> object:
    """A cuobjdump stand-in that completes and finds no device code."""
    executable = Path(directory) / "cuobjdump"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then '
        "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
        "echo 'does not contain device code' >&2; exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return resolve_cuobjdump(str(executable), timeout_seconds=10, fallback_mode="off")
