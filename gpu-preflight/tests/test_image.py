from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from gpu_preflight.cuda_inspection import (
    CudaArtifactEvidence,
    CuobjdumpTool,
    kernel_catalog_group,
    resolve_cuobjdump,
)
from gpu_preflight.facts import ImageFacts
from gpu_preflight.image import (
    DockerImageClient,
    ImageInspectionError,
    ImageManifestSet,
    KernelCatalogEvidence,
    NcclVersionEvidence,
    PlatformManifest,
    PulledImageFacts,
    RootfsScan,
    SkippedCudaCandidate,
    _cuda_artifact_findings,
    _cuda_compatibility_exceptions,
    _cuda_exception_assessment,
    _digest_reference,
    _docker_error,
    _exact_reference,
    _redact_sensitive_diagnostic,
    _toolkit_release,
    inspect_image_reference,
    inspect_with_image,
    parse_manifest_output,
    scan_rootfs_tar,
)
from gpu_preflight.profiles import load_profiles
from gpu_preflight.render import render_html, render_text


def manifest(platform: str, digest_suffix: str, size: int = 1024) -> PlatformManifest:
    return PlatformManifest(
        platform=platform,
        digest=f"sha256:{digest_suffix}",
        config_digest=f"sha256:config-{digest_suffix}",
        compressed_size_bytes=size,
    )


def image_facts(reference: str = "registry.example/pytorch:1") -> ImageFacts:
    return ImageFacts(
        path=f"image://{reference}",
        reference=reference,
        final_platform=None,
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        detected_packages=(),
        environment={},
    )


def rootfs_with(
    artifacts: tuple[CudaArtifactEvidence, ...] = (),
    skipped: tuple[SkippedCudaCandidate, ...] = (),
) -> RootfsScan:
    return RootfsScan(
        known_packages=(),
        installed_packages=(),
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        files_seen=len(artifacts) + len(skipped),
        bytes_scanned=0,
        metadata_bytes_scanned=0,
        truncated=False,
        metadata_truncated=False,
        cuda_candidate_files=len(artifacts) + len(skipped),
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
        skipped_artifact_limit=sum(item.reason == "artifact-limit" for item in skipped),
        skipped_file_limit=0,
        skipped_cuda_candidates=skipped,
    )


class FakeImageClient:
    def __init__(
        self,
        manifests: tuple[PlatformManifest, ...],
        pulled: PulledImageFacts | None = None,
        error: str | None = None,
    ) -> None:
        self.manifests = manifests
        self.pulled = pulled
        self.error = error

    def inspect_manifest(self, reference: str) -> ImageManifestSet:
        return ImageManifestSet(reference=reference, platforms=self.manifests)

    def pull_config(
        self,
        reference: str,
        selected: PlatformManifest,
    ) -> PulledImageFacts:
        if self.error:
            raise ImageInspectionError(self.error)
        if self.pulled is None:
            raise AssertionError("pull_config was not expected")
        return self.pulled

    def scan_filesystem(
        self,
        reference: str,
        selected: PlatformManifest,
        config: PulledImageFacts,
        max_scan_bytes: int,
    ) -> PulledImageFacts:
        return config


class ImageInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profiles()

    def test_preflight_requires_docker_cli(self) -> None:
        with patch("gpu_preflight.image.shutil.which", return_value=None):
            client = DockerImageClient(cuobjdump_mode="off")
        with self.assertRaisesRegex(ImageInspectionError, "docker CLI was not found"):
            client.preflight(require_daemon=True, require_cuobjdump=True)

    def test_preflight_requires_cuobjdump(self) -> None:
        client = DockerImageClient(cuobjdump_mode="off")
        client.docker_binary = "/usr/bin/docker"
        with self.assertRaisesRegex(ImageInspectionError, "cuobjdump was not found"):
            client.preflight(require_daemon=True, require_cuobjdump=True)

    def test_preflight_requires_accessible_docker_daemon(self) -> None:
        client = DockerImageClient(cuobjdump_mode="off")
        client.docker_binary = "/usr/bin/docker"
        client.cuobjdump = CuobjdumpTool(
            executable="/usr/local/cuda/bin/cuobjdump",
            version="CUDA 13.0",
            timeout_seconds=30,
        )
        with (
            patch.object(
                client,
                "_run",
                side_effect=ImageInspectionError("cannot connect to daemon"),
            ),
            self.assertRaisesRegex(ImageInspectionError, "Docker daemon is not accessible"),
        ):
            client.preflight(require_daemon=True, require_cuobjdump=True)

    def test_pull_failure_is_an_operational_error(self) -> None:
        arm64 = manifest("linux/arm64", "arm64")
        with self.assertRaisesRegex(ImageInspectionError, "registry denied"):
            inspect_image_reference(
                "registry.example/private:1",
                (self.profiles["gb300"],),
                platform="linux/arm64",
                scan_mode="filesystem",
                client=FakeImageClient((arm64,), error="registry denied"),
            )

    def test_verbose_manifest_parser_extracts_platforms_and_sizes(self) -> None:
        payload = [
            {
                "Descriptor": {
                    "digest": "sha256:amd64",
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
                "SchemaV2Manifest": {
                    "config": {"digest": "sha256:config-amd64"},
                    "layers": [{"size": 100}, {"size": 200}],
                },
            },
            {
                "Descriptor": {
                    "digest": "sha256:arm64",
                    "platform": {"architecture": "arm64", "os": "linux"},
                },
                "OCIManifest": {
                    "config": {"digest": "sha256:config-arm64"},
                    "layers": [{"size": 400}],
                },
            },
        ]
        parsed = parse_manifest_output("registry.example/image:1", json.dumps(payload))
        self.assertEqual(
            tuple(item.platform for item in parsed.platforms),
            ("linux/amd64", "linux/arm64"),
        )
        self.assertEqual(parsed.platforms[0].compressed_size_bytes, 300)
        self.assertEqual(parsed.platforms[1].config_digest, "sha256:config-arm64")

    def test_registry_diagnostics_redact_credentials(self) -> None:
        sanitized = _redact_sensitive_diagnostic(
            "Authorization: Bearer secret-value password=hunter2 "
            "https://user:private@registry.example/v2/"
        )
        self.assertNotIn("secret-value", sanitized)
        self.assertNotIn("hunter2", sanitized)
        self.assertNotIn("private", sanitized)
        self.assertEqual(sanitized.count("[REDACTED]"), 3)

    def test_pull_logs_progress_to_stderr_before_downloading(self) -> None:
        client = DockerImageClient()
        selected = manifest("linux/arm64", "arm64")
        inspected = json.dumps(
            [
                {
                    "Os": "linux",
                    "Architecture": "arm64",
                    "Config": {"Env": ["CUDA_VERSION=12.9.0", "NCCL_VERSION=2.27.5-1+cuda12.9"]},
                }
            ]
        )
        stderr = io.StringIO()
        with (
            patch.object(client, "_run", side_effect=("", inspected)),
            redirect_stderr(stderr),
        ):
            pulled = client.pull_config("registry.example/pytorch:1", selected)
        self.assertIn(
            "gpu-preflight: pulling image registry.example/pytorch:1 for linux/arm64;",
            stderr.getvalue(),
        )
        self.assertIn("compressed layers are approximately", stderr.getvalue())
        self.assertIn("is free in the local temporary filesystem", stderr.getvalue())
        self.assertEqual(pulled.nccl_versions, ("2.27.5",))
        self.assertEqual(len(pulled.nccl_evidence), 1)
        self.assertEqual(pulled.nccl_evidence[0].version, "2.27.5")
        self.assertEqual(pulled.nccl_evidence[0].source, "image-config")
        self.assertEqual(pulled.nccl_evidence[0].name, "NCCL_VERSION")

    def test_nccl_version_sources_are_exposed_in_all_report_formats(self) -> None:
        arm64 = manifest("linux/arm64", "arm64")
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"NCCL_VERSION": "2.28.3-1"},
            known_packages=("nccl",),
            installed_packages=(
                "libnccl2==2.28.3-1+cuda13.0",
                "nvidia-nccl-cu13==2.28.9",
            ),
            cuda_targets=(),
            cuda_versions=("13.0",),
            nccl_versions=("2.28.3", "2.28.9"),
            user=None,
            entrypoint=(),
            command=(),
            nccl_evidence=(
                NcclVersionEvidence(
                    version="2.28.3",
                    source="image-config",
                    name="NCCL_VERSION",
                ),
                NcclVersionEvidence(
                    version="2.28.3",
                    source="system-package",
                    name="libnccl2",
                    path="/var/lib/dpkg/status",
                ),
                NcclVersionEvidence(
                    version="2.28.9",
                    source="python-package",
                    name="nvidia-nccl-cu13",
                    path=(
                        "/usr/local/lib/python3.12/dist-packages/"
                        "nvidia_nccl_cu13-2.28.9.dist-info/METADATA"
                    ),
                ),
            ),
        )
        report = inspect_with_image(
            image_facts(reference="registry.example/sglang:1"),
            (self.profiles["gb300"],),
            scan_mode="pull",
            client=FakeImageClient((arm64,), pulled=pulled),
        )

        self.assertEqual(len(report.observations["nccl_evidence"]), 3)
        text_report = render_text(report)
        self.assertIn("NCCL installations and version evidence:", text_report)
        self.assertIn("System package: libnccl2 (/var/lib/dpkg/status)", text_report)
        self.assertIn("Python package: nvidia-nccl-cu13", text_report)
        self.assertIn("Effective loaded NCCL version: not statically established", text_report)
        html_report = render_html(report)
        self.assertIn("NCCL installations and version evidence", html_report)
        self.assertIn("nvidia-nccl-cu13", html_report)

    def test_manifest_enrichment_resolves_gb300_architecture(self) -> None:
        facts = image_facts()
        client = FakeImageClient(
            (
                manifest("linux/amd64", "amd64"),
                manifest("linux/arm64", "arm64"),
            )
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="manifest",
            client=client,
        )
        target = report.targets[0]
        finding_ids = {finding.id for finding in target.findings}
        self.assertEqual(report.observations["final_platform"], "linux/arm64")
        self.assertEqual(report.observations["available_platforms"], ["linux/amd64", "linux/arm64"])
        self.assertIn("GPF-IMAGE-001", finding_ids)
        self.assertIn("GPF-CPU-002", finding_ids)
        self.assertNotIn("GPF-CPU-003", finding_ids)

    def test_missing_target_platform_is_blocked(self) -> None:
        facts = image_facts()
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="manifest",
            client=FakeImageClient((manifest("linux/amd64", "amd64"),)),
        )
        self.assertEqual(report.targets[0].compatibility_status, "INCOMPATIBLE")
        self.assertIn(
            "GPF-IMAGE-002",
            {finding.id for finding in report.targets[0].findings},
        )

    def test_pulled_config_adds_packages_and_cuda_targets(self) -> None:
        facts = image_facts()
        arm64 = manifest("linux/arm64", "arm64")
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"TORCH_CUDA_ARCH_LIST": "10.3+PTX"},
            known_packages=("pytorch", "nccl"),
            installed_packages=("torch==2.8.0",),
            cuda_targets=("compute_103", "sm_103"),
            cuda_versions=("12.9.0",),
            nccl_versions=("2.28.3",),
            user=None,
            entrypoint=(),
            command=("python",),
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="pull",
            client=FakeImageClient((arm64,), pulled=pulled),
        )
        finding_ids = {finding.id for finding in report.targets[0].findings}
        self.assertIn("GPF-IMAGE-003", finding_ids)
        self.assertIn("GPF-CUDA-003", finding_ids)
        self.assertEqual(report.observations["image_cuda_targets"], ["compute_103", "sm_103"])
        self.assertEqual(report.observations["image_cuda_versions"], ["12.9.0"])
        self.assertEqual(report.observations["image_nccl_versions"], ["2.28.3"])
        self.assertEqual(report.observations["image_packages"], ["nccl", "pytorch"])
        self.assertIn("GPF-CUDA-007", finding_ids)
        rendered = render_text(report)
        self.assertIn("CUDA versions: 12.9.0", rendered)
        self.assertNotIn("Image CUDA versions:", rendered)

    def test_direct_image_can_select_arm64_on_any_host(self) -> None:
        client = FakeImageClient(
            (
                manifest("linux/amd64", "amd64"),
                manifest("linux/arm64", "arm64"),
            )
        )
        report = inspect_image_reference(
            "registry.example/pytorch:1",
            (self.profiles["gb300"],),
            platform="linux/arm64",
            scan_mode="manifest",
            client=client,
        )
        target = report.targets[0]
        self.assertEqual(report.source, "image://registry.example/pytorch:1")
        self.assertEqual(report.observations["final_platform"], "linux/arm64")
        self.assertIn("GPF-IMAGE-001", {finding.id for finding in target.findings})
        self.assertNotEqual(target.compatibility_status, "INCOMPATIBLE")

    def test_explicit_platform_conflict_is_blocked(self) -> None:
        report = inspect_image_reference(
            "registry.example/pytorch:1",
            (self.profiles["gb300"],),
            platform="linux/amd64",
            scan_mode="manifest",
            client=FakeImageClient((manifest("linux/amd64", "amd64"),)),
        )
        self.assertEqual(report.targets[0].compatibility_status, "INCOMPATIBLE")
        self.assertIn(
            "GPF-CPU-001",
            {finding.id for finding in report.targets[0].findings},
        )

    def test_rootfs_stream_finds_inventory_and_cuda_markers(self) -> None:
        stream = io.BytesIO()
        entries = {
            "usr/lib/python3/site-packages/torch-2.8.0.dist-info/METADATA": (
                b"Name: torch\nVersion: 2.8.0\n"
            ),
            "usr/local/lib/python3.12/dist-packages/nvidia_nccl_cu13-2.28.9.dist-info/METADATA": (
                b"Name: nvidia-nccl-cu13\nVersion: 2.28.9\n"
            ),
            "usr/local/cuda/lib/libtorch_cuda.so": b"binary-prefix sm_103 compute_103 suffix",
            "usr/local/cuda/version.json": b'{"cuda": {"version": "12.9.0"}}',
            "var/lib/dpkg/status": (b"Package: libnccl2\nVersion: 2.27.3-1+cuda12.9\n\n"),
            "usr/include/nccl.h": (
                b"#define NCCL_MAJOR 2\n#define NCCL_MINOR 28\n#define NCCL_PATCH 3\n"
            ),
        }
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, data in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        stream.seek(0)

        scan = scan_rootfs_tar(stream, max_scan_bytes=1024 * 1024)
        self.assertIn("torch==2.8.0", scan.installed_packages)
        self.assertIn("libnccl2==2.27.3-1+cuda12.9", scan.installed_packages)
        self.assertIn("pytorch", scan.known_packages)
        self.assertIn("nccl", scan.known_packages)
        self.assertEqual(scan.cuda_targets, ("compute_103", "sm_103"))
        self.assertEqual(scan.cuda_versions, ("12.9.0",))
        self.assertEqual(scan.nccl_versions, ("2.27.3", "2.28.3", "2.28.9"))
        evidence = {(item.version, item.source, item.name) for item in scan.nccl_evidence}
        self.assertIn(("2.27.3", "system-package", "libnccl2"), evidence)
        self.assertIn(("2.28.3", "header", "NCCL version macros"), evidence)
        self.assertIn(("2.28.9", "python-package", "nvidia-nccl-cu13"), evidence)
        self.assertFalse(scan.truncated)
        self.assertGreater(scan.metadata_bytes_scanned, 0)
        self.assertFalse(scan.metadata_truncated)

    def test_metadata_inventory_continues_after_binary_scan_limit(self) -> None:
        stream = io.BytesIO()
        entries = {
            "usr/local/cuda/lib/libtorch_cuda.so": b"sm_103" * 100,
            "usr/lib/python3/site-packages/torch-2.8.0.dist-info/METADATA": (
                b"Name: torch\nVersion: 2.8.0\n"
            ),
            "var/lib/dpkg/status": b"Package: libnccl2\nVersion: 2.27.3\n\n",
        }
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, data in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        stream.seek(0)

        scan = scan_rootfs_tar(stream, max_scan_bytes=8)
        self.assertTrue(scan.truncated)
        self.assertIn("torch==2.8.0", scan.installed_packages)
        self.assertIn("libnccl2==2.27.3", scan.installed_packages)
        self.assertGreater(scan.metadata_bytes_scanned, 0)
        self.assertFalse(scan.metadata_truncated)

    def test_invalid_tar_is_rejected(self) -> None:
        with self.assertRaisesRegex(ImageInspectionError, "invalid rootfs tar"):
            scan_rootfs_tar(io.BytesIO(b"not a tar archive"), max_scan_bytes=1024)

    def test_parent_path_tar_member_is_ignored(self) -> None:
        stream = io.BytesIO()
        data = b"sm_103"
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("../../torch/libtorch_cuda.so")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        stream.seek(0)
        scan = scan_rootfs_tar(stream, max_scan_bytes=1024)
        self.assertEqual(scan.cuda_candidate_files, 0)
        self.assertEqual(scan.cuda_artifacts, ())

    def test_cuobjdump_records_targets_for_each_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then '
                "echo 'Cuda compilation tools, release 13.0, V13.0.88'; exit 0; fi\n"
                "echo 'ELF file    1: kernel.1.sm_103.cubin'\n"
                "echo 'PTX file    1: kernel.1.sm_103.ptx'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            tool = resolve_cuobjdump(str(executable), timeout_seconds=2)
            self.assertIsNotNone(tool)
            self.assertEqual(tool.version, "CUDA 13.0 (V13.0.88)")

            stream = io.BytesIO()
            data = b"ELF payload without plain-text architecture markers"
            with tarfile.open(fileobj=stream, mode="w") as archive:
                info = tarfile.TarInfo("usr/lib/libtorch_cuda.so")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            stream.seek(0)

            with patch("gpu_preflight.image.BINARY_FILE_LIMIT_BYTES", 8):
                scan = scan_rootfs_tar(
                    stream,
                    max_scan_bytes=1024 * 1024,
                    cuobjdump=tool,
                    max_cuobjdump_files=4,
                    max_artifact_bytes=1024 * 1024,
                )
            self.assertTrue(scan.cuobjdump_available)
            self.assertEqual(scan.cuobjdump_files_analyzed, 1)
            self.assertEqual(scan.cuobjdump_targets_found, 1)
            self.assertEqual(scan.cuobjdump_no_device_code, 0)
            self.assertEqual(scan.cuobjdump_incomplete, 0)
            self.assertEqual(scan.bytes_scanned, len(data))
            self.assertEqual(len(scan.cuda_artifacts), 1)
            artifact = scan.cuda_artifacts[0]
            self.assertEqual(artifact.path, "/usr/lib/libtorch_cuda.so")
            self.assertEqual(artifact.targets, ("compute_103", "sm_103"))
            self.assertEqual(artifact.method, "cuobjdump")
            self.assertTrue(artifact.complete)

    def test_cuda_candidate_accounting_explains_scan_budget_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then '
                "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
                "echo 'ELF file 1: kernel.sm_103.cubin'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            tool = resolve_cuobjdump(str(executable), timeout_seconds=2)

            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name in ("usr/lib/libtorch_cuda.so", "usr/lib/libflash_cuda.so"):
                    data = b"payload"
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            stream.seek(0)

            scan = scan_rootfs_tar(stream, max_scan_bytes=1, cuobjdump=tool)
            self.assertEqual(scan.cuda_candidate_files, 2)
            self.assertEqual(scan.cuobjdump_files_analyzed, 0)
            self.assertEqual(scan.skipped_scan_budget, 2)
            self.assertEqual(
                [candidate.path for candidate in scan.skipped_cuda_candidates],
                ["/usr/lib/libtorch_cuda.so", "/usr/lib/libflash_cuda.so"],
            )
            self.assertEqual(
                scan.cuda_candidate_files,
                scan.cuobjdump_files_analyzed
                + scan.skipped_no_cuobjdump
                + scan.skipped_non_runtime
                + scan.skipped_scan_budget
                + scan.skipped_artifact_limit
                + scan.skipped_file_limit,
            )

    def test_kernel_catalog_inspects_one_representative_per_declared_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then '
                "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
                "echo 'ELF file 1: kernel.sm_103.cubin'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            tool = resolve_cuobjdump(str(executable), timeout_seconds=2)

            stream = io.BytesIO()
            names = (
                "opt/flashinfer/cubins/release/fmha/shape_a_sm100f.cubin",
                "opt/flashinfer/cubins/release/fmha/shape_b_sm100f.cubin",
                "opt/flashinfer/cubins/release/fmha/shape_c_sm103.cubin",
            )
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name in names:
                    data = b"cubin payload"
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            stream.seek(0)

            scan = scan_rootfs_tar(
                stream,
                max_scan_bytes=1024 * 1024,
                cuobjdump=tool,
                max_cuobjdump_files=8,
                max_artifact_bytes=1024 * 1024,
            )

        self.assertEqual(scan.cuda_candidate_files, 3)
        self.assertEqual(scan.cuobjdump_files_analyzed, 2)
        self.assertEqual(scan.kernel_catalog_members, 3)
        self.assertEqual(scan.kernel_catalog_grouped_members, 1)
        self.assertEqual(len(scan.kernel_catalogs), 1)
        catalog = scan.kernel_catalogs[0]
        self.assertEqual(catalog.target_variants, 2)
        self.assertEqual(catalog.samples_inspected, 2)
        self.assertEqual(catalog.verified_targets, ("sm_103",))
        self.assertEqual({artifact.role for artifact in scan.cuda_artifacts}, {"kernel-catalog"})
        self.assertEqual(scan.skipped_file_limit, 0)

        arm64 = manifest("linux/arm64", "arm64")
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "13.4", "NCCL_VERSION": "2.31.2"},
            known_packages=("sglang",),
            installed_packages=(),
            cuda_targets=("sm_103",),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=scan,
        )
        report = inspect_with_image(
            image_facts(reference="registry.example/sglang:1"),
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )
        target = report.targets[0]
        finding_ids = {finding.id for finding in target.findings}
        self.assertEqual(target.artifact_inspection_coverage_percent, 100.0)
        self.assertEqual(target.artifact_inspection_covered_candidates, 2)
        self.assertEqual(target.artifact_inspection_eligible_candidates, 2)
        self.assertIn("GPF-CUDA-018", finding_ids)
        self.assertNotIn("GPF-CUDA-016", finding_ids)
        catalog_finding = next(
            finding for finding in target.findings if finding.id == "GPF-CUDA-018"
        )
        self.assertIn("3 selectable cubin member(s)", catalog_finding.evidence)
        text_report = render_text(report)
        self.assertIn("Members grouped as selectable alternatives: 1", text_report)
        catalog_payload = report.to_dict()["observations"]["kernel_catalogs"][0]
        self.assertEqual(catalog_payload["member_count"], 3)
        self.assertEqual(catalog_payload["target_variants"], 2)
        self.assertEqual(catalog_payload["verified_targets"], ["sm_103"])
        self.assertEqual(len(catalog_payload["member_paths"]), 3)
        self.assertTrue(
            all(
                path.startswith("/opt/flashinfer/cubins/")
                for path in catalog_payload["member_paths"]
            )
        )

    def test_flashinfer_architecture_variants_are_evaluated_as_one_dispatch_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then '
                "echo 'Cuda compilation tools, release 13.4, V13.4.92'; exit 0; fi\n"
                'cat "$3"\n',
                encoding="utf-8",
            )
            executable.chmod(0o755)
            tool = resolve_cuobjdump(str(executable), timeout_seconds=2)

            root = "usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache"
            members = {
                f"{root}/batch_prefill_x/batch_prefill_x.so": (
                    b"ELF file    1: batch_prefill_x.1.sm_103a.cubin\n"
                ),
                f"{root}/batch_prefill_x_sm90/batch_prefill_x_sm90.so": (
                    b"ELF file    1: batch_prefill_x_sm90.1.sm_90.cubin\n"
                ),
                f"{root}/selective_state_update/selective_state_update.so": (
                    b"ELF file    1: selective_state_update.1.sm_80.cubin\n"
                ),
            }
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            stream.seek(0)

            scan = scan_rootfs_tar(
                stream,
                max_scan_bytes=1024 * 1024,
                cuobjdump=tool,
                max_cuobjdump_files=8,
                max_artifact_bytes=1024 * 1024,
            )

        self.assertEqual(scan.cuobjdump_files_analyzed, 3)
        self.assertEqual(scan.dispatch_module_groups, 1)
        self.assertEqual(scan.dispatch_module_members, 2)
        self.assertEqual(len(scan.kernel_catalogs), 1)
        catalog = scan.kernel_catalogs[0]
        self.assertEqual(catalog.kind, "architecture-module")
        self.assertEqual(catalog.member_count, 2)
        self.assertEqual(catalog.verified_targets, ("sm_103a", "sm_90"))
        singleton = next(
            artifact
            for artifact in scan.cuda_artifacts
            if "selective_state_update" in artifact.path
        )
        self.assertEqual(singleton.role, "runtime")

        arm64 = manifest("linux/arm64", "arm64")
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "13.4", "NCCL_VERSION": "2.31.2"},
            known_packages=("flashinfer",),
            installed_packages=(),
            cuda_targets=("sm_103a", "sm_90"),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=scan,
        )
        report = inspect_with_image(
            image_facts(reference="registry.example/flashinfer:1"),
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )
        exceptions = report.observations["cuda_compatibility_exceptions"]
        self.assertEqual(len(exceptions), 1)
        self.assertIn("selective_state_update", exceptions[0]["path"])
        dispatch_finding = next(
            finding for finding in report.targets[0].findings if finding.id == "GPF-CUDA-018"
        )
        self.assertIn("2 architecture-selected module(s)", dispatch_finding.evidence)
        warning = next(
            finding for finding in report.targets[0].findings if finding.id == "GPF-CUDA-016"
        )
        self.assertIn("1 binary review unit(s)", warning.evidence)

    def test_alternate_architecture_cache_with_ptx_fallback_does_not_drive_review(
        self,
    ) -> None:
        root = "/usr/lib/python3/dist-packages/flashinfer_jit_cache/jit_cache"
        generic = f"{root}/paged_kv/paged_kv.so"
        sm90 = f"{root}/paged_kv_sm90/paged_kv_sm90.so"
        group = kernel_catalog_group(generic)
        self.assertIsNotNone(group)
        artifacts = (
            CudaArtifactEvidence(
                path=generic,
                size_bytes=1024,
                targets=("compute_90",),
                method="cuobjdump",
                complete=True,
                role="kernel-catalog",
            ),
            CudaArtifactEvidence(
                path=sm90,
                size_bytes=1024,
                targets=("sm_90",),
                method="cuobjdump",
                complete=True,
                role="kernel-catalog",
            ),
        )
        rootfs = RootfsScan(
            known_packages=("flashinfer",),
            installed_packages=(),
            cuda_targets=("compute_90", "sm_90"),
            cuda_versions=("13.4",),
            nccl_versions=(),
            files_seen=2,
            bytes_scanned=2048,
            metadata_bytes_scanned=0,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=2,
            cuda_artifacts=artifacts,
            cuobjdump_available=True,
            cuobjdump_version="CUDA 13.4",
            cuobjdump_files_analyzed=2,
            cuobjdump_targets_found=2,
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=0,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
            kernel_catalogs=(
                KernelCatalogEvidence(
                    group=str(group),
                    kind="architecture-module",
                    member_count=2,
                    target_variants=2,
                    declared_targets=("sm_90",),
                    samples_inspected=2,
                    samples_incomplete=0,
                    verified_targets=("compute_90", "sm_90"),
                ),
            ),
            dispatch_module_members=2,
            dispatch_module_groups=1,
        )

        exceptions = _cuda_compatibility_exceptions(
            rootfs,
            self.profiles["gb300"],
            platform="linux/arm64",
            analyzer_matches_image=True,
        )
        self.assertEqual(exceptions, [])
        findings = _cuda_artifact_findings(
            rootfs,
            self.profiles["gb300"],
            ("13.4",),
            max_scan_bytes=1024 * 1024,
            max_cuobjdump_files=None,
            max_artifact_bytes=1024 * 1024,
            cuobjdump_timeout_seconds=30,
        )
        finding_ids = {finding.id for finding in findings}
        self.assertIn("GPF-CUDA-009", finding_ids)
        self.assertNotIn("GPF-CUDA-016", finding_ids)

    def test_cuda_exception_component_and_risk_classification(self) -> None:
        for path in (
            "/usr/local/cuda/lib/libcufft.so.12",
            "/usr/local/cuda/lib/libnppisu.so.13",
        ):
            self.assertEqual(
                _cuda_exception_assessment(path, "runtime-artifact"),
                (
                    "CUDA Toolkit libraries",
                    "likely-runtime-compatibility-gap",
                    "no-compatible-target",
                ),
            )
        self.assertEqual(
            _cuda_exception_assessment(
                "/opt/pytorch/ao/torchao/_C_cutlass_90a.abi3.so",
                "runtime-artifact",
                targets=("sm_90a",),
            ),
            (
                "PyTorch extensions",
                "optional-alternate-gpu-implementation",
                "alternate-architecture-only",
            ),
        )
        # sm_90a code in a module whose name does not declare it stays a normal review item.
        self.assertEqual(
            _cuda_exception_assessment(
                "/opt/project/custom_attention.so",
                "runtime-artifact",
                targets=("sm_90a",),
            )[1],
            "runtime-relevance-not-statically-established",
        )
        self.assertEqual(
            _cuda_exception_assessment(
                "/opt/flashinfer_jit_cache/jit_cache/example_sm90/example_sm90.so",
                "runtime-artifact",
            ),
            (
                "FlashInfer",
                "optional-alternate-gpu-implementation",
                "alternate-architecture-only",
            ),
        )

    def test_bounded_role_excluded_marker_read_is_not_budget_exhaustion(self) -> None:
        stream = io.BytesIO()
        data = b"sm_90 marker payload"
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("opt/tests/sample_cuda.so")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        stream.seek(0)

        with patch("gpu_preflight.image.BINARY_FILE_LIMIT_BYTES", 4):
            scan = scan_rootfs_tar(stream, max_scan_bytes=1024)

        self.assertFalse(scan.truncated)
        self.assertEqual(scan.skipped_scan_budget, 0)

    def test_incompatible_runtime_artifact_requires_review_despite_aggregate_native_target(
        self,
    ) -> None:
        facts = image_facts()
        arm64 = manifest("linux/arm64", "arm64")
        rootfs = RootfsScan(
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_90", "sm_100", "sm_103"),
            cuda_versions=("12.9",),
            nccl_versions=(),
            files_seen=3,
            bytes_scanned=3072,
            metadata_bytes_scanned=0,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=3,
            cuda_artifacts=(
                CudaArtifactEvidence(
                    path="/usr/lib/libcustom_cuda.so",
                    size_bytes=1024,
                    targets=("sm_90",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                CudaArtifactEvidence(
                    path="/usr/lib/libportable_cuda.so",
                    size_bytes=1024,
                    targets=("sm_100",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                CudaArtifactEvidence(
                    path="/usr/lib/libnative_cuda.so",
                    size_bytes=1024,
                    targets=("sm_103",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
            ),
            cuobjdump_available=True,
            cuobjdump_version="cuobjdump 13.0",
            cuobjdump_files_analyzed=3,
            cuobjdump_targets_found=3,
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=0,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
        )
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "12.9"},
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_90", "sm_100", "sm_103"),
            cuda_versions=("12.9",),
            nccl_versions=(),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )
        target = report.targets[0]
        finding_ids = {finding.id for finding in target.findings}
        self.assertEqual(target.compatibility_status, "REVIEW_REQUIRED")
        self.assertEqual(target.artifact_inspection_coverage_percent, 100.0)
        self.assertEqual(target.artifact_inspection_covered_candidates, 3)
        self.assertEqual(target.artifact_inspection_eligible_candidates, 3)
        self.assertIn("GPF-CUDA-010", finding_ids)
        self.assertIn("GPF-CUDA-016", finding_ids)
        self.assertIn("GPF-CUDA-017", finding_ids)
        self.assertNotIn("GPF-CUDA-009", finding_ids)
        ordered_ids = [finding.id for finding in target.findings]
        self.assertLess(ordered_ids.index("GPF-IMAGE-001"), ordered_ids.index("GPF-CPU-002"))
        self.assertLess(ordered_ids.index("GPF-CPU-002"), ordered_ids.index("GPF-CUDA-007"))
        self.assertLess(ordered_ids.index("GPF-CUDA-007"), ordered_ids.index("GPF-NCCL-001"))
        self.assertLess(ordered_ids.index("GPF-NCCL-001"), ordered_ids.index("GPF-CUDA-003"))
        self.assertLess(ordered_ids.index("GPF-CUDA-010"), ordered_ids.index("GPF-CUDA-016"))
        self.assertLess(ordered_ids.index("GPF-CUDA-017"), ordered_ids.index("GPF-CUDA-016"))
        self.assertLess(ordered_ids.index("GPF-CUDA-016"), ordered_ids.index("GPF-RUNTIME-001"))
        self.assertNotIn("Inspected runtime CUDA artifacts are compatible", target.passed_checks)
        compatibility = next(finding for finding in target.findings if finding.id == "GPF-CUDA-010")
        incompatible = next(finding for finding in target.findings if finding.id == "GPF-CUDA-016")
        self.assertEqual(compatibility.severity, "info")
        self.assertEqual(incompatible.severity, "warning")
        self.assertIn("GPF-CUDA-016", compatibility.remediation)
        compatibility_details = {detail.label: detail.value for detail in compatibility.details}
        self.assertEqual(
            compatibility_details["Runtime inspection units with enumerated targets"], 3
        )
        self.assertEqual(compatibility_details["Compatible inspection units"], 2)
        self.assertEqual(compatibility_details["Inspection units requiring review"], 1)
        self.assertNotIn("GPF-CUDA-010", " ".join(target.evidence_limitations))
        self.assertNotIn("GPF-CUDA-016", " ".join(target.evidence_limitations))
        incompatible_details = {detail.label: detail for detail in incompatible.details}
        self.assertEqual(
            incompatible_details["Other CUDA extensions"].value,
            1,
        )
        self.assertEqual(
            incompatible_details["Full inventory"].value,
            "JSON or HTML appendix",
        )
        exception = report.observations["cuda_compatibility_exceptions"][0]
        self.assertEqual(exception["path"], "/usr/lib/libcustom_cuda.so")
        self.assertEqual(exception["component"], "Other CUDA extensions")
        self.assertEqual(
            exception["classification"], "runtime-relevance-not-statically-established"
        )
        self.assertEqual(exception["status_effect"], "review-required")
        self.assertFalse(exception["has_compatible_sibling"])
        self.assertFalse(exception["has_jit_fallback"])
        group = report.observations["cuda_compatibility_exception_groups"][0]
        self.assertEqual(group["exception_count"], 1)
        self.assertEqual(group["file_count"], 1)
        optimization = next(finding for finding in target.findings if finding.id == "GPF-CUDA-017")
        optimization_details = {detail.label: detail for detail in optimization.details}
        self.assertEqual(optimization_details["Architecture-specific cubin"].value, 0)
        self.assertEqual(optimization_details["Architecture-specific cubin"].context, "sm_103a")
        self.assertEqual(optimization_details["Family-specific cubin"].value, 0)
        self.assertEqual(optimization_details["Exact base cubin"].value, 1)
        self.assertEqual(optimization_details["Exact base cubin"].context, "sm_103")
        self.assertEqual(optimization_details["Compatible older base cubin"].value, 1)
        self.assertEqual(
            optimization_details["Compatible older base cubin"].context,
            "sm_100",
        )
        self.assertEqual(optimization_details["No compatible code"].value, 1)

        text_report = render_text(report)
        self.assertNotIn("Assessment confidence", text_report)
        self.assertIn(
            "  CUDA artifact inspection coverage: 100.0% (3/3 eligible runtime inspection units)",
            text_report,
        )
        self.assertIn("  Verified embedded CUDA targets: 3 unique", text_report)
        self.assertIn(
            "  Target relevance (gb300): native base sm_103 present; specialized targets "
            "absent: sm_103f, sm_103a",
            text_report,
        )
        self.assertIn(
            "    Evidence: cuobjdump compatibility summary for sm_103.\n"
            "    Breakdown:\n"
            "      - Runtime inspection units with enumerated targets: 3\n"
            "      - Compatible inspection units: 2\n"
            "      - Inspection units requiring review: 1",
            text_report,
        )
        self.assertIn("CUDA compatibility review summary:", text_report)
        self.assertIn(
            "    - Other CUDA extensions: Runtime relevance not statically established",
            text_report,
        )
        self.assertIn("        - /usr/lib/libcustom_cuda.so", text_report)
        self.assertIn("open the sibling report.json file", text_report)
        self.assertIn("open the sibling report.html file", text_report)
        self.assertIn("not operations, features, or distinct workload code paths", text_report)
        self.assertIn("which NCCL installation or dynamic library", text_report)
        self.assertNotIn("Additional artifacts or catalogs not shown", text_report)
        self.assertIn("      - Compatible older base cubin: 1 (sm_100)", text_report)
        self.assertIn("      - No compatible code: 1", text_report)
        html_report = render_html(report)
        self.assertIn('<div class="table-wrap finding-details"><table>', html_report)
        self.assertIn("Compatible older base cubin", html_report)
        self.assertIn("3/3 eligible runtime inspection units", html_report)
        self.assertIn("Target relevance (gb300)", html_report)
        self.assertIn("CUDA compatibility review summary", html_report)
        self.assertIn("Complete CUDA compatibility inventory", html_report)
        self.assertIn("/usr/lib/libcustom_cuda.so", html_report)
        payload = report.to_dict()
        payload_findings = {item["id"]: item for item in payload["targets"][0]["findings"]}
        self.assertEqual(
            payload_findings["GPF-CUDA-017"]["details"][4],
            {"label": "Compatible older base cubin", "value": 1, "context": "sm_100"},
        )

    def test_cuda_compatibility_appendix_retains_every_exception(self) -> None:
        facts = image_facts()
        arm64 = manifest("linux/arm64", "arm64")
        incompatible_count = 12
        artifacts = tuple(
            CudaArtifactEvidence(
                path=f"/opt/extensions/liboptional_{index:02d}.so",
                size_bytes=1024 + index,
                targets=("sm_90",),
                method="cuobjdump",
                complete=True,
                role="runtime",
            )
            for index in range(incompatible_count)
        )
        rootfs = RootfsScan(
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_90",),
            cuda_versions=("12.9",),
            nccl_versions=("2.27.5",),
            files_seen=incompatible_count,
            bytes_scanned=incompatible_count * 1024,
            metadata_bytes_scanned=0,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=incompatible_count,
            cuda_artifacts=artifacts,
            cuobjdump_available=True,
            cuobjdump_version="cuobjdump 13.0",
            cuobjdump_files_analyzed=incompatible_count,
            cuobjdump_targets_found=incompatible_count,
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=0,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
        )
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "12.9", "NCCL_VERSION": "2.27.5"},
            known_packages=("pytorch", "nccl"),
            installed_packages=("libnccl2==2.27.5",),
            cuda_targets=("sm_90",),
            cuda_versions=("12.9",),
            nccl_versions=("2.27.5",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )

        exceptions = report.observations["cuda_compatibility_exceptions"]
        self.assertEqual(len(exceptions), incompatible_count)
        self.assertEqual(
            {item["path"] for item in exceptions},
            {f"/opt/extensions/liboptional_{index:02d}.so" for index in range(incompatible_count)},
        )
        text_report = render_text(report)
        html_report = render_html(report)
        for index in range(incompatible_count):
            path = f"/opt/extensions/liboptional_{index:02d}.so"
            self.assertIn(path, html_report)
            if index < 3:
                self.assertIn(path, text_report)
            else:
                self.assertNotIn(path, text_report)
        groups = report.observations["cuda_compatibility_exception_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["exception_count"], incompatible_count)
        self.assertEqual(groups[0]["file_count"], incompatible_count)
        self.assertEqual(len(groups[0]["representative_paths"]), 3)
        self.assertIn("Complete CUDA compatibility inventory (12 binary file records)", html_report)
        self.assertNotIn("Additional artifacts or catalogs not shown", text_report)

    def test_bounded_cuda_scan_explains_incomplete_inspection_and_rerun_flags(self) -> None:
        facts = image_facts()
        arm64 = manifest("linux/arm64", "arm64")
        rootfs = RootfsScan(
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_103",),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            files_seen=303215,
            bytes_scanned=2048 * 1024 * 1024,
            metadata_bytes_scanned=0,
            truncated=True,
            metadata_truncated=False,
            cuda_candidate_files=588,
            cuda_artifacts=(
                CudaArtifactEvidence(
                    path="/opt/tests/sample_cuda.so",
                    size_bytes=1024,
                    targets=("sm_90",),
                    method="marker-scan",
                    complete=False,
                    role="test",
                    diagnostic="classified as test",
                ),
                CudaArtifactEvidence(
                    path="/usr/lib/libinspection_cuda.so",
                    size_bytes=1024,
                    targets=(),
                    method="cuobjdump",
                    complete=False,
                    role="runtime",
                    diagnostic="cuobjdump timed out after 30 seconds",
                    incomplete_reason="timeout",
                ),
            ),
            cuobjdump_available=True,
            cuobjdump_version="CUDA 13.4 (V13.4.92)",
            cuobjdump_files_analyzed=78,
            cuobjdump_targets_found=9,
            cuobjdump_no_device_code=68,
            cuobjdump_incomplete=1,
            marker_only_files=43,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=185,
            skipped_scan_budget=325,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
            skipped_cuda_candidates=(
                SkippedCudaCandidate(
                    path="/usr/lib/liboptional_cuda.so",
                    size_bytes=12 * 1024 * 1024,
                    reason="scan-budget",
                ),
            ),
        )
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "13.4"},
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_103",),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )

        findings = {finding.id: finding for finding in report.targets[0].findings}
        evidence_gap = findings["GPF-CUDA-012"]
        evidence_details = {detail.label: detail for detail in evidence_gap.details}
        self.assertEqual(evidence_details["Timeouts"].value, 1)
        self.assertEqual(
            evidence_details["/usr/lib/libinspection_cuda.so"].value,
            "cuobjdump timed out after 30 seconds",
        )
        self.assertIn("--cuobjdump-timeout 120", evidence_gap.remediation)

        coverage = findings["GPF-CUDA-014"]
        coverage_details = {detail.label: detail.value for detail in coverage.details}
        self.assertEqual(coverage_details["Observed CUDA candidates"], 588)
        self.assertEqual(coverage_details["Inspected with cuobjdump"], 78)
        self.assertEqual(coverage_details["Role-excluded"], 185)
        self.assertEqual(coverage_details["Not inspected: scan budget"], 325)
        self.assertEqual(coverage_details["Incomplete cuobjdump inspections"], 1)
        self.assertIn("--max-scan-mb 8192", coverage.remediation)
        self.assertIn("--max-cuobjdump-files 512", coverage.remediation)
        self.assertNotIn("--max-artifact-mb", coverage.remediation)

        high_coverage_rootfs = replace(
            rootfs,
            cuobjdump_files_analyzed=376,
            cuobjdump_incomplete=0,
            skipped_scan_budget=24,
            skipped_artifact_limit=1,
        )
        high_coverage_findings = _cuda_artifact_findings(
            high_coverage_rootfs,
            self.profiles["gb300"],
            ("13.4",),
            max_scan_bytes=8192 * 1024 * 1024,
            max_cuobjdump_files=512,
            max_artifact_bytes=1536 * 1024 * 1024,
            cuobjdump_timeout_seconds=30,
        )
        high_coverage = next(
            finding for finding in high_coverage_findings if finding.id == "GPF-CUDA-014"
        )
        self.assertIn("93.8%", high_coverage.remediation)
        self.assertNotIn("--max-scan-mb", high_coverage.remediation)

        text_report = render_text(report)
        self.assertIn(
            "  Retained CUDA artifact evidence:\n"
            "    Total records: 2\n"
            "    - With cuobjdump-enumerated targets: 0\n"
            "    - With unverified target strings: 1\n"
            "    - Incomplete cuobjdump inspections: 1",
            text_report,
        )
        self.assertIn("Unverified target-string records: 43", text_report)
        self.assertIn("      - Incomplete inspections: 1", text_report)
        self.assertIn("Incomplete inspection reasons: 1 timeout", text_report)
        self.assertNotIn("failed", text_report.lower())
        html_report = render_html(report)
        self.assertIn("incomplete: timeout", html_report)
        self.assertIn("unverified target-like strings", html_report)
        self.assertIn("      - Role-excluded: 185", text_report)
        self.assertIn("      - Not inspected: scan budget: 325", text_report)
        self.assertIn(
            "/usr/lib/liboptional_cuda.so (candidate-content budget; 12.0 MiB)",
            text_report,
        )
        self.assertIn("CUDA candidate outcomes", html_report)
        self.assertIn("candidate-content budget", html_report)
        self.assertIn("Artifact-size limit", html_report)
        self.assertIn(
            "    Terms:\n"
            "      - Embedded CUDA targets: cuobjdump enumerated at least one cubin or PTX "
            "target.\n"
            "      - Without embedded device code: inspection completed but found no cubin or "
            "PTX target.\n"
            "      - Unverified target strings: raw string evidence only.",
            text_report,
        )
        self.assertIn(
            "    Notes:\n"
            "      - Role-excluded: path heuristics classified the files as tests, tooling, or "
            "host-side runtime.\n"
            "      - Candidate-content budget: cumulative 2048 MiB read cap; this is not the "
            "image size or host-memory limit.\n"
            "      - Unverified target-string records: subset of observed candidates, not "
            "additional files.",
            text_report,
        )

    def test_older_cuobjdump_suppresses_per_artifact_compatibility_claims(self) -> None:
        facts = image_facts()
        arm64 = manifest("linux/arm64", "arm64")
        rootfs = RootfsScan(
            known_packages=("pytorch",),
            installed_packages=(),
            cuda_targets=("sm_90", "sm_103"),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            files_seen=1,
            bytes_scanned=1024,
            metadata_bytes_scanned=0,
            truncated=False,
            metadata_truncated=False,
            cuda_candidate_files=1,
            cuda_artifacts=(
                CudaArtifactEvidence(
                    path="/usr/lib/libcustom_cuda.so",
                    size_bytes=1024,
                    targets=("sm_90",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
            ),
            cuobjdump_available=True,
            cuobjdump_version="CUDA 12.0 (V12.0.140)",
            cuobjdump_files_analyzed=1,
            cuobjdump_targets_found=1,
            cuobjdump_no_device_code=0,
            cuobjdump_incomplete=0,
            marker_only_files=0,
            skipped_no_cuobjdump=0,
            skipped_non_runtime=0,
            skipped_scan_budget=0,
            skipped_artifact_limit=0,
            skipped_file_limit=0,
        )
        pulled = PulledImageFacts(
            platform="linux/arm64",
            digest=arm64.digest,
            environment={"CUDA_VERSION": "13.4"},
            known_packages=("pytorch", "nccl"),
            installed_packages=(),
            cuda_targets=("sm_90", "sm_103"),
            cuda_versions=("13.4",),
            nccl_versions=("2.31.2",),
            user=None,
            entrypoint=(),
            command=(),
            rootfs=rootfs,
        )
        report = inspect_with_image(
            facts,
            (self.profiles["gb300"],),
            scan_mode="filesystem",
            client=FakeImageClient((arm64,), pulled=pulled),
        )
        target = report.targets[0]
        finding_ids = {finding.id for finding in target.findings}
        self.assertIn("GPF-CUDA-015", finding_ids)
        self.assertNotIn("GPF-CUDA-009", finding_ids)
        self.assertNotIn("GPF-CUDA-010", finding_ids)
        self.assertNotIn("GPF-CUDA-016", finding_ids)
        self.assertNotIn("GPF-CUDA-017", finding_ids)
        self.assertEqual(target.compatibility_status, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(target.artifact_inspection_coverage_percent)

    def test_tag_and_digest_reference_is_inspected_by_digest(self) -> None:
        # A customer Dockerfile pins the AWS DLC base this way; Docker rejects
        # `manifest inspect` of tag@digest for a single-platform manifest.
        reference = (
            "public.ecr.aws/deep-learning-containers/pytorch:2.13-cu133-amzn2023"
            "@sha256:e82f6817e9c6a5aaabb33c91df3ebd3c34dc34d1a0f1df32dd062ab8249faff0"
        )
        single_platform = json.dumps(
            {
                "Descriptor": {
                    "digest": "sha256:e82f6817",
                    "platform": {"os": "linux", "architecture": "amd64"},
                },
                "SchemaV2Manifest": {"config": {"digest": "sha256:c"}, "layers": [{"size": 7}]},
            }
        )
        client = DockerImageClient(cuobjdump_mode="off")
        client.docker_binary = "/usr/bin/docker"
        with patch.object(client, "_run", return_value=single_platform) as run:
            manifests = client.inspect_manifest(reference)
        run.assert_called_once_with(
            (
                "manifest",
                "inspect",
                "--verbose",
                "public.ecr.aws/deep-learning-containers/pytorch"
                "@sha256:e82f6817e9c6a5aaabb33c91df3ebd3c34dc34d1a0f1df32dd062ab8249faff0",
            )
        )
        self.assertEqual(manifests.reference, reference)
        self.assertEqual(manifests.platforms[0].platform, "linux/amd64")

    def test_digest_and_exact_references_keep_registry_ports(self) -> None:
        self.assertEqual(
            _digest_reference("registry.example:5000/team/img:tag@sha256:abc"),
            "registry.example:5000/team/img@sha256:abc",
        )
        self.assertEqual(
            _digest_reference("registry.example:5000/img:1"), "registry.example:5000/img:1"
        )
        self.assertEqual(_digest_reference("ubuntu@sha256:abc"), "ubuntu@sha256:abc")
        self.assertEqual(
            _exact_reference("registry.example:5000/team/img:tag", "sha256:platform"),
            "registry.example:5000/team/img@sha256:platform",
        )
        self.assertEqual(
            _exact_reference("nvcr.io/nvidia/pytorch:26.09-py3@sha256:index", "sha256:platform"),
            "nvcr.io/nvidia/pytorch@sha256:platform",
        )

    def test_docker_error_omits_raw_manifest_bytes(self) -> None:
        dump = " ".join(str(value) for value in range(30, 130) for _ in range(4))
        error = _docker_error(
            ("manifest", "inspect", "--verbose", "x"),
            f"unsupported manifest format: &{{[{dump}] application/vnd.docker}}",
        )
        self.assertIn("unsupported manifest format", str(error))
        self.assertIn("<raw bytes omitted>", str(error))
        self.assertLess(len(str(error)), 200)

    def test_toolkit_release_comes_from_path_then_wheel_then_image(self) -> None:
        self.assertEqual(
            _toolkit_release(
                "/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcufftw.so.11.0.12.1", ()
            ),
            "12.3",
        )
        self.assertEqual(
            _toolkit_release(
                "/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcufft.so.12",
                ("13.3", "13.3.0"),
            ),
            "13.3",
        )
        self.assertEqual(_toolkit_release("/usr/lib/libnppc.so.12", ("12.8", "12.8.0.038")), "12.8")
        self.assertIsNone(_toolkit_release("/usr/lib/libnppc.so.12", ("12.8", "13.0")))

    def test_toolkit_libraries_are_judged_by_cuda_release(self) -> None:
        path = "/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcufftw.so.11.0.12.1"
        self.assertEqual(
            _cuda_exception_assessment(
                path,
                "runtime-artifact",
                toolkit_release="12.3",
                minimum_cuda_version="11.8",
            ),
            (
                "CUDA Toolkit libraries",
                "vendor-library-supported-by-release",
                "toolkit-release-supports-target",
            ),
        )
        self.assertEqual(
            _cuda_exception_assessment(
                path,
                "runtime-artifact",
                toolkit_release="12.3",
                minimum_cuda_version="12.9",
            ),
            (
                "CUDA Toolkit libraries",
                "likely-runtime-compatibility-gap",
                "toolkit-release-predates-target",
            ),
        )

    def test_cufft_placeholder_cubin_does_not_require_review_on_a_supported_release(self) -> None:
        # AWS DLC PyTorch 2.13 base image: cuobjdump lists only one sm_75 cubin in cuFFT,
        # whose kernels live in a vendor-packaged section.
        rootfs = rootfs_with(
            (
                CudaArtifactEvidence(
                    path="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcufft.so.12",
                    size_bytes=273 * 1024 * 1024,
                    targets=("sm_75",),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
                CudaArtifactEvidence(
                    path="/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
                    size_bytes=448 * 1024 * 1024,
                    targets=("compute_120", "sm_100", "sm_103", "sm_90"),
                    method="cuobjdump",
                    complete=True,
                    role="runtime",
                ),
            )
        )
        profile = self.profiles["b300"]
        exceptions = _cuda_compatibility_exceptions(
            rootfs,
            profile,
            platform="linux/amd64",
            analyzer_matches_image=True,
            image_cuda_versions=("13.3", "13.3.0"),
        )
        self.assertEqual([item["status_effect"] for item in exceptions], ["informational"])
        findings = {
            finding.id: finding
            for finding in _cuda_artifact_findings(
                rootfs,
                profile,
                ("13.3", "13.3.0"),
                max_scan_bytes=1024,
                max_cuobjdump_files=None,
                max_artifact_bytes=1024,
                cuobjdump_timeout_seconds=30,
            )
        }
        self.assertNotIn("GPF-CUDA-016", findings)
        self.assertIn("CUDA 13.3", findings["GPF-CUDA-020"].evidence)
        # cuFFT is accounted for by release, so the remaining unit is fully compatible.
        self.assertIn("GPF-CUDA-009", findings)
        self.assertNotIn("GPF-CUDA-010", findings)


if __name__ == "__main__":
    unittest.main()
