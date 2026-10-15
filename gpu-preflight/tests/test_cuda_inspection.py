from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gpu_preflight.cuda_inspection import (
    CuobjdumpTool,
    classify_artifact_role,
    cuda_targets_from_listing,
    cuda_targets_from_text,
    kernel_catalog_declared_targets,
    kernel_catalog_group,
    kernel_catalog_kind,
    resolve_cuobjdump,
)


class CudaInspectionTests(unittest.TestCase):
    def test_parses_cuobjdump_listing_and_maps_ptx_to_compute_targets(self) -> None:
        # Verbatim `cuobjdump --list-elf --list-ptx` output for HPC-X hcoll's
        # hmca_gpu_cuda.so in nvcr.io/nvidia/pytorch:26.09-py3 (CUDA 13.4.92).
        listing = (
            "ELF file    1: hcoll_2609.1.sm_80.cubin\n"
            "PTX file    1: hcoll_2609.1.sm_80.ptx\n"
            "ELF file    2: hcoll_2609.2.sm_90.cubin\n"
            "PTX file    2: hcoll_2609.2.sm_90.ptx\n"
        )
        self.assertEqual(
            cuda_targets_from_listing(listing),
            ("compute_80", "compute_90", "sm_80", "sm_90"),
        )

    def test_listing_parser_reads_suffixes_and_ignores_input_names_and_info_lines(self) -> None:
        listing = (
            "ELF file    1: fp4_quantization_sm100.1.sm_103a.cubin\n"
            "ELF file    2: fp4_quantization_sm100.2.sm_100f.cubin\n"
            "PTX file    1: fp4_quantization_sm100.1.sm_100a.ptx\n"
            "cuobjdump info    : No PTX file found to extract from 'libcufft.so.12'.\n"
        )
        self.assertEqual(
            cuda_targets_from_listing(listing),
            ("compute_100a", "sm_100f", "sm_103a"),
        )

    def test_path_target_parser_still_reads_declared_names(self) -> None:
        targets = cuda_targets_from_text(
            "/opt/kernels/sm_090.cubin /opt/kernels/sm_103a.cubin /opt/compute_103.ptx"
        )
        self.assertEqual(targets, ("compute_103", "sm_103a", "sm_90"))

    def test_artifact_role_distinguishes_test_data_from_runtime_library(self) -> None:
        self.assertEqual(classify_artifact_role("/opt/project/tests/kernel.cubin"), "test")
        self.assertEqual(
            classify_artifact_role("/usr/bin/nvshmem_13/perftest/device/kernel.cubin"),
            "test",
        )
        self.assertEqual(
            classify_artifact_role("/opt/ucx/lib/libucx_perftest_cuda.so"),
            "test",
        )
        self.assertEqual(classify_artifact_role("/usr/lib/libtorch_cuda.so"), "runtime")
        self.assertEqual(
            classify_artifact_role("/usr/local/cuda/compat/libcuda.so.1"),
            "host-runtime",
        )
        self.assertEqual(
            classify_artifact_role(
                "/usr/local/cuda-13.4/targets/sbsa-linux/lib/libcufile.so.1.19.0"
            ),
            "host-runtime",
        )

    def test_precompiled_cubin_catalog_is_grouped_by_directory(self) -> None:
        first = "/opt/flashinfer/cubins/release/fmha/kernel_a_sm100f.cubin"
        second = "/opt/flashinfer/cubins/release/fmha/kernel_b_sm100f.cubin"
        self.assertEqual(
            kernel_catalog_group(first),
            "/opt/flashinfer/cubins/release/fmha",
        )
        self.assertEqual(kernel_catalog_group(first), kernel_catalog_group(second))
        self.assertEqual(classify_artifact_role(first), "kernel-catalog")
        self.assertIsNone(kernel_catalog_group("/opt/project/kernel_sm100f.cubin"))

    def test_flashinfer_architecture_modules_share_a_dispatch_group(self) -> None:
        root = "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache"
        generic = f"{root}/batch_prefill_x/batch_prefill_x.so"
        sm90 = f"{root}/batch_prefill_x_sm90/batch_prefill_x_sm90.so"
        self.assertEqual(kernel_catalog_group(generic), f"{root}/batch_prefill_x")
        self.assertEqual(kernel_catalog_group(generic), kernel_catalog_group(sm90))
        self.assertEqual(kernel_catalog_kind(sm90), "architecture-module")
        self.assertEqual(kernel_catalog_declared_targets(sm90), ("sm_90",))
        self.assertEqual(classify_artifact_role(sm90), "kernel-catalog")

    def test_flashinfer_plain_architecture_suffix_is_scoped_to_known_cache(self) -> None:
        root = "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache"
        sm100 = f"{root}/fp4_quantization_100/fp4_quantization_100.so"
        sm103 = f"{root}/fp4_quantization_103/fp4_quantization_103.so"
        self.assertEqual(kernel_catalog_group(sm100), kernel_catalog_group(sm103))
        self.assertEqual(kernel_catalog_declared_targets(sm100), ("sm_100",))
        self.assertIsNone(kernel_catalog_group("/opt/project/fp4_quantization_100.so"))

    def test_explicit_missing_cuobjdump_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "executable not found"):
            resolve_cuobjdump("/definitely/missing/cuobjdump", timeout_seconds=1)

    def test_explicit_non_executable_cuobjdump_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text("not executable\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cuobjdump is not executable"):
                resolve_cuobjdump(str(executable), timeout_seconds=1)

    def test_cuobjdump_timeout_is_bounded_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
            executable.chmod(0o755)
            artifact = Path(directory) / "library.so"
            artifact.write_bytes(b"payload")
            tool = CuobjdumpTool(
                executable=str(executable),
                version="test",
                timeout_seconds=1,
            )
            evidence = tool.inspect(artifact, "/usr/lib/library.so", len(b"payload"))
            self.assertFalse(evidence.complete)
            self.assertEqual(evidence.method, "cuobjdump")
            self.assertEqual(evidence.incomplete_reason, "timeout")
            self.assertIn("timed out", evidence.diagnostic or "")

    def test_no_device_code_is_a_completed_negative_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\n"
                "echo 'cuobjdump fatal : input file does not contain device code' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            artifact = Path(directory) / "host-library.so"
            artifact.write_bytes(b"payload")
            tool = CuobjdumpTool(str(executable), "test", timeout_seconds=1)
            evidence = tool.inspect(artifact, "/usr/lib/host-library.so", len(b"payload"))
            self.assertTrue(evidence.complete)
            self.assertEqual(evidence.targets, ())
            self.assertIsNone(evidence.incomplete_reason)
            self.assertIn("no embedded CUDA device code", evidence.diagnostic or "")

    def test_cuobjdump_error_has_an_aggregate_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(
                "#!/bin/sh\necho 'cuobjdump fatal : unsupported input' >&2\nexit 1\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            artifact = Path(directory) / "library.so"
            artifact.write_bytes(b"payload")
            tool = CuobjdumpTool(str(executable), "test", timeout_seconds=1)
            evidence = tool.inspect(artifact, "/usr/lib/library.so", len(b"payload"))
            self.assertFalse(evidence.complete)
            self.assertEqual(evidence.incomplete_reason, "analyzer-error")
            self.assertIn("unsupported input", evidence.diagnostic or "")


if __name__ == "__main__":
    unittest.main()
