from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gpu_preflight.cuda_inspection import (
    CuobjdumpTool,
    analyzer_error_detail,
    classify_artifact_role,
    cuda_targets_from_listing,
    cuda_targets_from_text,
    is_core_framework_library,
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

    def test_core_framework_libraries_are_recognized(self) -> None:
        for path in (
            "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so",
            "/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda_linalg.so",
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2.31.2",
            "/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so",
            "/opt/venv/lib/python3.12/site-packages/transformer_engine/wheel_lib/libtransformer_engine.so",
        ):
            self.assertTrue(is_core_framework_library(path), path)
        for path in (
            "/opt/amazon/ofi-nccl/lib64/libnccl-net.so",
            "/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublasLt.so.13",
            "/opt/pytorch/ao/torchao/_C_cutlass_90a.abi3.so",
        ):
            self.assertFalse(is_core_framework_library(path), path)

    def test_profiler_injection_libraries_are_tooling(self) -> None:
        self.assertEqual(
            classify_artifact_role(
                "/usr/local/cuda-12.3/targets/x86_64-linux/lib/libaccinj64.so.12.3.101"
            ),
            "tooling",
        )

    def test_analyzer_error_is_retried_with_fallback_cuobjdump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "cuobjdump-13.4"
            primary.write_text(
                "#!/bin/sh\n"
                "echo 'ELF file    1: libcublas.1.sm_50.cubin'\n"
                "echo \"cuobjdump fatal   : Invalid ELF in 'libcublas.so.12.3.4.1'\" >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            fallback = Path(directory) / "cuobjdump-13.0"
            fallback.write_text(
                "#!/bin/sh\n"
                "echo 'ELF file    1: libcublas.1.sm_50.cubin'\n"
                "echo 'ELF file    2: libcublas.2.sm_90.cubin'\n"
                "echo 'PTX file    1: libcublas.1.sm_90.ptx'\n",
                encoding="utf-8",
            )
            primary.chmod(0o755)
            fallback.chmod(0o755)
            artifact = Path(directory) / "libcublas.so"
            artifact.write_bytes(b"payload")
            tool = CuobjdumpTool(
                str(primary),
                "CUDA 13.4 (V13.4.92)",
                timeout_seconds=2,
                fallback=CuobjdumpTool(str(fallback), "CUDA 13.0 (V13.0.88)", timeout_seconds=2),
            )
            evidence = tool.inspect(artifact, "/usr/lib/libcublas.so", len(b"payload"))
            self.assertTrue(evidence.complete)
            self.assertEqual(evidence.targets, ("compute_90", "sm_50", "sm_90"))
            self.assertIn("Invalid ELF", evidence.diagnostic or "")
            self.assertIn("fallback cuobjdump CUDA 13.0", evidence.diagnostic or "")

    def test_fallback_is_attached_only_when_it_differs_from_the_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scripts = {}
            for name, release in (("primary", "13.4"), ("fallback", "13.0")):
                path = Path(directory) / name
                path.write_text(
                    f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.1'\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)
                scripts[name] = str(path)
            tool = resolve_cuobjdump(scripts["primary"], 2, fallback_mode=scripts["fallback"])
            self.assertIsNotNone(tool)
            assert tool is not None
            self.assertIsNotNone(tool.fallback)
            self.assertEqual(
                tool.fallback.version if tool.fallback else None, "CUDA 13.0 (V13.0.1)"
            )
            same = resolve_cuobjdump(scripts["primary"], 2, fallback_mode=scripts["primary"])
            self.assertIsNone(same.fallback if same else None)
            off = resolve_cuobjdump(scripts["primary"], 2, fallback_mode="off")
            self.assertIsNone(off.fallback if off else None)

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


class AnalyzerErrorDetailTests(unittest.TestCase):
    """A failed cuobjdump run must report why, not the listing that preceded it."""

    # Verbatim `cuobjdump --list-elf --list-ptx` output for HPC-X hcoll's
    # hmca_gpu_cuda.so in nvcr.io/nvidia/pytorch:24.01-py3, analyzed with
    # cuobjdump CUDA 13.4: a partial listing, then the failure.
    PARTIAL_LISTING_THEN_FAILURE = (
        "ELF file    1: gpu-preflight-4p3bwzt3.1.sm_50.cubin\n"
        "PTX file    1: gpu-preflight-4p3bwzt3.1.sm_50.ptx\n"
        "ELF file    2: gpu-preflight-4p3bwzt3.2.sm_52.cubin\n"
        "ELF file    3: gpu-preflight-4p3bwzt3.3.sm_60.cubin\n"
        "ELF file    4: gpu-preflight-4p3bwzt3.4.sm_61.cubin\n"
        "PTX file    2: gpu-preflight-4p3bwzt3.2.sm_70.ptx\n"
        "ELF file    5: gpu-preflight-4p3bwzt3.5.sm_70.cubin\n"
        "ELF file    6: gpu-preflight-4p3bwzt3.6.sm_80.cubin\n"
        "PTX file    3: gpu-preflight-4p3bwzt3.3.sm_80.ptx\n"
        "cuobjdump fatal   : Invalid ELF in "
        "'/opt/hpcx/hcoll/lib/hcoll/hmca_gpu_cuda.so'\n"
    )

    def test_the_failure_line_leads_the_diagnostic(self) -> None:
        detail = analyzer_error_detail(self.PARTIAL_LISTING_THEN_FAILURE, 1)
        self.assertTrue(
            detail.startswith("cuobjdump fatal : Invalid ELF"),
            f"diagnostic did not lead with the failure: {detail!r}",
        )
        # The temporary input file's name means nothing to a reader of the report.
        self.assertNotIn("gpu-preflight-", detail)
        self.assertNotIn("ELF file", detail)

    def test_repeated_failure_lines_are_collapsed(self) -> None:
        output = "cuobjdump fatal : Invalid ELF in 'x'\n" * 3
        self.assertEqual(analyzer_error_detail(output, 1), "cuobjdump fatal : Invalid ELF in 'x'")

    def test_output_without_a_failure_line_falls_back_to_the_tail(self) -> None:
        self.assertEqual(analyzer_error_detail("only a listing", 3), "only a listing")

    def test_empty_output_reports_the_exit_status(self) -> None:
        self.assertEqual(analyzer_error_detail("", 3), "exit status 3")

    def test_the_detail_stays_bounded(self) -> None:
        self.assertLessEqual(len(analyzer_error_detail("error : " + "x" * 4000, 1)), 500)
        self.assertLessEqual(len(analyzer_error_detail("y" * 4000, 1)), 500)

    def test_the_diagnostic_is_identical_between_runs(self) -> None:
        # cuobjdump names each embedded image after its input file, so its output
        # carries the temporary file's random stem. A report that embeds it differs
        # between two runs of the same image.
        first = self.PARTIAL_LISTING_THEN_FAILURE
        second = first.replace("4p3bwzt3", "d_dx8cjo")
        self.assertNotEqual(first, second)
        self.assertEqual(
            analyzer_error_detail(first, 1),
            analyzer_error_detail(second, 1),
        )

    def test_a_listing_without_a_failure_line_also_drops_the_stem(self) -> None:
        listing = "ELF file 1: gpu-preflight-xr_fs4lg.1.sm_60.cubin"
        detail = analyzer_error_detail(listing, 1)
        self.assertNotIn("gpu-preflight-", detail)
        self.assertIn("<artifact>", detail)

    def test_a_truncated_listing_is_still_identical_between_runs(self) -> None:
        # The fallback keeps only the tail of a long listing, and a stem can span
        # that boundary, so scrubbing covers the whole output before the tail is cut.
        def listing(stem: str) -> str:
            return "\n".join(
                f"ELF file {index}: gpu-preflight-{stem}.{index}.sm_{50 + index}.cubin"
                for index in range(1, 40)
            )

        first = analyzer_error_detail(listing("8uo_5ru"), 1)
        second = analyzer_error_detail(listing("_n1za4w"), 1)
        self.assertEqual(len(first), 500)
        self.assertEqual(first, second)
        for fragment in ("preflight-", "8uo_5ru", "_n1za4w"):
            self.assertNotIn(fragment, first)


class FlashInferProviderLayoutTests(unittest.TestCase):
    """Newer FlashInfer wheels interpose a per-architecture provider directory:
    flashinfer_jit_cache/providers/<sm>/jit_cache/<module>/<module>.so. Without it the
    sm100a and sm103a builds of one operation land in different dispatch families and
    neither is recognized as the other's compatible alternative.
    """

    ROOT = "/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache"
    MODULE = "alphamoe_fused_router_nvrtc_a1b2"

    def _provider(self, provider: str) -> str:
        return f"{self.ROOT}/providers/{provider}/jit_cache/{self.MODULE}/{self.MODULE}.so"

    def test_sibling_providers_share_one_dispatch_family(self) -> None:
        self.assertEqual(
            kernel_catalog_group(self._provider("sm100a")),
            kernel_catalog_group(self._provider("sm103a")),
        )

    def test_the_provider_directory_declares_the_target(self) -> None:
        self.assertEqual(kernel_catalog_declared_targets(self._provider("sm103a")), ("sm_103a",))
        self.assertEqual(kernel_catalog_declared_targets(self._provider("sm100a")), ("sm_100a",))

    def test_the_flat_layout_still_groups_as_before(self) -> None:
        flat = f"{self.ROOT}/jit_cache/{self.MODULE}_sm90/{self.MODULE}_sm90.so"
        self.assertEqual(kernel_catalog_group(flat), f"{self.ROOT}/jit_cache/{self.MODULE}")
        self.assertEqual(kernel_catalog_declared_targets(flat), ("sm_90",))

    def test_both_layouts_resolve_to_the_same_family(self) -> None:
        # One image can carry both, and they are the same operation either way.
        flat = f"{self.ROOT}/jit_cache/{self.MODULE}/{self.MODULE}.so"
        self.assertEqual(kernel_catalog_group(flat), kernel_catalog_group(self._provider("sm103a")))

    def test_an_unrecognized_provider_name_does_not_declare_a_target(self) -> None:
        unknown = self._provider("rubin")
        self.assertEqual(kernel_catalog_declared_targets(unknown), ())

    def test_a_shared_library_outside_the_cache_is_not_grouped(self) -> None:
        # Suppressing an unpaired incompatible module would hide a real gap, so the
        # rule stays confined to the cache layout.
        self.assertIsNone(kernel_catalog_group("/usr/lib/libth_common.so"))
        self.assertIsNone(kernel_catalog_group(f"{self.ROOT}/providers/sm103a/jit_cache/a/b.so"))


class NoDeviceCodeClassificationTests(unittest.TestCase):
    """cuobjdump exits non-zero for two different situations and only one is a failure.

    Now that candidates are no longer filtered by path keyword, glibc's empty stub
    archives (libdl.a, libpthread.a, librt.a and friends) reach the analyzer. It
    answers "Invalid fatbin header", which means the complete file holds no CUDA
    container -- a result, not a failed look. Counting it as an incomplete inspection
    understated coverage for every image that ships a static libc.
    """

    def _evidence(self, message: str, image_path: str):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "cuobjdump"
            executable.write_text(f"#!/bin/sh\necho '{message}' >&2\nexit 1\n", encoding="utf-8")
            executable.chmod(0o755)
            artifact = Path(directory) / "artifact.a"
            artifact.write_bytes(b"!<arch>\n")
            tool = CuobjdumpTool(str(executable), "test", timeout_seconds=5)
            return tool.inspect(artifact, image_path, 8)

    def test_an_invalid_fatbin_header_is_a_complete_result(self) -> None:
        evidence = self._evidence(
            "cuobjdump fatal : Invalid fatbin header in <artifact>",
            "/usr/lib/aarch64-linux-gnu/libdl.a",
        )
        self.assertTrue(evidence.complete)
        self.assertIsNone(evidence.incomplete_reason)
        self.assertEqual(evidence.targets, ())

    def test_the_existing_no_device_code_answer_is_unchanged(self) -> None:
        evidence = self._evidence(
            "cuobjdump info : File <artifact> does not contain device code",
            "/usr/lib/libz.so.1",
        )
        self.assertTrue(evidence.complete)
        self.assertIsNone(evidence.incomplete_reason)

    def test_a_real_analyzer_failure_is_still_incomplete(self) -> None:
        # The distinction has to hold in both directions, or a genuinely unreadable
        # artifact would be reported as inspected.
        evidence = self._evidence(
            "cuobjdump fatal : Could not open input file", "/usr/lib/libtorch_cuda.so"
        )
        self.assertFalse(evidence.complete)
        self.assertEqual(evidence.incomplete_reason, "analyzer-error")


class StubDirectoryRoleTests(unittest.TestCase):
    """Link-time stub objects exist to satisfy the linker, not to be loaded. CUDA ships
    lib/stubs/libcuda.so, and wheels ship stubs/<target-triple>/ built for other
    architectures on purpose -- the JAX NGC image carries aarch64 CuTe DSL stubs inside
    an amd64 image. Treating those as runtime libraries produced an architecture warning
    on an image that is in fact correct.
    """

    def test_a_stub_directory_is_tooling(self) -> None:
        for path in (
            "/usr/local/cuda-13.4/targets/x86_64-linux/lib/stubs/libcuda.so",
            "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/cu13/lib/"
            "stubs/aarch64-unknown-linux-gnu/libcute_dsl_runtime.so",
        ):
            with self.subTest(path=path):
                self.assertEqual(classify_artifact_role(path), "tooling")

    def test_the_real_library_beside_a_stub_is_still_runtime(self) -> None:
        self.assertEqual(
            classify_artifact_role(
                "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/cu13/lib/"
                "libcute_dsl_runtime.so"
            ),
            "runtime",
        )

    def test_compute_sanitizer_interceptors_remain_tooling(self) -> None:
        # The other half of the same false positive: 32-bit interceptor shims that
        # NVIDIA ships deliberately.
        self.assertEqual(
            classify_artifact_role(
                "/usr/local/cuda-13.4/compute-sanitizer/x86/libInterceptorInjectionTarget.so"
            ),
            "tooling",
        )
