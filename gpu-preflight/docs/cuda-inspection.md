# CUDA artifact inspection and scan limits

What GPU Preflight inspects inside a container image, the terms the report uses for what
it finds, and the bounded limits that govern a scan.

Back to the [GPU Preflight README](../README.md).

## How CUDA artifact inspection works

Container images usually do not have one directory per operation or GPU architecture. Frameworks and libraries install **CUDA artifacts** such as shared libraries (`.so`), Python native extensions, executables, and fat binaries. One artifact can contain CPU-side code plus many GPU **kernels** (device functions launched on the GPU), packaged as one or more architecture-specific target images:

- `sm_XX` is a native cubin containing GPU machine code for that SM target.
- `compute_XX` is PTX virtual code that a compatible NVIDIA driver may JIT-compile for a newer GPU.
- Architecture-specific targets such as `sm_90a` require that exact architecture family and do not carry forward like baseline PTX.

GPU Preflight uses file paths and names only to find candidates and estimate their role. It then uses `cuobjdump` to inspect the contents of each selected artifact. Therefore, **CUDA artifact** or **CUDA binary** is the correct term for a reported `.so`; **kernel** refers to one GPU entry point inside it.

A file is a candidate if its suffix is one only CUDA produces — `.cubin`, `.fatbin`, `.ptx` — wherever it sits, so an application's own kernels, a Triton JIT cache entry and a dispatch catalog under a vendor-neutral directory are all inspected. The generic binary suffixes `.so`, `.a` and `.bin` identify nothing on their own, and an image carries thousands of them, so one of those is a candidate only where the path names a CUDA-bearing project: `cuda`, `flash`, `jax`, `nccl`, `nvidia`, `nvshmem` or `torch`. A CUDA extension built into a `.so` under a path mentioning none of those is therefore not inspected, and not counted against coverage either. If an image ships device code that way, give the path or the file name one of those names, or inspect it separately.

Report terms:

- **with embedded CUDA targets**: `cuobjdump` completed and enumerated at least one cubin or PTX target.
- **without device code**: `cuobjdump` completed but found no embedded CUDA device code. Broad candidate matching produces some of these by design.
- **unverified target strings**: a bounded raw-byte scan found target-like strings, but `cuobjdump` did not verify them as device-code records. Treat these as discovery hints.
- **role-excluded**: path heuristics classified the file as a test, benchmark, tool, sample, or host-side CUDA runtime component rather than a loadable workload artifact.
- **incomplete inspection**: `cuobjdump` timed out, returned an analyzer error, encountered an I/O error, or did not otherwise complete. The report aggregates those reasons and retains example paths and diagnostics.
- **CUDA Toolkit libraries** (cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NPP, nvJPEG): judged by the CUDA release that shipped them rather than by the `cuobjdump` listing, because NVIDIA packages several of them in ways `cuobjdump` does not fully list (cuFFT keeps most kernels in a vendor-packaged section; cuFFTW and the NPP core library carry one placeholder cubin). A library from a release at or above the profile's minimum CUDA version is informational (GPF-CUDA-020); an older release remains a likely runtime gap.
- **kernel catalog member**: a standalone cubin below an explicit `cubins` dispatch tree. Members sharing a catalog directory and declared target are treated as selectable shape-specific alternatives.
- **architecture-variant module group**: multiple shared-library modules in a recognized runtime cache that represent selectable builds of the same operation family for different GPU targets. GPU Preflight applies this rule to FlashInfer's `flashinfer_jit_cache` layout only, and requires at least two variants before grouping.

Catalog-level compatibility means the runtime has at least one compatible selectable path in the inspected group. It does not prove that every operation, shape, or data type has an optimized path or that the workload selects that path.

For each hardware target, the report also classifies the best code path found in every successfully inspected runtime artifact. From most target-specific to least, these are an exact architecture-specific cubin (`sm_XXa`), exact family-specific cubin (`sm_XXf`), exact base cubin (`sm_XX`), compatible family cubin from an earlier target, compatible base cubin from an earlier target, and PTX JIT fallback. This is compilation coverage—not proof of runtime selection or performance—and should be combined with workload-path tracing and hardware benchmarks.

An artifact present in an image may never be loaded by a particular workload. A reported target mismatch is therefore a risk to trace to the workload's import, dependency, and feature paths—not proof that every use of the image will fail.

## Resource limits

A default scan sets none of the three budgets below, so coverage reflects the image rather than a number chosen in advance. Each is available when a run's duration or footprint matters more than its completeness, and the report names the limit that stopped it.

- `--max-scan-mb` caps candidate-file content read during the filesystem scan; it never limited the image download or the rootfs stream. Candidate bytes pass through a temporary file one at a time rather than accumulating, so an unbudgeted scan costs time, not space.
- `--max-artifact-mb` caps one file passed to `cuobjdump`. Without it, the free space in the temporary filesystem is the bound: an artifact that would not leave 2 GiB free is refused and recorded, rather than filling the disk. For scale, the largest real artifact measured is FlashAttention's `flash_ops.abi3.so` at 1677 MiB, with PyTorch's `libtorch_cuda.so` at 0.6–1.4 GiB across NGC 24.01 to 26.09.
- `--max-cuobjdump-files` caps the number of `cuobjdump` invocations. Capping costs coverage, and does not reliably buy time in exchange: inspecting all 1458 units of `sglang-runtime:1.3.0` took 155s, while the same scan capped at 1024 reached 70.5% in 176s. Files past the cap are still read, for byte markers instead of for the analyzer, so a cap removes evidence more dependably than it removes work.
- Core framework libraries (PyTorch CUDA, NCCL, FlashAttention, Transformer Engine) are always inspected: they do not count against `--max-scan-mb` or `--max-cuobjdump-files` when those are set, only against `--max-artifact-mb`, the free temporary space and the timeout. If one cannot be inspected, the report names it (GPF-CUDA-019) and the result cannot be ready for hardware testing.

The two timeouts are a different kind of limit and are set by default, because they recover a stuck run rather than trading away coverage. Neither has been observed to fire: no inspection in testing has reached 30 seconds, including on that 1677 MiB artifact.

- `--cuobjdump-timeout` bounds each artifact analysis. Default: 30 seconds.
- `--command-timeout` bounds individual Docker metadata and pull commands. Default: 3600 seconds.

The pull preflight prints the manifest's approximate compressed size and available temporary-filesystem capacity. Large framework images can still require substantial network time, Docker storage, and temporary space.

When a report shows skipped candidates, review the retained paths first and change only the limit relevant to files the workload may load. `--max-scan-mb` is a cumulative candidate-content read budget; it is not the image size, downloaded bytes, Docker storage limit, or host memory limit. At high coverage, GPU Preflight does not recommend a broad multiplier. At lower coverage, a larger scan budget can expose enough additional files that a `--max-cuobjdump-files` you set becomes the next limit reached, so the report may suggest both values as a bounded deeper-scan starting point. Skipped candidates appear only on a run that set a budget, or on one whose temporary filesystem could not hold an artifact. No run executes image contents.
