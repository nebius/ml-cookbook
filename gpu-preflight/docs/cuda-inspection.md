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

- `--max-scan-mb` limits candidate-file content read during the filesystem scan; it does not limit the image download or rootfs stream. Default: 16384 MiB.
- `--max-artifact-mb` limits one file passed to `cuobjdump`. Default: 2048 MiB, which covers PyTorch's `libtorch_cuda.so` (0.6–1.4 GiB across NGC 24.01 to 26.09, and differing between the arm64 and amd64 builds of one release).
- `--max-cuobjdump-files` bounds the number of `cuobjdump` invocations. Default: 1024.
- Core framework libraries (PyTorch CUDA, NCCL, FlashAttention, Transformer Engine) are always inspected: they do not count against `--max-scan-mb` or `--max-cuobjdump-files`, only against `--max-artifact-mb` and the timeout. If one cannot be inspected, the report names it (GPF-CUDA-019) and the result cannot be ready for hardware testing.
- `--exhaustive` removes only the `cuobjdump` file-count cap. Candidate-byte, per-artifact-size, per-invocation-timeout, and Docker-command safeguards remain active.
- `--cuobjdump-timeout` bounds each artifact analysis. Default: 30 seconds.
- `--command-timeout` bounds individual Docker metadata and pull commands. Default: 3600 seconds.

The pull preflight prints the manifest's approximate compressed size and available temporary-filesystem capacity. Large framework images can still require substantial network time, Docker storage, and temporary space.

When a report shows skipped candidates, review the retained paths first and change only the limit relevant to files the workload may load. `--max-scan-mb` is a cumulative candidate-content read budget; it is not the image size, downloaded bytes, Docker storage limit, or host memory limit. At high coverage, GPU Preflight does not recommend a broad multiplier. At lower coverage, a larger scan budget can expose enough additional files to hit `--max-cuobjdump-files`, so the report may suggest both values as a bounded deeper-scan starting point. Use `--exhaustive` for a one-time run that should inspect every eligible ordinary artifact and every representative catalog target that fits the remaining safeguards; do not treat it as permission to execute image contents.
