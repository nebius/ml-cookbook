# CUDA artifact inspection

What GPU Preflight inspects inside a container image, the terms the report uses for what
it finds, and what can refuse a file.

Back to the [GPU Preflight README](../README.md).

## How CUDA artifact inspection works

Container images usually do not have one directory per operation or GPU architecture. Frameworks and libraries install **CUDA artifacts** such as shared libraries (`.so`), Python native extensions, executables, and fat binaries. One artifact can contain CPU-side code plus many GPU **kernels** (device functions launched on the GPU), packaged as one or more architecture-specific target images:

- `sm_XX` is a native cubin containing GPU machine code for that SM target.
- `compute_XX` is PTX virtual code that a compatible NVIDIA driver may JIT-compile for a newer GPU. "Compatible" has a floor as well as a ceiling: CUDA 12.8 assembles `compute_50` and later, CUDA 13 assembles `compute_75` and later, and the driver's JIT assembles what its own release does. PTX below `compute_50` therefore runs nowhere on current hardware, and `compute_50` through `compute_74` depends on which CUDA release the host driver came from -- something the image cannot tell you.
- Architecture-specific targets such as `sm_90a` require that exact architecture family and do not carry forward like baseline PTX.

GPU Preflight uses file paths and names only to estimate an artifact's role. It then uses `cuobjdump` to inspect the contents of each selected artifact. Therefore, **CUDA artifact** or **CUDA binary** is the correct term for a reported `.so`; **kernel** refers to one GPU entry point inside it.

Candidates are found by reading bytes, not by matching names. A suffix only CUDA produces — `.cubin`, `.fatbin`, `.ptx` — makes a file a candidate wherever it sits; everything else is settled by its first twenty bytes, so anything carrying the ELF magic number is a candidate whatever it is called. Host binaries become candidates too and `cuobjdump` answers "no device code" for each, which costs roughly a tenth of the scan's time.

Report terms:

- **with embedded CUDA targets**: `cuobjdump` completed and enumerated at least one cubin or PTX target.
- **without device code**: `cuobjdump` completed but found no embedded CUDA device code. Broad candidate matching produces some of these by design.
- **unverified target strings**: a raw-byte scan found target-like strings, but `cuobjdump` did not verify them as device-code records. Treat these as discovery hints.
- **role-excluded**: path heuristics classified the file as a test, benchmark, tool, sample, or host-side CUDA runtime component rather than a loadable workload artifact.
- **incomplete inspection**: `cuobjdump` timed out, returned an analyzer error, encountered an I/O error, or did not otherwise complete. The report aggregates those reasons and retains example paths and diagnostics.
- **CUDA Toolkit libraries** (cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NPP, nvJPEG): judged by the CUDA release that shipped them rather than by the `cuobjdump` listing, because NVIDIA packages several of them in ways `cuobjdump` does not fully list (cuFFT keeps most kernels in a vendor-packaged section; cuFFTW and the NPP core library carry one placeholder cubin). A library from a release at or above the profile's minimum CUDA version is informational (GPF-CUDA-020); an older release remains a likely runtime gap.
- **kernel catalog member**: a standalone cubin below an explicit `cubins` dispatch tree. Members sharing a catalog directory and declared target are treated as selectable shape-specific alternatives.
- **architecture-variant module group**: multiple shared-library modules in a recognized runtime cache that represent selectable builds of the same operation family for different GPU targets. GPU Preflight applies this rule to FlashInfer's `flashinfer_jit_cache` layout only, and requires at least two variants before grouping.

Catalog-level compatibility means the runtime has at least one compatible selectable path in the inspected group. It does not prove that every operation, shape, or data type has an optimized path or that the workload selects that path.

For each hardware target, the report also classifies the best code path found in every successfully inspected runtime artifact. From most target-specific to least, these are an exact architecture-specific cubin (`sm_XXa`), exact family-specific cubin (`sm_XXf`), exact base cubin (`sm_XX`), compatible family cubin from an earlier target, compatible base cubin from an earlier target, and PTX JIT fallback. This is compilation coverage—not proof of runtime selection or performance—and should be combined with workload-path tracing and hardware benchmarks.

An artifact present in an image may never be loaded by a particular workload. A reported target mismatch is therefore a risk to trace to the workload's import, dependency, and feature paths—not proof that every use of the image will fail.

## Resource limits

The scan reads every artifact it finds. There is no byte budget, no file count, and
no per-artifact size ceiling, so coverage reflects the image rather than a number
chosen in advance.

Free space in the temporary filesystem is the only reason an artifact goes unread. An
artifact that would not leave 2 GiB free is skipped rather than filling the disk, and
the report names it, its size, and the reason (`temporary-space`), counting it against
coverage so the figure says what went uninspected. For scale, the largest real
artifact measured is FlashAttention's `flash_ops.abi3.so` at 1677 MiB, with PyTorch's
`libtorch_cuda.so` at 0.6–1.4 GiB across NGC 24.01 to 26.09. Free space on the scan
host, or point `TMPDIR` at a larger filesystem, and repeat the scan.

Core framework libraries (PyTorch CUDA, NCCL, FlashAttention, Transformer Engine) are
named explicitly in the report when they cannot be inspected (GPF-CUDA-019), and the
result cannot be ready for hardware testing while one is unread.

The two timeouts are a different kind of limit and are set by default, because they
recover a stuck run rather than trading away coverage. Neither has been observed to
fire: no inspection in testing has reached 30 seconds, including on that 1677 MiB
artifact.

- `--cuobjdump-timeout` bounds each artifact analysis. Default: 30 seconds.
- `--command-timeout` bounds individual Docker metadata and pull commands. Default: 3600 seconds.

The pull preflight prints the manifest's approximate compressed size and available temporary-filesystem capacity. Large framework images can still require substantial network time, Docker storage, and temporary space.

When a report shows skipped candidates, the cause is always the same one: the temporary filesystem could not hold the artifact. Review the named paths and sizes, give the scan host room for the largest of them, and repeat. No run executes image contents.
