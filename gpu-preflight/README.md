# Nebius GPU Preflight

GPU Preflight statically inspects a customer-owned container image for readiness on Nebius NVIDIA H100, H200, B200, B300, GB300, and VR hardware. It resolves and pulls the selected platform, inventories the image configuration and merged filesystem, and inspects CUDA artifacts without starting the image. GPU Preflight accepts built image references and performs validation including installed packages, copied files, native-library architecture, and embedded CUDA targets.

This is a preflight assessment, not hardware certification. CUDA execution, collective correctness, NVLink or MNNVL use, InfiniBand and GPUDirect RDMA selection, distributed behavior, correctness, and performance still require tests on the target hardware.

## Prerequisites

Run GPU Preflight from a Linux VM with:

- Git
- Python 3.11 or newer
- the Docker CLI and an accessible Docker daemon
- Docker registry access to pull the image
- NVIDIA `cuobjdump` from the same CUDA Toolkit release as the newest image being inspected, or a newer compatible release. An older `cuobjdump` can fail to decode newer fat-binary formats and GPU targets, producing incomplete or misleading evidence. Newer is not always safe either: `cuobjdump` 13.4 reports `Invalid ELF` on some CUDA 12.x libraries (for example cuBLAS 12.3 and HPC-X UCC) and lists only part of them. The installer therefore also installs `cuobjdump` 13.0, and GPU Preflight retries any analyzer error with `/usr/local/cuda-13.0/bin/cuobjdump` (`--cuobjdump-fallback`).

### Install `cuobjdump` on Ubuntu or Debian

Ubuntu may suggest `apt install nvidia-cuda-toolkit` when `cuobjdump` is missing. Do not use that package for GPU Preflight: the distribution package can lag current CUDA releases. Run the included minimal installer from the repository instead:

```bash
./install_cuobjdump.sh
cuobjdump --version
```

The script:

- detects the Ubuntu or Debian release and maps AMD64 to NVIDIA's `x86_64` repository or Arm64 to its `sbsa` repository
- configures NVIDIA's official CUDA network repository
- discovers and installs the newest available versioned `cuda-cuobjdump` package, plus `cuda-cuobjdump-13-0` as the fallback analyzer (`CUDA_CUOBJDUMP_FALLBACK_PACKAGE=none` skips it)
- creates `/usr/local/bin/cuobjdump` without replacing an existing regular file
- prints the resolved package, executable path, and analyzer version

The script uses `sudo` for package installation. It does not install a GPU driver or the complete CUDA Toolkit, and a GPU is not required. To pin an analyzer release instead of selecting the newest available package:

```bash
CUDA_CUOBJDUMP_PACKAGE=cuda-cuobjdump-13-4 ./install_cuobjdump.sh
```

Verify on the host:

```bash
python3 --version
docker info
cuobjdump --version
```

GPU Preflight does not require a GPU, NVIDIA driver, CUDA runtime, QEMU, or access to the target hardware. It runs directly from the cloned repository and does not require a virtual environment or Python package installation.

## What it checks

- image platforms, immutable platform digests, and compressed size
- CPU architecture compatibility for the selected hardware
- H100/H200 `sm_90`, B200 `sm_100`, B300/GB300 `sm_103`, and VR `sm_107`
- CUDA baseline, family (`f`), architecture-specific (`a`), cubin, and retained-PTX compatibility
- installed CUDA version against the hardware profile's compiler-support baseline
- installed NCCL versions against target-specific minimum platform-support baselines
- UCX and HPC-X installations against the support floor of the target platform's compute-fabric adapter, including whether a modular UCX build ships its CUDA transport
- image configuration and environment-derived CUDA targets
- Python, Debian, Alpine, and Conda package metadata
- per-file CUDA cubin and retained-PTX targets using NVIDIA `cuobjdump`
- precompiled dispatch catalogs, including standalone cubin trees and known
  architecture-selected module families, evaluated across their selectable variants
- mutable or non-digest-pinned image references
- target-specific runtime tests still required after static inspection

Image programs are never passed to the host loader. GPU Preflight copies one bounded candidate file at a time to a temporary location, invokes host `cuobjdump` without a shell, records the image path and targets, and removes the temporary file.

## How to read the result

The executive summary intentionally separates three static-assessment questions:

- **Compatibility**: `READY_FOR_HARDWARE_TESTING`, `REVIEW_REQUIRED`, `ACTION_REQUIRED`, `INCOMPATIBLE`, or `INSUFFICIENT_EVIDENCE`
- **CUDA artifact inspection coverage**: the percentage and count of eligible runtime inspection units successfully inspected with `cuobjdump`
- **Reproducibility**: whether the supplied image reference is digest-pinned

An unpinned tag is a reproducibility gap because the registry owner can change the image to which that tag points.

`READY_FOR_HARDWARE_TESTING` means no static blocker was detected; it never means the image was executed or the workload was certified. GPU Preflight does not track runtime test execution. The required hardware tests in the report are guidance for a separate validation process on the target system.

`REVIEW_REQUIRED` means static inspection found a potentially incompatible artifact, but cannot establish whether the workload loads it. `ACTION_REQUIRED` is reserved for compatibility findings that do not depend on that workload-relevance question. Coverage is reported without a subjective `LOW`, `MEDIUM`, or `HIGH` label and includes its numerator and denominator, such as `100.0% (401/401 eligible runtime inspection units)`. Role-excluded test, tooling, and host-runtime files are not part of that denominator. Shape-specific cubins in an explicit precompiled kernel catalog are selectable alternatives; GPU Preflight groups them and inspects one representative per catalog and declared target combination instead of counting every catalog member as an independent required artifact. Known architecture-selected module families, such as paired generic and `_sm90` modules in FlashInfer's JIT cache, are also evaluated as one runtime dispatch group. An alternate-architecture group drives `REVIEW_REQUIRED` only when the inspected group has neither a target-compatible sibling nor a compatible PTX/JIT fallback. A single unpaired module remains an ordinary runtime artifact, so grouping does not hide a standalone compatibility gap.

The human-readable report consolidates reference, configuration, and filesystem evidence into one observed stack. Its executive target summary reports the number of verified targets, whether the target GPU's native base and specialized targets are present, and how many artifacts contain unverified strings. Raw target lists remain available in JSON and the artifact appendix. NCCL versions include their source records—image configuration, system packages, Python or Conda packages, and development headers—so multiple installations are distinguishable even though the effective dynamically loaded version still requires runtime confirmation. CUDA candidate accounting distinguishes artifacts with enumerated CUDA targets, files without embedded device code, incomplete inspections with reason categories, unverified target-string evidence, role-based exclusions, and every bounded-limit skip reason. `GPF-CUDA-016` groups unresolved records by component, reason, and one of three static assessments: likely runtime compatibility gap, optional alternate-GPU implementation, or runtime relevance not statically established. Text output shows grouped totals and representative examples; JSON retains complete records and paths in `observations.cuda_compatibility_exceptions`; HTML provides the same full inventory in an expandable appendix. Counts in optimization coverage describe binary files or grouped dispatch catalogs, not operations, features, or distinct workload code paths. Unverified strings found only in role-excluded artifacts are retained as technical evidence but are not presented as an executive evidence limitation. Required hardware-test identifiers include a concise test objective; the exact commands remain workload- and cluster-specific.

Every report states the top-level static-analysis boundary explicitly: the scanner cannot establish which NCCL installation, native extension, dispatch catalog, or CUDA kernel variant the workload will actually load, and it does not prove runtime correctness or performance.

Detailed findings progress from broad prerequisites to specific evidence: image and platform, CPU architecture, CUDA and NCCL version baselines, aggregate CUDA target compatibility, per-artifact optimization coverage, specific artifact exceptions, evidence limitations, and required hardware tests.

### Exit codes

The process exit code is determined by finding severity, not directly by the compatibility label:

- `0`: the inspection completed and no finding met the configured failure threshold
- `1`: an input, profile, or operational error prevented successful inspection
- `2`: the report was generated, but a finding met the configured failure threshold

The default threshold is `warning`, so `REVIEW_REQUIRED`, `ACTION_REQUIRED`, and `INCOMPATIBLE` reports fail closed in CI. Use `--fail-on blocker` to allow warnings, or `--fail-on never` for an advisory report. `--fail-on never` prevents findings from producing exit code `2`; command errors can still produce exit code `1`.

## Quick start

```bash
git clone https://github.com/nebius/ml-cookbook.git
cd ml-cookbook/gpu-preflight

PYTHONPATH=src python3 -m gpu_preflight profiles

PYTHONPATH=src python3 -m gpu_preflight inspect \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  --target gb300 \
  --platform linux/arm64
```

At least one `--target` is required. If `--platform` is omitted, GPU Preflight selects the image platform compatible with each requested hardware target. Use `arm64`, `amd64`, `linux/arm64`, or `linux/amd64` to select a platform explicitly.

The inspection:

1. Verifies Docker daemon access and locates `cuobjdump`.
2. Resolves registry manifests and the immutable platform digest.
3. Pulls the selected platform using the customer's Docker configuration and credentials.
4. Inspects image configuration and package metadata.
5. Creates, but never starts, a temporary container and streams its merged root filesystem with `docker export`.
6. Runs host `cuobjdump` against bounded candidate CUDA artifacts.
7. Removes the temporary container and anonymous volumes.
8. Generates text, JSON, and HTML reports while leaving Docker's normal layer cache intact.

Pull and scan progress is written to stderr. By default, the human-readable report is also written to stdout and all three formats are saved together in a unique timestamped run directory:

```text
gpu-preflight-reports/
└── pytorch-26.09-py3-gb300-20261011T220102.123456Z/
    ├── report.txt
    ├── report.json
    └── report.html
```

This preserves earlier runs. Use `--output-dir PATH` to choose an exact report directory; if that directory already contains report files, GPU Preflight refuses to replace them unless `--overwrite` is supplied. The command prints the absolute path of every generated report when it finishes. Pulling and exporting a foreign-architecture image does not require emulation because no image program is executed.

For a pipeline that deliberately needs only one format, select it explicitly. `--output` continues to name the single output file in this mode:

```bash
PYTHONPATH=src python3 -m gpu_preflight inspect IMAGE \
  --target gb300 \
  --format json \
  --output report.json

PYTHONPATH=src python3 -m gpu_preflight inspect IMAGE \
  --target gb300 \
  --target vr \
  --format html \
  --output report.html \
  --fail-on never
```

JSON reports implement the strict, versioned `schemas/report.schema.json` contract. They include the tool version, profile schema version and hash, scan limits and capabilities, selected image digest, duration, per-artifact evidence provenance, and required hardware test plan. Finding breakdowns are retained as structured `details` records with `label`, `value`, and optional `context` fields; text renders them as indented rows and HTML renders them as tables.

## Comparing hardware targets

Repeat `--target` to assess the same image against several hardware profiles:

```bash
PYTHONPATH=src python3 -m gpu_preflight inspect IMAGE \
  --target gb300 \
  --target vr
```

GPU Preflight pulls each distinct required platform at most once. Comparing targets with different CPU architectures can therefore pull more than one platform variant.

### Compute fabric profiles

Each hardware profile names the compute-fabric adapter of its Nebius platform under `[fabric]`, and the adapters themselves live in `src/gpu_preflight/data/fabrics/`. Adapter support is a property of the NIC generation rather than the GPU, so one adapter entry and one citation cover every platform that ships it: ConnectX-7 for H100, H200, and B200; ConnectX-8 for B300 and GB300; ConnectX-9 for Vera Rubin.

An adapter entry records `minimum_hpcx_version`, the first HPC-X release whose change log adds that adapter, and `minimum_ucx_version`, the UCX release bundled with it. The UCX floor is therefore derived rather than separately published: NVIDIA states which HPC-X release supports an adapter, not a standalone UCX minimum, and each entry's notes say so. ConnectX-9 carries no floor, because no HPC-X release has announced support for it.

### UCX and HPC-X checks

UCX is optional. NCCL reaches InfiniBand through its own plugin, so an image with no UCX produces no finding at all. When an image does ship UCX or HPC-X, GPU Preflight reports:

- `GPF-UCX-001` / `GPF-UCX-006`: the installed releases and transport-module directories, or that the release could not be established. A bare `libucp.so.0` soname identifies the ABI, not the UCX release, so an installation without package metadata, an `HPCX_VERSION`-style environment variable, or an `/opt/hpcx/VERSION` file is reported as undetermined rather than guessed.
- `GPF-UCX-002`: a modular UCX build whose transport modules exclude CUDA. Such a build loads and passes connectivity tests while staging every GPU buffer through host memory. The finding requires a module directory that already holds `libuct_*` objects, because a UCX built with `--enable-modules=no` links CUDA support into `libuct` itself and legitimately ships no separate module.
- `GPF-UCX-003`: NIXL installed with no UCX runtime at all. UCX is NIXL's default transfer backend.
- `GPF-UCX-004` / `GPF-UCX-005`: detected releases against the adapter floor described above.

As everywhere else in the report, these are static observations. GPU Preflight cannot establish which UCX installation a workload loads, nor which transport it selects at runtime; `ucx-transport` is in each profile's required hardware checks for that reason.

## How CUDA artifact inspection works

Container images usually do not have one directory per operation or GPU architecture. Frameworks and libraries install **CUDA artifacts** such as shared libraries (`.so`), Python native extensions, executables, and fat binaries. One artifact can contain CPU-side code plus many GPU **kernels** (device functions launched on the GPU), packaged as one or more architecture-specific target images:

- `sm_XX` is a native cubin containing GPU machine code for that SM target.
- `compute_XX` is PTX virtual code that a compatible NVIDIA driver may JIT-compile for a newer GPU.
- Architecture-specific targets such as `sm_90a` require that exact architecture family and do not carry forward like baseline PTX.

GPU Preflight uses file paths and names only to find candidates and estimate their role. It then uses `cuobjdump` to inspect the contents of each selected artifact. Therefore, **CUDA artifact** or **CUDA binary** is the correct term for a reported `.so`; **kernel** refers to one GPU entry point inside it.

Report terms:

- **with embedded CUDA targets**: `cuobjdump` completed and enumerated at least one cubin or PTX target.
- **without device code**: `cuobjdump` completed but found no embedded CUDA device code. Broad candidate matching intentionally produces some of these.
- **unverified target strings**: a bounded raw-byte scan found target-like strings, but `cuobjdump` did not verify them as device-code records. Treat these as discovery hints.
- **role-excluded**: path heuristics classified the file as a test, benchmark, tool, sample, or host-side CUDA runtime component rather than a loadable workload artifact.
- **incomplete inspection**: `cuobjdump` timed out, returned an analyzer error, encountered an I/O error, or did not otherwise complete. The report aggregates those reasons and retains example paths and diagnostics.
- **CUDA Toolkit libraries** (cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NPP, nvJPEG): judged by the CUDA release that shipped them rather than by the `cuobjdump` listing, because NVIDIA packages several of them in ways `cuobjdump` does not fully list (cuFFT keeps most kernels in a vendor-packaged section; cuFFTW and the NPP core library carry one placeholder cubin). A library from a release at or above the profile's minimum CUDA version is informational (GPF-CUDA-020); an older release remains a likely runtime gap.
- **kernel catalog member**: a standalone cubin below an explicit `cubins` dispatch tree. Members sharing a catalog directory and declared target are treated as selectable shape-specific alternatives.
- **architecture-variant module group**: multiple shared-library modules in a recognized runtime cache that represent selectable builds of the same operation family for different GPU targets. GPU Preflight currently applies this conservative rule to FlashInfer's `flashinfer_jit_cache` layout and requires at least two variants before grouping.

Catalog-level compatibility means the runtime has at least one compatible selectable path in the inspected group. It does not prove that every operation, shape, or data type has an optimized path or that the workload selects that path.

For each hardware target, the report also classifies the best code path found in every successfully inspected runtime artifact. From most target-specific to least, these are an exact architecture-specific cubin (`sm_XXa`), exact family-specific cubin (`sm_XXf`), exact base cubin (`sm_XX`), compatible family cubin from an earlier target, compatible base cubin from an earlier target, and PTX JIT fallback. This is compilation coverage—not proof of runtime selection or performance—and should be combined with workload-path tracing and hardware benchmarks.

An artifact present in an image may never be loaded by a particular workload. A reported target mismatch is therefore a risk to trace to the workload's import, dependency, and feature paths—not proof that every use of the image will fail.

## Resource limits

- `--max-scan-mb` limits candidate-file content read during the filesystem scan; it does not limit the image download or rootfs stream. Default: 16384 MiB.
- `--max-artifact-mb` limits one file passed to `cuobjdump`. Default: 2048 MiB, which covers PyTorch's `libtorch_cuda.so` (about 1.3–1.4 GiB in NGC 24.01–25.01).
- `--max-cuobjdump-files` bounds the number of `cuobjdump` invocations. Default: 1024.
- Core framework libraries (PyTorch CUDA, NCCL, FlashAttention, Transformer Engine) are always inspected: they do not count against `--max-scan-mb` or `--max-cuobjdump-files`, only against `--max-artifact-mb` and the timeout. If one cannot be inspected, the report names it (GPF-CUDA-019) and the result cannot be ready for hardware testing.
- `--exhaustive` removes only the `cuobjdump` file-count cap. Candidate-byte, per-artifact-size, per-invocation-timeout, and Docker-command safeguards remain active.
- `--cuobjdump-timeout` bounds each artifact analysis. Default: 30 seconds.
- `--command-timeout` bounds individual Docker metadata and pull commands. Default: 3600 seconds.

The pull preflight prints the manifest's approximate compressed size and available temporary-filesystem capacity. Large framework images can still require substantial network time, Docker storage, and temporary space.

When a report shows skipped candidates, review the retained paths first and change only the limit relevant to files the workload may load. `--max-scan-mb` is a cumulative candidate-content read budget; it is not the image size, downloaded bytes, Docker storage limit, or host memory limit. At high coverage, GPU Preflight does not recommend a broad multiplier. At lower coverage, a larger scan budget can expose enough additional files to hit `--max-cuobjdump-files`, so the report may suggest both values as a bounded deeper-scan starting point. Use `--exhaustive` when a deliberate one-time run should inspect every eligible ordinary artifact and every representative catalog target that fits the remaining safeguards; do not treat it as permission to execute image contents.
