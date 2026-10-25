# Nebius GPU Preflight

GPU Preflight statically inspects a customer-owned container image for readiness on Nebius NVIDIA H100, H200, B200, B300, GB300, and VR hardware. It resolves and pulls the selected platform, inventories the image configuration and merged filesystem, and inspects CUDA artifacts without starting the image. GPU Preflight accepts built image references and performs validation including installed packages, copied files, native-library architecture, and embedded CUDA targets.

This is a preflight assessment, not hardware certification. CUDA execution, collective correctness, NVLink or MNNVL use, InfiniBand and GPUDirect RDMA selection, distributed behavior, correctness, and performance still require tests on the target hardware.

## Prerequisites

Run GPU Preflight from a Linux VM with:

- Git
- Python 3.11 or newer
- the Docker CLI and an accessible Docker daemon
- Docker registry access to the image, including read access to its manifest

  GPU Preflight resolves every image through its registry, even when a copy is already in the local Docker cache, because the report identifies the image by its immutable registry digest. Two consequences are worth planning for:

  - An image that was built locally and never pushed cannot be inspected. Push it to a registry the host can read first; otherwise Docker reports the reference as unauthorized, since it looks for it on Docker Hub.
  - Anonymous Docker Hub pulls are rate limited, and the limit covers manifest resolution as well as layer downloads, so a batch of scans can exhaust it and fail with `toomanyrequests`. Run `docker login` before scanning Docker Hub images in a loop or in CI. Registries such as `nvcr.io` apply their own limits.
- NVIDIA `cuobjdump`, in two versions: one from the same CUDA Toolkit release as the newest image being inspected, or newer, and 13.0 as a fallback.

  No single version reads every image. An older `cuobjdump` cannot decode newer fat-binary formats and GPU targets, so it produces incomplete or misleading evidence. A newer one has the opposite problem: `cuobjdump` 13.4 reports `Invalid ELF` on some CUDA 12.x libraries, cuBLAS 12.3 and HPC-X UCC among them, and lists only part of them. GPU Preflight therefore retries any analyzer error with `/usr/local/cuda-13.0/bin/cuobjdump` (`--cuobjdump-fallback`), and the installer puts both on the host.

  On a host that scans CUDA 12.x images, the fallback decides the verdict, not just its detail. Scanning `nvcr.io/nvidia/pytorch:24.01-py3` for H100 with the newer analyzer alone leaves nine HPC-X and cuBLAS 12.3 libraries unreadable and reports `INSUFFICIENT_EVIDENCE` at 96.6% coverage. A host carrying both reads the same image completely and reports `READY_FOR_HARDWARE_TESTING` at 100%.

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

The executive summary answers three independent questions:

- **Compatibility**: one of the five labels below.
- **CUDA artifact inspection coverage**: the share of eligible runtime inspection units successfully inspected with `cuobjdump`, reported as `100.0% (401/401 eligible runtime inspection units)`.
- **Reproducibility**: whether the supplied image reference is digest-pinned. An unpinned tag is a reproducibility gap, because the registry owner can change the image that tag points to.

Every report also states the static-analysis boundary explicitly: the scanner cannot establish which NCCL installation, native extension, dispatch catalog, or CUDA kernel variant the workload will actually load, and it does not prove runtime correctness or performance.

### Compatibility labels

GPU Preflight assigns the first label that applies, in this order:

- `INCOMPATIBLE`: a compatibility finding of blocker severity. Fix it before hardware testing.
- `ACTION_REQUIRED`: a compatibility warning that does not depend on whether the workload loads the artifact in question.
- `REVIEW_REQUIRED`: static inspection found a potentially incompatible artifact but cannot establish whether the workload loads it.
- `INSUFFICIENT_EVIDENCE`: no compatibility finding stands, but unresolved inspection or profile findings leave the assessment unsupported.
- `READY_FOR_HARDWARE_TESTING`: no static blocker was detected. It never means the image was executed or the workload certified. GPU Preflight does not track runtime test execution; the required hardware tests in the report are guidance for a separate validation process on the target system.

### What coverage counts

The denominator is eligible runtime inspection units, not files. Role-excluded test, tooling, and host-runtime files are outside it, and two kinds of artifact are grouped rather than counted individually:

- **Kernel catalogs**: shape-specific cubins in an explicit precompiled catalog are selectable alternatives. GPU Preflight inspects one representative per catalog and declared target combination.
- **Architecture-selected module families**, such as paired generic and `_sm90` modules in FlashInfer's JIT cache, are evaluated as one runtime dispatch group.

An alternate-architecture group drives `REVIEW_REQUIRED` only when the inspected group has neither a target-compatible sibling nor a compatible PTX/JIT fallback. A single unpaired module remains an ordinary runtime artifact, so grouping never hides a standalone compatibility gap. [CUDA artifact inspection and scan limits](docs/cuda-inspection.md) defines each term the report uses for what was found.

### How the report is organized

Detailed findings progress from broad prerequisites to specific evidence: image and platform, CPU architecture, CUDA and NCCL version baselines, aggregate CUDA target compatibility, per-artifact optimization coverage, specific artifact exceptions, evidence limitations, and required hardware tests.

The human-readable report consolidates reference, configuration, and filesystem evidence into one observed stack:

- The executive target summary reports the number of verified targets, whether the target GPU's native base and specialized targets are present, and how many artifacts contain unverified strings. Raw target lists remain in JSON and the artifact appendix.
- NCCL versions carry their source records — image configuration, system packages, Python or Conda packages, development headers — so multiple installations stay distinguishable. The effective dynamically loaded version still requires runtime confirmation.
- CUDA candidate accounting separates artifacts with enumerated targets, files without device code, incomplete inspections by reason, unverified target-string evidence, role-based exclusions, and every bounded-limit skip reason. `GPF-CUDA-016` groups unresolved records by component, reason, and one of three static assessments: likely runtime compatibility gap, optional alternate-GPU implementation, or runtime relevance not statically established.
- Optimization coverage counts binary files or grouped dispatch catalogs — not operations, features, or distinct workload code paths.

Text output shows grouped totals and representative examples, JSON retains complete records and paths in `observations.cuda_compatibility_exceptions`, and HTML provides the same inventory in an expandable appendix. Unverified strings found only in role-excluded artifacts are kept as technical evidence but are not presented as an executive evidence limitation. Required hardware-test identifiers include a concise test objective; the exact commands remain workload- and cluster-specific.

### Exit codes

The process exit code is determined by finding severity, not directly by the compatibility label:

- `0`: the inspection completed and no finding met the configured failure threshold
- `1`: an input, profile, or operational error prevented successful inspection
- `2`: the report was generated, but a finding met the configured failure threshold

A malformed command line — an unknown flag, an invalid `--format` or `--fail-on` value, a missing image argument — is rejected during argument parsing and also exits `2`, by the usual convention for usage errors. The two cases are distinguishable by output: a threshold failure writes a report and prints where, while a usage error prints the usage message and writes nothing. A CI job that treats `2` as "findings present" should check that a report exists.

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

For a pipeline that needs only one format, select it explicitly. In this mode `--output` names the single output file:

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

Each profile also names the compute-fabric adapter of its Nebius platform, which sets the UCX and HPC-X support floor for that target.

## Reference

- [CUDA artifact inspection and scan limits](docs/cuda-inspection.md): what counts as a CUDA artifact, the terms the report uses, and the `--max-*` scan limits.
- [Compute fabric, UCX, and HPC-X checks](docs/fabric-checks.md): adapter support floors and the `GPF-UCX-*` findings.
