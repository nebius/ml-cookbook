# Nebius GPU Preflight

GPU Preflight statically inspects a customer-owned container image for readiness on Nebius NVIDIA H100, H200, B200, B300, GB300, RTX PRO 6000, and Vera Rubin hardware. It resolves and pulls the selected platform, inventories the image configuration and merged filesystem, and inspects CUDA artifacts without starting the image.

## Disclaimer

GPU Preflight uses heuristic static analysis and can make mistakes: it may flag a compatible component or miss an incompatibility. Read every finding and compatibility label alongside its evidence. `READY_FOR_HARDWARE_TESTING`, or 100% inspection coverage, does not guarantee compatibility, correctness, or performance. CUDA execution, collective correctness, NVLink or MNNVL use, InfiniBand and GPUDirect RDMA selection, distributed behavior, correctness, and performance are established by running your own workload on the target hardware.

## Prerequisites

- A Linux VM with:
  - Git
  - Python 3.11 or newer
  - Docker CLI and a reachable Docker daemon
  - NVIDIA `cuobjdump`

```bash
./install_cuobjdump.sh
cuobjdump --version
```

> [!IMPORTANT]
> `cuobjdump` is installed twice: the newest release and 13.0 as a fallback. No single version reads every image. An older analyzer cannot decode newer fat-binary formats; a newer one reports `Invalid ELF` on some CUDA 12.x libraries, cuBLAS 12.3 and HPC-X UCC among them. GPU Preflight retries any analyzer error with the fallback (`--cuobjdump-fallback`).

> [!NOTE]
> Every image is resolved through its registry, even one already in the local Docker cache because the report identifies it by immutable digest. The host needs manifest read access and an image built locally must be pushed first.

<details>
<summary>Installer details, and pinning an analyzer release</summary>

Ubuntu may suggest `apt install nvidia-cuda-toolkit` when `cuobjdump` is missing. Do not use that package: it can lag current CUDA releases.

`install_cuobjdump.sh` detects the Ubuntu or Debian release, maps AMD64 to NVIDIA's `x86_64` repository or Arm64 to its `sbsa` repository, configures NVIDIA's official CUDA network repository, and installs the newest versioned `cuda-cuobjdump` package plus `cuda-cuobjdump-13-0` as the fallback (`CUDA_CUOBJDUMP_FALLBACK_PACKAGE=none` skips it). It creates `/usr/local/bin/cuobjdump` without replacing an existing regular file, prints the resolved package, path, and version, and uses `sudo` for package installation. It installs neither a GPU driver nor the full CUDA Toolkit.

To pin a release instead of taking the newest:

```bash
CUDA_CUOBJDUMP_PACKAGE=cuda-cuobjdump-13-4 ./install_cuobjdump.sh
```

No GPU, driver, CUDA runtime, QEMU, or access to the target hardware is required, and nothing is installed into Python: GPU Preflight runs from the clone.

</details>

## What it checks

| Area | Checks |
| --- | --- |
| Image | platforms, immutable platform digests, compressed size, and whether the reference is digest-pinned |
| CPU | architecture compatibility for the selected hardware |
| CUDA targets | H100/H200 `sm_90`, B200 `sm_100`, B300/GB300 `sm_103`, RTX PRO 6000 `sm_120`, VR `sm_107`; baseline, family (`f`), architecture-specific (`a`), cubin, and retained-PTX compatibility |
| Versions | CUDA against the profile's compiler-support baseline, NCCL against the target's platform-support baseline, and UCX and HPC-X against the support floor of the target's compute-fabric adapter, including whether a modular UCX build ships its CUDA transport |
| Packages | Python, Debian, Alpine, and Conda metadata, plus image configuration and environment-derived CUDA targets |
| Artifacts | per-file cubin and retained-PTX targets via `cuobjdump`, and precompiled dispatch catalogs — standalone cubin trees and known architecture-selected module families — across their selectable variants |
| Next steps | the target-specific runtime tests still required after static inspection |

Image programs are never passed to the host loader. GPU Preflight copies one candidate file at a time to a temporary location, invokes host `cuobjdump` without a shell, records the image path and targets, and removes the temporary file.

## How to read the result

Every target gets an executive summary that answers three independent questions:

```text
NVIDIA B300 (b300)
  Compatibility: ACTION_REQUIRED
  Discovered artifacts inspected: 71.4% (5/7 inspection units successfully analyzed)
  Reproducibility: UNPINNED
  Summary: Address the listed compatibility risks before hardware testing.
```

- **Compatibility** is one of the five labels below.
- **Coverage** is the share of eligible runtime inspection units `cuobjdump` read successfully.
- **Reproducibility** is whether the reference is digest-pinned. An unpinned tag is a gap, because the registry owner can change what that tag points to.

Every report also states the static-analysis boundary explicitly: the scanner cannot establish which NCCL installation, native extension, dispatch catalog, or CUDA kernel variant the workload will actually load, and it does not prove runtime correctness or performance.

### Compatibility labels

GPU Preflight assigns the first label that applies, in this order:

| Label | What it means |
| --- | --- |
| `INCOMPATIBLE` | A compatibility finding of blocker severity. Fix it before hardware testing. |
| `ACTION_REQUIRED` | A compatibility warning that does not depend on whether the workload loads the artifact. |
| `REVIEW_REQUIRED` | A potentially incompatible artifact was found, but whether the workload loads it cannot be established statically. |
| `INSUFFICIENT_EVIDENCE` | No compatibility finding stands, but unresolved inspection or profile findings leave the assessment unsupported. |
| `READY_FOR_HARDWARE_TESTING` | No static blocker was detected. |

`READY_FOR_HARDWARE_TESTING` never means the image was executed or the workload certified. GPU Preflight does not track runtime test execution; the required hardware tests in the report are guidance for a separate validation process on the target system.

### What coverage counts

The denominator is eligible runtime inspection units, not files. Role-excluded test, tooling, and host-runtime files are outside it, and two kinds of artifact are grouped rather than counted individually:

- **Kernel catalogs**: shape-specific cubins in an explicit precompiled catalog are selectable alternatives. GPU Preflight inspects one representative per catalog and declared target combination.
- **Architecture-selected module families**, such as paired generic and `_sm90` modules in FlashInfer's JIT cache, are evaluated as one runtime dispatch group.

An alternate-architecture group drives `REVIEW_REQUIRED` only when the inspected group has neither a target-compatible sibling nor a compatible PTX/JIT fallback. A single unpaired module remains an ordinary runtime artifact, so grouping never hides a standalone compatibility gap. [CUDA artifact inspection](docs/cuda-inspection.md) defines each term the report uses for what was found.

### How the report is organized

Detailed findings progress from broad prerequisites to specific evidence: image and platform, CPU architecture, CUDA and NCCL version baselines, aggregate CUDA target compatibility, per-artifact optimization coverage, specific artifact exceptions, evidence limitations, and required hardware tests.

The human-readable report consolidates reference, configuration, and filesystem evidence into one observed stack:

- The executive target summary reports the number of verified targets, whether the target GPU's native base and specialized targets are present, and how many artifacts contain unverified strings. Raw target lists remain in JSON and the artifact appendix.
- NCCL versions carry their source records — image configuration, system packages, Python or Conda packages, development headers — so multiple installations stay distinguishable. The effective dynamically loaded version still requires runtime confirmation.
- CUDA candidate accounting separates artifacts with enumerated targets, files without device code, incomplete inspections by reason, unverified target-string evidence, role-based exclusions, and every bounded-limit skip reason. `GPF-CUDA-016` groups unresolved records by component, reason, and one of three static assessments: likely runtime compatibility gap, optional alternate-GPU implementation, or runtime relevance not statically established.
- Optimization coverage counts binary files or grouped dispatch catalogs — not operations, features, or distinct workload code paths.

Text output shows grouped totals and representative examples, JSON retains complete records and paths in `observations.cuda_compatibility_exceptions`, and HTML provides the same inventory in an expandable appendix. Unverified strings found only in role-excluded artifacts are kept as technical evidence but are not presented as an executive evidence limitation. Required hardware-test identifiers include a concise test objective; the exact commands remain workload- and cluster-specific.

### Exit codes

The exit code is determined by finding severity, not directly by the compatibility label:

| Code | Meaning |
| --- | --- |
| `0` | The inspection completed and no finding met the failure threshold. |
| `1` | An input, profile, or operational error prevented inspection. |
| `2` | A report was generated, but a finding met the failure threshold. |

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

JSON reports implement the strict, versioned `schemas/report.schema.json` contract. They include the tool version, profile schema version and hash, scan settings and capabilities, selected image digest, duration, per-artifact evidence provenance, and required hardware test plan. Finding breakdowns are retained as structured `details` records with `label`, `value`, and optional `context` fields; text renders them as indented rows and HTML renders them as tables.

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

- [CUDA artifact inspection](docs/cuda-inspection.md): what counts as a CUDA artifact, the terms the report uses, and what can refuse a file.
- [Compute fabric, UCX, and HPC-X checks](docs/fabric-checks.md): adapter support floors and the `GPF-UCX-*` findings.

## Known limitations

- **The scan does not run the image.** It reads the filesystem and metadata, so it produces no evidence of correctness, stability, or performance.
- **Which library the workload loads is not knowable statically.** An image can hold several NCCL installations or several builds of the same extension; the scan reports all of them and cannot say which one the process will resolve at run time.
- **Kernels compiled at run time are not inspected.** Triton, CuTe DSL and similar leave no cubin to read, so they are judged by installed version against a per-hardware floor (`GPF-PKG-*`) rather than by inspecting code.
- **Coverage measures what was found, not what should be there.** A library absent from the image is not counted as missing, so 100% coverage does not mean the image is complete.
- **A sampled catalog represents operations that were not individually inspected.** Catalog-level compatibility means at least one compatible selectable path exists in the group, not that every operation, shape, and dtype has one.
- **Free temporary space can leave an artifact unread.** An artifact that would not leave 2 GiB free is skipped and named in the report, counting against coverage. See [CUDA artifact inspection](docs/cuda-inspection.md).
