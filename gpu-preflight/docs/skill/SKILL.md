---
name: gpu-preflight
description: Use when someone needs to know whether a container image will run on Nebius NVIDIA GPU hardware (H100, H200, B200, B300, GB300, RTX PRO 6000, Vera Rubin) before any GPU time is booked. Runs the Nebius GPU Preflight scanner from the ml-cookbook repo, which statically inspects a Docker image - CUDA sm_ targets, CUDA/NCCL/UCX/HPC-X version baselines, CPU architecture, package metadata - without starting it, then reads the report. Trigger on "will this image run on B300/GB300", "image compatibility", "sm_100 / sm_103 / sm_90", "CUDA target check", "preflight", "arm64 image for GB300", "NCCL version baseline", "will my image run on Blackwell", or a request to check an image before a PoC, a capacity request, or a hardware test.
---

# GPU Preflight: static container image readiness for Nebius GPU hardware

GPU Preflight pulls the image, exports its filesystem, and inspects CUDA artifacts with host `cuobjdump` — it never executes the image. It answers "is this image obviously wrong for this hardware?" before GPU hours are spent discovering it on the cluster.

It is **not** certification. CUDA execution, NCCL collectives, NVLink/MNNVL, GPUDirect RDMA, performance, and correctness still need real tests on the target hardware. Say this in every summary, whoever is reading it.

## Required context

- **The image reference**, exactly as it is used in the workload (`nvcr.io/nvidia/pytorch:25.04-py3`, a private registry path, or a digest).
- **The target hardware**: `h100`, `h200`, `b200`, `b300`, `gb300`, `rtx6000`, or `vr`. Aliases work (`gpu-gb300`, `nvidia-h100`, `gb300-nvl72`, `gpu-b300-sxm`). `PYTHONPATH=src python3 -m gpu_preflight profiles` lists them with their SM target, CPU arch, CUDA/NCCL floors, and fabric adapter. `vr` is marked **provisional**; the rest are `supported`.
- **The image's CUDA era** (from the tag, or ask). This decides which analyzer you need — see "Pick the analyzer before you trust a verdict", the most important section here.
- **Registry access** from the scanning host. Every image is resolved through its registry even when it is already in the local Docker cache, because the report pins the immutable digest.

If the request names a Nebius platform rather than a GPU (`gpu-b300-sxm`, `gpu-gb300`), map it to the target id yourself; don't ask. If it names no hardware at all, ask — the whole report is target-relative.

## Setup (once per host)

Linux, Python 3.11+, Docker CLI with a reachable daemon, and NVIDIA `cuobjdump`. **No GPU, driver, or CUDA runtime is needed.** No virtualenv or `pip install` — run with `PYTHONPATH=src`.

The repo does **not** install Docker for you. On a bare VM, install Docker Engine first (`docker-ce-cli docker-ce containerd.io` from Docker's apt repo), add your user to the `docker` group, and confirm `docker info` works as a non-root user.

```bash
git clone https://github.com/nebius/ml-cookbook.git
cd ml-cookbook/gpu-preflight
./install_cuobjdump.sh && cuobjdump --version
```

The installer adds the newest versioned `cuda-cuobjdump` plus `cuda-cuobjdump-13-0` as fallback. That pair is **not sufficient for every image** — read the next section.

Verify before the first scan: `python3 --version`, `docker info`, `cuobjdump --version`.

## Pick the analyzer before you trust a verdict

No single `cuobjdump` reads every image, and **the analyzer changes the verdict, not just the detail.** Two confirmed cases on the same host:

| Image | Analyzers | Coverage | Label |
| --- | --- | --- | --- |
| `pytorch:24.01-py3` → h100 | latest only | 96.7% (263/272) | `INSUFFICIENT_EVIDENCE` |
| `pytorch:24.01-py3` → h100 | latest + 13.0 fallback | **100%** (272/272) | `READY_FOR_HARDWARE_TESTING` |

Same image, same target, same host — only the analyzer differs. Nine HPC-X and cuBLAS 12.3 libraries are unreadable to a current 13.x analyzer alone, and an artifact the analyzer cannot read is recorded as missing evidence, not as a compatibility result, so a blocker inside it would not appear.

**Rules:**

1. **Never report a verdict from a scan below 100% coverage.** Sub-100% means the scanner could not read something, and what it could not read may be a blocker.
2. **`GPF-CUDA-019` (core framework libraries were not inspected) invalidates the verdict.** The label comes back `INSUFFICIENT_EVIDENCE` rather than something more reassuring; fix the analyzer and rerun.
3. **Keep both analyzers installed.** CUDA 13.x images: the primary. CUDA 12.x: the 13.0 fallback handles them, where a current 13.x analyzer alone reports `Invalid ELF` on cuBLAS 12.3 and HPC-X UCC. `install_cuobjdump.sh` sets both up, and `apt-cache search cuda-cuobjdump` lists what is available. CUDA 11.x images are out of scope for GPU Preflight; an image that old should be rebuilt on a supported CUDA release before it is assessed.
4. **Read `Incomplete inspection reasons` to tell the causes apart.** The scan sets no limits of its own, so a shortfall is one of three things, and only the first two are yours to fix. `N analyzer error` is the wrong analyzer — install the fallback and rerun. `temporary filesystem space` means an artifact did not fit: free disk on the host, or point `TMPDIR` at a larger filesystem. `N unreadable architecture` is different in kind: `cuobjdump` refused a Fermi-era artifact *while naming the architecture it holds*, so those targets do count as listed and the artifact is not unread. An older toolkit (`--cuobjdump-fallback`) lists it in full; without one, report it as an image too old to assess rather than as a scanner failure.

## Workflow

1. `PYTHONPATH=src python3 -m gpu_preflight profiles` if unsure of the target id.
2. Run the inspection. At least one `--target` is required; repeat it to compare hardware in one run.
   ```bash
   PYTHONPATH=src python3 -m gpu_preflight inspect \
     nvcr.io/nvidia/pytorch:25.04-py3 \
     --target gb300 \
     --platform linux/arm64
   ```
   Omit `--platform` to let it pick the platform matching each target. Progress goes to stderr and the text report to stdout. **`--format` defaults to `all`, which writes `report.txt`, `report.json` and `report.html` together** into a timestamped directory under `./gpu-preflight-reports/`, whose absolute paths are printed at the end. Do not pass `--format` unless a pipeline genuinely needs one file; the JSON is what you triage from and the HTML is what you hand to someone else, so a run that produced only one of them has to be repeated.
3. **Check coverage first.** If it is not 100%, fix the analyzer or free disk on the host and rerun before reading anything else.
4. **Triage `REVIEW_REQUIRED` in the JSON** rather than quoting the label (see below).
5. Summarize: label, coverage, reproducibility, specific `GPF-*` findings — package floors included (see below) — and the explicit statement that hardware testing is still required.

## Reading the result

- **Compatibility label**, first that applies: `INCOMPATIBLE` (blocker; fix before hardware testing) → `ACTION_REQUIRED` (warning that holds regardless of what the workload loads) → `REVIEW_REQUIRED` (possibly incompatible artifact, relevance not statically provable) → `INSUFFICIENT_EVIDENCE` (no finding stands, but evidence is incomplete) → `READY_FOR_HARDWARE_TESTING` (no static blocker; never means it ran).
- **Coverage** — share of eligible *runtime inspection units*, e.g. `100.0% (302/302)`. Not files: kernel catalogs count one representative per catalog/target pair, and architecture-selected module families (generic + `_sm90` siblings) count as one dispatch group. It reads **`unavailable`** when the image has no CUDA artifacts at all.
- **Reproducibility** — `PINNED` or `UNPINNED`. A tag is `UNPINNED` and raises `GPF-REPRO-002`.

### `REVIEW_REQUIRED` is the normal result for stock vendor images

How much sits behind the label varies by an order of magnitude, so always open the JSON before you say anything about it.

- **A stock framework image is usually one file.** `nvcr.io/nvidia/pytorch:25.04-py3` on h100 returns `REVIEW_REQUIRED` at 100% coverage driven by exactly **one** file: `group_norm_v2_cuda...so`, built only for sm_100. Most other exceptions in that report are informational (cuFFT/NPP libraries whose kernels `cuobjdump` does not fully list).
- **A large inference image is hundreds.** `vllm/vllm-openai:v0.33.0` at 100% coverage produced **550 review-required exceptions for h100 and 220 for b300**; SGLang and TensorRT-LLM land in the same range. Triage these by grouping, not file by file — and note the asymmetry: far more on the older target, because artifacts built only for newer architectures cannot run there.

Do not report the bare label. Name the files (or the groups), and establish whether the workload actually loads them.

Triage straight from the JSON:

The default run writes into a timestamped bundle, so point at that rather than the working directory:

```bash
BUNDLE=$(ls -dt gpu-preflight-reports/*/ | head -1)   # or the path the run printed
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for r in d['observations']['cuda_compatibility_exceptions']:
    if r['status_effect'] == 'review-required':
        print(r['path'], r['targets'], r['reason_summary'])
```

`status_effect` is `review-required` for the ones that moved the label and `informational` for the rest.

### Package floors: the part coverage cannot tell you

Artifact scanning reads kernels that are already compiled. Triton, CuTe DSL and similar
compile at run time, leaving nothing to inspect, so the tool judges them by installed
version against a per-hardware floor with a release-note citation. **A 100% coverage
figure says nothing about these**, which is why `GPF-PKG-002` matters even on an
otherwise clean report.

| Finding | Meaning |
| --- | --- |
| `GPF-PKG-001` | meets the published floor |
| `GPF-PKG-002` | **predates published support for this hardware** — warning, raises the label |
| `GPF-PKG-003` | present, no floor published, not assessed |
| `GPF-PKG-004` | several versions of one package installed; which one loads is a runtime question |
| `GPF-PKG-005` | published for other hardware, not this one |

The floors differ per target and live in the profile TOML, so `--profile-dir` can
override them without touching code. B300 requires `triton` 3.5.1 and
`nvidia-cutlass-dsl` 4.3.1; GB300 needs CuTe DSL 4.4.0, a notch higher, because SM103
reached the DSL in 4.3.1 but the GB300 platform only in 4.4.0. Quote the citation in the
finding rather than the bare version — it names the release note that set the floor.

### Exit codes

- `0` — nothing met the failure threshold.
- `1` — input, profile, or operational error; **no report written**. Includes a missing or unknown `--target`, an unreadable image, a malformed `--profile-dir` profile, and reusing `--output-dir` without `--overwrite`.
- `2` — a finding met the threshold **or** an argparse usage error (bad flag, bad `--format`/`--platform` value). Tell them apart by whether a report exists.

Reproducibility findings do not set the exit code: an unpinned tag is reported as `GPF-REPRO-002` but does not fail the run by itself. Pin the digest when the validation evidence needs to identify exact layers.

Read the label rather than the exit code. `REVIEW_REQUIRED` and `INSUFFICIENT_EVIDENCE` both exit 2 under the default threshold, and they mean different things.

## Gotchas

- **A locally built, never-pushed image cannot be scanned.** It fails with `denied: requested access to the resource is denied / unauthorized`, exit 1. Push it to a readable registry first.
- **The Docker Hub anonymous rate limit arrives fast.** Manifest resolution counts, and every run re-resolves even a cached image — roughly ten scans of the *same cached* `ubuntu:24.04` exhausted it and failed with `toomanyrequests`. `docker login` before any repeated Docker Hub work. `nvcr.io` was unaffected across 12+ scans.
- **Budget time and disk.** NGC PyTorch images are 8–11 GiB compressed per platform and scan in **2–8 minutes** (115s–489s observed). Large inference images cost far more: vLLM, SGLang and TensorRT-LLM scanned for h100+b300 took **291–504s each and consumed 33–52 GiB of disk per image** (the pulled image plus the merged rootfs export), walking 200k+ files. Budget **~60 GiB free per concurrent scan** and reclaim between images — `docker image prune -f` only removes dangling layers, so remove the scanned image by tag. A multi-target run across architectures pulls each platform once (amd64 *and* arm64 for `--target b300 --target gb300`).
- **Cross-architecture scanning needs no emulation.** An arm64 image inspects cleanly on an amd64 host, because nothing is executed.
- **SM targets**: H100/H200 `sm_90`, B200 `sm_100`, B300/GB300 `sm_103`, RTX PRO 6000 `sm_120`, VR `sm_107`. GB300 and VR require **arm64** images; the rest amd64.
- **Current NGC images already miss some floors.** `pytorch:25.04-py3` ships NCCL 2.26.3, below GB300's 2.27.5 baseline → `ACTION_REQUIRED` with `GPF-NCCL-003`. Expect this on stock images; it is a real finding worth acting on.
- **Discovery is by bytes, not by name, so coverage counts custom code too.** A `.cubin`, `.fatbin` or `.ptx` is inspected wherever it sits, and everything else is settled by its first twenty bytes: anything carrying the ELF magic number is a candidate whatever it is called and wherever it lives. A vLLM-style `_C.abi3.so`, a team's own `libfused.so` under `/opt/app`, an extensionless executable like `/usr/bin/tensorflow_model_server` — all in scope, all in the denominator. Nothing needs renaming to get inspected, and the percentage can be read at face value.
- `--profile-dir DIR` loads extra hardware profiles from TOML (useful for an unreleased or site-specific target); it is a **global** flag, before the subcommand.

## Risks and approval boundaries

- The scan **pulls the image** onto the host — tens of GiB each. Confirm the host is an appropriate place for that image's contents and has disk headroom. When the image belongs to someone else, whether it may be pulled there is their decision, not yours.
- It uses the host's existing Docker credentials. Never add someone else's registry credential to a shared host without their explicit approval, and never put a registry token into a ticket, a summary, or a commit.
- Reports contain full filesystem paths and package inventories from the image. Treat them as sensitive: review before pasting one into a ticket or a shared channel, and do not forward a report about someone else's image further than they expect.
- The tool creates and removes a temporary container and anonymous volumes. It never starts the image and never passes image programs to the host loader — don't work around that.

## How to verify

- `docker info` succeeds as a non-root user and `cuobjdump --version` prints a version.
- The run prints absolute report paths, and `report.txt`, `report.json`, `report.html` exist.
- **Coverage is 100% and no `GPF-CUDA-019`.** Otherwise the verdict is not reportable.
- The JSON validates against `schemas/report.schema.json` and carries the tool version, profile schema hash, and selected image digest.

## Reference

- `docs/cuda-inspection.md` — what counts as a CUDA artifact, report terminology, and what can refuse a file.
- `docs/fabric-checks.md` — adapter floors and `GPF-UCX-*` findings (ConnectX-7/8/9).
- `README.md` — full flag reference.

Validated on Ubuntu 24.04 / Python 3.12 / Docker 29.9.2 with the latest cuobjdump plus the 13.0 fallback, 2026-11-03: 18+ scans across H100, B300, GB300, RTX PRO 6000 and non-CUDA images. Re-checked 2026-11-08 on a fresh VM against five large inference images (vLLM, SGLang, TensorRT-LLM) for h100+b300, all at 100% coverage with the then-current primary analyzer and no `GPF-CUDA-019`. The analyzer table above was re-measured on that date against the current scanner, which sets no scan budgets by default.
