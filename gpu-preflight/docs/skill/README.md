# GPU Preflight

This directory is the canonical copy of the GPU Preflight agent skill. Copy it into an agent's skill directory to install it:

```bash
cp -r gpu-preflight/docs/skill ~/.claude/skills/gpu-preflight
```

`SKILL.md` is what the agent reads; `evals/evals.json` holds the trigger and behavior checks.

## What it is

Agent skill for the Nebius GPU Preflight scanner, which statically checks whether a container image is ready to be tested on Nebius NVIDIA GPU hardware. It is written for both Nebius engineers checking an image on someone's behalf and teams checking their own image before they book GPU time.

It teaches the agent to run `gpu-preflight` from the Nebius [ml-cookbook](https://github.com/nebius/ml-cookbook) repo against an image and a hardware target, and to read the result correctly: the compatibility label, the CUDA inspection coverage number, and the reproducibility verdict.

The scanner pulls the image, exports its filesystem, and inspects CUDA artifacts with host `cuobjdump`. It never starts the image, and it does not need a GPU, a driver, a CUDA runtime, or access to the target hardware. It checks CPU architecture, CUDA `sm_` targets and retained PTX, CUDA/NCCL version baselines, UCX and HPC-X against the target's compute-fabric adapter, package metadata, and whether the image reference is digest-pinned.

## When to use it

- Someone needs to know whether an image will run on H100, H200, B200, B300, GB300, RTX PRO 6000, or Vera Rubin before a PoC or a hardware booking — your own image, or one you are checking for someone else.
- Before GPU time is booked, to catch an amd64-only image aimed at GB300, or a CUDA/NCCL build that predates the target's support floor.
- Comparing one image across two hardware targets ("will the same image work on B300 and GB300?").
- Triaging a failure on the cluster that looks like a missing `sm_` target rather than a workload bug.

Do not use it to claim an image is certified. It is a preflight assessment; runtime tests on the hardware are still required.

## Requirements

- Required tools: Linux host, Python 3.11+, Docker Engine (the repo does **not** install it — set it up yourself and confirm non-root `docker info`), NVIDIA `cuobjdump` via the repo's `install_cuobjdump.sh`, which installs both the primary analyzer and the 13.0 fallback.
- Required access: Docker registry read access to the image, including its manifest. `docker login` for Docker Hub — the anonymous limit is exhausted by roughly ten scans, even of an already-cached image.
- Required environment variables: none. Run from the cloned repo with `PYTHONPATH=src`; no virtualenv or install step.
- Required internal systems: none. The tool is self-contained.
- Host budget: 8–11 GiB and 2–8 minutes for an NGC framework image; large inference images (vLLM, SGLang, TensorRT-LLM) take 5–8 minutes and 33–52 GiB of disk apiece. Allow ~60 GiB free per concurrent scan.
- Source: `gpu-preflight/` in `nebius/ml-cookbook`.

## How to use

Clone the repo, install `cuobjdump`, then:

```bash
PYTHONPATH=src python3 -m gpu_preflight profiles

PYTHONPATH=src python3 -m gpu_preflight inspect \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  --target gb300 \
  --platform linux/arm64
```

Output is a timestamped directory under `./gpu-preflight-reports/` holding `report.txt`, `report.json`, and `report.html`.

**Check coverage before the label.** A scan below 100% means the analyzer could not read some artifact, and what it could not read may be a blocker. When a core framework library is among them, the label comes back `INSUFFICIENT_EVIDENCE` — fix the analyzer and rerun rather than reporting the findings that happened to be readable.

**Do not read the exit code as the verdict.** Exit `2` covers every compatibility and evidence finding at or above the threshold, so `REVIEW_REQUIRED` and `INSUFFICIENT_EVIDENCE` both produce it and mean different things — and a usage error produces it too. Reproducibility findings do not set the exit code, so an unpinned tag does not fail a build on its own; the finding is still reported, and pinning the digest is still what reproducible validation evidence requires.

## Example prompts

- "Will `nvcr.io/nvidia/pytorch:25.04-py3` run on GB300?"
- "Our training image fails on B300 with no kernel image available — can you check it before we book more GPU time?"
- "Compare this image against B300 and GB300 and tell me what breaks."
- "Why does the preflight report say INSUFFICIENT_EVIDENCE at 96% coverage?"
- "Preflight came back INSUFFICIENT_EVIDENCE with GPF-CUDA-019 naming libnccl, but it also lists an NCCL version warning — is bumping NCCL enough?"

## Safety and approval boundaries

- The scan pulls the image onto the host — confirm it is an appropriate host and has disk space before scanning large images. When the image is not yours, that is the owner's call.
- It uses the host's existing Docker credentials. Do not install someone else's credentials on a shared host without their approval, and keep registry tokens out of tickets and summaries.
- Reports contain full image filesystem paths and package inventories. Treat them as sensitive; review before pasting into a ticket or a shared channel, and do not forward a report about someone else's image further than they expect.
- The scan is read-only with respect to the image owner's environment. It creates and removes one temporary container and never executes image programs.

## Known limitations

- Static analysis only. It cannot establish which NCCL installation, native extension, dispatch catalog, or kernel variant the workload actually loads, and it proves nothing about correctness or performance.
- **Kernels compiled at run time are not inspected.** Triton, CuTe DSL and similar leave no cubin to read, so they are judged by installed version against a per-hardware floor (`GPF-PKG-*`) rather than by the artifact scan. Coverage says nothing about them.
- An image that was built locally and never pushed cannot be scanned.
- **Verdicts depend on the analyzer version.** Keep both analyzers the installer sets up: a current 13.x analyzer alone leaves CUDA 12.x libraries unreadable and drops coverage below 100%. Never report a verdict from a scan below 100% coverage. CUDA 11.x images are out of scope.
- `REVIEW_REQUIRED` is the common result for real images, and its scale varies by an order of magnitude: one alternate-architecture extension on a stock NGC image, several hundred on a large inference image. Either way it needs triage in the JSON, not a bare label in a summary.
- Not yet exercised on a live escalation. Authenticated Docker Hub pulls have since been used across a large scan corpus; private-registry authentication remains untested.

## Validation

Validated end to end on a clean Ubuntu 24.04 VM with 18+ scans across H100, B300, GB300, RTX PRO 6000 and non-CUDA images. Since exercised against a 399-image corpus for H100 and B300, and re-run through the documented setup on a fresh VM against five large inference images, all at 100% coverage.
