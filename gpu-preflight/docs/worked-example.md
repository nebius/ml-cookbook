# A worked example: NGC PyTorch on GB300

One real scan, read end to end: what it settles, what it leaves open, and what to do
with the result.

Back to the [GPU Preflight README](../README.md).

## The question

A team wants to run `nvcr.io/nvidia/pytorch:26.10-py3` on GB300 and asks whether
the image is ready. GB300 is Arm, so the image needs an arm64 variant; its kernels
need to cover `sm_103`; and its NCCL, UCX and HPC-X need to clear the platform's
floors.

```bash
PYTHONPATH=src python3 -m gpu_preflight inspect \
  nvcr.io/nvidia/pytorch:26.10-py3 \
  --target gb300
```

The scan took 321.7 seconds on a VM with no GPU.

## What the report establishes

```text
NVIDIA GB300 NVL72 (gb300)
  Compatibility: REVIEW_REQUIRED
  Discovered artifacts inspected: 100.0% (8499/8499 inspection units successfully analyzed)
  Reproducibility: UNPINNED
```

| Question | Answer |
| --- | --- |
| Will it start on Arm? | Yes. The registry carries amd64 and arm64; `linux/arm64` resolved and every native library matches. |
| Is there `sm_103` code? | Yes. 32 unique targets, including base `sm_103` and architecture-specific `sm_103a`. `sm_103f` is absent. |
| Do the version floors pass? | Yes. CUDA 13.4, NCCL 2.32.3, UCX and HPC-X all clear the GB300 floors, as do Triton 3.8.0, CuTe DSL 4.8.0 and NVSHMEM 3.8.0. |
| How much was read? | 8955 candidates found, 456 role-excluded, 8499 inspected, none skipped or incomplete. 158 of them hold device code. |
| Is the reference reproducible? | No. The tag is not digest-pinned, so the registry owner can change what it points to. |

One artifact is flagged for review:

```text
- PyTorch extensions: Optional alternate-GPU implementation
  Reason: Architecture-specific implementation has no identified compatible sibling or PTX/JIT fallback
  Representative examples:
    - /opt/pytorch/ao/torchao/_C_cutlass_90a.abi3.so
```

That is torchao's Hopper-only CUTLASS extension. It holds `sm_90a` and nothing a GB300
can run.

## What the report does not establish

**Whether that extension matters.** torchao ships per-architecture extensions and loads
the one matching the GPU it finds. On GB300 it should never open the `90a` build — but
"should" is a claim about torchao's dispatch code, not about the image. A CUDA extension
is usually imported by path rather than resolved by SONAME, so neither the loader search
path nor its absence proves a library is dead. This is why the label is
`REVIEW_REQUIRED` and not `INCOMPATIBLE`.

**Which NCCL the process loads.** The scan found 2.32.3 in the image configuration, the
system packages and the headers. A process can still load something else from
`LD_LIBRARY_PATH` or a venv.

**Whether run-time-compiled kernels will target `sm_103`.** Triton and CuTe DSL compile
at startup and leave no cubin to inspect, so they are judged by installed version
against a published floor. A floor says the release can target GB300. It does not say
the workload's kernels compile, or that they are fast.

**Two packages are reported but not judged.** `flash-attn-4` 4.0.0 is a beta whose
suffix the version parser drops, and `numba` 0.64.0 publishes no per-architecture floor.
Both are listed as present, with no verdict.

**Performance.** `sm_103a` being present means an architecture-specific path was
compiled, not that the operations a given workload runs take it.

## What to do next

The next step is a smoke run on one GB300 node, capturing what the image actually does:

- `NCCL_DEBUG=VERSION` to confirm the loaded NCCL, and `/proc/<pid>/maps` or `ldd` on
  the entrypoint to confirm whether `_C_cutlass_90a.abi3.so` is ever opened
- a framework smoke test: import, enumerate GPUs, run a tensor operation
- a reduced run of the real workload, for correctness and throughput

Then pin the image by digest, so the evidence names the layers it was gathered from.

## Provenance

`nvcr.io/nvidia/pytorch:26.10-py3`, analyzer `cuobjdump` 13.5 with the 13.0 fallback,
report schema v1. Counts move between image releases and between tool versions; the
shape of the answer does not.
