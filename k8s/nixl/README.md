# Cross-rack NIXL-EP on GB300

Run SGLang's NIXL expert-parallel all-to-all (`MOE_A2A_BACKEND=nixl`) across two racks on GB300 over InfiniBand with `DMA-BUF`.

Cross-rack NIXL-EP requires two things:

1. **An image bundling NIXL 1.4.0 or later** (SGLang 0.5.20+). The GB300 managed node image has `nvidia_peermem` disabled on purpose so GPU memory is mapped for the NIC through `DMA-BUF`. NIXL < 1.4.0 cannot register device memory through DMA-BUF (its DEVX UMEM path does not support it) so expert-parallel setup dies with a misleading "load Nvidia peermem" message. See [`past-runs/`](./past-runs/) for a passing v0.5.20 run and a failing v0.5.16 run.
2. **`UCX_TLS=^cuda_ipc`.** Cross-rack is cross-IMEX-domain so CUDA IPC cannot reach the peer. The leading `^` means "everything except"; without it the job asserts in `prepMemView` or silently generates garbage.

To run this example, you need:

1. A **GB300 two-rack** slice: 4 nodes, 4 GPUs + 4 fabric NICs each, on the rack-local NVLink tier (`topology.nebius.com/tier-0`).
2. The **NVIDIA DRA driver** (ComputeDomain / IMEX) and the Nebius IB DRA device class (`ib.networking.nebius.ai`) installed on the cluster.
3. `kubectl` with access to the cluster, and a namespace to deploy into.

Note: This recipe does not install the cluster-side DRA driver; it assumes a Nebius GB300 cluster where it is already present. This setup uses DRA, but DRA is not a requirement.

## Layout

```
k8s/nixl/
├── nixl-ep-gb300/
│   ├── manifest.yaml        # headless Service + two Indexed Jobs (one per clique)
│   ├── dra/
│   │   ├── compute-domains.yaml   # 2 ComputeDomains -> the IMEX channel templates
│   │   └── claim-template.yaml    # 4 GPUs + 4 IB NICs per node
│   └── scripts/             # mounted at /opt/validation via a ConfigMap
└── past-runs/               # real pass/fail rank-0 logs from the writeup
```

## Run it

Apply everything into one namespace of your choice. The commands below use an `NS` variable:

```bash
export NS=nixl-ep
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
cd k8s/nixl/nixl-ep-gb300
```

**1. Apply the DRA resources.** The NVIDIA DRA driver reconciles each ComputeDomain and then creates the channel `ResourceClaimTemplate` it names so this must happen before the workload:

```bash
kubectl apply -n "$NS" -f dra/
```

**2. Wait for the channel templates** the driver generates:

```bash
kubectl -n "$NS" wait --for=create \
  resourceclaimtemplate/dra-two-clique-a-channel \
  resourceclaimtemplate/dra-two-clique-b-channel --timeout=120s
```

**3. Create the scripts ConfigMap** the pods mount at `/opt/validation`:

```bash
kubectl -n "$NS" create configmap gpu-validation-scripts \
  --from-file=scripts/
```

**4. Launch the workload** (headless Service + both Jobs):

```bash
kubectl apply -n "$NS" -f manifest.yaml
```

**5. Watch it.** Rank 0 is `sglang-nixl-ep-0`; the `-b` Job carries ranks 2–3:

```bash
kubectl -n "$NS" get pods -l validation.nebius.ai/test-id=sglang-nixl-ep -w
kubectl -n "$NS" logs -f job/sglang-nixl-ep
```

A pass ends with `distributed SGLang generation passed` and the NIXL-EP assertion:

```text
Using NIXL EP
NIXL-EP GPU-initiated path: PASS
distributed SGLang generation passed
```

### Clean up

```bash
kubectl -n "$NS" delete -f manifest.yaml
kubectl -n "$NS" delete configmap gpu-validation-scripts
kubectl -n "$NS" delete -f dra/
```

## Verifying the transport (and DMA-BUF)

The expert-parallel dispatch/combine rides UCX, not NCCL. The UCX trace is where you confirm it is on the `rc_gda` / `rc_mlx5` IB lanes. To emit that trace, add `UCX_LOG_LEVEL=debug` to the container env. The smoke script redirects the SGLang server and its UCX output to `/tmp/sglang-<rank>.log` inside each pod and only echoes a summary to stdout so `kubectl logs` will not show it. `kubectl exec` into a pod while the server is up and read that file. See [`past-runs/job-sglang0520-nixl-ep-success.log`](./past-runs/job-sglang0520-nixl-ep-success.log) for a full captured example.

`UCX_IB_GDA_DMABUF_ENABLE` defaults to `try`: DMA-BUF first, silent fallback to `nvidia_peermem`. Because the fallback is silent, a pass in the default mode does not prove DMA-BUF was used. To force it, add this to the container env and re-run:

```yaml
- name: UCX_IB_GDA_DMABUF_ENABLE
  value: "yes"      # DMA-BUF only, no peermem fallback
```

Then the log shows DMA-BUF being selected for the GPU-initiated transport:

```text
mlx5_0: dmabuf is supported
mlx5_0: using dmabuf for gda transport
ep: lane[7]: 19:rc_gda/cuda0-mlx5_2:1.0 ... device
```

## Past runs

Logs from two runs this recipe is based on.

| Log | Outcome | Image |
| --- | --- | --- |
| [`past-runs/job-sglang0520-nixl-ep-success.log`](./past-runs/job-sglang0520-nixl-ep-success.log) | Pass — NIXL-EP up, generates "Paris" | `lmsysorg/sglang:v0.5.20` (NIXL 1.4.1 / UCX 1.22.0) |
| [`past-runs/job-sglang0516-nixl-ep-fail.log`](./past-runs/job-sglang0516-nixl-ep-fail.log) | Fail — DEVX UMEM has no DMA-BUF; `prepMemView` assertion | `lmsysorg/sglang:v0.5.16-cu130` (NIXL 1.3.2 / UCX 1.21.0) |

The passing run forces `UCX_IB_GDA_DMABUF_ENABLE=yes` (so DMA-BUF cannot silently fall back) with `UCX_LOG_LEVEL=debug`, so the transport trace is captured in full.

**Pass** — DMA-BUF selected, NIXL-EP up, generation verified:

```text
gdaki.c:289  UCX  DEBUG mlx5_0: using dmabuf for gda transport
[TP0 EP0] Using NIXL EP (world_size=16, max_ep_size=16, ... num_experts=64, num_experts_per_rank=4)
distributed SGLang generation passed
```

**Fail** — DMA-BUF unavailable through DEVX UMEM, ending in the setup assertion:

```text
gdaki.c:191  UCX  DEBUG DEVX UMEM doesn't support DMA-BUF
ucx_backend.cpp:1467 Failed to prepare remote memory view: Failed to create device memory list(remote): Input/output error
RuntimeError: Assertion error nixl_ep.cpp:1330 '... prepMemView(...) == NIXL_SUCCESS'
```
