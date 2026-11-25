# NVIDIA Run:ai on Nebius MK8S — Driverful GPU Nodes

This recipe installs NVIDIA Run:ai 3.0.20 on an **existing** Nebius Managed
Kubernetes cluster whose GPU nodes use the **driverful** node image. It
supplements the [driverless installation path](https://docs.nebius.com/3p-integrations/run-ai)
in the Nebius documentation; that path remains unchanged.

## 1. Purpose and scope

Nebius MK8S offers two GPU node-image variants:

| Image type | Driver source | Container-runtime integration |
|---|---|---|
| **Driverless** | GPU Operator / NVIDIA driver container | Toolkit sidecar |
| **Driverful** | Baked into the node OS | Provided by the node image |

On driverful nodes the NVIDIA driver and container-runtime hook are already
present on the host. Nebius also supplies the NVIDIA device plugin so that
Kubernetes can schedule `nvidia.com/gpu` resources. This recipe adds **only**
the components that are still missing for Run:ai:

- Node Feature Discovery (NFD) and GPU Feature Discovery (GFD) labels
- A standalone DCGM Exporter for GPU monitoring
- kube-prometheus-stack for metrics collection
- HAProxy Kubernetes Ingress Controller
- Run:ai control plane and cluster agent

It deliberately avoids reinstalling the NVIDIA driver, container toolkit,
runtime-class configuration, or device plugin, all of which are already
managed by Nebius.

## 2. Tested configuration

### Full validation

| Component | Version / detail |
|---|---|
| Nebius MK8S | 1.36.3 |
| GPU nodes | 2 × Ubuntu 24.04 driverful, 1 B200 each |
| NVIDIA driver | 580.173.02 |
| CUDA UMD (nvidia-smi) | 13.0 |
| GPU memory | 183,359 MiB per card |
| NVIDIA Run:ai | 3.0.20 |
| nvidia-device-plugin chart (NFD/GFD only) | 0.20.2 |
| DCGM Exporter chart | 4.9.0 |
| kube-prometheus-stack chart | 88.6.5 |
| HAProxy Kubernetes Ingress chart | 1.57.1 |
| Shared storage | Nebius SFS via csi-mounted-fs-path |
| GPU allocation mode | Whole GPU |

The CUDA version reported by `nvidia-smi` reflects the driver's User-Mode
Driver (UMD) compatibility level, not the CUDA Toolkit version installed in
workload containers. Each container image brings its own CUDA runtime.

### Later regression check (driver upgrade only)

A subsequent node-image update with driver **610.57.04** (CUDA UMD 13.3) was
checked for:

- B200 visibility and `nvidia.com/gpu` allocation in Kubernetes
- NFD/GFD pod recovery after node recreation
- DCGM Exporter pod recovery
- Both GPU nodes continuing to advertise one GPU each

The complete Run:ai scheduling, quota, and preemption test suite was **not**
repeated on the R610 driver. Treat the R610 result as a compatibility signal,
not a full re-validation.

## 3. Prerequisites

Before starting, verify that you have:

- **Nebius MK8S cluster** already provisioned and reachable
- **Driverful GPU node group** (at least one node) with `nebius.com/gpu: "true"` label
- **CPU node group** for control-plane and ingress workloads
- `kubectl` configured with cluster-admin access
- `helm` ≥ 3.14
- [Nebius CLI](https://docs.nebius.com/cli/install) installed and authenticated
- `jq` (used by the validation script)
- **Linux filesystem with POSIX permissions** for the Run:ai installer directory
  (permissions such as `0700` are required; Windows-mounted WSL paths such as
  `/mnt/c/…` may not satisfy this)
- A **production DNS name** that you control, resolvable to your load-balancer IP
- A **trusted TLS certificate** for that domain (or accept the lab trade-offs in
  Section 11)
- Sufficient GPU quota in your Nebius project
- Security-group rules allowing HTTPS (443) and HTTP (80) inbound to the
  cluster's load-balancer

## 4. Inventory the existing GPU stack

Establish what is already running before adding anything. Unexpected
duplicates (two device plugins, two Prometheus instances, two DCGM exporters)
cause conflicts that are difficult to diagnose.

```bash
# All nodes and their labels
kubectl get nodes --show-labels

# Confirm driverful nodes
kubectl get nodes -l nebius.com/gpu=true -o wide

# GPU model and driver labels (populated after NFD/GFD — may be absent now)
kubectl get nodes -l nebius.com/gpu=true \
  -o custom-columns='NAME:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product,DRIVER:.metadata.labels.nvidia\.com/cuda\.driver\.major'

# Allocatable GPU resources
kubectl get nodes -l nebius.com/gpu=true \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'

# NVIDIA device plugin (Nebius-managed)
kubectl get pods -A -l app=nvidia-device-plugin-daemonset

# Existing GPU Operator (expect: none on driverful clusters)
kubectl get pods -A -l app=gpu-operator

# Nebius-managed DCGM DaemonSet
kubectl get daemonset -A | grep dcgm

# Existing NFD / GFD
kubectl get pods -A | grep -E 'node-feature|gpu-feature'

# Existing Prometheus stacks
kubectl get pods -A | grep prometheus

# Storage classes
kubectl get storageclass

# All Helm releases
helm list -A
```

Document every component you find. Decide whether you own it or Nebius does,
and do not duplicate Nebius-managed components.

## 5. Validate the driverful GPU

Before installing any Run:ai components, confirm that bare Kubernetes GPU
allocation works. This test bypasses Run:ai entirely and verifies:

1. The scheduler can assign `nvidia.com/gpu: 1` to a pod
2. The container runtime injects device access
3. The host driver is usable inside a container

```bash
kubectl apply -f manifests/gpu-smoke-test.yaml
kubectl wait pod/gpu-smoke-test --for=condition=Ready --timeout=60s
kubectl logs gpu-smoke-test
```

Expected output includes the GPU model, driver version, and memory:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI ...          Driver Version: 580.173.02   CUDA Version: 13.0                |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================================================================|
|   0  NVIDIA B200                    Off | ...                Off | ...                  |
|  ...                                   |    0MiB / 183359MiB    |      0%      Default  |
+-----------------------------------------+------------------------+----------------------+
```

Wait for `Succeeded`, then clean up:

```bash
kubectl wait pod/gpu-smoke-test --for=condition=Succeeded --timeout=120s
kubectl delete pod gpu-smoke-test
```

> **Stop here if the pod does not reach `Succeeded`.** The issue is in the
> base Kubernetes GPU stack, not in Run:ai. Fix it before proceeding.

## 6. Persistent and shared storage

Run:ai supports two storage patterns:

| StorageClass | Access mode | Use case |
|---|---|---|
| `compute-csi-default-sc` | ReadWriteOnce | Per-pod scratch / model cache |
| `csi-mounted-fs-path-sc` | ReadWriteMany | Shared datasets across pods and nodes |

Multi-node and multi-pod workloads need RWX storage. The validated approach
uses a Nebius SFS (serverless filesystem) exposed as a virtiofs mount on each
node group, with the `csi-mounted-fs-path` CSI driver bridging it into
Kubernetes.

### 6.1 Create and attach a Nebius SFS

> **Warning:** Modifying a node-group template (to add a filesystem mount)
> triggers a **rolling recreation** of all nodes in that group. Plan this
> step during a maintenance window and ensure your workloads tolerate node
> replacement.

Follow the [Nebius documentation](https://docs.nebius.com/storage/serverless-filesystem/create)
to:

1. Create a Nebius SFS in the same availability zone as your cluster.
   Note the filesystem ID (`${RUNAI_FS_ID}`).

2. Attach the filesystem in **read-write** mode to both the CPU node group
   (`${CPU_NODE_GROUP_ID}`) and the GPU node group (`${GPU_NODE_GROUP_ID}`).
   Set the mount point to `/mnt/runai-shared` and the filesystem type to
   `virtiofs` in the node-group template.

3. The mount is automatically added to `/etc/fstab` on each node when the
   template is applied.

Consult the
[Nebius CLI reference](https://docs.nebius.com/cli/reference/mk8s/node-group/update)
for the exact `nebius mk8s node-group update` command syntax, as the flags
may change between CLI releases.

### 6.2 Install or verify csi-mounted-fs-path

Check whether the CSI driver is already installed:

```bash
kubectl get storageclass csi-mounted-fs-path-sc 2>/dev/null \
  && echo "Already present" || echo "Needs installation"
```

If it is absent, install from the Nebius-provided chart:

```bash
helm repo add nebius-csi https://nebius.github.io/csi-mounted-fs-path
helm repo update
helm install csi-mounted-fs-path nebius-csi/csi-mounted-fs-path \
  --namespace kube-system
```

> Verify the chart name and repo URL against the
> [Nebius GitHub repository](https://github.com/nebius/csi-mounted-fs-path)
> before running this command.

### 6.3 Create the RWX PVC

```bash
kubectl apply -f manifests/shared-storage-pvc.example.yaml
```

> **Important:** The `1Gi` capacity in the PVC spec is passed to the driver
> for API conformance only. The PVC will expose the full capacity of the
> underlying SFS, not a 1 GiB slice. Size your SFS according to actual
> workload needs.

The PVC will remain in `Pending` with `WaitForFirstConsumer`; this is
expected until a pod mounts it.

### 6.4 Test cross-node read-write

```bash
kubectl apply -f manifests/shared-storage-test.yaml -n runai
kubectl wait pod/shared-storage-writer -n runai \
  --for=condition=Succeeded --timeout=60s
kubectl logs shared-storage-reader -n runai
# Expected: "read-test OK"
kubectl delete -f manifests/shared-storage-test.yaml -n runai
```

## 7. GPU discovery (NFD and GFD)

Run:ai uses NVIDIA feature labels — applied by Node Feature Discovery (NFD)
and GPU Feature Discovery (GFD) — to match GPU nodes and report GPU capacity.
On a fresh driverful cluster these labels are absent, which causes Run:ai to
display **0 GPUs** even though the nodes and device plugin are working.

Add NFD and GFD using the NVIDIA device-plugin chart, with the device plugin
itself **disabled** (Nebius already provides it):

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvdp nvdp/nvidia-device-plugin \
  --version 0.20.2 \
  --namespace gpu-operator \
  --create-namespace \
  -f values/gpu-feature-discovery-values.yaml
```

Key settings in `values/gpu-feature-discovery-values.yaml`:

```yaml
devicePlugin:
  enabled: false   # Nebius already manages the device plugin
gfd:
  enabled: true
nfd:
  enabled: true
runtimeClassName: ""   # No toolkit RuntimeClass on driverful nodes
migStrategy: none
nodeSelector:
  nebius.com/gpu: "true"
```

Verify after a few minutes:

```bash
# NFD worker pods (should run on ALL nodes)
kubectl get pods -n gpu-operator -l app=node-feature-discovery

# GFD pods (should run only on GPU nodes)
kubectl get pods -n gpu-operator -l app=gpu-feature-discovery

# NVIDIA labels on GPU nodes
kubectl get nodes -l nebius.com/gpu=true \
  -o custom-columns='NAME:.metadata.name,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product,DRIVER:.metadata.labels.nvidia\.com/cuda\.driver\.major,CUDA:.metadata.labels.nvidia\.com/cuda\.runtime\.major'

# Confirm nvidia.com/gpu is still allocatable
kubectl get nodes -l nebius.com/gpu=true \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

Expected: each GPU node shows `nvidia.com/gpu=1` in allocatable resources and
the NVIDIA product/driver labels are populated.

## 8. GPU monitoring (standalone DCGM Exporter)

The Nebius-managed `nebius-dcgm` DaemonSet uses node selectors and runtime
assumptions (toolkit-ready path, NVIDIA RuntimeClass) that are not present on
driverful nodes, so it schedules zero pods on them. Do not modify or delete
this managed release.

Install a separate, standalone DCGM Exporter in the `gpu-monitoring`
namespace:

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/gpu-helm-charts
helm repo update

kubectl create namespace gpu-monitoring --dry-run=client -o yaml | kubectl apply -f -

helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --version 4.9.0 \
  --namespace gpu-monitoring \
  -f values/dcgm-exporter-values.yaml
```

Key settings in `values/dcgm-exporter-values.yaml`:

```yaml
runtimeClassName: ""          # No toolkit RuntimeClass on driverful nodes
nodeSelector:
  nebius.com/gpu: "true"
extraEnv:
  - name: NVIDIA_VISIBLE_DEVICES
    value: all
  - name: NVIDIA_DRIVER_CAPABILITIES
    value: compute,utility
serviceMonitor:
  enabled: true
  interval: 30s
  scrapeTimeout: 25s
  additionalLabels:
    release: prometheus       # Must match both Prometheus instances
```

> **Note:** We evaluated the `1.x-compatibility-metrics.csv` profile but it
> did not provide the exact metric names that Run:ai 3.0.20 expects (see
> Section 10). The compatibility PrometheusRule in Section 10 is the correct
> solution.

Verify:

```bash
# One pod per GPU node
kubectl get pods -n gpu-monitoring -l app.kubernetes.io/name=dcgm-exporter -o wide

# Metrics service
kubectl get svc -n gpu-monitoring

# Sample metrics via port-forward
kubectl port-forward -n gpu-monitoring svc/dcgm-exporter 9400:9400 &
sleep 3
curl -s http://localhost:9400/metrics | grep -E \
  'DCGM_FI_DEV_GPU_TEMP|DCGM_FI_DEV_GPU_UTIL|DCGM_FI_DEV_FB_USED|DCGM_FI_DEV_POWER'
kill %1
```

Confirm that GPU temperature, utilization, and framebuffer metrics appear, and
that workload-attribution labels (`pod`, `namespace`, `container`) are present
on metrics for running jobs.

## 9. Prometheus

Install kube-prometheus-stack with Grafana disabled (optional) to keep the
footprint small. Run:ai 3.0.20 also deploys its own internal Prometheus
instance; both must be able to discover the DCGM Exporter ServiceMonitor.

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

helm install prometheus prometheus-community/kube-prometheus-stack \
  --version 88.6.5 \
  --namespace monitoring \
  --set grafana.enabled=false \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false
```

The two flags ending in `NilUsesHelmValues=false` tell Prometheus to discover
ServiceMonitors and PrometheusRules across all namespaces, which is necessary
for it to pick up the DCGM Exporter in `gpu-monitoring` and the compatibility
rule in `runai`.

Verify both Prometheus instances:

```bash
# kube-prometheus-stack instance
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus

# Run:ai internal Prometheus (created after Run:ai installation)
kubectl get pods -n runai -l app.kubernetes.io/name=prometheus

# Check DCGM metrics are visible in the kube-prometheus-stack instance
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090 &
sleep 3
curl -sg 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL_RATIO' \
  | jq '.data.result | length'
# Expected: > 0
kill %1
```

## 10. DCGM metric compatibility

**Problem:** DCGM Exporter 4.9.0 emits these metric names:

- `DCGM_FI_DEV_GPU_UTIL_RATIO` — fraction 0.0–1.0
- `DCGM_FI_DEV_FB_USED_BYTES` — bytes

**Run:ai 3.0.20 queries:**

- `DCGM_FI_DEV_GPU_UTIL` — integer 0–100
- `DCGM_FI_DEV_FB_USED` — MiB

The mismatch caused the Run:ai UI GPU-compute and GPU-memory graphs to remain
blank. The fix is a PrometheusRule that derives the expected names, preserving
all existing labels:

```bash
kubectl apply -f manifests/dcgm-compatibility-rule.yaml
```

The rule lives in the `runai` namespace so the Run:ai internal Prometheus
discovers it automatically.

Verify all four metrics are queryable:

```bash
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090 &
sleep 3

for metric in \
  DCGM_FI_DEV_GPU_UTIL_RATIO \
  DCGM_FI_DEV_FB_USED_BYTES \
  DCGM_FI_DEV_GPU_UTIL \
  DCGM_FI_DEV_FB_USED; do
  count=$(curl -sg "http://localhost:9090/api/v1/query?query=${metric}" \
    | jq '.data.result | length')
  echo "${metric}: ${count} series"
done
kill %1
```

Expected: each metric shows ≥ 1 series. `DCGM_FI_DEV_GPU_UTIL` and
`DCGM_FI_DEV_FB_USED` are computed from the source metrics by the recording
rules; they appear in Prometheus after the first evaluation interval (30 s).

## 11. Ingress, DNS, and TLS

Ingress routes external browser and API traffic into the Run:ai control plane.
DNS maps your domain to the load-balancer IP. TLS protects all traffic.

> **Production guidance:** Use a real domain name you control and a trusted
> certificate (e.g. from Let's Encrypt or your organisation's CA). The nip.io
> wildcard service and self-signed certificates are laboratory shortcuts only
> and must not be used in production.

### 11.1 Install HAProxy Ingress Controller

```bash
helm repo add haproxytech https://haproxytech.github.io/helm-charts
helm repo update

kubectl create namespace haproxy --dry-run=client -o yaml | kubectl apply -f -

helm install haproxy haproxytech/kubernetes-ingress \
  --version 1.57.1 \
  --namespace haproxy \
  -f values/haproxy-values.yaml
```

Key settings in `values/haproxy-values.yaml`:

```yaml
controller:
  nodeSelector:
    nebius.com/node-group-id: "${CPU_NODE_GROUP_ID}"
  service:
    type: LoadBalancer
```

HAProxy runs on CPU nodes to preserve GPU capacity for workloads.

### 11.2 Retrieve the load-balancer IP

```bash
kubectl get svc -n haproxy -l app.kubernetes.io/name=kubernetes-ingress \
  -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}'
```

Note the IP as `${INGRESS_IP}`.

### 11.3 Configure DNS

Create an A record (or wildcard A record) in your DNS zone:

```
${RUNAI_DOMAIN}       A  ${INGRESS_IP}
*.${RUNAI_DOMAIN}     A  ${INGRESS_IP}
```

Verify with:

```bash
dig +short ${RUNAI_DOMAIN}
# Expected: ${INGRESS_IP}
```

### 11.4 TLS certificate

**Production** — create a Kubernetes Secret from your certificate before
running the installer:

```bash
kubectl create namespace runai --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret tls runai-tls \
  --cert=fullchain.pem \
  --key=privkey.pem \
  -n runai
```

Then reference it in `values/runai-config.example.yaml` under the `tls.secretName` key.

**Lab only** — set `tls.generateSelfSigned: true` in the installer
config. Browsers will show a certificate warning and workloads that validate
TLS against the system CA will fail.

## 12. Run:ai installer

### 12.1 Download and verify

Download the Run:ai 3.0.20 installer from
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/runai/resources/runai-installer)
and verify the checksum as documented on the NGC download page.

> **Filesystem requirement:** The installer must be run from a directory on a
> Linux filesystem that supports `0700` permissions. If you are using WSL with
> the working directory on a Windows drive (`/mnt/c/…`), copy the installer to
> a native Linux path first:
>
> ```bash
> cp runai-installer ~/runai-installer
> chmod 0700 ~/runai-installer
> ```

### 12.2 Prepare the configuration

```bash
cp values/runai-config.example.yaml runai-config.yaml
```

Edit `runai-config.yaml` and replace every `${...}` placeholder:

| Placeholder | Value |
|---|---|
| `${RUNAI_DOMAIN}` | Your DNS name, e.g. `runai.example.com` |
| `${RUNAI_CLUSTER_NAME}` | A short name for this cluster in the Run:ai UI |

Choose **one** TLS block (production or lab) and delete the other.

### 12.3 Preflight check

```bash
./runai-installer preflight --config runai-config.yaml 2>&1 | tee preflight.log
```

Review the output. Address all `FAIL` entries before continuing. `WARN`
entries about GPU Operator or missing RuntimeClass are expected on driverful
nodes and do not block installation if `gpuOperator.mode: skip` is set.

### 12.4 Dry run

```bash
./runai-installer install --config runai-config.yaml --dry-run 2>&1 | tee dryrun.log
```

Review the dry-run output for any unexpected namespace creation, Helm
releases, or CRD changes.

### 12.5 Install

Only proceed after a clean preflight and dry run:

```bash
./runai-installer install --config runai-config.yaml 2>&1 | tee install.log
```

### 12.6 Verify the installation

```bash
# Run:ai namespaces
kubectl get namespaces | grep runai

# Helm releases
helm list -A | grep runai

# Control-plane pods
kubectl get pods -n runai

# Cluster agent pods
kubectl get pods -n runai-backend 2>/dev/null || kubectl get pods -n runai

# Ingress objects
kubectl get ingress -A

# Open the Run:ai UI
echo "https://${RUNAI_DOMAIN}"
```

The UI should load and show the cluster with GPU nodes visible. GPU count
will initially show as 0 until Run:ai has collected metrics; allow a few
minutes.

## 13. Run:ai configuration after installation

> Production environments should plan OIDC/SAML integration and SMTP
> configuration according to customer requirements. These are not required for
> a lab evaluation.

### 13.1 Create a project and assign GPU quota

In the Run:ai UI:

1. Navigate to **Projects** → **New Project**.
2. Set the project name (e.g. `demo`).
3. Under **GPU resources**, set **GPU quota** to the number of whole GPUs you
   want this project to use.
4. Save the project.

Or using the Run:ai CLI:

```bash
runai project create demo
runai project set-gpu demo --gpu 2
```

### 13.2 Create a data source from the shared PVC

In the Run:ai UI:

1. Navigate to **Data Sources** → **New Data Source** → **PVC**.
2. Select the `runai-shared-storage` PVC in the `runai` namespace.
3. Set the mount path to `/shared`.
4. Save and assign to the `demo` project.

### 13.3 Submit test workloads

**Whole-GPU nvidia-smi job:**

```bash
runai submit nvidia-smi-test \
  --project demo \
  --image nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04 \
  --gpu 1 \
  -- nvidia-smi
```

```bash
runai logs nvidia-smi-test -p demo
runai delete job nvidia-smi-test -p demo
```

**CUDA vector-add kernel test:**

```bash
runai submit cuda-vector-add \
  --project demo \
  --image nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0 \
  --gpu 1
```

```bash
runai logs cuda-vector-add -p demo
# Expected: "Test PASSED"
runai delete job cuda-vector-add -p demo
```

Check workload metrics in the Run:ai UI under **Workloads** → select the
job → **Metrics**. GPU compute and GPU memory graphs should populate within
30 s of the job starting.

## 14. Validation checklist

Work through this checklist in order. Each item depends on the ones above it.

- [ ] GPU smoke test pod reaches `Succeeded` and `nvidia-smi` shows the B200
- [ ] Both GPU nodes show `nvidia.com/gpu=1` in `kubectl get nodes`
- [ ] NFD labels (`nvidia.com/gpu.product`, `nvidia.com/cuda.driver.major`) present on GPU nodes
- [ ] GFD pods running only on GPU nodes; NFD workers on all nodes
- [ ] Run:ai UI shows both GPU nodes with correct GPU count
- [ ] `DCGM_FI_DEV_GPU_UTIL_RATIO` and `DCGM_FI_DEV_FB_USED_BYTES` visible in Prometheus
- [ ] `DCGM_FI_DEV_GPU_UTIL` and `DCGM_FI_DEV_FB_USED` visible in Prometheus (compat rules)
- [ ] Run:ai UI GPU-compute and GPU-memory graphs show data during a workload
- [ ] Whole-GPU `nvidia-smi` Run:ai job reaches `Completed`
- [ ] CUDA vector-add workload prints "Test PASSED"
- [ ] RWX writer/reader test passes across two different GPU nodes
- [ ] Ingress is reachable via HTTPS at `https://${RUNAI_DOMAIN}`
- [ ] TLS certificate is trusted by the browser (production) or warning is accepted (lab)
- [ ] Run:ai project `demo` exists with GPU quota assigned
- [ ] A third workload queues when both GPUs are occupied (quota enforcement)
- [ ] A high-priority preemptible workload preempts a lower-priority preemptible workload

## 15. GPU fractions and MIG

> **What was validated:** Whole-GPU allocation only.

The following were **not** configured or validated in this recipe:

- MIG (Multi-Instance GPU) partitioning
- Run:ai GPU fractions (time-slicing / spatial partitioning)

MIG and Run:ai GPU fractions are independent features with different
prerequisites. Do not conflate them.

### GPU fractions — planning notes

As of the time of writing, Run:ai GPU fractions require (verify against the
[Run:ai support matrix](https://run-ai-docs.nvidia.com/self-hosted/latest/admin/hardware/hardware-requirements/)
before planning):

| Requirement | Note |
|---|---|
| CUDA driver ≥ R615 | Host driver version |
| CUDA UMD ≥ 13.4 | As reported by `nvidia-smi` |
| NVIDIA GPU Operator ≥ 26.7.1 | With `cdi.nriPluginEnabled=false` |
| cgroup v2 | On all GPU nodes |

Consult the
[NVIDIA Run:ai GPU fractions documentation](https://run-ai-docs.nvidia.com/self-hosted/latest/admin/workloads/gpu-fractions/)
for authoritative requirements, as version numbers change between releases.

### Driverful integration model for GPU fractions

The driverful node image retains ownership of the NVIDIA driver. Any future
GPU Operator deployment must set `driver.enabled=false`. Components already
managed by Nebius — device plugin, container toolkit, NFD, DCGM — must not
be duplicated. The ownership of CDI/DRA device management and the fractioning
operator must be agreed with your Nebius support team before deployment.

**Before attempting GPU fractions on a driverful cluster:**

1. Test on a **separate, non-production node group**.
2. Confirm component ownership with Nebius.
3. Validate the complete workflow end to end before updating the production
   node group.

This recipe does not claim GPU-fraction support until it has been validated.

## 16. Troubleshooting

### Run:ai UI shows 0/0 GPUs

NFD and GFD labels are missing. Run:

```bash
kubectl get nodes -l nebius.com/gpu=true \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.nvidia\.com/gpu\.present}{"\n"}{end}'
```

If labels are absent, check NFD/GFD pods in `gpu-operator`:

```bash
kubectl get pods -n gpu-operator
kubectl describe pod -n gpu-operator -l app=gpu-feature-discovery
```

### GPU nodes show "Missing prerequisites" in Run:ai

This message typically refers to the GPU fractioning operator, not to whole
GPUs. Whole-GPU workloads may still schedule and run correctly. Inspect the
Run:ai cluster status in the UI and check whether the message specifically
references fractioning or dynamic MIG. The validation checklist in Section 14
will confirm whether whole-GPU paths work.

### Nebius-managed nebius-dcgm schedules zero pods on GPU nodes

The managed DaemonSet has node selectors or tolerations that assume GPU
Operator labels or toolkit paths that do not exist on driverful nodes. This is
expected. Do not modify the Nebius-managed release. The standalone DCGM
Exporter installed in Section 8 handles Run:ai monitoring.

### Run:ai CPU metrics work but GPU graphs are blank

The compatibility PrometheusRule is missing or its labels do not match the
Prometheus `ruleSelector`. Check:

```bash
kubectl get prometheusrule dcgm-compat -n runai
kubectl describe prometheusrule dcgm-compat -n runai
```

Also verify that the Run:ai internal Prometheus discovers rules in the `runai`
namespace:

```bash
kubectl get prometheus -n runai -o yaml | grep -A5 ruleSelector
```

### RWX PVC remains Pending

`csi-mounted-fs-path-sc` uses `WaitForFirstConsumer`. The PVC will not bind
until a pod mounts it. This is expected; submit the shared-storage test pod
to trigger binding.

### Run:ai installer rejects directory permissions

The installer requires the working directory to have `0700` permissions. If
you are running from a Windows-mounted WSL path:

```bash
cp runai-installer ~/runai-installer
chmod 0700 ~/runai-installer
cd ~ && ./runai-installer install --config runai-config.yaml
```

### InvalidImageName error in pods

The `image` field in a pod spec must contain only the image reference string
(e.g. `nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04`). It must not include
shell quoting characters, `run:` prefixes, or command fragments. Check the
workload YAML and the Run:ai UI's workload submission form.

### Node-group storage attachment triggers rolling node recreation

This is expected behaviour. Attaching a filesystem to an existing node group
modifies the node template, which Nebius applies by recreating each node
one at a time. Plan this step during a maintenance window, ensure your
workloads are preemptible or have been drained first, and monitor the
node-group status until all nodes are `Ready`.
