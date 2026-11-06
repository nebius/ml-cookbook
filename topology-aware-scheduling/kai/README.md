# KAI Scheduler

KAI can express the two 16-node rack segments in two ways. Its automatic
segment integration converts annotations on a supported workload into a
hierarchical `PodGroup`; use `10-pytorchjob.yaml` or
`11-leaderworkerset.yaml` for that path. A native Indexed Job is not one of
those integrations, so `12-indexedjob.yaml` shows the explicit equivalent:
one root `PodGroup`, two rack-level subgroups, and one 16-Pod Job per subgroup.

The automatic path is shorter and preserves the workload controller's own
lifecycle. The explicit path exposes the rank-to-segment mapping directly and
works with native Jobs, at the cost of maintaining two coordinated Jobs.

## Requirements

- KAI Scheduler 0.17.0 or a compatible release with topology scheduling and
  the PodGrouper enabled;
- Kubeflow Training Operator V1 for `10-pytorchjob.yaml`, LeaderWorkerSet for
  `11-leaderworkerset.yaml`, or neither additional controller when only
  running `12-indexedjob.yaml`;
- 32 schedulable nodes labeled
  `node.kubernetes.io/instance-type=gpu-gb300`, all in the intended
  `topology.nebius.com/gpu-cluster-id`, with two usable
  `topology.nebius.com/nvl-instance-group-id` rack domains;
- at least 4 allocatable `nvidia.com/gpu`, 108 allocatable CPUs, and 720 GiB of
  allocatable memory on every selected node;
- the Nebius NVIDIA device plugin, rack-local static IMEX configuration, and
  host paths `/dev/infiniband` and `/dev/nvidia-caps-imex-channels`;
- permission to run privileged Pods and network access to pull
  `nvcr.io/nvidia/nemo:26.06` and download the Hugging Face model.

The workload requests and limits exactly four GPUs per Pod. CPU and memory are
Burstable: each Pod requests 108 CPUs and 720 GiB and is limited to 112 CPUs
and 760 GiB. If the VM shape differs, update every chosen workload manifest
and the matching Queue quota in `00-config.yaml`.

## Install KAI Scheduler 0.17

The production chart enables the PodGrouper required by the automatic segment
examples. Install it directly with the pinned release tag:

```bash
helm install kai-scheduler \
  oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  --version v0.17.0 \
  --namespace kai-scheduler \
  --create-namespace \
  --set podgrouper.enabled=true \
  --set topologyMigration.enabled=false \
  --set defaultQueue.createDefaultQueue=false \
  --wait \
  --timeout 10m
kubectl -n kai-scheduler wait deployment \
  --all \
  --for=condition=Available \
  --timeout=5m
kubectl -n kai-scheduler get deployments,pods
```

KAI's control plane uses the `kai-scheduler` namespace. Workloads must use a
different namespace; all examples use `tas-kai`.

The managed Nebius NVIDIA device plugin provides the `nvidia.com/gpu` resources
used by these examples; do not install a second GPU device plugin.

## Install the selected workload controller

The automatic segment path depends on the controller that owns the workload.
Install only the controllers for the examples you intend to run. The explicit
Indexed Job example needs no controller beyond Kubernetes and KAI.

### PyTorchJob

`10-pytorchjob.yaml` uses the legacy `kubeflow.org/v1` `PyTorchJob` API, so it
requires Kubeflow Training Operator V1 rather than the Trainer V2 `TrainJob`
API. Install the standalone stable controller to run this variant:

```bash
kubectl apply --server-side -k \
  'github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1'
kubectl -n kubeflow wait deployment/training-operator \
  --for=condition=Available \
  --timeout=5m
kubectl get crd pytorchjobs.kubeflow.org
```

The Training Operator Python SDK is not required because this guide applies a
YAML manifest directly.

### LeaderWorkerSet

`11-leaderworkerset.yaml` uses the stable
`leaderworkerset.x-k8s.io/v1` API. Install LeaderWorkerSet 0.10.0 to run this
variant:

```bash
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/lws/releases/download/v0.10.0/manifests.yaml
kubectl -n lws-system wait deployment/lws-controller-manager \
  --for=condition=Available \
  --timeout=5m
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io
```

## Verify the cluster prerequisites

KAI installs `Topology` in the `kai.scheduler` API group. `PodGroup` and
`Queue` remain in `scheduling.run.ai`; checking
`topologies.scheduling.run.ai` is therefore incorrect for KAI 0.17.

```bash
kubectl get crd \
  topologies.kai.scheduler \
  podgroups.scheduling.run.ai \
  queues.scheduling.run.ai
kubectl -n kai-scheduler get deployments,pods
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -o 'custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'
```

Every eligible node must have both Nebius topology labels. Static IMEX means
these examples do not require a DRA driver, `ResourceClaimTemplate`, or
per-Pod `ResourceClaim`.

## Configure KAI topology and queue

KAI does not need a separate label-discovery controller for this setup. The
`Topology` object in `00-config.yaml` tells the scheduler to read, in order,
`gpu-cluster-id`, `nvl-instance-group-id`, and hostname directly from each
node. The same file creates the dedicated workload namespace and a Queue with
quota for the entire 128-GPU job.

KAI represents Queue CPU in millicores and memory in decimal megabytes. The
provided quota is the 32-Pod request total: 3,456,000 millicores,
24,739,012 MB, and 128 GPUs.

```bash
kubectl apply -f 00-config.yaml
kubectl get topologies.kai.scheduler nebius-gb300-kai-guide
kubectl get queues.scheduling.run.ai gb300-tas-kai
kubectl get topologies.kai.scheduler nebius-gb300-kai-guide -o yaml
```

The KAI chart includes the PodGrouper that interprets the PyTorchJob and
LeaderWorkerSet segment annotations; no separate TAS feature gate is needed.
The `kai.scheduler/topology` annotation must resolve to
`nebius-gb300-kai-guide`, or KAI ignores the segment annotations. The provided
manifests already select `schedulerName: kai-scheduler` and the
`gb300-tas-kai` Queue.

Create the shared launcher and Hugging Face token once before running any
variant:

```bash
kubectl -n tas-kai create secret generic hf-token \
  --from-literal=HF_TOKEN='...'
kubectl -n tas-kai apply -f ../common/megatron-scripts.yaml
```

## PyTorchJob with automatic segments

This is the preferred batch example. It uses 32 Worker replicas and no
separate Master replica, keeping the total at 32 GPU Pods. KAI derives two
16-Pod rack segments from the Worker annotations.

```bash
kubectl get crd pytorchjobs.kubeflow.org
kubectl apply -f 10-pytorchjob.yaml
kubectl -n tas-kai get pytorchjobs,podgroups,pods -o wide
kubectl -n tas-kai logs -f \
  -l training.kubeflow.org/job-name=megatron-tas-kai-pytorch,training.kubeflow.org/replica-index=0 \
  --all-containers=true \
  --max-log-requests=1
kubectl delete -f 10-pytorchjob.yaml --ignore-not-found
```

## LeaderWorkerSet with automatic segments

LeaderWorkerSet models one 32-Pod replica group divided into 16-Pod
subgroups. It is a service controller rather than a batch completion API, so
the containers remain alive after successful training until the
LeaderWorkerSet is deleted.

```bash
kubectl get crd leaderworkersets.leaderworkerset.x-k8s.io
kubectl apply -f 11-leaderworkerset.yaml
kubectl -n tas-kai get leaderworkersets,podgroups,pods -o wide
kubectl -n tas-kai logs -f megatron-tas-kai-lws-0 --all-containers=true
kubectl delete -f 11-leaderworkerset.yaml --ignore-not-found
```

## Native Indexed Jobs with an explicit PodGroup

Use this fallback when the application must remain a native Job. Segment 0
owns node ranks 0-15 (global ranks 0-63), and segment 1 owns node ranks 16-31
(global ranks 64-127).

```bash
kubectl apply -f 12-indexedjob.yaml
kubectl -n tas-kai get podgroups,jobs,pods -o wide
kubectl -n tas-kai logs -f \
  -l batch.kubernetes.io/job-name=megatron-tas-kai-job-segment-0,batch.kubernetes.io/job-completion-index=0 \
  --all-containers=true \
  --max-log-requests=1
kubectl delete -f 12-indexedjob.yaml --ignore-not-found
```

## Verify placement

For any variant, verify the actual placement against the node labels:

```bash
kubectl -n tas-kai get pods \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase
kubectl get nodes \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id
kubectl -n tas-kai get podgroups -o yaml
```

KAI treats each generated segment as a topology-constrained subgroup. Always
check the assigned node labels: the intended result is two rack domains, not
merely two logical groups that happen to fit within one domain.

## Clean up the example

Delete the selected workload first, then remove the shared inputs and example
scheduler objects. This does not uninstall KAI, Training Operator, or
LeaderWorkerSet.

```bash
kubectl delete -f 10-pytorchjob.yaml --ignore-not-found
kubectl delete -f 11-leaderworkerset.yaml --ignore-not-found
kubectl delete -f 12-indexedjob.yaml --ignore-not-found
kubectl -n tas-kai delete configmap megatron-tas-scripts --ignore-not-found
kubectl -n tas-kai delete secret hf-token --ignore-not-found
kubectl delete -f 00-config.yaml --ignore-not-found
```

## Uninstall KAI and optional controllers

```bash
helm uninstall kai-scheduler --namespace kai-scheduler
```

Remove the optional controllers that you installed for these examples:

```bash
kubectl delete -k \
  'github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1'
kubectl delete \
  -f https://github.com/kubernetes-sigs/lws/releases/download/v0.10.0/manifests.yaml
```

References: [KAI installation](https://github.com/kai-scheduler/KAI-Scheduler#installation),
[KAI 0.17 release](https://github.com/kai-scheduler/KAI-Scheduler/releases/tag/v0.17.0),
[KAI segments](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/topology/segments.md),
[KAI multilevel topology](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/topology/multilevel.md),
[Training Operator V1 installation](https://www.kubeflow.org/docs/components/trainer/legacy-v1/installation/),
and [LeaderWorkerSet installation](https://lws.sigs.k8s.io/docs/installation/).
