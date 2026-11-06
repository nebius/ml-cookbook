# Volcano

This example uses one 32-replica Volcano task. A job-level hard topology
constraint keeps every Pod in one `gpu-cluster-id`; the task's
`partitionPolicy` divides task indices into two 16-Pod partitions and places
each partition at the rack (`nvl-instance-group-id`) tier.

## Requirements

- 32 schedulable nodes labeled
  `node.kubernetes.io/instance-type=gpu-gb300`, all in the intended
  `gpu-cluster-id`, with two usable `nvl-instance-group-id` rack domains;
- at least 4 allocatable `nvidia.com/gpu`, 108 allocatable CPUs, and 720 GiB of
  allocatable memory on every selected node;
- the Nebius NVIDIA device plugin, rack-local static IMEX configuration, and
  host paths `/dev/infiniband` and `/dev/nvidia-caps-imex-channels`;
- permission to run privileged Pods and network access to pull
  `nvcr.io/nvidia/nemo:26.06` and download the Hugging Face model.

The workload requests and limits exactly four GPUs per Pod. CPU and memory are
Burstable: each Pod requests 108 CPUs and 720 GiB and is limited to 112 CPUs
and 760 GiB. Adjust the resource values in `10-training.yaml` if the VM shape
differs.

## Install Volcano 1.15

Run this from the `volcano` directory. `helm-values.yaml` enables the
`network-topology-aware` scheduler plugin and configures HyperNode discovery
from the Nebius GPU-cluster, NVLink rack, and hostname labels.

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts --force-update
helm repo update volcano-sh
helm install volcano volcano-sh/volcano \
  --version 1.15.0 \
  --namespace volcano-system \
  --create-namespace \
  --values helm-values.yaml \
  --wait \
  --timeout 10m
kubectl -n volcano-system wait deployment \
  --all \
  --for=condition=Available \
  --timeout=5m
kubectl -n volcano-system get deployments,pods
kubectl get hypernodes.topology.volcano.sh \
  -l volcano.sh/network-topology-source=label \
  -o wide
```

The generated `spec.tierName` values must include
`topology.nebius.com/gpu-cluster-id` and
`topology.nebius.com/nvl-instance-group-id`. The `highestTierName` fields in
`10-training.yaml` resolve against those exact names. If a tier name is
missing or mismatched, Volcano cannot enforce that named boundary.

## Verify the cluster prerequisites

Run these checks after installation:

```bash
kubectl get crd \
  jobs.batch.volcano.sh \
  queues.scheduling.volcano.sh \
  podgroups.scheduling.volcano.sh \
  hypernodes.topology.volcano.sh
kubectl -n volcano-system get deployments,pods
kubectl get hypernodes.topology.volcano.sh
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id \
  -L kubernetes.io/hostname
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -o 'custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'
```

Every eligible node must have both Nebius topology labels, and the
device-plugin check must report at least four GPUs per node. Static IMEX means
this example does not require a DRA driver, `ResourceClaimTemplate`, or
per-Pod `ResourceClaim`.

## Configure the queue and run the workload

Run the commands below from this directory. `00-config.yaml` creates the
dedicated workload namespace and the Volcano Queue used by the Job.

```bash
kubectl apply -f 00-config.yaml
kubectl get queues.scheduling.volcano.sh gb300-tas-volcano
kubectl -n tas-volcano create secret generic hf-token \
  --from-literal=HF_TOKEN='...'
kubectl -n tas-volcano apply -f ../common/megatron-scripts.yaml
kubectl apply -f 10-training.yaml
```

## Inspect placement

Inspect the Volcano Job, generated PodGroup, and actual placement:

```bash
kubectl -n tas-volcano get vcjob,podgroup,pods -o wide
kubectl -n tas-volcano get vcjob megatron-tas-volcano -o yaml
kubectl -n tas-volcano get podgroup -o yaml
kubectl -n tas-volcano get pods \
  -l volcano.sh/job-name=megatron-tas-volcano \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,PHASE:.status.phase
kubectl get nodes \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id
kubectl -n tas-volcano logs -f megatron-tas-volcano-trainer-0 \
  --all-containers=true
```

`minPartitions: 2` requires two complete logical partitions, but it does not
itself require two different rack label values. The current rack sizes cannot
fit both 16-Pod partitions in one rack; verify the Pod-to-node labels before
relying on the result on any other topology.

## Why this does not use Volcano's PyTorch plugin

The PyTorch plugin is useful for its conventional one-Master-plus-N-Worker
layout, but that layout does not model this exact 32-node segmentation. The
plugin generates a one-replica `master` task for rank 0 and a 31-replica
`worker` task for ranks 1-31. Volcano's `partitionPolicy` belongs to one task,
and the generated subgroup identity includes both task name and partition.
Consequently, the 31-Worker task cannot be divided into two complete
16-replica partitions (`31 != 2 x 16`), while the Master cannot join a Worker
partition.

The plugin also injects one process-level `RANK` and `WORLD_SIZE` per Pod. This
workload deliberately runs four local GPU processes per Pod with `torchrun`,
so `VC_TASK_INDEX` is the node rank and the global rank count remains 128.
Using one `trainer` task makes both the topology partition and the rank mapping
unambiguous.

## Clean up the example

Delete the workload and its inputs when finished. This does not uninstall
Volcano.

```bash
kubectl delete -f 10-training.yaml --ignore-not-found
kubectl -n tas-volcano delete configmap megatron-tas-scripts --ignore-not-found
kubectl -n tas-volcano delete secret hf-token --ignore-not-found
kubectl delete -f 00-config.yaml --ignore-not-found
```

## Uninstall Volcano

```bash
helm uninstall volcano --namespace volcano-system
```

References: [Volcano installation](https://volcano.sh/docs/gettingstarted/installation/),
[Volcano 1.15 release](https://volcano.sh/blog/volcano-1.15.0-release/),
[network-topology-aware scheduling and discovery](https://volcano.sh/docs/v1.15.0/keyfeatures/networktopologyaware/),
[network topology design](https://github.com/volcano-sh/volcano/blob/v1.15.0/docs/design/Network%20Topology%20Aware%20Scheduling.md),
and [PyTorch plugin](https://volcano.sh/docs/v1.13.0/userguide/user_guide_how_to_use_pytorch_plugin/).
