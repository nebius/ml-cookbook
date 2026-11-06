# Kueue

This example uses Kueue 0.19 multi-layer TAS with a native Indexed Job. Kueue
admits the complete 32-Pod workload into one `gpu-cluster-id` and divides it
into two 16-Pod `nvl-instance-group-id` slices.

## Requirements

- Kueue 0.19 with the `batch/job` integration and the
  `TopologyAwareScheduling` and `TASMultiLayerTopology` feature gates enabled;
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
and 760 GiB. Adjust both the Pod resources in `10-training.yaml` and the
matching ClusterQueue quotas in `00-config.yaml` if the VM shape differs.

## Install Kueue 0.19

Run this from the `kueue` directory. `helm-values.yaml` enables both TAS feature
gates, the native Job integration, and the all-or-nothing Pod readiness policy
used by this workload.

```bash
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version 0.19.0 \
  --namespace kueue-system \
  --create-namespace \
  --values helm-values.yaml \
  --wait \
  --timeout 5m
kubectl wait deployment/kueue-controller-manager \
  --namespace kueue-system \
  --for=condition=Available \
  --timeout=5m
```

## Verify the cluster prerequisites

Run these checks before creating the example queue. `Topology`,
`ResourceFlavor`, and `ClusterQueue` use the `v1beta2` API in these manifests.

```bash
kubectl get crd \
  topologies.kueue.x-k8s.io \
  resourceflavors.kueue.x-k8s.io \
  clusterqueues.kueue.x-k8s.io \
  localqueues.kueue.x-k8s.io
kubectl -n kueue-system get deployments,pods
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id
kubectl get nodes -l node.kubernetes.io/instance-type=gpu-gb300 \
  -o 'custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'
```

Every eligible node must have both Nebius topology labels. The device-plugin
check must report at least four GPUs per node. Static IMEX means this example
does not require a DRA driver, `ResourceClaimTemplate`, or per-Pod
`ResourceClaim`.

## Configure TAS and run the workload

Run the commands below from this directory. `00-config.yaml` creates the
workload namespace, the three-level Nebius `Topology`, a topology-backed
GB300 `ResourceFlavor`, a ClusterQueue with quota for all 32 Pods, and the
namespaced LocalQueue referenced by the Job.

```bash
kubectl apply -f 00-config.yaml
kubectl get topologies.kueue.x-k8s.io nebius-gb300-kueue
kubectl get resourceflavors.kueue.x-k8s.io gb300-kueue
kubectl get clusterqueues.kueue.x-k8s.io gb300-tas-kueue
kubectl -n tas-kueue get localqueues.kueue.x-k8s.io gb300
```

Create the workload inputs and submit the common training workload:

```bash
kubectl -n tas-kueue create secret generic hf-token \
  --from-literal=HF_TOKEN='...'
kubectl -n tas-kueue apply -f ../common/megatron-scripts.yaml
kubectl apply -f 10-training.yaml
```

## Inspect placement

Inspect Kueue's authoritative topology assignment and the resulting Pods:

```bash
kubectl -n tas-kueue get workloads.kueue.x-k8s.io
kubectl -n tas-kueue get workloads.kueue.x-k8s.io -o yaml
kubectl -n tas-kueue get pods \
  -l batch.kubernetes.io/job-name=megatron-tas-kueue \
  -o 'custom-columns=INDEX:.metadata.labels.batch\.kubernetes\.io/job-completion-index,NODE:.spec.nodeName,PHASE:.status.phase'
kubectl get nodes \
  -L topology.nebius.com/gpu-cluster-id \
  -L topology.nebius.com/nvl-instance-group-id
kubectl -n tas-kueue logs -f \
  -l batch.kubernetes.io/job-name=megatron-tas-kueue,batch.kubernetes.io/job-completion-index=0 \
  --all-containers=true \
  --max-log-requests=1
```

The current two-rack test topology cannot fit both 16-Pod slices in one rack.
Still verify the assigned rack labels rather than assuming that two logical
slices always imply two distinct topology domains.

## Clean up the example

Delete the workload and its namespaced inputs when finished. This does not
uninstall Kueue.

```bash
kubectl delete -f 10-training.yaml --ignore-not-found
kubectl -n tas-kueue delete configmap megatron-tas-scripts --ignore-not-found
kubectl -n tas-kueue delete secret hf-token --ignore-not-found
kubectl delete -f 00-config.yaml --ignore-not-found
```

## Uninstall Kueue

```bash
helm uninstall kueue --namespace kueue-system
```

References: [Kueue installation](https://kueue.sigs.k8s.io/v0.19/docs/getting-started/installation/),
[topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/),
[running a TAS workload](https://kueue.sigs.k8s.io/v0.19/docs/tasks/run/topology_aware_scheduling/),
and [all-or-nothing startup policy](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/).
