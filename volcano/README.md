# Running distributed workloads with Volcano installed on Nebius Managed k8s cluster

This section contains example workloads using Volcano scheduler on a Nebius Managed Kubernetes cluster. 

## About Volcano:

**Volcano** is a batch scheduling system designed for Kubernetes, specifically optimized for AI/ML, deep learning, big data, and high-performance computing (HPC) workloads. It extends Kubernetes' native scheduling capabilities by providing advanced job management features necessary for large-scale distributed computing. [Official documentation here](https://volcano.sh/en/docs/).

Volcano introduces some new **CRD**s to make it possible:

- **Queue** in Volcano acts as a logical grouping mechanism that organizes jobs based on priority and resource quotas. It helps in fair scheduling and resource sharing across teams or workloads. It is similar in a certain way to partitions in Slurm, however it is not bound to workers (physical nodes or worker pods in Soperator) but rather to a set of resources (CPU, RAM, GPU), thus abstracting the physical infrastructure. It is possible to define priority, preemption and resource limits. [More about Queues](https://volcano.sh/en/docs/queue/).

- **VolcanoJob** is a logical development of k8s **Job** adapted to handle complex, distributed, and high-performance computing (HPC) workloads. It introduces advanced scheduling, resource management, and job execution features that are essential for AI/ML training, deep learning, and large-scale batch processing. Most notable features include:

    - Gang scheduling makes sure all required pods of a job are scheduled together before execution starts, which allows to prevent hosting partial jobs and wasting resources.

    - Multi-task jobs allow defining multiple task roles within a job, like master and worker nodes for distributed training.

    - Priority and Preemption allow to set the priority for job execution and evict low priority jobs.

    - Resource-aware scheduling allows to manage GPU allocation, use NUMA-aware scheduling as well as Network Topology aware scheduling. 

    [More about Jobs in Volcano](https://volcano.sh/en/docs/vcjob/).

- PodGroup is the resource which is used by VolcanoJob to manage gang scheduling, which guarantees that all pods in a job are launched simultaneously, avoiding resource fragmentation and partial execution. It is typically used in together with **VolcamoJob**. [More about PodGroup](https://volcano.sh/en/docs/podgroup/).

## Prerequisites:

Provision a managed k8s cluster from **k8s-training** solution with at least 2 GPU nodes and IB from [Nebius Solutions Library](https://github.com/nebius/nebius-solution-library/tree/main/k8s-training) and connect to it [following documentation](https://docs.nebius.com/kubernetes/quickstart#connect). 

## Installing Volcano:

Volcano can be installed on a Nebius Managed Kubernetes cluster using Helm. Follow these steps to install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

## Example wokloads:

**NOTE**: to run example workloads, first create a `Queue` (including 2 nodes with 8 GPUs each) by running `kubectl apply -f queue2.yaml`.
Use queue4.yaml as example of queue for 4 nodes, and apply changes for more nodes if needed.
- [NCCL tests in pure PyTorch](./nccl-test-pytorch/)
- [LLM finetuning with *llama-cookbook*](./llama-cookbook-finetuning/)

## Cleaning up:

- To delete the `Queue`, run `kubectl delete -f queue.yaml`.
- To delete your Volcano deployment, run `helm uninstall -n volcano-system volcano`