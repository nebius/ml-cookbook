# Running NCCL tests with PyTorch
This document provides instructions on how to run NCCL (NVIDIA Collective Communications Library) tests using pure PyTorch.

## Prerequisites

Before you start, make sure you have the following:
- A Kubernetes cluster with GPU operator, network operator and Volcano installed.
- Created resource [`Queue` named `test-queue` in Volcano](../queue.yaml).

## Steps

For running the actual test, we will use [pure PyTorch implementation of AllReduce NCCL test](https://github.com/stas00/ml-engineering/blob/a0f2d508309027d036fde400cbc6060e8679c70d/network/benchmarks/all_reduce_bench.py).
For launching the job, we will use the `pytorch` plugin of Volcano, which takes care of setting up the environment variables `RANK`, `MASTER_ADDR`, `MASTER_PORT` for you. All you need to do is to split yopur job in 2 tasks:
1. Master task: This task will run the on the master node which will coordinate the other worker nodes. It has a single node, so `replicas` should be set to `1`.
2. Worker task: This task will run on the worker nodes. The number of nodes is determined by the `replicas` parameter and should be set to `N-1` if `N` is the total number of nodes you want to use for the test.

### [Optional] Configure the job

Examine `nccl-test-pytorch.yaml` and modify the following parameters as needed:
- `replicas` in the `worker` task: Number of nodes you want to use for the test.
- `image`: base Docker image to use for the test. It is recommended to use PyTorch images supplied by NVIDIA since they have the necessary NCCL, RDMA, and CUDA libraries pre-installed.
- `-nnodes` parameter of `torchrun`: Total number of nodes participating in the test.

Optionally, you may configure NCCL environment variables for a more verbose output:
```yaml
env:
  - name: NCCL_DEBUG
    value: INFO
```

### Start the job

Submit the job to Volcano with the following command:
```bash
kubectl apply -f nccl-test-pytorch.yaml
```

### Monitor the job

You can monitor the job logs with the following command:

```bash
kubectl logs -f nccl-test-pytorch-master-0
```

### Clean up

To delete the job, use the following command:

```bash
kubectl delete -f nccl-test-pytorch.yaml
```
