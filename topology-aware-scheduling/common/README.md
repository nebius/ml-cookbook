# Common Megatron Bridge launcher

All Kubernetes examples use `megatron-scripts.yaml` as the single source of
truth for the training command. It runs Qwen3 235B-A22B on 32 GB300 nodes with
four local processes per node and 128 ranks in total.

Each scheduler manifest supplies only two launcher inputs:

- `NODE_RANK`, `JOB_COMPLETION_INDEX`, or `VC_TASK_INDEX`;
- `MASTER_ADDR` for its controller-specific rendezvous DNS name.

Apply the ConfigMap to the workload namespace before applying a scheduler's
training manifest:

```bash
kubectl -n <workload-namespace> apply -f ../common/megatron-scripts.yaml
```

The Pod templates deliberately remain in the scheduler manifests because the
different workload APIs place them at different schema paths. They use the same
image, resource requests, environment, security context, and device mounts.
