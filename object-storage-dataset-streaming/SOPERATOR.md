# Mount a dataset bucket on Soperator

This guide is for cluster operators using Soperator 4.1.6. It attaches a
dataset bucket to the shared jail through the Mountpoint CSI driver, so jobs
can read `/mnt/docvqa` on every node. Training instructions are in the
[dataset streaming recipe](README.md).

If your platform team manages the cluster, request a read-only Mountpoint
CSI volume at `/mnt/docvqa` on login and worker nodes, with the dataset
prefix selected and a read-only access key stored in a Kubernetes secret.

## Attach the bucket to the jail

Soperator supports [Kubernetes volumes as jail submounts](https://github.com/nebius/soperator/blob/4.1.6/docs/features.md#L54-L55).
The CSI driver manages the mount's lifecycle; jobs read the mounted path.

1. On the Soperator's Kubernetes cluster, install the driver and create the
   PersistentVolume and claim using the
   [Nebius Mountpoint instructions](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3).
   Use the `aws-secret` key-pair secret in `kube-system`, a `ReadOnlyMany`
   volume, and a claim named `s3-mountpoint` in `namespace: soperator`.
   Set the volume's `mountOptions` for `endpoint-url`, `region`,
   `maximum-throughput-gbps 10000`, `max-threads 64`, and `allow-other`.
   Choose a memory target for the Mountpoint pod's allocation
   ([memory sizing](README.md#size-the-memory-budget)). Add the dataset prefix
   and read-only flag to `spec.mountOptions`:

   ```yaml
   - prefix datasets/docmatix-demo/
   - read-only
   ```

   Without the prefix, `/mnt/docvqa` exposes the bucket root and the example
   cannot find `train/`. Keep `storageClassName: ""` on both the volume and
   claim so they bind to each other. Create the claim before referencing it
   from SlurmCluster and NodeSet resources; pods wait while it is missing or
   unbound.

2. Reference the claim from the
   [SlurmCluster resource](https://github.com/nebius/soperator/blob/4.1.6/api/v1/slurmcluster_types.go#L49)
   for the login nodes:

   ```yaml
   spec:
     volumeSources:
       - name: docvqa
         persistentVolumeClaim:
           claimName: s3-mountpoint
           readOnly: true
     slurmNodes:
       login:
         volumes:
           jailSubMounts:
             - name: docvqa
               mountPath: /mnt/docvqa
               readOnly: true
               volumeSourceName: docvqa
   ```

   In Soperator 4.1.6, workers are separate `NodeSet` resources, and each
   NodeSet that trains carries its own entry under `spec.slurmd.volumes`,
   with the source inlined instead of named:

   ```yaml
   spec:
     slurmd:
       volumes:
         jailSubMounts:
           - name: docvqa
             mountPath: /mnt/docvqa
             readOnly: true
             volumeSource:
               persistentVolumeClaim:
                 claimName: s3-mountpoint
                 readOnly: true
   ```

   For installations built from the
   [Nebius solutions library](https://github.com/nebius/nebius-solutions-library/tree/main/soperator),
   the referenced Terraform template has no claim-backed submount option.
   Add both entries to the managed Helm configuration
   (the `slurm-cluster` values for `volumeSources` and the login node, the
   `nodesets` values for each NodeSet). The
   [existing jail and spool values](https://github.com/nebius/nebius-solutions-library/blob/1b0b25476bb67099cc60ce6bf30d5b1e24b491d7/soperator/modules/slurm/templates/helm_values/terraform_fluxcd_values.yaml.tftpl#L511)
   show how claims are rendered. Persist the changes in Helm/Flux or Terraform
   so reconciliation does not overwrite them.

3. In jobs, read `/mnt/docvqa` as in
   [Step 2 of the recipe](README.md#step-2-read-it-with-the-loader-you-already-have).
   The driver attaches the claim again when pods are rescheduled after node
   replacement or preemption.

## Verify the mount

Mountpoint CSI driver 2.x runs `mount-s3` in separate pods
(by default in the `mount-s3` namespace) and can share a mount between compatible
workloads on a node. Verify that the driver and mount pods run on every target
node, including nodes with Soperator taints.

Inside the jail, check that `/mnt/docvqa` is a read-only `mountpoint-s3` FUSE
mount with `allow_other`. Check from a login shell, an `srun` step, and an
`sbatch` job; each must see the same path and read the dataset.

If login or worker pods stay `Pending`, run
`kubectl get pvc -n soperator s3-mountpoint`. The claim must show `Bound`, with
`storageClassName: ""` matching the static volume.

## Fallback: mount inside the job

If a CSI mount is unavailable, first check the deployed jail and cluster
policy. Soperator 4.1.6 renders
[privileged workers with `SYS_ADMIN`](https://github.com/nebius/soperator/blob/4.1.6/internal/render/worker/container.go#L256-L262)
and [bind-mounts their `/dev` into the jail](https://github.com/nebius/soperator/blob/4.1.6/images/common/scripts/complement_jail.sh#L268).
The [jail image](https://github.com/nebius/soperator/blob/4.1.6/images/jail/jail.dockerfile)
does not include `mount-s3`.

Run the [preflight checks](README.md#before-you-mount) inside an `srun` step
on a worker, as well as on the login node. If `/dev/fuse` is available and
policy permits job-managed mounts, install `mount-s3` and `fuse3` in the jail.

Each step has its own mount namespace, so run the mount and trainer in the
same step. An [upstream issue](https://github.com/awslabs/mountpoint-s3/issues/846)
reports daemonized `mount-s3` dying under `srun`; use `--foreground` with a
supervisor that waits for the mount to become ready and handles startup failure.
Keep the daemon alive until the trainer exits, then unmount and stop it.
Root can use `umount`; an unprivileged mount needs `fusermount3`.

A surviving daemon can retain the step's mount namespace, so step teardown
alone is insufficient cleanup. Killing the daemon first leaves a stale mount
until it is detached or the namespace is destroyed. Pass
`--maximum-throughput-gbps` explicitly as in
[Step 1](README.md#step-1-mount-the-bucket): Nebius hosts do not provide the
EC2 metadata Mountpoint uses to choose its default.

## Fallback: no FUSE

Stream directly from the bucket in the training process. Hugging Face
Datasets reads `s3://` URIs through `fsspec` and `s3fs`. Install `s3fs` in your
training environment and set the endpoint in `storage_options`:

```python
import os

from datasets import load_dataset

documents = load_dataset(
    "parquet",
    data_files="s3://YOUR_BUCKET/datasets/docmatix-demo/train/*.parquet",
    split="train",
    streaming=True,
    storage_options={
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {"endpoint_url": "https://storage.eu-north1.nebius.cloud"},
    },
)
```

PyArrow readers can use `pyarrow.fs.S3FileSystem(endpoint_override=...)`;
the [S3 Connector for PyTorch](https://github.com/awslabs/s3-connector-for-pytorch)
is another option. Apply the same
[shard-assignment rules](README.md#how-shards-reach-processes).

Nebius' own [Downloading data](https://docs.nebius.com/slurm-soperator/storage/download-data)
page also describes staging the dataset on the shared filesystem with the
AWS CLI or rclone as a Slurm job. Consider this for
[repeated epochs](README.md#several-epochs).
