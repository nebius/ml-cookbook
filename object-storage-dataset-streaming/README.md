# Train from a dataset in Object Storage

Mount a Nebius Object Storage dataset as a read-only directory and train from
it using ordinary file paths, without staging the full dataset locally. This
recipe uses [Mountpoint for Amazon S3](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3)
and includes an optional Qwen3.8-27B QLoRA example for document question answering.

```text
Dataset shards in Object Storage
  -> read-only mount at /mnt/docvqa on each node
  -> file-based data loader
  -> trainer
```

Mountpoint is independent of your training framework and GPU driver. Use a
loader that streams files without requiring locks or writes beside the input
files. For distributed training, also check how it assigns shards to processes.

- **Use your own dataset:** complete the prerequisites, then follow
  [Step 1](#step-1-mount-the-bucket) and
  [Step 2](#step-2-read-it-with-the-loader-you-already-have).
- **Run the bundled example:** [prepare the sample](#prepare-the-example-optional),
  then follow all three steps.
- **Use a container or cluster:** see [Where this runs](#where-this-runs)
  for mounting instructions.

## Prerequisites

- A Linux environment where you can mount the bucket: a Compute VM,
  Managed Kubernetes cluster, or Soperator cluster. For cluster setup with
  the Mountpoint CSI driver, see [Where this runs](#where-this-runs).
- An Object Storage bucket in the same region as the training nodes.
- An access key that can list the dataset prefix and read its objects.
  Publishing the sample also requires write access and the AWS CLI.

The VM installation and mount commands below target Ubuntu 24.04.
Replace `YOUR_BUCKET` with your bucket name. The examples use `eu-north1`;
change both the region and endpoint if your bucket is elsewhere.

### Provide credentials

Follow the [AWS CLI setup guide](https://docs.nebius.com/object-storage/interfaces/aws-cli#aws-cli-setup-instructions)
to configure access. Mountpoint uses the standard AWS credential chain: export
the key pair in the shell that runs it, or use `~/.aws/credentials` with
`AWS_PROFILE`.

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

Use a read-only key for training jobs. Keep credentials out of the repository
and container images.

## Prepare the example (optional)

Skip this section if you are using your own dataset and training environment.
The bundled example needs Linux x86_64 and Python 3.12. Training targets an
NVIDIA GPU with 80 GB of memory. The pinned PyTorch wheel requires a CUDA 13
driver; on Nebius, use the `ubuntu24.04-cuda13.0` image. Allow disk space for
the model weights and saved adapter.

`--max-pixels` limits page resolution, but page count and text length also
affect GPU memory use. Lowering the resolution alone may not make a document fit.

### Set up the example's environment

From a clone of this repository:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev python3.12-venv
cd ml-cookbook/object-storage-dataset-streaming || exit
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip==26.2.1
.venv/bin/python -m pip install --require-hashes -r requirements.lock
export HF_HOME=/persistent/path/huggingface
```

`requirements.lock` pins dependencies for Linux x86_64 and Python 3.12;
`--require-hashes` verifies the downloaded packages against those pins.
Set `HF_HOME` to a writable location with room for the model weights.

### Publish the sample dataset

The publisher writes six Docmatix documents as separate Parquet shards: four
for training and two for validation. It can run on a CPU-only machine using
the environment above and needs access to the Hugging Face Hub. Use a fresh
output directory and credentials with write access for the upload:

```bash
.venv/bin/python prepare_docmatix.py \
  --output-dir docmatix-demo \
  --train-documents 4 \
  --validation-documents 2 \
  --documents-per-file 1

aws s3 sync docmatix-demo s3://YOUR_BUCKET/datasets/docmatix-demo/ \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Rows are written in the source order of the pinned
[`HuggingFaceM4/Docmatix`](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)
stream, one document per row group. For a real corpus, raise
`--documents-per-file` so each shard is hundreds of MiB, shuffle rows when you
publish, and review the source dataset's licence and contents before training
on it.

## Step 1. Mount the bucket

### Install Mountpoint

Install Mountpoint 1.24.0 and `fusermount3`, the tool for unmounting without root:

```bash
sudo apt-get install -y fuse3
wget https://s3.amazonaws.com/mountpoint-s3-release/1.24.0/x86_64/mount-s3-1.24.0-x86_64.deb
sudo apt-get install -y ./mount-s3-1.24.0-x86_64.deb
mount-s3 --version
```

For RPM-based images, see the
[Nebius installation instructions](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3).

### Before you mount

Check that the tools and FUSE device are available and that the target is not
already mounted:

```bash
command -v mount-s3 fusermount3          # both print a path
ls -l /dev/fuse                          # crw-rw-rw- ... /dev/fuse
mountpoint -q /mnt/docvqa && echo "already mounted"
```

Mount on an empty directory you own. If it is already mounted, reuse it or
unmount it first. If `/dev/fuse` is missing, ask the platform team to provide
it or a managed mount; see [Where this runs](#where-this-runs).

### Mount and verify

The commands below use the sample published above. For your own dataset,
replace the prefix and verification filename. The prefix must end with `/`.

```bash
sudo mkdir -p /mnt/docvqa
sudo chown "$(id -u):$(id -g)" /mnt/docvqa

mount-s3 YOUR_BUCKET /mnt/docvqa \
  --prefix datasets/docmatix-demo/ \
  --read-only \
  --region eu-north1 \
  --endpoint-url https://storage.eu-north1.nebius.cloud:443 \
  --force-path-style \
  --maximum-throughput-gbps 10000 \
  --max-threads 64 \
  --metadata-ttl 120 \
  --memory-target 4096

grep " /mnt/docvqa " /proc/mounts                # mountpoint-s3 /mnt/docvqa fuse ro,...
ls /mnt/docvqa/train | head
head -c 4 /mnt/docvqa/train/part-00000.parquet   # prints PAR1 for a Parquet file
```

`mount-s3` prints one line and returns only once the mount is ready, then
keeps running as a daemon:

```text
prefix datasets/docmatix-demo/ of bucket YOUR_BUCKET is mounted at /mnt/docvqa
```

The options set a 4 GiB memory target and a two-minute metadata cache.
See [Mount options](#mount-options) for tuning and access-control details.

## Step 2. Read it with the loader you already have

Point your loader at the mounted path. For example, read Parquet batches
with PyArrow:

```python
import pyarrow.parquet as pq

for batch in pq.ParquetFile("/mnt/docvqa/train/part-00000.parquet").iter_batches(64):
    ...
```

With Hugging Face Datasets, use `streaming=True`:

```python
from datasets import load_dataset

documents = load_dataset(
    "parquet", data_files="/mnt/docvqa/train/*.parquet", split="train", streaming=True
)
```

Without streaming, `load_dataset` reads the full corpus and builds a local
Arrow cache. For JSONL, use the `"json"` loader and your file pattern.

For distributed reads, publish at least one shard per training process,
ideally `processes × max(1, workers_per_process)` shards. Let one layer of the stack
assign disjoint files to processes. With the pinned TRL version, keep
`shuffle_dataset=False`; see [How shards reach processes](#how-shards-reach-processes)
for framework settings and the streaming-shuffle limitation.

Mountpoint works best with sequential reads of large objects. For tar shards
or memory-mapped tokenized data, see
[WebDataset and tokenized shards](#webdataset-and-tokenized-shards).

**Optional: inspect the sample without a GPU.** Using the environment created
above, decode the first document and print its question:

```bash
.venv/bin/python - <<'PY'
from datasets import Image, List, load_dataset
rows = load_dataset("parquet", data_files="/mnt/docvqa/train/*.parquet", split="train", streaming=True)
row = next(iter(rows.cast_column("images", List(Image()))))
print(rows.n_shards, "shards;", len(row["images"]), "page(s);", row["prompt"][0]["content"][:80])
PY
```

With the sample dataset this prints:

```text
4 shards; 3 page(s); What is the purpose of the Confirmation Statement mentioned in the document?
```

## Step 3. Train from the mount (optional Qwen3.8-27B example)

With the example environment prepared and the bucket mounted, run
[`train_sft.py`](train_sft.py) for five optimizer steps on one GPU:

```bash
.venv/bin/python train_sft.py /mnt/docvqa \
  --max-steps 5 \
  --gradient-accumulation 1 \
  --num-workers 1 \
  --output-dir outputs/qwen-docvqa-smoke
```

On two GPUs of one node, use `torchrun`. Each process loads its own model
replica and reads two of the four training shards:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 train_sft.py /mnt/docvqa \
  --max-steps 5 --gradient-accumulation 1 --num-workers 1 \
  --output-dir outputs/qwen-docvqa-two-gpu
```

### Expected output

On one GPU with the six-document sample, the log looks like this after the
model loads; the values depend on your data, hardware, and run:

```text
trainable params: ... || all params: ... || trainable%: ...
{'loss': '...', 'grad_norm': '...', 'learning_rate': '0', ..., 'epoch': '0.2'}
{'loss': '...', 'grad_norm': '...', 'learning_rate': '0.0002', ..., 'epoch': '0.4'}
...
{'train_runtime': '...', 'train_samples_per_second': '...', 'train_steps_per_second': '...', 'train_loss': '...', 'epoch': '...'}
finished 5 optimizer steps in ...s (... documents/s, mean training loss ..., peak reserved GPU memory ... GiB at --max-pixels 1048576)
saved the LoRA adapter and processor to outputs/qwen-docvqa-smoke
```

The first step is warm-up, so its logged learning rate is zero. The logged
`epoch` tracks progress towards `max_steps`, not passes over the repeating
data stream. The output directory holds the LoRA adapter and processor files.

This short run checks the training pipeline. The script does not evaluate
model quality or use the validation split. It saves only at the end and has
no resume option, so an interrupted run must start again.

### What the example does

- Lists `train/part-*.parquet` and requires at least one shard per process.
- Randomizes shard order with the seed, repeats the stream until `--max-steps`,
  and decodes page images. Repeating lets gradient accumulation continue
  across passes over a small dataset.
- Loads the pinned model in 4-bit NF4, adds rank-16 LoRA to the language model, and
  freezes the vision tower.
- Trains with completion-only loss, gradient checkpointing, and a cosine
  schedule with 5% warmup, then saves the adapter and processor.

TRL selects its vision-language collator from the `images` column. To use
your own corpus with this example, publish these columns or adapt the loader
and TRL input schema:

```text
images: list<binary>                          # encoded JPEG or PNG pages
prompt: [{"role": "user", "content": "..."}]
completion: [{"role": "assistant", "content": "..."}]
chat_template_kwargs: {"enable_thinking": false}
```

## Operating the mount

### Mount options

| Flag | Why |
| --- | --- |
| `--read-only` | Prevent writes through the dataset mount. Save training outputs elsewhere. |
| `--prefix datasets/docmatix-demo/` | Expose only the dataset prefix. Include the trailing `/`. |
| `--endpoint-url` / `--region` | Select the bucket's regional Nebius endpoint. |
| `--force-path-style` | Use path-style addressing. Nebius also supports the default virtual-hosted addressing. |
| `--maximum-throughput-gbps 10000 --max-threads 64` | Nebius's documented client tuning settings. They do not guarantee throughput or bypass storage-class limits. |
| `--metadata-ttl 120` | Cache file lookups and sizes for two minutes. Use `indefinite` only for an immutable dataset. |
| `--memory-target 4096` | Target 4 GiB of memory use; this is neither a hard cap nor an upfront allocation. See [memory sizing](#size-the-memory-budget). |
| `--cache /local/nvme/mountpoint` | Optionally cache object data on node-local disk for later epochs. |
| `--allow-other` | Allow other users to read the mount. Non-root mounts also need `user_allow_other` in `/etc/fuse.conf`. |

### Size the memory budget

Mountpoint 1.24 defaults to a target of 95% of total or cgroup-limited memory.
Leave room for model loading, DataLoader workers, and other processes. Start
with the example's 4 GiB target and adjust after measuring throughput and
memory use. The minimum accepted target is 512 MiB.

On Linux, inspect host memory and the current process's cgroup membership:

```bash
awk '/^MemTotal:/ {print}' /proc/meminfo
cat /proc/self/cgroup
```

For cgroup v2, inspect `memory.max` in that cgroup and its ancestors, resolved
relative to the host's cgroup mount. `max` means no limit at that level; a
parent may still impose one. Reading only `/sys/fs/cgroup/memory.max` can miss
a nested Slurm job limit. Cgroup v1 uses `memory.limit_in_bytes`.

Use the smallest applicable limit. If a container hides the ancestors, ask
the operator for the effective allocation. CSI-managed Mountpoint pods have
their own memory allocations.

### Wrap a job

On a Compute VM, wrap a command so ordinary exits and catchable cancellation
signals clean up the mount. This Bash example uses Linux `setsid` (from
`util-linux`) to keep the trainer and its workers in a process group. Replace
`python your_train.py --data "$MOUNT_DIR"` with your training command,
retaining `setsid` and the background launch. For the bundled example, use
the command from [Step 3](#step-3-train-from-the-mount-optional-qwen38-27b-example).
Start from an unmounted directory, after completing Step 1's directory setup:

```bash
#!/usr/bin/env bash
set -euo pipefail
MOUNT_DIR=/mnt/docvqa
trainer_pid=
mkdir -p "$MOUNT_DIR"
mountpoint -q "$MOUNT_DIR" && { echo "$MOUNT_DIR is already mounted" >&2; exit 1; }

mount-s3 YOUR_BUCKET "$MOUNT_DIR" --prefix datasets/docmatix-demo/ --read-only \
  --region eu-north1 --endpoint-url https://storage.eu-north1.nebius.cloud:443 \
  --force-path-style --maximum-throughput-gbps 10000 --max-threads 64 \
  --metadata-ttl 120 --memory-target 4096
cleanup() {
  status=$?
  trap '' INT TERM HUP
  if [[ -n "$trainer_pid" ]]; then
    kill -TERM -- "-$trainer_pid" 2>/dev/null || kill -TERM "$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid" 2>/dev/null || true
  fi
  fusermount3 -u "$MOUNT_DIR" || fusermount3 -uz "$MOUNT_DIR"
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

setsid python your_train.py --data "$MOUNT_DIR" &
trainer_pid=$!
wait "$trainer_pid"
```

The wrapper preserves the trainer's exit code. On cancellation it sends
`SIGTERM` to the trainer's process group, waits for the trainer to exit, and
then unmounts. The trainer must terminate its workers before exiting. The
`-uz` fallback detaches a busy or stale mount. No shell trap runs after
`SIGKILL`, an OOM kill of the wrapper, or VM loss; use
[Unmount and recover](#unmount-and-recover) if the host survives. A trainer
that ignores `SIGTERM` needs a supervisor with a shutdown deadline.

Run this wrapper on each participating VM around that node's launcher. On
Soperator, use the [CSI setup](SOPERATOR.md); a mount created outside an
`srun` step may not be visible inside its private mount namespace.

With `--foreground`, `mount-s3` stays attached to the launching process.
A supervisor must run it in the background, wait for the mount to be ready,
and stop waiting if the daemon exits or a startup deadline expires.

### Unmount and recover

```bash
fusermount3 -u /mnt/docvqa      # normal unmount, no root needed
fusermount3 -uz /mnt/docvqa     # lazy: detaches a stale or busy mount
```

If `mount-s3` dies, the mount can remain listed in `/proc/mounts` while reads
fail with "Transport endpoint is not connected". Detach it with
`fusermount3 -uz`, then mount again. Without `fuse3`, use `sudo umount -l`.
Unmount as the user who created the mount.

To investigate failures, create a log directory and remount with
`--log-directory "$HOME/mountpoint-logs" --debug`. Otherwise Mountpoint logs
to syslog.

## Scaling out

Measure time spent waiting for data before tuning storage. Image decoding,
tokenization, or model computation may dominate the training step.

### One mount per node

A FUSE mount is local to the host. Make the same path available on every
node before training starts. For mounts managed by the job, use
[Wrap a job](#wrap-a-job) to mount at startup and unmount at exit, including
after a node replacement. On Soperator, let the CSI driver manage the mount.

### How shards reach processes

Give each rank a disjoint slice of the file list and each worker a slice of
that. Let one layer own the split. For custom iterable loaders:

| Framework | Rank and process count |
| --- | --- |
| Plain PyTorch | Use the rank and world size from `torch.distributed`. |
| PyTorch Lightning | Use `trainer.global_rank` and `trainer.world_size`; Lightning leaves `IterableDataset` sharding to you. |
| JAX | Use `jax.process_index()` and `jax.process_count()`. |

**Hugging Face Datasets with Accelerate 1.14.** Accelerate splits files when
the input is a Datasets `IterableDataset`, there are at least as many shards
as processes, and both `dispatch_batches` and `split_batches` are `False`.
The bundled example meets these conditions.

With dispatch enabled, process 0 reads the entire stream and broadcasts
batches. With dispatch disabled but too few shards, each process reads the
entire stream and keeps every n-th batch. Configure your trainer as follows:

- TRL's `SFTTrainer` turns dispatch off for iterable datasets itself.
- Plain `Trainer` leaves dispatch on for iterable datasets; pass
  `TrainingArguments(accelerator_config={"dispatch_batches": False, "split_batches": False})`.
- A hand-written Accelerate loop uses
  `Accelerator(dataloader_config=DataLoaderConfiguration(dispatch_batches=False, split_batches=False))`.

Alternatively, split manually with `datasets.distributed.split_dataset_by_node`
and leave the dataloader out of `accelerator.prepare()`. This function splits
by file only when the file count is a multiple of the world size; otherwise
every rank reads every file.

The example refuses to start with fewer files than processes. Inside a
process, `DataLoader` workers split the process's files again, so a file count
below processes times workers leaves workers idle.

Use nonempty shards with similar row counts as well as similar byte sizes.
Each rank and worker repeats its assigned shards independently in this example;
smaller partitions therefore replay their documents more often. Equal file
counts alone do not guarantee uniform sampling. These details are specific to
the pinned library versions; check shard assignment again when upgrading.

In `datasets` 5.0, `IterableDataset.shuffle()` reports a single shard,
regardless of buffer size. This prevents Accelerate from assigning disjoint
files to multiple processes. TRL's `shuffle_dataset=True` calls that method,
so the example explicitly keeps the TRL 1.10 default, `False`, and randomizes
the file list instead. Randomize rows within shards when publishing them.

### WebDataset and tokenized shards

WebDataset assigns shards itself. Publish enough shards for every DataLoader
worker in every process, or one per process if there are no workers:

```python
import webdataset as wds

dataset = wds.WebDataset(
    "/mnt/data/shards/shard-{000000..000255}.tar",
    nodesplitter=wds.split_by_node,  # one slice of the shard list per rank
    shardshuffle=100,  # shuffles shard order, not rows
).decode("pil")
```

`split_by_worker` is applied by default inside each process. With
`resampled=True` every rank samples shards with replacement instead, which
removes the divisibility constraint at the cost of exact epochs.

Exactly one layer must own the split across ranks. A WebDataset pipeline is a
plain PyTorch `IterableDataset`, so if its `DataLoader` goes through
`accelerator.prepare()`, or through `Trainer` and TRL, Accelerate wraps it in
another sampling or dispatch layer, which can skip data or centralize reading
on rank 0. Keep the WebDataset loader out of `accelerator.prepare()`; prepare
the model and optimizer, and move batches to the device yourself.

If Accelerate must own the split instead,
set `nodesplitter=None` explicitly: omitting it selects WebDataset's
`single_node_only` default, which rejects distributed use. That approach gives
up disjoint file reads. With `Trainer`, retaining WebDataset's own split
requires overriding `get_train_dataloader`. The bundled SFT example expects
a Hugging Face dataset.

For tokenized shards (`.bin`, `.npy`), assign separate files to each process
and read them sequentially. If your loader uses random offsets into one huge
file through `np.memmap`, stage that file locally or measure performance on
your data before adopting the mount.

### Shard size

Aim for hundreds of MiB per object to amortize request and first-byte latency.
Keep enough shards for all processes and workers, with similar row counts
and byte sizes to balance their work.

### Storage class

Nebius documents a per-tenant, per-region download throughput limit for
Standard and Intelligent, shared by jobs in that tenant and region.
Enhanced Throughput has no such limit. See the
[Object Storage classes](https://docs.nebius.com/object-storage/storage-classes)
page for current limits, regional availability, and pricing. Compare your
jobs' aggregate read rate with the throughput limits and account for request
costs. Achieved throughput also depends on client resources, the network,
and access patterns.

### Several epochs

Without `--cache`, later passes fetch data that is no longer in memory from
the bucket again. Add `--cache <node-local directory>` to cache object data
on disk. This changes the default metadata TTL to one minute unless you set
it explicitly, as the example does.

Use a separate cache directory per mount. Mountpoint clears it at startup
and normal exit, so it cannot be reused across remounts. For many epochs over
a dataset that fits local NVMe, consider staging with `aws s3 sync`. The
[Nebius storage guide](https://nebius.com/blog/posts/choosing-storage-for-deep-learning)
recommends local NVMe for single-node fine-tuning and a shared filesystem for
multi-node jobs.

### Small files

Millions of small objects add request and listing overhead.
`torchvision.datasets.ImageFolder` also walks the full directory tree before
reading a sample. Pack small files into WebDataset tar or Parquet shards
using the layout your loader expects.

## Where this runs

| Environment | Mount setup |
| --- | --- |
| Compute VM | Run `mount-s3` on the host as in [Step 1](#step-1-mount-the-bucket). |
| Container on a Compute VM | Mount on the host, then bind-mount the directory into the container. See below. |
| Managed Kubernetes | Use the [Mountpoint CSI driver setup](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3) to expose the bucket as a PersistentVolume with `allow-other`. |
| Soperator (Slurm) | Attach the CSI volume to the jail on login and worker nodes. See the [Soperator setup guide](SOPERATOR.md). |

For a container, bind-mount the host directory with
`docker run -v /mnt/docvqa:/mnt/docvqa:ro ...`. If the container runs as a
different user, add `--allow-other` when mounting on the host. A non-root
mount also needs `user_allow_other` in `/etc/fuse.conf`; FUSE otherwise denies
access to other users, including root.

Running `mount-s3` inside the container requires `--device /dev/fuse`,
`--cap-add SYS_ADMIN`, and `fuse3` in the image. If your platform cannot
provide FUSE, see [in-process streaming](SOPERATOR.md#fallback-no-fuse).

## Limitations and gotchas

- **Shuffle before publishing.** The example randomizes shard order, but reads
  rows within each shard in order; see [shard assignment](#how-shards-reach-processes).
- **Add checkpointing for longer runs.** The example has no recovery support.
  Integrate checkpoints and decide how to restore or replay the data stream. The
  [Slurm checkpointing recipe](../slurm/object-storage-checkpointing/) demonstrates
  a separate DCP implementation; it is not integrated with this trainer.
- **Read-only by design.** Write checkpoints and outputs elsewhere; Mountpoint
  supports only sequential single-writer uploads and no renames or locks.
- **Metadata caching relaxes consistency.** Directory listings are not cached,
  so a newly uploaded shard shows up in `ls` at once, but a path whose lookup
  was cached before the object changed stays stale until the TTL expires, or
  until you remount when the TTL is `indefinite`.
- **Transport errors surface as I/O errors.** When Mountpoint exhausts its
  retries the read fails with `EIO` and the worker raises. Rely on checkpoint
  cadence, not on retry code in the loader.
- **One slow node slows the training step.** In distributed training,
  watch per-rank data-wait time when a run slows down.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `mount-s3` fails at start with an access or signature error | Credentials are exported in this shell; `--region eu-north1`; `--endpoint-url` ends with `:443`; toggle `--force-path-style`. |
| The mount is empty or the loader finds no `train/part-*.parquet` | The `--prefix` value ends with `/` and matches the object keys; `ls` the mount and compare with `aws s3 ls`. |
| `fusermount3: command not found` or `/dev/fuse: No such file` | Install `fuse3`; on a managed cluster ask the platform team (see [Where this runs](#where-this-runs)). In a Soperator jail, `apt-get install fuse3` as root adds `fusermount3`. |
| `Transport endpoint is not connected` | The `mount-s3` process died (OOM, preemption, cgroup teardown). `fusermount3 -uz <dir>`, then mount again. |
| Throughput is far below expectations | Object size (aim for hundreds of MiB), `--maximum-throughput-gbps 10000 --max-threads 64` present, enough shards for processes times workers, bucket in the same region, `dispatch_batches` off. |
| Host memory pressure after mounting | Lower `--memory-target`; the default is a large share of RAM. |
| `RuntimeError: The NVIDIA driver on your system is too old` | The example's pinned PyTorch needs a CUDA 13 driver. Use `ubuntu24.04-cuda13.0` or upgrade the driver. |
| The example refuses to start with "fewer shards than processes" | Publish at least one shard per process, or reduce the process count. |
| A file that was just uploaded cannot be opened, or reads stale metadata | A lookup of that path was cached before the object existed or changed; wait out the metadata TTL or remount. Listings themselves are never cached. |

## Files

| File | Role |
| --- | --- |
| [`train_sft.py`](train_sft.py) | QLoRA example that streams Parquet shards from a directory |
| [`prepare_docmatix.py`](prepare_docmatix.py) | Writes the sample shards locally for upload |
| [`requirements.txt`](requirements.txt) / [`requirements.lock`](requirements.lock) | Direct dependencies and hash-locked environment for Linux x86_64 and Python 3.12 |
| [`SOPERATOR.md`](SOPERATOR.md) | Cluster setup and streaming fallbacks for Soperator |

## Cleanup

Unmount with `fusermount3 -u /mnt/docvqa`. To remove the sample data, list the
exact prefix first and delete only that prefix; never target the bucket root:

```bash
aws s3 ls s3://YOUR_BUCKET/datasets/docmatix-demo/ --recursive --endpoint-url https://storage.eu-north1.nebius.cloud
aws s3 rm s3://YOUR_BUCKET/datasets/docmatix-demo/ --recursive --endpoint-url https://storage.eu-north1.nebius.cloud
```

## References

- [Mountpoint for Amazon S3 on Nebius](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3)
- [Mountpoint 1.24 configuration reference](https://github.com/awslabs/mountpoint-s3/blob/mountpoint-s3-1.24.0/doc/CONFIGURATION.md), [filesystem semantics](https://github.com/awslabs/mountpoint-s3/blob/mountpoint-s3-1.24.0/doc/SEMANTICS.md), and the [Mountpoint CSI driver](https://github.com/awslabs/mountpoint-s3-csi-driver)
- [Object Storage classes](https://docs.nebius.com/object-storage/storage-classes)
- [Choosing storage for deep learning](https://nebius.com/blog/posts/choosing-storage-for-deep-learning)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) and [Hugging Face Datasets streaming](https://huggingface.co/docs/datasets/stream)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
