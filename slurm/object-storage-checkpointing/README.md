# Running PyTorch training with persistent Object Storage checkpoints on Slurm (Soperator)

Use this recipe when training checkpoints are filling the shared Soperator jail,
or when a job interruption would otherwise lose hours of progress. Every rank
writes its checkpoint shard directly to Nebius Object Storage: checkpoint data
does not stage in, or consume capacity from, the jail.

After an interruption, the job reads one small `latest` marker and resumes from
the newest complete checkpoint. A hard kill during an upload cannot replace that
marker with an incomplete checkpoint.

## What this gives you

| Situation | Result |
| --- | --- |
| Checkpoints are too large for the shared jail | Shards are uploaded directly to Object Storage; no checkpoint directory is created in the jail |
| Slurm requeues the job after a node interruption | The same job ID resumes automatically from its latest complete checkpoint |
| You stop and resubmit a job | Pass the old prefix and resume from the same checkpoint |
| A rank is killed during an upload | The previous committed checkpoint remains safe and resumable |
| A cluster is replaced | An operator can attach the old bucket and the researcher can reuse the prefix |

The included model is intentionally small. The reusable part is the FSDP2 and
PyTorch Distributed Checkpoint (DCP) pattern: asynchronous sharded uploads, an
atomic commit marker, retention, consistent failure handling across ranks, and
signal-safe shutdown.

## Interruptions, signals, and checkpoint recovery

A signal is a short operating-system notification sent to a running process. It
does not contain checkpoint data, and receiving one does not by itself mean that
Slurm will restart the job. Signals only give the trainer a chance to react
before its processes disappear.

The signals relevant to this recipe are:

- `SIGUSR1`: an application-defined warning. This recipe asks Slurm to send it
  before the job time limit and interprets it as "finish the current safe step,
  commit a checkpoint, and exit."
- `SIGTERM`: a request to terminate cleanly. Slurm uses it when a job reaches
  its time limit or is cancelled. The trainer handles it like `SIGUSR1`, but
  there may be much less time available.
- `SIGCONT`: a wake-up signal that Slurm sends before some warnings and
  termination sequences. The trainer does not need a handler for it.
- `SIGKILL`: an immediate, uncatchable stop. No handler, cleanup callback, or
  final checkpoint can run after it.

There are therefore two recovery paths:

```text
planned stop: SIGUSR1/SIGTERM -> safe step boundary -> final commit -> exit
hard stop:    node loss/SIGKILL -> previous marker stays valid -> restart -> resume
```

### What can happen on Nebius Soperator

| Event | What the training ranks can rely on | Recommended checkpoint behavior | Restart behavior |
| --- | --- | --- | --- |
| Job approaches its Slurm time limit | Because the job requests `--signal=USR1@120`, Slurm signals all job steps approximately 120 seconds before the limit. Slurm may send the warning up to 60 seconds earlier than requested. At the limit, remaining tasks receive `SIGCONT` and `SIGTERM`, followed later by `SIGKILL` if they do not exit. | Treat `SIGUSR1` or `SIGTERM` as a request to stop collectively at the next safe step, drain any upload, and commit the current step. Measure your worst-case step and save time before choosing the warning window. | A time limit is not automatically requeued by `--requeue` alone; site policy may differ. Resubmit with the same prefix if needed. |
| A preemptible Nebius worker VM is stopped | Nebius Compute gives the VM a 60-second termination notice, but that is a VM-level lifecycle event, not a guaranteed Slurm signal to every training rank. In Soperator, assume no usable in-band warning unless your platform team has explicitly validated propagation to the ranks. | Depend on periodic committed checkpoints, not a final signal-triggered save. A partial upload cannot replace `latest`. | `--requeue` makes the job eligible to return to the queue when Slurm records the node failure. The job can resume only after the platform has restored enough worker capacity. |
| Hardware, network, Kubernetes-node, or VM failure | Usually no warning. One rank may disappear while the others report NCCL/Gloo errors or stall until Slurm detects the failed node. | Use the same hard-stop path as preemption: retain the previous marker and resume from it. | Requeue depends on how Slurm classifies the failure and on site policy. Node repair or replacement remains a platform responsibility. |
| [`scancel <job-id>`](https://slurm.schedmd.com/scancel.html) | Slurm sends `SIGCONT` and `SIGTERM`, waits the site `KillWait`, then sends `SIGKILL` if tasks remain. | The trainer attempts a final checkpoint, but cancellation is not a generous warning mechanism. For a deliberate graceful stop, prefer `scancel --signal=USR1 <job-id>` and wait for the committed-checkpoint log line. | `scancel` does not requeue the job. Resubmit with the old explicit prefix to continue. |
| [`scontrol requeue <job-id>`](https://slurm.schedmd.com/scontrol.html#OPT_requeue) | Slurm ends the current allocation and returns the same job ID to the queue. Do not assume the termination window is long enough for a new checkpoint. | The final signal save is useful when it completes, but correctness still comes from the latest checkpoint committed before requeue. | Yes. The default prefix contains the unchanged job ID, so the restarted job resumes automatically. |

[Nebius documents](https://docs.nebius.com/compute/virtual-machines/preemptible)
a 60-second `SIGTERM` before a preemptible VM is stopped, and then a forced stop
if it does not exit. That notice protects VM shutdown; it must not be confused
with Slurm's delivery of a signal to every `srun` task. For distributed
training, the safe assumption is that preemption and node loss are hard stops.

Slurm's [`--signal=USR1@120`](https://slurm.schedmd.com/sbatch.html#OPT_signal)
is different: it is an explicit scheduler warning requested by this job. The
directive intentionally has no `B:` prefix, because `B:` would signal only the
batch shell; without it, Slurm signals the job steps that contain the training
ranks. Slurm documents its full
[job-termination sequence](https://slurm.schedmd.com/job_launch.html#job_termination)
separately.

### How the trainer handles a catchable signal

The signal handler only sets a flag. It does not call PyTorch, Object Storage,
or a distributed collective while the signal is executing asynchronously.
Then, at each training-step boundary:

1. all ranks combine their stop flags on a dedicated control process group
2. all ranks leave the training loop together
3. any in-flight asynchronous checkpoint is allowed to finish
4. if that checkpoint does not already cover the current step, all ranks write
   one synchronous final checkpoint
5. rank zero advances `latest` only after the checkpoint is complete

This collective decision is essential. If only the rank that received the
signal exits, its peers can deadlock in the next training or checkpoint
collective.

### What to do after an interruption

First inspect the end of the job log. If it contains
`SIGUSR1 observed at a safe step boundary` or
`SIGTERM observed at a safe step boundary`, followed by `checkpoint committed`,
the graceful path completed. If those lines are absent, assume a hard stop and
use the previous committed marker.

Check Slurm's view of the event and restart count:

```bash
sacct -j <job-id> -X \
  --format=JobID,State,ExitCode,Restarts,Elapsed,NodeList
```

- If the same job returns to `PENDING` or `RUNNING`, `--requeue` is handling the
  restart and the default prefix will resume automatically.
- If the job is terminal, such as `CANCELLED` or `TIMEOUT`, submit a new job with
  the prior prefix:

  ```bash
  TRAIN_ARGS="--prefix checkpointing-demo-<old-job-id>" sbatch checkpoint_train.sh
  ```

The practical rule is simple: use periodic checkpoints for failures that give
no warning, and use signals only to reduce lost work when Slurm offers a planned
shutdown window.

## Prerequisites

Before you start, make sure you have:

- access to a [Nebius Soperator cluster](https://nebius.com/services/soperator)
  with two GPU worker nodes
- this repository cloned to the shared filesystem
- Linux and Python 3.10 or newer with `venv`, `curl`, `tar`, `sha256sum`, and
  `flock` on the login node
- a readable `/etc/nebius-checkpoints.env` prepared by the platform operator

Standard Soperator 4.1.x jail images include these command-line tools. The setup
script checks them explicitly so that older or customized images fail with a
clear request for the platform operator.

Check the platform handoff without displaying any credentials:

```bash
if test -r /etc/nebius-checkpoints.env; then
  echo "Object Storage checkpointing is ready"
else
  echo "Ask the platform operator to enable Object Storage checkpointing"
fi
```

If the file is missing or unreadable, send the operator this request:

> Enable Soperator Object Storage checkpointing for this cluster and make
> `/etc/nebius-checkpoints.env` readable by my Slurm user.

You do not need to create a bucket, access key, or IAM policy. Those platform
details are collected in
[For platform operators](#for-platform-operators).

## Steps

### Set up the environment

From the shared filesystem on the login node:

```bash
cd ml-cookbook/slurm/object-storage-checkpointing
bash setup.sh
```

The setup script:

1. takes an exclusive setup lock and creates a shared `.env` virtual environment
2. installs the pinned pip version and exact versions from `requirements.txt`
3. downloads checksum-verified `s5cmd` 2.3.0 into `bin/`
4. creates `outputs/` for Slurm logs
5. verifies create, read, and delete access to the checkpoint bucket

A successful setup ends with output similar to:

```text
Nebius Object Storage access OK: bucket=<bucket> endpoint=<endpoint>
torch 2.13.0 / s3torchconnector DCP import OK
Setup done.
```

### Examine and configure the job

`checkpoint_train.sh` requests two nodes with eight GPUs per node, runs one task
per GPU, and has a 30-minute time limit. It also enables:

- `--requeue` for scheduler-driven node failure or preemption recovery
- `--open-mode=append` so logs survive a requeue
- `--signal=USR1@120` so ranks can commit a final checkpoint before a time limit

Trainer arguments can be passed through `TRAIN_ARGS`. Slurm resource settings can
be overridden on the `sbatch` command line.

Useful trainer options include:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--save-every-seconds` | `120` | Time-based checkpoint cadence |
| `--save-every-steps` | unset | Step-based cadence; overrides time-based cadence |
| `--keep-last` | `3` | Number of committed checkpoints to retain; `0` keeps all committed checkpoints, while stale uncommitted steps are still removed |
| `--prefix` | job name and job ID | Object key namespace used for resume |
| `--steps` | `10000000` | Maximum training steps |

Run `.env/bin/python train_fsdp.py --help` for the full argument list.

### Submit the job

For a small cross-node demonstration, use one GPU on each of two nodes and
checkpoint every 60 seconds:

```bash
TRAIN_ARGS="--save-every-seconds 60 --keep-last 2" \
  sbatch --gpus-per-node=1 --ntasks-per-node=1 checkpoint_train.sh
```

For the default two-node, eight-GPU-per-node allocation, run:

```bash
sbatch checkpoint_train.sh
```

The default prefix is `<job-name>-<job-id>`. It isolates independent submissions
while remaining unchanged when Slurm requeues the same job.

### Monitor the job

Check the queue:

```bash
squeue --me
```

Follow the combined stdout/stderr log:

```bash
tail -f outputs/checkpointing-demo-<job-id>.out
```

Inspect checkpoint objects:

```bash
source /etc/nebius-checkpoints.env
bin/s5cmd --endpoint-url "$NEBIUS_OBJECT_STORAGE_ENDPOINT" \
  ls "s3://$NEBIUS_CHECKPOINT_BUCKET/*"
```

### Expected output

Exact steps and timings depend on the GPU type and Object Storage throughput.
A healthy run shows the initial state, asynchronous staging time, upload
completion, and marker commit:

```text
[12:00:00] world=16 params=...M (FSDP2-sharded) checkpoint~...GB bucket=... prefix=checkpointing-demo-123
[12:00:00] no checkpoint found, starting fresh
[12:02:01] step 42 loss ...: async_save blocked training for ...ms
[12:02:05] checkpoint step 42: upload finished in ...s, marker committed
```

On a requeue or explicit-prefix resubmission, it instead reports:

```text
resuming from checkpoint step 42
```

### Try an interruption and resume

Wait until the log says `marker committed`, then requeue the job:

```bash
scontrol requeue <job-id>
```

The job keeps the same ID and prefix. Slurm allocates nodes again, the log is
appended instead of truncated, and the restarted ranks print:

```text
resuming from checkpoint step <step>
```

There is still no model-checkpoint directory in the jail. The committed shards
are under
`s3://<bucket>/checkpointing-demo-<job-id>/step-<step>/` in Object Storage.

## Verify checkpoint recovery

After setup, run the automated verifier from the login node:

```bash
bash verify.sh
```

The verifier submits the real job with one GPU on each of two nodes and checks:

1. a complete checkpoint is committed
2. a hard `SIGKILL` cannot advance `latest` to an incomplete upload
3. a new submission with the same prefix resumes and commits more progress
4. `SIGUSR1` causes a final checkpoint before exit

Each phase prints `PASS` or `FAIL`. The script exits non-zero on failure and
cleans its unique `verify-*` prefix, including incomplete multipart uploads.
Typical runtime is 5–10 minutes after both nodes are allocated; queue time is
additional. Override queue and phase deadlines with `VERIFY_START_DEADLINE`,
`VERIFY_COMMIT_DEADLINE`, and `VERIFY_RESUME_DEADLINE` if needed.

Automatic Slurm requeue, real VM preemption, node replacement, and cross-cluster
recovery remain manual tests because they depend on cluster policy and operator
actions.

## How checkpointing works

### Sharded asynchronous saves

The model is wrapped with FSDP2 `fully_shard`. DCP therefore gives every rank only
its model and optimizer shard to upload. `dcp.async_save` blocks training for
GPU-to-host staging, while Object Storage transfer continues in a background
thread.

Only one save is allowed in flight. Before another save begins, all ranks confirm
that the previous upload and marker commit succeeded. A checkpoint upload or
marker failure on one rank is converted into a fatal error on every rank instead
of leaving peers stuck in a later collective.

### Atomic commit marker

Nebius Object Storage does not provide an atomic directory rename. The recipe
uses this protocol instead:

1. write a new checkpoint below `<prefix>/step-<step>/`
2. wait for DCP to finish every shard and the `.metadata` object
3. overwrite the small `<prefix>/latest` object with the committed step

Resume reads only `latest`. It never guesses the newest checkpoint by listing
step directories. If marker access fails because of authentication, networking,
or throttling, the job fails instead of silently starting from step zero.

A hard kill during upload can leave objects or multipart uploads under a newer
step, but `latest` continues to identify the previous complete checkpoint.
Cleanup removes these stale step directories after the next successful commit,
including when `--keep-last=0`. Retention runs only after `latest` is committed;
if cleanup fails, the committed checkpoint remains valid, the job logs a warning,
and cleanup is retried after the next commit.

### Prefixes and resume behavior

A prefix is a single-writer namespace. Never run two active jobs with the same
explicit prefix because both would update `latest` and apply retention.

- Slurm requeue: the default prefix is stable because the job ID is unchanged.
- Separate submission: pass the earlier prefix explicitly.
- Replacement cluster: reuse the bucket through the Soperator infrastructure,
  then pass the earlier prefix explicitly.

For example:

```bash
TRAIN_ARGS="--prefix checkpointing-demo-123" sbatch checkpoint_train.sh
```

DCP can reshard a checkpoint when the new job uses a different rank count.
The model and optimizer definitions must remain checkpoint-compatible.

## Apply the protocol to another workload

A production trainer does not need to copy the toy model, but it should preserve
these invariants:

- shard checkpoint state rather than gathering it on rank zero
- serialize saves or otherwise prevent overlapping commits
- publish the marker only after every checkpoint object is complete
- resume only from the marker
- treat marker read errors as fatal
- make checkpoint upload and marker failures fatal on all ranks
- treat retention cleanup as retryable housekeeping after a successful commit
- coordinate stop decisions at safe collective boundaries
- keep a prefix single-writer
- measure staging and upload time before choosing a cadence

Frameworks without direct Object Storage support can save a completed checkpoint
to local or shared storage, upload that completed directory, and advance the
marker only after the upload succeeds.

## Sizing and cadence

- Checkpoint size is approximately bytes per parameter times parameter count.
  FP32 weights plus two FP32 AdamW moments are roughly 12 bytes per parameter.
- Bucket capacity should cover `keep_last × checkpoint size` plus any separately
  retained milestone checkpoints.
- The log reports both training-blocking staging time and total upload time, and
  reports the complete duration of a final synchronous save through marker commit.
- A starting interval can use the Young/Daly approximation
  `sqrt(2 × checkpoint blocking time × mean time between failures)`.
- The background upload must finish before the next interval. Otherwise host
  memory and pending-upload wait time can erase the benefit of asynchronous saves.
- The Slurm warning window must cover the worst-case training step, any in-flight
  upload drain, and the final synchronous save.

## Cleanup

`verify.sh` cleans only its own unique prefix. To remove a demo prefix manually,
first confirm the exact prefix and then run:

```bash
source /etc/nebius-checkpoints.env
PREFIX="checkpointing-demo-<job-id>"
bin/s5cmd --endpoint-url "$NEBIUS_OBJECT_STORAGE_ENDPOINT" \
  rm "s3://$NEBIUS_CHECKPOINT_BUCKET/$PREFIX/*"
```

This removes objects under that prefix; it does not delete the bucket. Interrupted
uploads may also leave incomplete multipart uploads, which require an
S3-compatible multipart-abort operation. Bucket retention, destruction, and
cross-cluster reuse belong to the linked Soperator infrastructure workflow.

## Troubleshooting

- `/etc/nebius-checkpoints.env` is missing or unreadable: confirm checkpoint
  storage is enabled in the Soperator installation, its credential-rendering job
  completed, and the Slurm user has read access.
- Setup reports an Object Storage permission error: verify the service account has
  create, read, and delete access on the configured bucket.
- The job refuses to start fresh after a marker error: this is intentional. Fix
  credentials, endpoint, DNS, or transport access and resubmit with the same prefix.
- A job hangs after only one rank receives a signal: make sure the workload keeps
  the dedicated control process group and collective stop decision.
- Resume finds incompatible state: use the same model and optimizer definitions,
  or start a new prefix after an intentional incompatible change.
- Output files cannot be opened: run `bash setup.sh` first so `outputs/` exists
  before Slurm opens the log file.

## For platform operators

Researchers can skip this section once the readiness check passes.

The platform-side reference is
[nebius-solutions-library PR #1074](https://github.com/nebius/nebius-solutions-library/pull/1074),
with an
[immutable implementation commit](https://github.com/nebius/nebius-solutions-library/commit/56d3a164c31956cc134dbfd98f720d2e5dcc220f).
Enable `checkpoint_storage_enabled = true` in the Soperator installation. The
platform workflow provisions or attaches a bucket, creates least-privilege
workload access, and renders `/etc/nebius-checkpoints.env` into the jail.

The environment file must provide:

| Variable | Purpose |
| --- | --- |
| `NEBIUS_CHECKPOINT_BUCKET` | Object Storage bucket name |
| `NEBIUS_OBJECT_STORAGE_ENDPOINT` | Regional Object Storage endpoint |
| `NEBIUS_OBJECT_STORAGE_REGION` | Signing region |
| `AWS_ACCESS_KEY_ID` | S3-compatible SDK credential |
| `AWS_SECRET_ACCESS_KEY` | S3-compatible SDK credential |

The trainer maps `NEBIUS_OBJECT_STORAGE_ENDPOINT` to the `AWS_ENDPOINT_URL`
compatibility variable before constructing the DCP reader or writer. The
reference infrastructure also exports `AWS_ENDPOINT_URL` and `AWS_REGION` for
other S3-compatible tools. The target remains Nebius Object Storage. Do not log
the file. The infrastructure defaults to owner `root:root` and mode `600`;
configure a numeric user/group and suitable mode when researchers submit as
non-root users.

Bucket retention, safe installation teardown, and attaching the bucket to a
replacement cluster remain platform responsibilities. Existing buckets can be
reused so researchers only need the old checkpoint prefix to resume.

## Version notes

The versions in `requirements.txt` were validated together on Soperator 4.1.3 and
4.1.4 H100 clusters in the reference implementation.

`s3torchconnector[dcp]`, `S3StorageReader`, `S3StorageWriter`,
`S3ClientConfig`, `boto3`, `s5cmd`, `AWS_*`, and `s3://` retain their upstream
S3-compatible API names. They are configured with the Nebius Object Storage
endpoint, region, bucket, and credentials.

`dcp.async_save` needs a CPU process-group backend alongside NCCL. The trainer
initializes the default group with `cpu:gloo,cuda:nccl` and uses a separate Gloo
group for main-thread control collectives.
