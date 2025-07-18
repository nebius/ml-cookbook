# Running fault-tolerant training with `torchft`
This document provides a step-by-step guide to launching a fault tolerant training job for the Llama-3.1-8B model leveraging [`torchft`](https://github.com/pytorch/torchft) integration with the [`torchtitan`](https://github.com/pytorch/torchtitan) framework on Slurm (Soperator) cluster. 

## Prerequisites

Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator).

## Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Sopeartor has `/` mounted as a shared filesystem).

### Install environment with `torchtitan` and `torchft`

*Note: `torchft` functionality is experimental so you will have to use nightly builds.*

1. Create a virtual environment and activate it with `python3 -m venv .env`;
2. Install [nightly build of `torch`](https://pytorch.org/get-started/locally/);
3. Follow [these instructions](https://github.com/pytorch/torchtitan#from-source) to install the latest `torchtitan` from source into the current directory;
4. Install `torchft` by [following the documentation in GitHub repository](https://github.com/pytorch/torchft#installing-from-pypi).

After completing these steps, you will need to [download the Llama-3.1 tokenizer](https://github.com/pytorch/torchtitan#downloading-a-tokenizer) (requires access to Llama model repository on Hugging Face).

Optionally, export your WANDB API key to enable logging to Weigh and Biases:
```
export WANDB_API_KEY=<your_wandb_api_key_here>
```


### [Optional] Launch classic disributed training with `torchtitan`:

Here we are starting a job on 2 nodes with 8 GPUs each, adjust `N` for your node count:

```
sbatch -N 2 llama3.1-8b-pretraining.sh
```

*Note: this job will run for 1000 steps which may take >20 minutes on 2 nodes of H100 GPUs. Use `scancel` to abort the job if needed.*

### Launch `torchft` training

We are going to reuse the same training config as for the classic distributed training job, but now we will use `torchft` to make the training more resilient to node failures. Please see the [`torchft` documentation](https://docs.pytorch.org/torchft/) for integrating it in your workloads.

#### Start the `Lighthouse` service

Before launching `torchft` training, you need to start the `Lighthouse` service (in this example we are using the login node. If you want to use a worker node or an external server, alter the `TORCHFT_LIGHTHOUSE` env variable accordingly):

```
source .env/bin/activate
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

#### Start the training job

We changed the job script (`llama3.1-8b-pretraining-torchft.sh`) to launch an array of separate, smaller pytorch jobs (1 per node) to make it easier to requeue failed jobs. To start the training job, launch the `torchft` training job (alter the size of array to match the number of nodes you want to use (`N-1`)):

```
sbatch --array=0-1 llama3.1-8b-pretraining-torchft.sh
```

We can see that this creates an array of 2 jobs, each running on one node:

```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              53_1      main tft_repl ckondrat  R       1:34      1 worker-0
              53_0      main tft_repl ckondrat  R       2:05      1 worker-1
```
Once both replicas are up, we can see the corresponding `Quorum` messages in the logs of `Lighthouse` service:
```
2025-07-16T13:48:29.913 [INFO] [torchft::lighthouse] - Quorum! Quorum { quorum_id: 2, participants: [QuorumMember { replica_id: "torchtitan_ft_0:6cc4c926-56d9-4bb9-96f6-544803cae49c", address: "http://worker-1:41093", store_address: "worker-1.soperator-worker-svc.soperator.svc.cluster.local:33745", step: 20, world_size: 8, shrink_only: false, commit_failures: 0, data: "" }, QuorumMember { replica_id: "torchtitan_ft_1:fc5fbe3f-6287-4d62-9b18-aa2f5ad85c4e", address: "http://worker-0:40269", store_address: "worker-0.soperator-worker-svc.soperator.svc.cluster.local:43199", step: 20, world_size: 8, shrink_only: false, commit_failures: 0, data: "" }], created: Some(Timestamp { seconds: 1752673709, nanos: 913262125 }) }
```

#### Simulating node failure

Let's bring down one worker node to simulate a node failure:
```
sudo scontrol update NodeName=worker-0 State=DOWN Reason="manual maintenance"
```
This is enough to knock out one of the replicas and trigger job requeuing:

```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              53_1      main tft_repl ckondrat PD       0:00      1 (BeginTime)
              53_0      main tft_repl ckondrat  R       2:46      1 worker-1
```

The job `53_1` is now in the pending state due to the "failure" of `worker-0`. The remaining replica `53_0` continues to run. 

Logs from `53_0` show that an error occured on step 21:
```
[titan] 2025-07-16 13:49:32,462 - torchft.manager - INFO - [torchtitan_ft_0:6cc4c926-56d9-4bb9-96f6-544803cae49c/0 - step 21] should_commit=False enough_replicas=True, errored=aborted
NoneType: None
```

`Lighthouse` logs reflect the `Quorum` change due to one replica failure:
```
2025-07-16T13:49:41.512 [INFO] [torchft::lighthouse] - Detected quorum change, bumping quorum_id to 3
2025-07-16T13:49:41.512 [INFO] [torchft::lighthouse] - Quorum! Quorum { quorum_id: 3, participants: [QuorumMember { replica_id: "torchtitan_ft_0:6cc4c926-56d9-4bb9-96f6-544803cae49c", address: "http://worker-1:41093", store_address: "worker-1.soperator-worker-svc.soperator.svc.cluster.local:33745", step: 21, world_size: 8, shrink_only: false, commit_failures: 1, data: "" }], created: Some(Timestamp { seconds: 1752673781, nanos: 512580203 }) }
```

Let's bring the node back up to see if we can recover the original training setup:
```
sudo scontrol update NodeName=worker-0 State=RESUME
```

It will take a minute or two for the scheduler to requeue the job and for the replica to start:
```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              53_1      main tft_repl ckondrat  R       1:02      1 worker-0
              53_0      main tft_repl ckondrat  R       5:59      1 worker-1
$ tail ./outputs/torchft/tft-53/tft-53_1.out 
[2025-07-16 13:51:42]  launching replica 1 (attempt 1) on worker-0
```

This replica will catch up with the rest of the training loop and it will be included in the `Quorum`:
```
2025-07-16T13:52:24.923 [INFO] [torchft::lighthouse] - Detected quorum change, bumping quorum_id to 4
2025-07-16T13:52:24.923 [INFO] [torchft::lighthouse] - Quorum! Quorum { quorum_id: 4, participants: [QuorumMember { replica_id: "torchtitan_ft_0:6cc4c926-56d9-4bb9-96f6-544803cae49c", address: "http://worker-1:41093", store_address: "worker-1.soperator-worker-svc.soperator.svc.cluster.local:33745", step: 142, world_size: 8, shrink_only: false, commit_failures: 0, data: "" }, QuorumMember { replica_id: "torchtitan_ft_1:5ea849a1-5cd7-40c6-a1e3-b307716c8088", address: "http://worker-0:40525", store_address: "worker-0.soperator-worker-svc.soperator.svc.cluster.local:44595", step: 0, world_size: 8, shrink_only: false, commit_failures: 0, data: "" }], created: Some(Timestamp { seconds: 1752673944, nanos: 923882447 }) }
```

These `Lighthouse` logs indicate that the system was running on a single replica from step 21 to 142 while the Slurm job corresponding to the second replica was being requeued and restarted.

If we look into the logs of the job `53_0` which was running all this time, we can see that its model state dict is being used for recovery of the second replica:

```
[titan] 2025-07-16 13:52:26,769 - torchft.manager - INFO - [torchtitan_ft_0:6cc4c926-56d9-4bb9-96f6-544803cae49c/1 - step 142] peers need recovery from us [1]
```

After this recovery, the training proceeds as usual with both replicas participating. 

*Note: this job will run for 1000 steps which may take >20 minutes on 2 nodes of H100 GPUs. Use `scancel` to abort the job if needed.*
