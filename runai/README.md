# Run:ai examples on Nebius infrastructure

## Prerequisites

You will need to setup an integration between Run:ai platform and Nebius Managed Kubernetes. Follow [our documentation on how to set it up](https://docs.nebius.com/3p-integrations/run-ai).

## Run:ai CLI setup

You will need Run:ai CLI to submit the jobs to your cluster. To configure the CLI follow [this procedure](https://run-ai-docs.nvidia.com/self-hosted/2.21/reference/cli/install-cli).

Once the CLI is installed you will need to authenticate, run the following command:
```
runai login
```

## Steps

### [Optional] Examine the script

In this example, we are using MPI to submit a distributed job via [runai mpi submit](https://run-ai-docs.nvidia.com/self-hosted/reference/cli/runai/runai_mpi_submit) command.
We use the following arguments:
- `--image` corresponding to the container image (different CUDA version for Hopper/Blackwell platforms)
- We use `--environment` to pass environment variables to the container
- `--node-pools` corresponds to capacity node pool in which you want to run the job. You may configure it in your Run:ai console.
- `--auto-deletion-time-after-completion` may be configured to keep your job pods/logs after completetion.
- `--large-shm`, `--capability "IPC_LOCK"` are used to configure the container for NCCL test.
- `--workers`, `--gpu-devices-request` and `--slots-per-worker` are used to allocate resource to the workload.
- `--master-command` and `--master-args` are used to launch NCCL test on multiple nodes via `MPI`.
- `--command -- bash -c "/usr/sbin/sshd -De"` runs an SSH daemon in worker pods so `mpirun` can launch ranks remotely (`-D` don’t daemonize, `-e` log to stderr).

### Submit the job

To submit the MPI job, naviage to the `mpi` directory, make script executable with `chmod +x nccl-tests.sh` and launch it. You can override some options:
```
$ ./nccl-tests.sh -h                                                                
Usage: nccl-tests.sh [options]

Options:
  -p, --project   <name>   Project name (default: demo)
  -g, --gpus      <int>    GPUs per node (default: 8)
  -n, --nodes     <int>    Number of nodes / workers (default: 2)
      --platform  <str>    Platform: hopper | blackwell (default: hopper)
  -h, --help                Show this help

Examples:
  nccl-tests.sh --platform hopper -p demo -g 8 -n 2
  nccl-tests.sh --platform blackwell -p bw -g 16 -n 4
```
### Get the job logs

To visualize the logs of the job, run the following:
```
runai training mpi logs -p <project> <job-name>
```

To get the job name, you may list the jobs in the project with `runai training mpi list -p <project>`.

#### Example output:
```
$ runai training mpi logs -p demo demo-nccl-hopper-8g2n
Warning: Permanently added 'demo-nccl-test-16-h200-worker-0.demo-nccl-test-16-h200.runai-demo.svc' (ED25519) to the list of known hosts.
Warning: Permanently added 'demo-nccl-test-16-h200-worker-1.demo-nccl-test-16-h200.runai-demo.svc' (ED25519) to the list of known hosts.
# nThread 1 nGpus 1 minBytes 1073741824 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid     25 on demo-nccl-test-16-h200-worker-0 device  0 [0x8d] NVIDIA H200
#  Rank  1 Group  0 Pid     26 on demo-nccl-test-16-h200-worker-0 device  1 [0x91] NVIDIA H200
#  Rank  2 Group  0 Pid     27 on demo-nccl-test-16-h200-worker-0 device  2 [0x95] NVIDIA H200
#  Rank  3 Group  0 Pid     28 on demo-nccl-test-16-h200-worker-0 device  3 [0x99] NVIDIA H200
#  Rank  4 Group  0 Pid     29 on demo-nccl-test-16-h200-worker-0 device  4 [0xab] NVIDIA H200
#  Rank  5 Group  0 Pid     30 on demo-nccl-test-16-h200-worker-0 device  5 [0xaf] NVIDIA H200
#  Rank  6 Group  0 Pid     33 on demo-nccl-test-16-h200-worker-0 device  6 [0xb3] NVIDIA H200
#  Rank  7 Group  0 Pid     34 on demo-nccl-test-16-h200-worker-0 device  7 [0xb7] NVIDIA H200
#  Rank  8 Group  0 Pid     20 on demo-nccl-test-16-h200-worker-1 device  0 [0x8d] NVIDIA H200
#  Rank  9 Group  0 Pid     21 on demo-nccl-test-16-h200-worker-1 device  1 [0x91] NVIDIA H200
#  Rank 10 Group  0 Pid     22 on demo-nccl-test-16-h200-worker-1 device  2 [0x95] NVIDIA H200
#  Rank 11 Group  0 Pid     23 on demo-nccl-test-16-h200-worker-1 device  3 [0x99] NVIDIA H200
#  Rank 12 Group  0 Pid     24 on demo-nccl-test-16-h200-worker-1 device  4 [0xab] NVIDIA H200
#  Rank 13 Group  0 Pid     25 on demo-nccl-test-16-h200-worker-1 device  5 [0xaf] NVIDIA H200
#  Rank 14 Group  0 Pid     27 on demo-nccl-test-16-h200-worker-1 device  6 [0xb3] NVIDIA H200
#  Rank 15 Group  0 Pid     29 on demo-nccl-test-16-h200-worker-1 device  7 [0xb7] NVIDIA H200
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
  1073741824     268435456     float     sum      -1   4508.0  238.19  446.60      0   4489.7  239.16  448.42      0
  2147483648     536870912     float     sum      -1   8680.6  247.39  463.85      0   8688.9  247.15  463.41      0
  4294967296    1073741824     float     sum      -1    17054  251.85  472.21      0    17105  251.09  470.79      0
  8589934592    2147483648     float     sum      -1    34104  251.87  472.26      0    33842  253.83  475.93      0
 17179869184    4294967296     float     sum      -1    67369  255.01  478.15      0    67212  255.61  479.26      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 467.089 
#
```

### Cleanup

To delete the job, run the following:
```
runai training mpi delete -p <project> <job-name>
```