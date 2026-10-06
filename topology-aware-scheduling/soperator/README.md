# Soperator (Slurm)

Soperator exposes the same physical topology through Slurm rather than a
Kubernetes workload CRD. The included
[`megatron-pretraining.sh`](megatron-pretraining.sh) runs the same
32-node, four-GPU-per-node Megatron Bridge pretraining workload used by the
Kubernetes examples. It launches the Python workload as 128 native Slurm tasks
and does not use `torchrun`.



For `--segment`, the GB300 cluster must use Slurm's `topology/block` plugin
and each base block must represent one NVLink rack. `scontrol show config`
should report `TopologyPlugin=topology/block`; Soperator GB300 clusters also
normally use `BlockAsNodeRank`.

Copy the batch file into the login node, export the Hugging Face token there,
and submit it:

```bash

export HF_TOKEN='replace-with-hugging-face-token'
export TAS_SLURM_ACCOUNT='replace-with-slurm-account'
export TAS_SLURM_PARTITION='replace-with-slurm-partition'
sbatch --account="${TAS_SLURM_ACCOUNT}" --partition="${TAS_SLURM_PARTITION}" megatron-pretraining.sh
squeue --me
```

`#SBATCH --segment=16` divides the 32-node allocation into two segments. `srun` starts four tasks per node; each task selects
its GPU through `LOCAL_RANK`. The installed Enroot PyTorch hook derives `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` from the Slurm step. With
block task distribution and `BlockAsNodeRank`, tasks 0-63 run on the first
16-node segment and tasks 64-127 run on the second.

The script uses Soperator's Pyxis/Enroot integration for `nvcr.io/nvidia/nemo:26.06`. Megatron Bridge receives the allocation's `SLURM_JOB_ACCOUNT` and
`SLURM_JOB_PARTITION` values. Adjust memory, wall time, or container cache
settings to match the target Slurm cluster.

See [Slurm topology](https://slurm.schedmd.com/topology.html),
[running jobs on Soperator](https://docs.nebius.com/slurm-soperator/jobs), and
[Pyxis/Enroot containers](https://docs.nebius.com/slurm-soperator/jobs/containers/pyxis-enroot).
The rank environment is provided by NVIDIA's
[Slurm PyTorch hook](https://github.com/NVIDIA/enroot/blob/main/conf/hooks/extra/50-slurm-pytorch.sh).
