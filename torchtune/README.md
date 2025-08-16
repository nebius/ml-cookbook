# TorchTune on Soperator
This guide will walk you through setting up a TorchTune finetuning workload on Nebius AI Cloud.

## Quickstart
2. Create a SLURM cluster with Soperator. Make sure it has at least 4 nodes.
2. Clone this repository to your login node.
3. Run the bootstrap script: `/bin/sh bootstrap.sh`
   1. Make sure to answer any prompts that the script produces.
4. Submit the SLURM batch job: `sbatch torchtune_multinode.slurm`