#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- adjust NCCL_BUFFSIZE if you encounter memory constraint issues or to tune for improved performance.

#SBATCH --job-name=torchtitan_multi_node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_out/flux-%j.out
#SBATCH --error=slurm_out/flux-%j.err
###SBATCH --partition=train

source .env/bin/activate  # Edit this path to your venv path

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# optional debug settings and flags (optional)
export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0"

# Set working directory
export PYTHONPATH="$(pwd)/torchtitan:$PYTHONPATH"

# dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below to your specific node count, and update target launch file.
CONFIG_FILE=${CONFIG_FILE:-"$(pwd)/flux_schnell_model.toml"}

# Set working directory
cd $(pwd)/torchtitan

srun torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.experiments.flux.train --job.config_file ${CONFIG_FILE}
# dcgmi profile --resume