#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------- SBATCH commands ---------- #
#SBATCH --job-name=torchtune-multi-node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_out/torchtune-lora-%j.out

# ---------- Set env variables ---------- #
# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# Specify the IP and port for the processes to communicate
# The port should be between 29500 and 65535
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
# Set triton cache to ext4 filesystem to avoid VirtioFS concurrent access issues
export TRITON_CACHE_DIR=/tmp/triton_cache

# You might need to explicitly set the network interface for distributed backends:
# export NCCL_SOCKET_IFNAME=...
# export GLOO_SOCKET_IFNAME=...
# export NCCL_DEBUG=INFO

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO

# ---------- Launch training ---------- #
source .env/bin/activate

SHARED_FS=$(pwd)
CHECKPOINT_DIR="$SHARED_FS/checkpoint"
OUTPUT_DIR="$SHARED_FS/output_lora"

srun tune run \
     --nnodes 2 \
     --nproc_per_node 8 \
     --rdzv_backend c10d \
     --rdzv_endpoint "$head_node_ip:29500" \
     lora_finetune_distributed \
     --config ./llama3_3_70B_lora_multinode.yaml \
     checkpoint_dir=$CHECKPOINT_DIR \
     output_dir=$OUTPUT_DIR