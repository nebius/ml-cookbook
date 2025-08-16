#!/bin/bash

# ---------- SBATCH commands ---------- #
#SBATCH --job-name=torchtune-multi-node
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

# ---------- Set env variables ---------- #
# Grab the IP for head node:
# You may need to set this to the fully qualified domain name of your head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# You might need to explicitly set the network interface for distributed backends:
# export NCCL_SOCKET_IFNAME=...
# export GLOO_SOCKET_IFNAME=...

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO

# ---------- Launch training ---------- #
# You probably want to load in a virtual env w/ conda...
# module load conda
# conda activate torchtune
# ...or venv
source torchtune/bin/activate

SHARED_FS="/root/model" # <-- Replace w/ your filesystem
CHECKPOINT_DIR="$SHARED_FS/model"
OUTPUT_DIR="$SHARED_FS/Llama3.3-70B-fft-output"

# Adjust sbatch --ntasks and sbatch --nodes above and --nnodes below to your specific node count
srun tune run --nnodes 4 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    full_finetune_distributed --config config.yaml checkpoint_dir=$CHECKPOINT_DIR output_dir=$OUTPUT_DIR
