#!/bin/bash

#SBATCH --job-name=torchtitan_multi_node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_out/flux-%j.out
#SBATCH --error=slurm_out/flux-%j.err

source .env/bin/activate

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo Node IP: $head_node_ip

# debug
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="eth0"
export LOGLEVEL=INFO

CONFIG_FILE=${CONFIG_FILE:-"$(pwd)/flux_schnell_model.toml"}

srun \
    torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.experiments.flux.train \
    --job.config_file ${CONFIG_FILE} \
    --encoder.autoencoder_path "$(pwd)/assets/ae.safetensors" \
    --parallelism.data_parallel_replicate_degree $SLURM_JOB_NUM_NODES #\
    # --training.dataset_path "$(pwd)/dataset_cc12m-wds" \
    # --metrics.enable_wandb
