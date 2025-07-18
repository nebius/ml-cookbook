#!/bin/bash

#SBATCH --job-name=torchtitan_multi_node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
#SBATCH --export=ALL

source .env/bin/activate  # Edit this path to your venv path

export HEAD_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo Node IP: $HEAD_NODE_IP

export LOGLEVEL=INFO

# debugging flags (optional)
export PYTHONFAULTHANDLER=1

# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"

export PYTHONPATH="$(pwd)/torchtitan:$PYTHONPATH"
export CONFIG_FILE="$(pwd)/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml"

METRICS_ARG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    METRICS_ARG="--metrics.enable-wandb"
fi

# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.

cd torchtitan

srun \
    torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint "$HEAD_NODE_IP:29500" \
    --role rank \
    --tee 3 \
    ./torchtitan/train.py \
    --job.config_file $CONFIG_FILE \
    --training.dataset="c4_test" \
    --profiling.no-enable-profiling \
    --comm.trace-buf-size 0 \
    --comm.train-timeout-seconds 300\
    $METRICS_ARG