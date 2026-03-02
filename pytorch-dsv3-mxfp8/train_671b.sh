#!/bin/bash
set -euo pipefail

#SBATCH --job-name=torchtitan_DSV3_671B_AO
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --ntasks=32
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96

# Select a recipe and expert-parallel backend via environment variables:
RECIPE=${RECIPE:-"bf16"}          # bf16 | mxfp8_default
EP_BACKEND=${EP_BACKEND:-"standard"}  # standard | deepep

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DUMP_FOLDER="$SCRIPT_DIR/outputs_671b_${RECIPE}_${EP_BACKEND}_${SLURM_JOB_ID}"
source "$SCRIPT_DIR/.env/bin/activate"
cd "$SCRIPT_DIR/torchtitan"

# Environment settings
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -r | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | head -n 1 | cut -d " " -f 1)
echo "MASTER_ADDR: $MASTER_ADDR"

export UCX_NET_DEVICES=ens7
export NCCL_IB_HCA=mlx5

# NVSHMEM / DeepEP settings (required when EP_BACKEND=deepep)
if [ "$EP_BACKEND" = "deepep" ]; then
    export NVSHMEM_REMOTE_TRANSPORT=ibrc
    export NVSHMEM_IB_ENABLE_IBGDA=1
    export NVSHMEM_IBGDA_NIC_HANDLER=gpu
fi

# Training hyperparameters
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml"
TP=2
PP=2
EP=32
ETP=1
DP_REPL=1
DP_SHARD=-1
LBS=64
STEPS=20
SEQLEN=8192
COMPILE="loss,model"
AC="full"

export WANDB_PROJECT="Nebius Torchtitan mxfp8 vs DeepEP DeepSeek V3 671B"
export WANDB_NAME="DSV3_671B_${RECIPE}_LBS=${LBS}_PP=${PP}_EP=${EP}_TP=${TP}_backend=${EP_BACKEND}_AC=${AC}_COMPILE=${COMPILE}_SEQ=${SEQLEN}"

# Build recipe-specific MXFP8 arguments
MXFP8_ARGS=()
if [ "$RECIPE" = "mxfp8_default" ]; then
    MXFP8_ARGS+=(
        --model.converters="quantize.grouped_mm.mx"
        --quantize.grouped_mm.mx.fqns="experts"
        --quantize.grouped_mm.mx.recipe_name="mxfp8"
    )
fi

# Launch
srun torchrun \
    --nnodes "${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node 8 \
    --rdzv_id "${SLURM_JOB_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:29501" \
    -m torchtitan.train \
    --job.config_file "${CONFIG_FILE}" \
    --metrics.log_freq=1 \
    --training.steps="${STEPS}" \
    --parallelism.data_parallel_replicate_degree="${DP_REPL}" \
    --parallelism.data_parallel_shard_degree="${DP_SHARD}" \
    --parallelism.expert_parallel_degree="${EP}" \
    --parallelism.tensor_parallel_degree="${TP}" \
    --parallelism.pipeline_parallel_degree="${PP}" \
    --parallelism.expert_parallel_comm_backend="${EP_BACKEND}" \
    --parallelism.expert_tensor_parallel_degree="${ETP}" \
    --training.local_batch_size="${LBS}" \
    --training.seq_len="${SEQLEN}" \
    --lr_scheduler.warmup_steps=2000 \
    --optimizer.lr=1e-4 \
    --optimizer.eps=1e-8 \
    --model.print_after_conversion \
    --compile.enable \
    --compile.components="${COMPILE}" \
    --debug.moe_force_load_balance \
    --metrics.enable_wandb \
    --activation_checkpoint.mode="${AC}" \
    --job.dump_folder="${DUMP_FOLDER}" \
    "${MXFP8_ARGS[@]}" \
    "$@"
