#!/bin/bash
set -euo pipefail

#SBATCH --job-name=torchtitan_DSV3_16B_AO
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --ntasks=32
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96

# Select a recipe via environment variable:
RECIPE=${RECIPE:-"bf16"}  # bf16 | mxfp8_wgrad_hp

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DUMP_FOLDER="$SCRIPT_DIR/outputs_16b_${RECIPE}_${SLURM_JOB_ID}"
source "$SCRIPT_DIR/.env/bin/activate"
cd "$SCRIPT_DIR/torchtitan"

# Environment settings
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -r | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_ADDR" | head -n 1 | cut -d " " -f 1)
echo "MASTER_ADDR: $MASTER_ADDR"

export UCX_NET_DEVICES=ens7
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1

# Training hyperparameters
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"
TP=1
EP=32
ETP=1
DP_REPL=1
DP_SHARD=-1
BS=16
STEPS=1500
SEQLEN=8192

export WANDB_PROJECT="Nebius Torchtitan DeepSeek V3 16B"
export WANDB_NAME="${RECIPE}_tp${TP}_ep${EP}_bs${BS}_steps${STEPS}_seq${SEQLEN}"
echo "WANDB_NAME=$WANDB_NAME"

# Recipe-specific MXFP8 arguments
MXFP8_ARGS=()
if [ "$RECIPE" = "mxfp8_wgrad_hp" ]; then
    MXFP8_ARGS+=(
        --model.converters="quantize.grouped_mm.mx,quantize.linear.mx"
        --quantize.grouped_mm.mx.fqns="experts"
        --quantize.grouped_mm.mx.recipe_name="mxfp8_wgrad_with_hp"
        --parallelism.expert_parallel_a2a_dispatch_fwd_precision="mxfp8"
        --parallelism.expert_parallel_a2a_combine_bwd_precision="mxfp8"
        --quantize.linear.mx.recipe_name="mxfp8_cublas_rceil"
        --quantize.linear.mx.filter_fqns="output,router.gate,wq,wkv,wo,feed_forward.w2,shared_experts"
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
    --parallelism.expert_tensor_parallel_degree="${ETP}" \
    --training.local_batch_size="${BS}" \
    --training.seq_len="${SEQLEN}" \
    --lr_scheduler.warmup_steps=2000 \
    --optimizer.lr=1e-4 \
    --optimizer.eps=1e-8 \
    --model.print_after_conversion \
    --compile.enable \
    --compile.components="model,loss" \
    --metrics.enable_wandb \
    --activation_checkpoint.mode="full" \
    --debug.moe_force_load_balance \
    --job.dump_folder="${DUMP_FOLDER}" \
    "${MXFP8_ARGS[@]}" \
    "$@"
