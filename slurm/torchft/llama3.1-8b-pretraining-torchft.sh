#!/bin/bash

#SBATCH --job-name=tft_replica
###SBATCH --array=0-7                           # N independent tasks = N replicas
#SBATCH --nodes=1                               # one node per replica
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --output=outputs/torchft/tft-%A/tft-%A_%a.out
#SBATCH --error=outputs/torchft/tft-%A/tft-%A_%a.err
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --export=ALL

export REPLICA_ID=${SLURM_ARRAY_TASK_ID}        # 0 … N-1
export GROUP_SIZE=${SLURM_ARRAY_TASK_COUNT}     # N

echo "[`date '+%F %T'`]  launching replica $REPLICA_ID (attempt ${SLURM_RESTART_COUNT:-0}) on $(hostname)"
source .env/bin/activate

export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="eth0"
export GLOO_SOCKET_IFNAME="eth0"

export PYTHONPATH="$(pwd)/torchtitan:$PYTHONPATH"
export CONFIG_FILE="$(pwd)/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml"

export TORCHFT_LIGHTHOUSE="http://login-0:29510"

# enable wandb only on replica 0
METRICS_ARG=""
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ -n "${WANDB_API_KEY:-}" ]; then
    METRICS_ARG="--metrics.enable-wandb"
fi

cd torchtitan

torchrun \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:0" \
  --rdzv_id="tt_$SLURM_ARRAY_JOB_ID" \
  ./torchtitan/train.py \
  --job.config_file="${CONFIG_FILE}" \
  --fault_tolerance.enable \
  --fault_tolerance.group_size="${GROUP_SIZE}" \
  --fault_tolerance.replica_id="${REPLICA_ID}" \
  --parallelism.data_parallel_shard_degree=8 \
  --training.dataset="c4_test" \
  --profiling.no-enable-profiling \
  --comm.trace-buf-size 0 \
  --comm.train-timeout-seconds 300 \
  $METRICS_ARG