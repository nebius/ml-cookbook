#!/bin/bash
#SBATCH --job-name=verl-ppo-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -eou pipefail

SUBMIT_DIR=$(pwd)
export SUBMIT_DIR

verl_workdir_mount=$SUBMIT_DIR/verl:/workspace
train_files=$SUBMIT_DIR/data/gsm8k/train.parquet
val_files=$SUBMIT_DIR/data/gsm8k/test.parquet
model_path=$SUBMIT_DIR/models/deepseek-math-7b-instruct
image_path=$SUBMIT_DIR/verl-vllm012.latest.sqsh

# Nodes
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}

# Get head node IP
head_node_ip=$(
  srun --nodes=1 --ntasks=1 -w "$head_node" \
    hostname --ip-address
)

# Handle IPv6 case
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using $head_node_ip"
fi

port=6379
ip_head="${head_node_ip}:${port}"

echo "Head node: $head_node"
echo "Head IP:   $head_node_ip"
echo "IP Head:   $ip_head"

printenv

# Start Ray head
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image "$image_path" \
  --container-mounts "$verl_workdir_mount,$train_files,$val_files,$model_path" \
  --container-workdir /workspace \
  --container-name=ray-head-$SLURM_JOB_ID \
  --no-container-mount-home \
  bash -lc "
    set -eoux pipefail
    ray start --head \
      --node-ip-address='$head_node_ip' \
      --port=$port \
      --num-cpus '${SLURM_CPUS_PER_TASK}' \
      --num-gpus '${SLURM_GPUS_PER_NODE}' \
      --disable-usage-stats \
      --resources='{\"worker_units\": ${SLURM_GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}' \
      --block
  " &

# Start workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"

  srun --nodes=1 --ntasks=1 -w "$node_i" \
    --container-image "$image_path" \
    --container-mounts "$verl_workdir_mount,$train_files,$val_files,$model_path" \
    --container-workdir /workspace \
    --no-container-mount-home \
    bash -lc "
      set -eoux pipefail
      ray start --address '$ip_head' \
        --num-cpus '${SLURM_CPUS_PER_TASK}' \
        --num-gpus '${SLURM_GPUS_PER_NODE}' \
        --disable-usage-stats \
        --resources='{\"worker_units\": ${SLURM_GPUS_PER_NODE}, \"slurm_managed_ray_cluster\": 1}' \
        --block
    " &
done

# Allow time for ray to start up
sleep 20

# Wait until all workers are registered in Ray cluster
extract_worker_units() {
  status_output=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=ray-head-$SLURM_JOB_ID --no-container-mount-home \
    ray status)
  if echo "$status_output" | grep -q "worker_units"; then
    echo "$status_output" | grep "worker_units" | awk -F'[/. ]' '{print $4}'
  else
    echo 0
  fi
}

NUM_ACTORS=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
TIMEOUT=300
START_TIME=$SECONDS
while true; do
  worker_units=$(extract_worker_units)
  elapsed=$((SECONDS - START_TIME))
  echo "[INFO] Number of actors online: $worker_units/$NUM_ACTORS (${elapsed}s elapsed)"
  if [[ "$worker_units" -eq "$NUM_ACTORS" ]]; then
    break
  fi
  if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
    echo "[ERROR] Timed out after ${TIMEOUT}s waiting for all actors to come online ($worker_units/$NUM_ACTORS)"
    exit 1
  fi
  sleep 10
done

echo "All workers connected!"
echo "Launching training..."

export RAY_ADDRESS="$head_node_ip:6379"
echo "RAY_ADDRESS=$RAY_ADDRESS"

METRICS_ARG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_NOTES="${SLURM_JOB_NAME}-${SLURM_JOBID}-${SLURM_NNODES}nodes"
    METRICS_ARG="trainer.logger='[\"console\",\"wandb\"]'"
else
    METRICS_ARG="trainer.logger=console"
fi

# Attaching to a running container 'ray-head' to submit the job
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  --container-name=ray-head-$SLURM_JOB_ID \
  --no-container-mount-home \
  --container-workdir /workspace \
  --jobid "$SLURM_JOB_ID" \
  bash -lc "
    set -eoux pipefail
    ray status
    python3 -m verl.trainer.main_ppo \
        data.train_files='$train_files' \
        data.val_files='$val_files' \
        data.train_batch_size=1024 \
        data.max_prompt_length=512 \
        data.max_response_length=512 \
        actor_rollout_ref.model.path='$model_path' \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=512 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path='$model_path' \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=32 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.project_name='verl_example_gsm8k' \
        trainer.experiment_name='deepseek_llm_7b_function_rm-${SLURM_JOBID}' \
        trainer.n_gpus_per_node='${SLURM_GPUS_PER_NODE}' \
        trainer.nnodes='${SLURM_NNODES}' \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=2 \
        $METRICS_ARG
  " 2>&1 | tee logs/verl_ppo_slurm.log
