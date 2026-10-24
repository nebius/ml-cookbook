#!/bin/bash
#SBATCH --job-name=verl-grpo-gb300
#SBATCH --nodes=36
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -eou pipefail

SUBMIT_DIR=${VERL_ROOT:-$(pwd)}
export SUBMIT_DIR

verl_workdir_mount=$SUBMIT_DIR/verl:/workspace
hf_cache_mount=$SUBMIT_DIR/hf-cache:/workspace/hf-cache
train_files=$SUBMIT_DIR/data/gsm8k/train.parquet
val_files=$SUBMIT_DIR/data/gsm8k/test.parquet
model_path=$SUBMIT_DIR/models/deepseek-llm-7b-chat
image_path=$SUBMIT_DIR/artifacts/verl-cu130-arm64-sm103-base.sqsh

[[ -f "$image_path" ]] || {
  echo "Missing $image_path; run setup_gb300.sh first" >&2
  exit 1
}
[[ -x "$SUBMIT_DIR/verl/.venv/bin/python" ]] || {
  echo "Missing verl virtual environment; run setup_gb300.sh first" >&2
  exit 1
}

# GB300 CUDA, networking, and NCCL settings.
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=10.3
export CUDAARCHS=103
export CMAKE_CUDA_ARCHITECTURES=103
export HF_HOME=/workspace/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RAY_USAGE_STATS_ENABLED=0
export NCCL_DEBUG=WARN
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
export NCCL_BUFFSIZE=8388608

ray_cpus_per_node=$((SLURM_CPUS_PER_TASK - 4))

# Nodes
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}

# Select the 10.x fabric address used by the GB300 nodes.
head_node_ip=$(
  srun --overlap --nodes=1 --ntasks=1 --exact -w "$head_node" \
    bash -lc "hostname -I | tr ' ' '\\n' | awk '/^10[.]/ {print; exit}'"
)

if [[ -z "$head_node_ip" ]]; then
  echo "Could not find the head node's 10.x IP address" >&2
  exit 1
fi

port=6379
ip_head="${head_node_ip}:${port}"

echo "Head node: $head_node"
echo "Head IP:   $head_node_ip"
echo "IP Head:   $ip_head"

printenv

# Start Ray head
echo "Starting HEAD at $head_node"
srun --overlap --nodes=1 --ntasks=1 --exact -w "$head_node" \
  --container-image "$image_path" \
  --container-mounts "$verl_workdir_mount,$hf_cache_mount,$train_files,$val_files,$model_path,/tmp:/tmp" \
  --container-workdir /workspace \
  --container-name=ray-head-$SLURM_JOB_ID \
  --no-container-mount-home \
  bash -lc "
    set -eoux pipefail
    source /workspace/.venv/bin/activate
    ray start --head \
      --node-ip-address='$head_node_ip' \
      --port=$port \
      --num-cpus '$ray_cpus_per_node' \
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

  srun --overlap --nodes=1 --ntasks=1 --exact -w "$node_i" \
    --container-image "$image_path" \
    --container-mounts "$verl_workdir_mount,$hf_cache_mount,$train_files,$val_files,$model_path,/tmp:/tmp" \
    --container-workdir /workspace \
    --no-container-mount-home \
    bash -lc "
      set -eoux pipefail
      source /workspace/.venv/bin/activate
      node_ip=\$(hostname -I | tr ' ' '\\n' | awk '/^10[.]/ {print; exit}')
      [[ -n \"\$node_ip\" ]] || { echo 'Could not find a worker 10.x IP address' >&2; exit 1; }
      ray start --address '$ip_head' \
        --node-ip-address=\"\$node_ip\" \
        --num-cpus '$ray_cpus_per_node' \
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
  status_output=$(srun --overlap --nodes=1 --ntasks=1 --exact -w "$head_node" \
    --container-name=ray-head-$SLURM_JOB_ID --no-container-mount-home \
    bash -lc "source /workspace/.venv/bin/activate; ray status")
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
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 --exact -w "$head_node" \
  --container-name=ray-head-$SLURM_JOB_ID \
  --no-container-mount-home \
  --container-workdir /workspace \
  --jobid "$SLURM_JOB_ID" \
  bash -lc "
    set -eoux pipefail
    source /workspace/.venv/bin/activate
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    ray status
    python3 -m verl.trainer.main_ppo \
        --config-path=config \
        --config-name='ppo_megatron_trainer.yaml' \
        algorithm.adv_estimator=grpo \
        data.train_files='$train_files' \
        data.val_files='$val_files' \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path='$model_path' \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='deepseek_llm_7b_grpo-${SLURM_JOBID}' \
        trainer.n_gpus_per_node='${SLURM_GPUS_PER_NODE}' \
        trainer.nnodes='${SLURM_NNODES}' \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=2 \
        $METRICS_ARG
  " 2>&1 | tee logs/verl_grpo_gb300_slurm.log
