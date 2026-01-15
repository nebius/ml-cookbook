#!/bin/bash
#SBATCH --job-name=verl-grpo-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -eou pipefail
export WANDB_API_KEY=<wandb_...>

export SUBMIT_DIR=$(pwd)

verl_workdir_mount=$SUBMIT_DIR/verl:/workspace
train_files=$SUBMIT_DIR/data/gsm8k/train.parquet
val_files=$SUBMIT_DIR/data/gsm8k/test.parquet
model_path=$SUBMIT_DIR/models/qwen2.5-3b-instruct
image_path=$SUBMIT_DIR/verl-sgl056.latest.sqsh

CONFIG_PATH="/workspace/examples/sglang_multiturn/config"

# Nodes
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}

# Get head node IP
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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

ray_tmp_mount=/tmp:/tmp

# Start Ray head
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image "$image_path" \
  --container-mounts "$verl_workdir_mount,$train_files,$val_files,$model_path,$ray_tmp_mount" \
  --container-workdir /workspace \
  --no-container-mount-home \
  bash -lc "
    set -eoux pipefail
    pip install -U weave
    ray start --head \
      --node-ip-address='$head_node_ip' \
      --port=$port \
      --num-cpus '${SLURM_CPUS_PER_TASK}' \
      --num-gpus '${SLURM_GPUS_PER_NODE}' \
      --disable-usage-stats \
      --block
  " &

sleep 10

# Start workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"

  srun --nodes=1 --ntasks=1 -w "$node_i" \
    --container-image "$image_path" \
    --container-mounts "$verl_workdir_mount,$train_files,$val_files,$model_path,$ray_tmp_mount" \
    --container-workdir /workspace \
    --no-container-mount-home \
    bash -lc "
      set -eoux pipefail
      pip install -U weave
      ray start --address '$ip_head' \
        --num-cpus '${SLURM_CPUS_PER_TASK}' \
        --num-gpus '${SLURM_GPUS_PER_NODE}' \
        --disable-usage-stats \
        --block
    " &
  sleep 5
done
sleep 10

echo "Ray cluster ready!"
echo "Launching training..."

METRICS_ARG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_NOTES="${SLURM_JOB_NAME}-${SLURM_JOBID}-${SLURM_NNODES}nodes"
    METRICS_ARG="trainer.logger='[\"console\",\"wandb\"]'"
else
    METRICS_ARG="trainer.logger=console"
fi

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image "$image_path" \
  --container-mounts "$verl_workdir_mount,$train_files,$val_files,$model_path,$ray_tmp_mount" \
  --container-workdir /workspace \
  --no-container-mount-home \
  bash -lc "
    set -eoux pipefail
    pip install -U weave
    export RAY_ADDRESS='$ip_head'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    ray status
    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='gsm8k_multiturn_grpo' \
        actor_rollout_ref.model.path='$model_path' \
        actor_rollout_ref.rollout.trace.backend=weave \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=256 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='gsm8k_async_rl' \
        trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-async-sgl-multi-w-tool-${SLURM_JOBID}' \
        trainer.save_freq=-1 \
        trainer.test_freq=20 \
        trainer.total_epochs=15 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
        critic.ppo_max_token_len_per_gpu=8192 \
        critic.forward_max_token_len_per_gpu=8192 \
        data.train_files='$train_files' \
        data.val_files='$val_files' \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/gsm8k_tool_config.yaml" \
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/interaction_config/gsm8k_interaction_config.yaml" \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
        $METRICS_ARG
  " 2>&1 | tee verl_grpo_slurm.log