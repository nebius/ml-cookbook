#!/bin/bash
#SBATCH --job-name=megatron-tas
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --mem=760G
#SBATCH --time=06:00:00
#SBATCH --segment=16
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN before submitting this job}"
: "${SLURM_JOB_ACCOUNT:?Slurm job account is required}"
: "${SLURM_JOB_PARTITION:?Slurm job partition is required}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TOKENIZERS_PARALLELISM=False
export TORCH_NCCL_HIGH_PRIORITY=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NCCL_MNNVL_ENABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,NVLS

srun \
  --container-image="nvcr.io#nvidia/nemo:26.06" \
  python \
    /opt/Megatron-Bridge/scripts/performance/run_script.py \
      --account "${SLURM_JOB_ACCOUNT}" \
      --partition "${SLURM_JOB_PARTITION}" \
      --gpu gb300 \
      --num_gpus 128 \
      --gpus_per_node 4 \
      --model_family_name qwen \
      --model_recipe_name qwen3_235b_a22b \
      --cuda_graph_impl=transformer_engine \
      --cuda_graph_scope=moe_router,moe_preprocess \
      --hf_token "${HF_TOKEN}" \
      --max_steps 50 \
      --compute_dtype fp8_mx \
      -tp 1 -pp 4 -vp 12 -cp 1 -ep 32 -et 1 \
      -gb 8192 -mb 2 \
      train.manual_gc=true train.manual_gc_interval=100
