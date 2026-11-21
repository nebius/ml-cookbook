#!/bin/bash
#SBATCH --job-name=verl-gb300-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

ROOT=${VERL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
BASE_IMAGE=${VERL_BASE_IMAGE:-nvcr.io#nvidia/cuda:13.0.2-devel-ubuntu24.04}
IMAGE=${ROOT}/artifacts/verl-cu130-arm64-sm103-base.sqsh
VERL_REF=${VERL_REF:-3efe38c759c14622fd1b2c9e3679f2d02f86bdac}
UV_VERSION=${UV_VERSION:-0.11.16}

mkdir -p "${ROOT}"/{artifacts,bin,data,logs,models,hf-cache,uv-cache}

# Pyxis pulls the base image and saves a reusable image containing the OS
# packages needed to build verl for an ARM64 GB300 node.
if [[ ! -f "${IMAGE}" ]]; then
  srun \
    --container-image="${BASE_IMAGE}" \
    --container-remap-root \
    --container-writable \
    --container-save="${IMAGE}" \
    --container-mounts="${ROOT}:/workspace" \
    --container-workdir=/workspace \
    bash -s <<'BOOTSTRAP'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git git-lfs build-essential cmake ninja-build \
  python3 python3-dev python3-pip python3-venv \
  libnuma1 libnuma-dev libibverbs1 libibverbs-dev \
  librdmacm1 librdmacm-dev libopenmpi-dev numactl
rm -rf /var/lib/apt/lists/*
BOOTSTRAP
fi

# The environment, model, and data live in the shared workspace and are reused
# by every node in the training job.
srun \
  --container-image="${IMAGE}" \
  --container-mounts="${ROOT}:/workspace,/tmp:/tmp" \
  --container-workdir=/workspace \
  bash -s -- "${VERL_REF}" "${UV_VERSION}" <<'SETUP'
set -euo pipefail

VERL_REF=$1
UV_VERSION=$2
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=10.3
export CUDAARCHS=103
export CMAKE_CUDA_ARCHITECTURES=103
export MAX_JOBS=16
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache

if [[ ! -d /workspace/verl/.git ]]; then
  git clone https://github.com/volcengine/verl.git /workspace/verl
fi
git -C /workspace/verl fetch --tags origin
git -C /workspace/verl checkout --detach "${VERL_REF}"

if [[ ! -x /workspace/bin/uv ]]; then
  curl --fail --location --retry 3 \
    --output /tmp/uv.tar.gz \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz"
  tar -xzf /tmp/uv.tar.gz -C /tmp
  install -m 0755 /tmp/uv-aarch64-unknown-linux-gnu/uv /workspace/bin/uv
fi

cd /workspace/verl
/workspace/bin/uv sync --frozen --all-packages --extra vllm --extra megatron

PYTHONPATH=/workspace/verl .venv/bin/python examples/data_preprocess/gsm8k.py \
  --local_save_dir /workspace/data/gsm8k

.venv/bin/python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/deepseek-llm-7b-chat"
output_path = "/workspace/models/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path, safe_serialization=True)
PY

.venv/bin/python - <<'PY'
import importlib.metadata as metadata

for package in ("verl", "torch", "vllm", "ray", "flash-attn", "megatron-core"):
    print(f"{package}={metadata.version(package)}")
PY
SETUP

echo "Setup complete. Submit with: sbatch ml-cookbook/slurm/verl/grpo_multinode_gb300.sh"
