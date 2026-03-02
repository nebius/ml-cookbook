#!/bin/bash
set -euo pipefail
ROOT_DIR=$(pwd)

# Setup TorchTitan env with pinned dependancies
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

if [ ! -d "torchtitan" ]; then
    git clone https://github.com/pytorch/torchtitan.git
fi
cd torchtitan && git checkout 1a36996b8dffb464340c280d3f0a8139ca4c36b6  # Nightly accessed 2026-02-03
pip install -e .                

# Download tokenizers
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/deepseek-moe-16b-base', local_dir='assets/hf/deepseek-moe-16b-base', revision='521d2bc4fb69a3f3ae565310fcc3b65f97af2580', allow_patterns=['tokenizer*', 'special_tokens*'])
snapshot_download('deepseek-ai/DeepSeek-V3.1-Base', local_dir='assets/hf/DeepSeek-V3.1-Base', revision='d3d4eafdc470de44bbf6f0a74f852eb522357be8', allow_patterns=['tokenizer*', 'special_tokens*'])
"
cd "$ROOT_DIR"
mkdir -p slurm_logs
