#!/bin/bash
set -e

: "${HF_TOKEN:?Please export your HF_TOKEN environment variable before running this script}"

ROOT_DIR=$(pwd)

python -m venv .env
source .env/bin/activate

pip install torchtitan==0.2.0
pip install transformers==4.57.1 \
    einops==0.8.1 \
    sentencepiece==0.2.1 \
    wandb==0.22.3

python -m torchtitan.experiments.flux.scripts.download_autoencoder \
    --repo_id black-forest-labs/FLUX.1-dev \
    --ae_path ae.safetensors \
    --hf_token $HF_TOKEN \
    --local_dir ./assets/
 
cd $ROOT_DIR
mkdir -p outputs slurm_out
echo "Setup complete"
