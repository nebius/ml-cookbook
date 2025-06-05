#!/bin/bash

# 0) remember where we started
ROOT_DIR=$(pwd)

# 1) Install Python 3.11 + venv support if missing
if ! command -v python3.11 &> /dev/null; then
  echo ">>> python3.11 not found – installing via apt..."
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv

# 2) Create & activate a 3.11 venv
python3.11 -m venv .env
source .env/bin/activate

# 3) Clone and cd into the flux experiment, upgrade pip and install requirements
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install --upgrade pip
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
# pip install -e .

# Flux specific setup
python ./torchtitan/experiments/flux/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-dev --ae_path ae.safetensors --hf_token <YOUR HF TOKEN>
cd torchtitan/experiments/flux
pip install -r requirements-flux.txt

# 5) Make required directories 
mkdir -p output slurm_out
cd ../../../..
echo "✅ Setup complete! Activated venv with: $(python --version)"
