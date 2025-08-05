#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Ensure the user supplied a valid Hugging Face access token
: "${HF_TOKEN:?Please export your HF_TOKEN environment variable before running this script}"  # Abort if undefined or empty


# 0) remember where we started
ROOT_DIR=$(pwd)

# 1) Install Python 3.11 + venv support if missing
if ! command -v python3.11 &> /dev/null; then
  echo ">>> python3.11 not found – installing via apt..."
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv
fi

# 2) Create & activate a 3.11 venv
python3.11 -m venv .env
source .env/bin/activate

# 3) Clone and cd into the flux experiment, upgrade pip and install requirements
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install --upgrade pip
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# Flux specific setup
python ./torchtitan/experiments/flux/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-dev --ae_path ae.safetensors --hf_token $HF_TOKEN
cd torchtitan/experiments/flux
pip install -r requirements-flux.txt

# 5) Make required directories 
cd $ROOT_DIR
mkdir -p outputs slurm_out
echo "✅ Setup complete! Activated venv with: $(python --version)"
