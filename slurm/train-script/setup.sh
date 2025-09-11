#!/bin/bash
set -euo pipefail
echo "if the HF repo is private, make sure you have set up a token with access and"
echo "run: export HF_TOKEN=your_token"

SHARED_DIR="/shared"
VENV_DIR="$SHARED_DIR/venvs/distbert-train"


# Create project-specific Python virtual environment
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt


# Create full model, data, and checkpoint and log directories as in config.yaml to be prepopulated
MODEL_DIR="/shared/model/distilbert-base-uncased"
DATA_DIR="/shared/data/glue_sst2"
CHECKPOINT_DIR="/shared/checkpoints/distilbert-base-uncased"
SLURM_LOGS_DIR="/shared/slurm_logs/distilbert-base"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$SLURM_LOGS_DIR"

# Download model and dataset into these subfolders
python download_data_model.py \
	--model distilbert-base-uncased \
	--dataset glue \
	--subset sst2 \
	--shared_folder "$SHARED_DIR" \
	--model_dir "$MODEL_DIR" \
	--data_dir "$DATA_DIR"

echo "Download and Setup complete."
