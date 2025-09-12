#!/bin/bash
set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "if the HF repo is private, make sure you have set up a token with access and"
echo "run: export HF_TOKEN=your_token"

SHARED_DIR="/shared"
VENV_DIR="$SHARED_DIR/venvs/distbert-train"

# Check for Python 3.11, install if not present

if command -v python3.11 &> /dev/null; then
	PYTHON_BIN=$(command -v python3.11)
	echo -e "${GREEN}Found Python 3.11 at $PYTHON_BIN${NC}"
else
	echo -e "${RED}Python 3.11 not found. Attempting to install...${NC}"
	sudo apt-get update
	sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
	PYTHON_BIN=$(command -v python3.11)
	if [ -z "$PYTHON_BIN" ]; then
	echo -e "${RED}Python 3.11 installation failed. Exiting.${NC}"
		exit 1
	fi
fi

# Create project-specific Python virtual environment with Python 3.11
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt


# Create full model, data, and checkpoint and log directories as in config.yaml to be prepopulated
# Since the HF downloader creates a nested folder for the model, we create the parent folder here only for the model_dir
MODEL_DIR="/shared/model/distilbert-base-uncased"
DATA_DIR="/shared/data/sst2"
CHECKPOINT_DIR="/shared/checkpoints/distilbert-base-uncased"
SLURM_LOGS_DIR="/shared/slurm_logs/distilbert-base-uncased"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$SLURM_LOGS_DIR"

# Download model and dataset into these subfolders
python download_data_model.py \
	--model distilbert/distilbert-base-uncased \
	--dataset stanfordnlp/sst2 \
	--shared_folder "$SHARED_DIR" \
	--model_dir "$MODEL_DIR" \
	--data_dir "$DATA_DIR"

echo -e "${GREEN}Download and Setup complete.${NC}"
