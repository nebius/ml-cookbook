#!/bin/bash
set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "if the HF repo is private, make sure you have set up a token with access and"
echo "run: export HF_TOKEN=your_token"


# Set project name for flexible directory management
PROJECT_NAME="distilbert-train"
SHARED_DIR="/shared"

echo -e "${GREEN}Current PROJECT_NAME: $PROJECT_NAME${NC}"
echo -e "${GREEN}Current SHARED_DIR: $SHARED_DIR${NC}"
read -p "Proceed with these settings? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
	echo -e "${RED}Aborting. Please edit PROJECT_NAME and SHARED_DIR in setup.sh as needed.${NC}"
	exit 1
fi

# All key directories use PROJECT_NAME for easy switching
VENV_DIR="$SHARED_DIR/venvs/$PROJECT_NAME"
MODEL_DIR="$SHARED_DIR/model/$PROJECT_NAME"
DATA_DIR="$SHARED_DIR/data/$PROJECT_NAME"
SLURM_LOGS_DIR="$SHARED_DIR/slurm_logs/$PROJECT_NAME"
EXPERIMENTS_ROOT="$SHARED_DIR/experiments"

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

# Create project-specific Python virtual environment with Python 3.11 (idempotent)
if [ ! -d "$VENV_DIR" ]; then
	echo -e "${GREEN}Creating virtual environment at $VENV_DIR${NC}"
	$PYTHON_BIN -m venv "$VENV_DIR"
fi

# Write all key environment variables to a project.env file for use in Slurm and config.yaml
cat > "$VENV_DIR/project.env" <<EOF
export PROJECT_NAME="$PROJECT_NAME"
export SHARED_DIR="$SHARED_DIR"
export VENV_DIR="$VENV_DIR"
export MODEL_DIR="$MODEL_DIR"
export DATA_DIR="$DATA_DIR"
export SLURM_LOGS_DIR="$SLURM_LOGS_DIR"
export EXPERIMENTS_ROOT="$EXPERIMENTS_ROOT"
EOF

source "$VENV_DIR/project.env"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt


# Create model, data, log, and experiments root directories
# Per-run checkpoint lives under: $EXPERIMENTS_ROOT/<run_name>/checkpoint.pt
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$SLURM_LOGS_DIR" "$EXPERIMENTS_ROOT"

# Basic permission checks
for d in "$MODEL_DIR" "$DATA_DIR" "$SLURM_LOGS_DIR" "$EXPERIMENTS_ROOT"; do
	if [ ! -w "$d" ]; then
		echo -e "${RED}Warning: Directory $d is not writable by current user.$NC"
	fi
done

echo -e "${GREEN}Setup complete.${NC}"
echo "To download/refresh model + dataset run:"
echo "  source $VENV_DIR/project.env && ./download_data_model.sh" 
