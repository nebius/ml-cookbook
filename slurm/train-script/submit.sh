#!/bin/bash
# Minimal submit helper: prompts for VENV_DIR, sources project.env, then submits job.
# Usage:
#   ./submit.sh

set -euo pipefail

# Default path (edit as needed)
DEFAULT_VENV_DIR="/shared/venvs/distilbert-train"

# Use existing VENV_DIR if set, else default
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"

echo "Current VENV_DIR is: $VENV_DIR"
read -p "Is this correct? [Y/n]: " yn
case $yn in
	[Nn]*)
		read -p "Enter new VENV_DIR path: " VENV_DIR
		;;
esac


export VENV_DIR

if [ ! -d "$VENV_DIR" ]; then
	echo "[ERROR] VENV_DIR directory does not exist: $VENV_DIR. Run setup.sh to create it." >&2
	exit 1
fi

if [ ! -f "$VENV_DIR/project.env" ]; then
	echo "[ERROR] project.env not found at $VENV_DIR/project.env. Run setup.sh first." >&2
	exit 1
fi

source "$VENV_DIR/project.env"
sbatch --export=ALL train.slurm
