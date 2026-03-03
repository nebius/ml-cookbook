#!/bin/bash
# Usage:
# Set the project venv path in VENV_DIR variable below, then run:
#   ./submit.sh

set -euo pipefail

# Set the project venv path
VENV_DIR="/shared/venvs/distilbert-train"

echo "Project venv path: $VENV_DIR"
read -p "Proceed with this path? [Y/n]: " yn
case $yn in
	[Nn]*)
		echo "Aborted by user. Edit submit.sh to change VENV_DIR." >&2
		exit 0
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

# Ensure log directory exists (create if missing)
if [ -z "${SLURM_LOGS_DIR:-}" ]; then
		echo "[ERROR] SLURM_LOGS_DIR not set after sourcing project.env" >&2
		exit 2
fi
mkdir -p "$SLURM_LOGS_DIR"

echo "Submitting job with logs in: $SLURM_LOGS_DIR"
sbatch --export=ALL \
	--output="${SLURM_LOGS_DIR}/%x-%j.out" \
	--error="${SLURM_LOGS_DIR}/%x-%j.err" \
	train.slurm
