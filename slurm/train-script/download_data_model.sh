#!/bin/bash
set -euo pipefail
##########################################################################################
# Lightweight wrapper to download model + dataset using existing environment.
# Assumes `setup.sh` has already been run to create the venv & project.env.
#
# Usage:
#   source $VENV_DIR/project.env
#   ./download_data_model.sh
# Adjust MODEL_NAME, DATASET_NAME, FORCE_REFRESH defaults inside this script as needed.
##########################################################################################

# Environment variables (override as needed before invocation):
: "${MODEL_NAME:=distilbert/distilbert-base-uncased}"   # HF model repo id
: "${DATASET_NAME:=stanfordnlp/sst2}"                  # HF dataset repo id
: "${FORCE_REFRESH:=1}"                               # 1 to delete existing model/data dirs before download, 0 to skip if present

# Required base env created by source project.env
: "${VENV_DIR:?Run setup.sh first so VENV_DIR is defined}"
: "${MODEL_DIR:?MODEL_DIR missing (source project.env or run setup.sh)}"
: "${DATA_DIR:?DATA_DIR missing (source project.env or run setup.sh)}"
: "${SHARED_DIR:?SHARED_DIR missing (source project.env or run setup.sh)}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[ERROR] Virtualenv activate script not found at $VENV_DIR/bin/activate" >&2
  exit 2
fi

source "$VENV_DIR/bin/activate"

mkdir -p "$MODEL_DIR" "$DATA_DIR"
DL_ARGS=()
if [ "$FORCE_REFRESH" = "1" ]; then
  DL_ARGS+=(--force_download)
fi

python download_data_model.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET_NAME" \
  --shared_folder "$SHARED_DIR" \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  ${DL_ARGS[@]:-}

echo "[INFO] Download (model=$MODEL_NAME dataset=$DATASET_NAME) complete." >&2
