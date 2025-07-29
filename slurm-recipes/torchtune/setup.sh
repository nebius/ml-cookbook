#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Ensure the user supplied a valid Hugging Face access token
: "${HF_TOKEN:?Please export your HF_TOKEN environment variable before running this script}"  # Abort if undefined or empty

# Create directories for output and logs
mkdir -p output
mkdir -p slurm_out

# Create and activate virtual environment
python3 -m venv .env
source .env/bin/activate

# Upgrade pip and install dependencies
pip install -U pip
pip install torch torchvision torchao tensorboard

# Clone torchtune repo (if needed) and install in editable mode
if [ ! -d "torchtune" ]; then
    git clone https://github.com/pytorch/torchtune.git
fi
pip install -e torchtune

# Render TorchTune YAML configs with the absolute root directory
ROOT_DIR=$(pwd)

for template in llama3_3_70B_full_multinode.yaml.tpl \
                llama3_3_70B_lora_multinode.yaml.tpl; do
  if [ -f "$template" ]; then
    config="${template%.tpl}"
    sed "s|__ROOT_DIR__|${ROOT_DIR}/|g" "$template" > "$config"
    echo "Generated $config with root_dir=$ROOT_DIR/"
  else
    echo "Warning: template $template not found; skipping" >&2
  fi
done

# Download LLaMA-3 model to shared FS --> Syncs to all Soperoator nodes
tune download meta-llama/Llama-3.3-70B-Instruct \
  --ignore-patterns "original/consolidated*.pth" \
  --output-dir ./models/Llama-3.3-70B-Instruct