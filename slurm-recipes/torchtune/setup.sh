#!/bin/bash

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

# Download LLaMA-3 model
tune download meta-llama/Llama-3.3-70B-Instruct \
  --ignore-patterns "original/consolidated*.pth" \
  --output-dir ./models/Llama-3.3-70B-Instruct