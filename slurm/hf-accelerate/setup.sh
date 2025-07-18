#!/bin/bash

python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt

wget https://raw.githubusercontent.com/huggingface/accelerate/refs/tags/v1.6.0/examples/slurm/fsdp_config.yaml
wget https://raw.githubusercontent.com/huggingface/accelerate/refs/tags/v1.6.0/examples/complete_nlp_example.py
