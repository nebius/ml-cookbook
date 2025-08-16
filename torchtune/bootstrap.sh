#!/bin/sh
export MODEL_DIR=~/model/model"
export MODEL_HF='meta-llama/Llama-3.3-70B-Instruct'
pip install -U virtualenv
virtualenv torchtune
. torchtune/bin/activate
pip install -r requirements.txt
huggingface-cli login
mkdir -p ~/model/model
hf download --max-workers $(nproc) --local-dir $MODEL_DIR $MODEL_HF
