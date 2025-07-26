FROM nvcr.io/nvidia/pytorch:25.06-py3

# Upgrade pip and setuptools
RUN pip install -U pip setuptools

# Install wandb and clone llama-cookbook
RUN pip install wandb
RUN git clone https://github.com/meta-llama/llama-cookbook.git
WORKDIR /workspace/llama-cookbook

# Install llama-cookbook in editable mode
RUN pip install -e .