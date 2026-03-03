FROM nvcr.io/nvidia/pytorch:26.01-py3

# Upgrade pip and setuptools
RUN pip install -U pip setuptools

# Install wandb and clone llama-cookbook
RUN pip install "wandb==0.25.0"
RUN git clone --branch v0.0.5 --depth 1 https://github.com/meta-llama/llama-cookbook.git
WORKDIR /workspace/llama-cookbook

# Install llama-cookbook in editable mode
RUN pip install -e .