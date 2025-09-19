# Distributed LLM Training Stack

This project provides a robust, modular solution for distributed training of large language models (LLMs) using PyTorch Distributed Data Parallel (DDP) on multi-node, multi-GPU clusters managed by Slurm. It is designed for reproducibility, scalability, and ease of use, with clear separation of environment setup, data/model download, configuration, and job submission.

## Features
- Multi-node, multi-GPU training with PyTorch DDP
- Slurm integration for cluster scheduling
- Modular scripts for environment setup and data/model download
- Config-driven hyperparameter management
- Robust logging, checkpointing, and monitoring
- Hugging Face model and dataset support

---

## Usage Guide

### 1. Prerequisites
- Access to a Slurm-managed cluster with multiple GPUs per node
- Python 3.8+ and CUDA-compatible GPUs
- [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers) installed
- (Optional) Conda or virtualenv for environment isolation

> **Note:** All the prerequisites will be installed by `setup.sh`.

### 2. Environment Setup
Set the key variables `PROJECT_NAME`, `SHARED_DIR` inside the `setup.sh` script and run the setup script to create shared directories, configure the Python environment, and install dependencies:

```bash
bash setup.sh
```

### 3. Download Model and Data
Use the provided script to download pretrained models and datasets from Hugging Face repositories.
Make sure to set these variables inside the script `MODEL_NAME` and `DATASET_NAME`

```bash
bash download_data_model.sh
```
**Features**
  - Supports FORCE_REFRESH for download and skip it if exists
  - Only Hugging Face models and datasets are supported by default for download

### 4. Configure Hyperparameters
Edit `config.yaml` to set your model's hyperparameters:
- Batch size
- Learning rate
- Number of epochs
- Model and dataset names
- Other training options

### 5. Set Up Slurm Job Script
Edit `train.slurm` to specify cluster resources and job parameters:
- Number of nodes and GPUs
- Wall time
- Environment variables

### 6. Submit Training Job
Use the wrapper script to submit your job to Slurm:

```bash
bash submit.sh
```

This will launch distributed training using torchrun, with each process assigned a GPU and a unique data shard.

---

For more details, see the documentation in `doc/architecture.md` and `doc/overview.md`.