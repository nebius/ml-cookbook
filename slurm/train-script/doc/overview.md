# Project Overview

This project provides a robust, modular workflow for training large language models (LLMs) using PyTorch Distributed Data Parallel (DDP) across multiple nodes and GPUs. It supports environment setup, model and dataset download to a shared folder, and a configurable training script (`train.py`) driven by a YAML hyperparameter file. The workflow is compatible with various schedulers (Slurm, Kubernetes, RayJob) and includes TensorBoard logging for monitoring training metrics.

## Key Features

- Multi-node, multi-GPU distributed training with PyTorch DDP and AMP (Auto Mixed Precision)
- Checkpointing for fault tolerance and resuming training
- Modular configuration via `config.yaml`
- TensorBoard integration for real-time monitoring
- Scheduler-agnostic design
- Foundation for standardizing LLM training scripts

# Project Structure

```
ml-cookbook/
├── slurm/
│   └── train-script/
│       ├── train.py
│       ├── setup.sh
│       ├── config.yaml
│       ├── requirements.txt
│       ├── train.slurm
│       └── doc/
│           ├── overview.md
│           ├── architecture.md
│           └── images/
```

## Main Components

- **train.py**: Main training script implementing PyTorch DDP, AMP, checkpointing, and TensorBoard logging.
- **setup.sh**: Environment setup script for preparing shared folders and dependencies.
- **config.yaml**: Hyperparameter configuration file.
- **requirements.txt**: Python package requirements.
- **train.slurm**: Example Slurm job submission script.
- **doc/**: Documentation folder.

