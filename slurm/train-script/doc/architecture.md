# PyTorch Distributed Data Parallel (DDP) Explained

PyTorch DDP enables efficient multi-node, multi-GPU training by distributing model replicas across devices and synchronizing gradients. This section explains key concepts and how they are used in the project.

## Key Concepts

- **Rank**: Unique identifier for each process in the distributed setup.
- **Local Rank**: GPU index on the current node assigned to a process.
- **World Size**: Total number of processes (across all nodes).
- **PyTorch Workers**: Processes handling model training; each worker is assigned a rank and local rank.
- **GPU Group**: Set of GPUs used for distributed training.
- **NCCL**: NVIDIA Collective Communications Library, used for fast, efficient GPU communication.

## How DDP Works

1. **Initialization**: Each worker process initializes the distributed process group using NCCL as the backend.
2. **Device Assignment**: Each process sets its CUDA device using its local rank.
3. **Model Replication**: The model is wrapped with `torch.nn.parallel.DistributedDataParallel`, which handles gradient synchronization.
4. **Data Loading**: Each worker loads a subset of the data using `DistributedSampler` to avoid overlap.
5. **Training Loop**: Workers perform forward and backward passes independently; gradients are synchronized after each backward pass.
6. **Checkpointing**: Model and optimizer states are periodically saved for fault tolerance.

## NCCL

NCCL is optimized for multi-GPU and multi-node communication. It handles collective operations (e.g., all-reduce for gradient synchronization) efficiently, ensuring scalable training.

![NCCL Communication Diagram](https://pytorch.org/tutorials/_images/ddp.png)

*Image source: [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)*

# Training Workflow

This section describes the end-to-end workflow implemented in the project.

## 1. Environment Setup

- Run `setup.sh` to create shared directories, set up the Python environment, and install dependencies.

## 2. Model and Dataset Download

- Download pretrained models and datasets to the shared folder using provided scripts or manual steps.

## 3. Configuration

- Edit `config.yaml` to set hyperparameters (batch size, learning rate, epochs, etc.).

## 4. Launch Training

- Submit the job using the scheduler (e.g., `sbatch train.slurm` for Slurm).
- The scheduler launches multiple worker processes, each assigned a rank and local rank.

## 5. Distributed Training

- Each worker:
  - Initializes the process group with NCCL.
  - Loads its data shard.
  - Wraps the model with DDP.
  - Runs the training loop with AMP for mixed precision.
  - Logs metrics to TensorBoard.
  - Saves checkpoints periodically.

## 6. Monitoring

- Use TensorBoard to monitor training metrics (loss, accuracy) in real time.

## 7. Resuming Training

- If interrupted, training can resume from the latest checkpoint.

## 8. Evaluation and Export

- After training, evaluate the model and export using Hugging Face’s `save_pretrained()` for future use.
