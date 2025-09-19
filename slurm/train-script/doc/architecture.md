
# PyTorch Distributed Data Parallel (DDP) Explained

DDP is for data parallelism—each process gets a full copy of the model and a shard of the data, and synchronizes gradients. It uses torchrun to launch one process per GPU. Each process (on one GPU) gets a full copy of the model. DistributedSampler ensures each process gets a unique data shard. NCCL backend is used for fast GPU communication. This section explains key concepts and how they are used in the project.


## Key Concepts

- **Rank**: Unique identifier for each process in the distributed setup.
- **Local Rank**: GPU index on the current node assigned to a process.
- **World Size**: Total number of processes (across all nodes).
- **PyTorch Workers**: Processes handling model training; each worker is assigned a rank and local rank.
- **Process Group**: A collection of processes that can communicate with each other in distributed training. In PyTorch, process groups are initialized to manage communication (e.g., gradient synchronization) among participating worker processes.
- **MASTER_ADDR**: The hostname or IP address of the node designated as the rendezvous (master) for process group initialization. All processes use this address to find and connect to the master for coordination.
- **MASTER_PORT**: The TCP port on the master node used for communication and rendezvous. All processes must use the same port to join the same process group.
- **NCCL**: NVIDIA Collective Communications Library, used for fast, efficient GPU communication. NCCL is the recommended backend for multi-GPU training on CUDA devices.
- **Gloo/MPI**: Alternative backends for CPU or non-NVIDIA GPU training.
- **torchrun**: Recommended launcher for DDP jobs, which sets up environment variables and spawns processes correctly.

## How DDP Works

1. **Process Launch**: Use `torchrun` or `torch.multiprocessing.spawn` to launch one process per GPU. Each process is assigned a unique rank and local rank.
2. **Initialization**: Each worker process initializes the distributed process group using NCCL (or another backend) as the communication layer.
3. **Device Assignment**: Each process sets its CUDA device using its local rank.
4. **Model Replication**: The model is wrapped with `torch.nn.parallel.DistributedDataParallel`, which handles gradient synchronization. DDP broadcasts the initial model state from rank 0 to all other ranks.
5. **Data Loading**: Each worker loads a subset of the data using `DistributedSampler` to avoid overlap.
6. **Training Loop**: Workers perform forward and backward passes independently; gradients are synchronized after each backward pass using efficient bucketed all-reduce operations.
7. **Checkpointing**: Model and optimizer states are periodically saved for fault tolerance.
8. **Optimizer Step**: Each process updates its local model parameters using the averaged gradients.

## Internal Design Details

- **Reducer and Buckets**: DDP organizes gradients into buckets to optimize communication. When all gradients in a bucket are ready, DDP triggers an asynchronous all-reduce operation.
- **Autograd Hooks**: DDP registers hooks on each parameter to synchronize gradients as soon as they are computed.
- **find_unused_parameters**: If your model has conditional branches, set this to `True` in DDP to avoid hangs, but note the extra overhead.
- **Synchronization Order**: All processes must invoke all-reduce operations in the same order to avoid hangs or incorrect results.

![Communication Diagram](images/ddp.png)

*source: [PyTorch DDP Tutorial](https://docs.pytorch.org/docs/stable/notes/ddp.html)*

## Supported Backends

- **NCCL**: Recommended for CUDA GPUs.
- **Gloo**: For CPU or non-NVIDIA GPU training.
- **MPI**: For advanced users with MPI environments.

## Best Practices

- Always use one process per GPU (torchrun or mp.spawn).
- Set environment variables: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (torchrun does this automatically).
- Avoid calling `.cuda()` inside the model’s forward pass; set device before wrapping with DDP.
- Use `DistributedSampler` for your DataLoader.
- Wrap your model with DDP before compiling with TorchDynamo (if using).
- Save checkpoints on rank 0 only to avoid file contention.

## Limitations

- DDP does not support model parallelism (splitting a single model across multiple GPUs).
- All processes must follow the same code path and synchronize in the same order.
- Some PyTorch features (e.g., certain hooks, buffers) may require special handling.

## Debugging

- Set `TORCH_LOGS=distributed` for verbose DDP logs.
- Use `NCCL_DEBUG=INFO` for NCCL communication debugging.

## NCCL Communication

NCCL is optimized for multi-GPU and multi-node communication. It handles collective operations (e.g., all-reduce for gradient synchronization) efficiently, ensuring scalable training.



## References

- [PyTorch DDP Documentation](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [DDP API Reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

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
