#!/bin/bash
set -x
source "$VENV_DIR/project.env"
# -----------------------------------------------------------------------------
# Usage Examples:
#   export VENV_DIR=/shared/venvs/distilbert-train
#   export SLURM_LOGS_DIR=/shared/slurm_logs/distilbert-train/
#   sbatch --export=ALL train.slurm
# Or
# (with wrapper script)
#   ./submit.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=pt-ddp-train           # Set the job name
##SBATCH --account=research               # Specify the account to charge
#SBATCH --nodes=1                         # Number of nodes to use
#SBATCH --gpus-per-node=8                 # Number of GPUs per node
#SBATCH --ntasks-per-node=1               # Number of tasks per node, meaning one srun command per node
#SBATCH --cpus-per-task=64                # CPU cores per task, per each srun command
#SBATCH --mem=400G                        # Memory per node
#SBATCH --time=02:00:00                   # (2 hours) Wall time limit, days-hours:minutes:seconds
#SBATCH --output=${SLURM_LOGS_DIR}/%x-%j.out # Standard output log
#SBATCH --error=${SLURM_LOGS_DIR}/%x-%j.err  # Standard error log


# Expect VENV_DIR to be exported before submission (see usage header).


# Activate Python virtual environment
source "$VENV_DIR/bin/activate"

# Set NCCL and OMP env vars for performance
export OMP_NUM_THREADS=8      # Set number of OpenMP threads per process (set to cpus-per-task divided by number of GPUs per node)
export NCCL_DEBUG=WARN        # Enable NCCL debug logging for troubleshooting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # PyTorch expects this (replaces deprecated NCCL_ASYNC_ERROR_HANDLING warning)
export SLURM_CPU_BIND=cores      # Slurm will bind each task to a set of CPU cores rather than OS (improves performance)
export MKL_NUM_THREADS=$OMP_NUM_THREADS          # Align MKL thread pool with per-process OMP threads to prevent oversubscription
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS     # Keep OpenBLAS thread usage consistent (avoids CPU thrash / context switching)
export PYTHONFAULTHANDLER=1                      # Emit Python traceback immediately on fatal signals/exceptions (faster debugging)
export TOKENIZERS_PARALLELISM=false              # Disable HF tokenizers intra-process thread pool (avoids fork warning & CPU contention)

# Run distributed training using torchrun
# --nnodes: Total number of nodes to use (from Slurm allocation by srun)
# --nproc_per_node: Number of processes (tasks) to launch per node. Standard usage is to set it equal to the number of GPUs per node
# --rdzv_id: Unique rendezvous ID for this job (ensures all nodes join the same group)
# --rdzv_backend: Rendezvous backend for process group initialization (c10d is standard for PyTorch DDP)
# --rdzv_endpoint: Address and port for rendezvous (first node in allocation, port 29500)
# --max_restarts=2: means it will attempt up to 2 restarts before failing the job if any rank crashes
# Get the first node's hostname for rendezvous
RDZV_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)

echo "Running srun torchrun on $SLURM_JOB_NUM_NODES nodes with $SLURM_GRES gpus total"
echo "Rendezvous endpoint: $RDZV_HOST:29500"
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$RDZV_HOST:29500 \
    --max_restarts=2 \
    train.py --config config.yaml --resume
