# Running multi-node GRPO multiturn async experiment with VERL on Slurm

This guide describes how to run a **multi-node, multi-turn, asynchronous GRPO** experiment using VERL on a Slurm cluster. It mirrors `grpo_multinode.md`, adapted for **multiturn async** setup and **Qwen2.5-3B-Instruct**.

---

## Prerequisites

- Access to a Slurm (Soperator) cluster
- Enroot + Pyxis enabled
- Hugging Face access token configured (`huggingface-cli login`)
- Shared filesystem across nodes

---

## Setup

### Clone VERL

```bash
git clone https://github.com/volcengine/verl.git -b v0.7.0
```

### Download VERL container image

```bash
enroot import -o ./verl-sgl056.latest.sqsh docker://verlai/verl:sgl056.latest
```

### Create Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
```

---

## Prepare dataset (GSM8K multiturn with tool calls)

```bash
PYTHONPATH=./verl python3 gsm8k_multiturn_w_tool_call.py \
  --local_save_dir data/gsm8k
```

Expected output:
```
data/gsm8k/
├── train.parquet
└── test.parquet
```

---

## Download model checkpoint

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /root/models/qwen2.5-3b-instruct \
  --local-dir-use-symlinks False
```

---

## Directory structure

After setup, your working directory should look like:

```
.
├── data/
│   └── gsm8k/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   └── qwen2.5-3b-instruct/
├── verl/
├── grpo_multinode_multiturn_async.sh
├── gsm8k_multiturn_w_tool_call.py
├── verl-sgl056.latest.sqsh
└── .venv/
```

---

## Slurm job script

The script `grpo_multinode_multiturn_async.sh`:

- Launches a Ray cluster across multiple nodes
- Uses **async GRPO** with **multi-turn rollouts**
- Runs inside the VERL SGLang container
- Attaches the Ray driver to the Slurm job

You can edit `#SBATCH` parameters in the script to adjust:
- Number of nodes
- GPUs per node
- Time and memory limits

Before launching the Slurm job, export your `WANDB_API_KEY` inside the `grpo_multinode_multiturn_async.sh` script.

---

## Submit the job

```bash
sbatch grpo_multinode_multiturn_async.sh
```

---

## Monitoring

- Job status:
```bash
squeue -u $USER
```

- Slurm logs:
```
logs/*.out
logs/*.err
```

- Metrics and text output: `https://wandb.ai/`.

---

## Notes

- Async multiturn GRPO overlaps rollout generation and optimization steps
- Tool-call trajectories are handled during rollout
- Optional: enable Weights & Biases by exporting `WANDB_API_KEY`
