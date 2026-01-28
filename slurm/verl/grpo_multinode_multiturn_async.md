# Running multi-node GRPO multiturn async experiment with VERL on a Slurm cluster
This document provides a step-by-step guide to launching an async multi-turn GRPO (Group Relative Policy Optimization) training job using GSM8K with `deepseek-ai/deepseek-llm-7b-chat` with [verl](https://github.com/volcengine/verl) framework on a Slurm (Soperator) cluster using SGLang backend.

## Prerequisites

Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator).

## Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Soperator has `/` mounted as a shared filesystem).

### Setup the environment

- Clone VERL repository:
```bash
git clone https://github.com/volcengine/verl.git -b v0.7.0
```

- Download VERL container image:
```bash
enroot import -o ./verl-sgl056.latest.sqsh docker://verlai/verl:sgl056.latest
```

- Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
```

- Download GSM8K multiturn dataset with interaction feedback:
```bash
PYTHONPATH=./verl python3 verl/examples/data_preprocess/gsm8k_multiturn_w_interaction.py \
    --local_save_dir data/gsm8k_multiturn
```

- Download the model checkpoint:
```bash
hf download deepseek-ai/deepseek-llm-7b-chat --local-dir models/deepseek-llm-7b-chat
```

### Create logs directory

```bash
mkdir -p logs
```

### Verify directory structure

Correctly set up working directory will look like this:
```
.
├── data/
│   └── gsm8k_multiturn/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   └── deepseek-llm-7b-chat
│       ├── README.md
│       ├── config.json
│       ├── generation_config.json
│       ├── pytorch_model-00001-of-00002.bin
│       ├── pytorch_model-00002-of-00002.bin
│       ├── pytorch_model.bin.index.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── verl/
├── grpo_multinode_multiturn_async.sh
├── verl-sgl056.latest.sqsh
└── .venv/
```

### [Optional] Examine the `sbatch` script

The script `grpo_multinode_multiturn_async.sh` uses pyxis containers and configures:

- **Async GRPO** with **multi-turn rollouts** using SGLang backend
- Ray cluster deployment across multiple Slurm worker nodes
- The `GsmInteraction` class provides feedback after each model turn

The script contains a number of arguments which configure the Slurm job (starting with `#SBATCH`). If you want to change the job parameters (e.g. number of nodes, GPUs, etc.), you can modify the script accordingly.

This script deploys a Ray cluster on Slurm worker nodes (head node and worker nodes). Once the cluster is ready, we submit the job with Ray driver on the head node by attaching to the job (hence `--jobid "$SLURM_JOB_ID"` argument).

### Submit the job

To submit the job, simply run:
```bash
sbatch grpo_multinode_multiturn_async.sh
```

You may opt in for W&B to log your job metrics, you will need a valid `WANDB_API_KEY` in your environment to enable it.

### Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`*.out` and `*.err`). Ray job will log output in `verl_grpo_slurm.log`.

### Expected output

The script will run the async multi-turn GRPO training process on 2 nodes with 8 GPUs each (16 GPUs total). The output log at the end should show validation metrics with accuracy scores on GSM8K.

```bash
(TaskRunner pid=3033680) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=3033680)  "0.6618650492797574, 'val-core/openai/gsm8k/acc/mean@1': 0.6618650492797574, "
(TaskRunner pid=3033680)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=3033680)  "'val-aux/num_turns/mean': 2.0}")
Training Progress: 100%|██████████| 29/29 [29:31<00:00, 61.09s/it]
```
