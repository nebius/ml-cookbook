# Running multi-node GRPO experiment with VERL on a Slurm cluster
This document provides a step-by-step guide to launching a GRPO (Group Relative Policy Optimization) training job using GSM8K with `deepseek-ai/deepseek-llm-7b-chat` with [verl](https://github.com/volcengine/verl) framework on a Slurm (Soperator) cluster using Megatron backend.

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
enroot import -o ./verl-vllm012.latest.sqsh docker://verlai/verl:vllm012.latest
```

- Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
```

- Download GSM8K dataset:
```bash
PYTHONPATH=./verl python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

- Download the model checkpoint and convert to safetensors format:

The Megatron backend requires model weights in safetensors format. The following command downloads the model and saves it as safetensors:

```bash
srun --nodes=1 --time=00:30:00 --mem=64G \
  --container-image=./verl-vllm012.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'deepseek-ai/deepseek-llm-7b-chat'
output_path = 'models/deepseek-llm-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path, safe_serialization=True)
print('Download and conversion complete!')
"
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
│   └── gsm8k/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   └── deepseek-llm-7b-chat
│       ├── chat_template.jinja
│       ├── config.json
│       ├── generation_config.json
│       ├── model-00001-of-00003.safetensors
│       ├── model-00002-of-00003.safetensors
│       ├── model-00003-of-00003.safetensors
│       ├── model.safetensors.index.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── verl/
├── grpo_multinode.sh
└── verl-vllm012.latest.sqsh
```

### [Optional] Examine the `sbatch` script

The script `grpo_multinode.sh` uses pyxis containers and configures:

The script `grpo_multinode.sh` contains a number of arguments which configure the Slurm job (starting with `#SBATCH`). If you want to change the job parameters (e.g. number of nodes, GPUs, etc.), you can modify the script accordingly.

This script deploys a Ray cluster on Slurm worker nodes (head node and worker nodes). Once the cluster is ready, we submit the job with Ray driver on the head node by attaching to the job (hence `--jobid "$SLURM_JOB_ID"` argument).

### Submit the job

To submit the job, simply run:
```
sbatch grpo_multinode.sh
```

You may opt in for W&B to log your job metrics, you will need a valid `WANDB_API_KEY` in your environment to enable it.

### Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`*.out` and `*.err`). Ray job will log output in `verl_grpo_slurm.log`.

### Expected output

The script will run the GRPO training process on 2 nodes with 8 GPUs each (16 GPUs total). The output log at the end should show validation metrics with accuracy scores on GSM8K.

```bash
(TaskRunner pid=2520521) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=2520521)  "0.66868840030326, 'val-core/openai/gsm8k/acc/mean@1': 0.66868840030326, "
(TaskRunner pid=2520521)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=2520521)  "'val-aux/num_turns/mean': 2.0}")
Training Progress: 100%|██████████| 14/14 [20:56<00:00, 89.77s/it]
```
