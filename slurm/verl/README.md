# VERL Training on Slurm (Soperator)

## Overview

[VERL](https://github.com/volcengine/verl) is an open-source framework for reinforcement learning from human feedback (RLHF) training. These examples demonstrate how to run multi-node VERL training jobs on a [Nebius Soperator](https://nebius.com/services/soperator) cluster using Ray for distributed orchestration and Pyxis/Enroot for container management.

## Examples

- [PPO Multi-node](ppo_multinode.sh): PPO training with FSDP backend on GSM8K ([see details](#ppo-multi-node))
- [GRPO Multi-node](grpo_multinode.sh): GRPO training with Megatron backend on GSM8K ([see details](#grpo-multi-node))
- [GRPO Multi-turn Async Multi-node](grpo_multinode_multiturn_async.sh): Async multi-turn GRPO training with SGLang using FSDP on GSM8K ([see details](#grpo-multi-turn-async))

## Prerequisites

Before you start, make sure you have:

- Access to a [Soperator cluster](https://nebius.com/services/soperator)
- SSH access to the login node

## Common Setup

All examples share the same initial steps. SSH to the login node and run the following on the shared filesystem (by default, Soperator has `/` mounted as a shared filesystem).

### Clone VERL repository

```bash
git clone https://github.com/volcengine/verl.git -b v0.7.0
```

### Create the logs and models directories

```bash
mkdir -p logs models
```

---

## PPO Multi-node (FSDP)

PPO (Proximal Policy Optimization) training using `deepseek-ai/deepseek-math-7b-instruct` on GSM8K with FSDP backend.

### Setup

Download the container image:

```bash
enroot import -o ./verl-vllm012.latest.sqsh docker://verlai/verl:vllm012.latest
```

Download the GSM8K dataset:

```bash
srun --nodes=1 --ntasks=1 \
  --container-image=./verl-vllm012.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  bash -c "PYTHONPATH=$(pwd)/verl python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k"
```

Download the model checkpoint:

```bash
srun --nodes=1 --ntasks=1 \
  --container-image=./verl-vllm012.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  hf download deepseek-ai/deepseek-math-7b-instruct \
    --local-dir models/deepseek-math-7b-instruct
```

### Verify directory structure

```
.
├── data
│   └── gsm8k
│       ├── test.parquet
│       └── train.parquet
├── logs
├── ml-cookbook
├── models
│   └── deepseek-math-7b-instruct
│       ├── LICENSE
│       ├── README.md
│       ├── config.json
│       ├── generation_config.json
│       ├── pytorch_model-00001-of-00002.bin
│       ├── pytorch_model-00002-of-00002.bin
│       ├── pytorch_model.bin.index.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── verl
└── verl-vllm012.latest.sqsh
```

### Submit the job

```bash
sbatch ml-cookbook/slurm/verl/ppo_multinode.sh
```

Optionally enable W&B logging by exporting `WANDB_API_KEY` before submitting.

### Monitor

Check job status with `squeue`. Logs are written to `logs/slurm-<jobid>.out` / `logs/slurm-<jobid>.err` and `logs/verl_ppo_slurm.log`.

### Expected output

The script runs PPO training on 2 nodes with 8 GPUs each (16 GPUs total). On H100 GPUs the training portion takes about 5 minutes. The output log should show validation metrics:

```
(TaskRunner pid=358695) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=358695)  "0.8036391205458681, 'val-core/openai/gsm8k/acc/mean@1': 0.8036391205458681, "
(TaskRunner pid=358695)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=358695)  "'val-aux/num_turns/mean': 2.0}")
```

---

## GRPO Multi-node (Megatron)

GRPO (Group Relative Policy Optimization) training using `deepseek-ai/deepseek-llm-7b-chat` on GSM8K with Megatron backend.

### Setup

Download the container image:

```bash
enroot import -o ./verl-vllm012.latest.sqsh docker://verlai/verl:vllm012.latest
```

Download the GSM8K dataset:

```bash
srun --nodes=1 --ntasks=1 \
  --container-image=./verl-vllm012.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  bash -c "PYTHONPATH=$(pwd)/verl python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k"
```

Download the model checkpoint and convert to safetensors format (required by the Megatron backend):

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

### Verify directory structure

```
.
├── data
│   └── gsm8k
│       ├── test.parquet
│       └── train.parquet
├── logs
├── ml-cookbook
├── models
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
├── verl
└── verl-vllm012.latest.sqsh
```

### Submit the job

```bash
sbatch ml-cookbook/slurm/verl/grpo_multinode.sh
```

Optionally enable W&B logging by exporting `WANDB_API_KEY` before submitting.

### Monitor

Check job status with `squeue`. Logs are written to `logs/slurm-<jobid>.out` / `logs/slurm-<jobid>.err` and `logs/verl_grpo_slurm.log`.

### Expected output

The script runs GRPO training on 2 nodes with 8 GPUs each (16 GPUs total). On H100 GPUs the training portion takes about 20 minutes. The output log should show validation metrics:

```
(TaskRunner pid=395646) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=395646)  "0.6755117513267627, 'val-core/openai/gsm8k/acc/mean@1': 0.6755117513267627, "
(TaskRunner pid=395646)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=395646)  "'val-aux/num_turns/mean': 2.0}")
```

---

## GRPO Multi-turn Async (FSDP)

Async multi-turn GRPO training using `deepseek-ai/deepseek-llm-7b-chat` on GSM8K with SGLang. The `GsmInteraction` class provides feedback after each model turn, enabling multi-turn rollouts with interaction feedback.

### Setup

Download the container image (note: this uses the SGLang-based image, different from the vLLM image above):

```bash
enroot import -o ./verl-sgl056.latest.sqsh docker://verlai/verl:sgl056.latest
```

Download the GSM8K multiturn dataset with interaction feedback:

```bash
srun --nodes=1 --ntasks=1 \
  --container-image=./verl-sgl056.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  bash -c "PYTHONPATH=$(pwd)/verl python3 verl/examples/data_preprocess/gsm8k_multiturn_w_interaction.py \
    --local_save_dir data/gsm8k_multiturn"
```

Download the model checkpoint:

```bash
srun --nodes=1 --ntasks=1 \
  --container-image=./verl-sgl056.latest.sqsh \
  --container-mounts=$(pwd):$(pwd) \
  --container-workdir=$(pwd) \
  hf download deepseek-ai/deepseek-llm-7b-chat \
    --local-dir models/deepseek-llm-7b-chat
```

### Verify directory structure

```
.
├── data
│   └── gsm8k_multiturn
│       ├── test.parquet
│       └── train.parquet
├── ml-cookbook
├── models
│   └── deepseek-llm-7b-chat
│       ├── README.md
│       ├── config.json
│       ├── generation_config.json
│       ├── pytorch_model-00001-of-00002.bin
│       ├── pytorch_model-00002-of-00002.bin
│       ├── pytorch_model.bin.index.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── verl
├── verl-sgl056.latest.sqsh
```

### Submit the job

```bash
sbatch ml-cookbook/slurm/verl/grpo_multinode_multiturn_async.sh
```

Optionally enable W&B logging by exporting `WANDB_API_KEY` before submitting.

### Monitor

Check job status with `squeue`. Logs are written to `logs/slurm-<jobid>.out` / `logs/slurm-<jobid>.err` and `logs/verl_grpo_multiturn_async_slurm.log`.

### Expected output

The script runs GRPO multi-turn async training on 2 nodes with 8 GPUs each (16 GPUs total). On H100 GPUs the training portion takes about 20 minutes. The output log should show validation metrics:

```
(TaskRunner pid=458106) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=458106)  "0.6679302501895376, 'val-core/openai/gsm8k/acc/mean@1': 0.6679302501895376, "
(TaskRunner pid=458106)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=458106)  "'val-aux/num_turns/mean': 2.0}")
```
