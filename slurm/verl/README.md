# Running multi-node experiment with VERL on a Slurm cluster
This document provides a step-by-step guide to a launching a PPO training job using GSM8K with `deepseek-ai/deepseek-llm-7b-chat` from [VERL examples](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html#step-4-perform-ppo-training-with-your-model-on-gsm8k-dataset) on Slurm (Soperator) cluster.

## Prerequisites

Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator).

## Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Sopeartor has `/` mounted as a shared filesystem).

### Setup the environment

- clone VERL repository:
```
git clone https://github.com/volcengine/verl.git -b v0.7.0
```

- download VERL container image:
```
enroot import -o ./verl-vllm012.latest.sqsh docker://verlai/verl:vllm012.latest
```
- create a virtual environment and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
```

- download GSM8K dataset:
```
PYTHONPATH=./verl python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

- [optional] download the model checkpoint:
```
huggingface-cli download deepseek-ai/deepseek-math-7b-instruct --local-dir models/deepseek-math-7b-instruct --local-dir-use-symlinks False
```

Correctly set up working directory will look like this:
```
.
├── data
├── models
├── README.md
├── verl
├── verl_gsm8k_example.sh
└── verl-vllm012.latest.sqsh
```

### [Optional] Examine the `sbatch` script

The script `verl_gsm8k_example.sh `contains a number of arguments which configure Slurm job (starting with `#SBATCH`). If you want to change the job parameters (e.g. number of nodes, GPUs, etc.), you can modify the script accordingly.

This script deploys a Ray cluster on Slurm worker nodes (head node and worker nodes). Once the cluster is ready, we submit the job with Ray driver on the head node by attaching to the job (hence `--jobid "$SLURM_JOB_ID"` argument).

### Submit the job

To submitt the job, simply run:
```
sbatch verl_gsm8k_example.sh
```

You may opt in for W&B to log your job metrics, you will need a valid `WANDB_API_KEY` in your environment to enable it.

### Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`*.out` and `*.err`). Ray job will log output in `verl_demo_slurm.log`.

### Expected output

The script will run the PPO training process on 2 nodes with 8 GPUs each (16 GPUs total). On H200 GPUs this takes about 35 minutes. The output log at the end should look similar to the following:

```
(TaskRunner pid=902322) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=902322)  "0.8362395754359363, 'val-core/openai/gsm8k/acc/mean@1': 0.8362395754359363, "
(TaskRunner pid=902322)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=902322)  "'val-aux/num_turns/mean': 2.0}")
```