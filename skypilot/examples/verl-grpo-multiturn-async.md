# VERL GRPO Multi-turn Async Training on SkyPilot

This document provides a step-by-step guide to launching an async multi-turn GRPO (Group Relative Policy Optimization) training job using GSM8K with `deepseek-ai/deepseek-llm-7b-chat` using the [VERL](https://github.com/volcengine/verl) framework on Nebius AI Cloud via SkyPilot.

## Overview

This example demonstrates:
- **Async GRPO** with **multi-turn rollouts** using SGLang backend
- Ray cluster deployment across multiple SkyPilot nodes
- The `GsmInteraction` class provides feedback after each model turn
- Training on 2 nodes with 8 H100 GPUs each (16 GPUs total)

## Prerequisites

Before you start, ensure you have:

1. **Nebius Account and CLI**:
   - Create your Nebius account
   - Install and configure the [Nebius CLI](https://docs.nebius.com/cli)
   - Run the setup script:
     ```bash
     wget https://raw.githubusercontent.com/nebius/nebius-solution-library/refs/heads/main/skypilot/nebius-setup.sh
     chmod +x nebius-setup.sh 
     ./nebius-setup.sh
     ```

2. **Python Requirements**:
   - Python version 3.10 or higher
   - Install SkyPilot with Nebius support:
     ```bash
     pip install "skypilot-nightly[nebius]"
     ```

3. **[Optional] Weights & Biases API Key**:
   - For logging training metrics to W&B
   - Get your API key from [W&B Settings](https://wandb.ai/settings)

## Launch the Training Job

### Basic Launch

Launch the training job:

```bash
sky launch -c verl-grpo examples/verl-grpo-multiturn-async.yaml -y
```

### With W&B Logging

To enable Weights & Biases logging:

```bash
export WANDB_API_KEY=<your_wandb_api_key>

sky launch -c verl-grpo examples/verl-grpo-multiturn-async.yaml --env WANDB_API_KEY -y
```

## Monitor the Job

### Check Cluster Status

```bash
sky status
```

### View Logs

Stream the logs from the running job:

```bash
sky logs verl-grpo
```

### SSH into the Cluster

For debugging or interactive access:

```bash
ssh verl-grpo
```

## Expected Output

The training will run on 2 nodes with 8 GPUs each (16 GPUs total). At the end of training, you should see validation metrics similar to:

```
(TaskRunner pid=3033680) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=3033680)  "0.6618650492797574, 'val-core/openai/gsm8k/acc/mean@1': 0.6618650492797574, "
(TaskRunner pid=3033680)  "'val-aux/num_turns/min': 2, 'val-aux/num_turns/max': 2, "
(TaskRunner pid=3033680)  "'val-aux/num_turns/mean': 2.0}")
Training Progress: 100%|██████████| 29/29 [29:31<00:00, 61.09s/it]
```

## Configuration Details

The job uses the following configuration:

| Parameter | Value |
|-----------|-------|
| Model | `deepseek-ai/deepseek-llm-7b-chat` |
| Dataset | GSM8K multi-turn with interaction feedback |
| Nodes | 2 |
| GPUs per node | 8x H100 |
| Rollout backend | SGLang |
| Rollout mode | Async |
| Train batch size | 256 |
| Max prompt length | 1024 |
| Max response length | 1024 |
| Tensor parallel size | 2 |
| Total epochs | 1 |

## Cleanup

When finished, terminate the cluster to stop billing:

```bash
sky down verl-grpo
```

Or terminate all clusters:

```bash
sky down -a
```

## Troubleshooting

### Ray Cluster Issues

If the Ray cluster fails to start, check the logs for connection errors:

```bash
sky logs verl-grpo --status
```

### Out of Memory

If you encounter OOM errors, try reducing:
- `data.train_batch_size`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.gpu_memory_utilization`

### Model Download Issues

Check that the model can be downloaded from HuggingFace:

```bash
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --dry-run
```

## References

- [VERL GitHub Repository](https://github.com/volcengine/verl)
- [SkyPilot Documentation](https://docs.skypilot.co/)
- [Nebius AI Cloud](https://nebius.com/)
- [GSM8K Dataset](https://github.com/openai/grade-school-math)
