# ML Cookbook 🍳

Welcome to the **ML Cookbook** repository! This is your one-stop guide for common use cases in training and inference of machine learning models on the **Nebius.ai** cloud platform. Whether you're a seasoned data scientist or just starting your ML journey, this repository provides practical examples, best practices, and ready-to-use code snippets to help you get the most out of your ML workflows.

## Start Here

This repository is a collection of **recipes**, not a single application to deploy end-to-end.

To get started, first choose:

- your **environment**: `Slurm / Soperator`, `Kubernetes`, `SkyPilot`, or `Run:ai`
- your **goal**: `training`, `fine-tuning`, `inference`, `storage setup`, or `network validation`

### 1. Choose your environment

| If you are using... | Start here |
| --- | --- |
| Nebius Soperator / Slurm cluster | [`slurm/`](./slurm/) |
| Nebius Managed Kubernetes | [`volcano/`](./volcano/) and [`common/`](./common/) |
| SkyPilot | [`skypilot/`](./skypilot/) |
| Run:ai | [`runai/`](./runai/) |

### 2. Choose your goal

| Goal | Recommended starting point |
| --- | --- |
| Verify your GPU environment works | [`skypilot/examples/basic-job.yaml`](./skypilot/examples/basic-job.yaml) or [`volcano/nccl-test-pytorch/`](./volcano/nccl-test-pytorch/) |
| Run a simple distributed training example | [`slurm/hf-accelerate/`](./slurm/hf-accelerate/) |
| Persist and recover Slurm training checkpoints | [`slurm/object-storage-checkpointing/`](./slurm/object-storage-checkpointing/) |
| Use a generic PyTorch DDP training template | [`slurm/train-script/`](./slurm/train-script/) |
| Fine-tune an LLM on Slurm | [`slurm/torchtune/`](./slurm/torchtune/) |
| Run RLHF / VERL training | [`slurm/verl/`](./slurm/verl/) |
| Try agentic web intelligence workflows with Tavily | [`agents/tavily/`](./agents/tavily/) |
| Fine-tune on Kubernetes with Volcano | [`volcano/llama-cookbook-finetuning/`](./volcano/llama-cookbook-finetuning/) |
| Prepare shared filesystem storage on Kubernetes | [`common/shared-filesystem-mount/`](./common/shared-filesystem-mount/) |
| Download model weights or datasets to shared storage | [`common/hf-downloader/`](./common/hf-downloader/) |
| Serve Llama 4 with SkyPilot + SGLang | [`skypilot/examples/llama4-sglang.yaml`](./skypilot/examples/llama4-sglang.yaml) |
| Run the large-scale DeepSeek-V3 B200 recipe | [`pytorch-dsv3-mxfp8/`](./pytorch-dsv3-mxfp8/) |

### 3. Recommended first path for new users

If you are new to this repository, start with the smallest working example for your platform:

- **Slurm / Soperator**: [`slurm/hf-accelerate/`](./slurm/hf-accelerate/)
- **Kubernetes**: [`common/shared-filesystem-mount/`](./common/shared-filesystem-mount/) then [`volcano/nccl-test-pytorch/`](./volcano/nccl-test-pytorch/)
- **SkyPilot**: [`skypilot/examples/basic-job.yaml`](./skypilot/examples/basic-job.yaml)
- **Run:ai**: [`runai/`](./runai/)

These recipes are the fastest way to verify that your environment, credentials, GPUs, and networking are set up correctly before moving on to larger workloads.

### 4. Common prerequisites

Most recipes assume some combination of the following:

- access to Nebius infrastructure
- configured CLI access such as `kubectl`, `sky`, `runai`, or SSH access to a Slurm login node
- enough GPU quota for the selected workload
- a Hugging Face token for gated model downloads
- an optional Weights & Biases API key for logging
- shared filesystem or object storage for larger models and datasets

### 5. Typical workflow

Most recipes follow the same high-level flow:

1. Provision or connect to a cluster
2. Configure storage and credentials
3. Prepare the environment with `setup.sh` or a container image
4. Download model weights and/or datasets
5. Submit the job with `sbatch`, `kubectl apply`, `sky launch`, or `runai`
6. Monitor logs and metrics

## Repository Map

- [`common/`](./common/) - Shared Kubernetes utilities such as filesystem mounts and model download pods
- [`agents/tavily/`](./agents/tavily/) - Tavily-based agent recipes for search, extraction, mapping, crawling, and research
- [`deepep/`](./deepep/) - DeepEP installation and RDMA / NVSHMEM setup guidance
- [`pytorch-dsv3-mxfp8/`](./pytorch-dsv3-mxfp8/) - DeepSeek-V3 pre-training recipes for large B200 Slurm clusters
- [`runai/`](./runai/) - Run:ai examples for distributed MPI / NCCL validation
- [`skypilot/`](./skypilot/) - SkyPilot job examples for training, inference, storage, and migration
- [`slurm/`](./slurm/) - Slurm / Soperator recipes for distributed training, durable checkpoint recovery, fine-tuning, and RLHF
- [`volcano/`](./volcano/) - Volcano scheduler examples for Kubernetes batch workloads
- [`workload-samples/`](./workload-samples/) - Supporting container build examples for selected workloads

## 🚀 What's Inside?

This repository is organized into **recipes** that cover a variety of ML tasks, leveraging popular tools and technologies such as:

- **Kubernetes (K8s)**: Scalable and efficient orchestration of ML workloads.
- **NVIDIA GPUs**: Accelerate training and inference with CUDA-enabled hardware.
- **Linux**: Optimized environments for ML development.
- **Open Source Tools**: Leverage the power of open-source libraries and frameworks.

Each recipe includes:
- **Step-by-step instructions** for setup and execution.
- **Code examples** for training and inference.
- **Tips and tricks** to optimize performance and avoid common pitfalls.

## 📚 Recipes

Here’s a sneak peek of the recipes available in this cookbook:

### 1. **DeepSeek-V3 Pre-training with MXFP8 + DeepEP** ([pytorch-dsv3-mxfp8](pytorch-dsv3-mxfp8/))
   - TorchTitan recipes for DeepSeek-V3 671B and 16B MoE on 256 NVIDIA B200 GPUs.
   - Combines MXFP8 via TorchAO with DeepEP expert-parallel communication for **+41% throughput** over BF16.
   - Reproducible Slurm scripts for tested configurations.

### 2. **Training a Deep Learning Model on Kubernetes**
   - Deploy a distributed training job using Kubernetes.
   - Leverage NVIDIA GPUs for accelerated training.
   - Monitor and scale your training workload.

### 3. **Agentic Web Intelligence with Tavily** ([agents/tavily](agents/tavily/))
   - Search, extract, map, crawl, and research workflows using Tavily APIs.
   - Nebius-shaped recipes for SkyPilot and Slurm.
   - Good fit for agent demos, corpus building, and research workflows.

## Quick Navigation by Use Case

- **I want the simplest possible first run**
  - Start with [`skypilot/examples/basic-job.yaml`](./skypilot/examples/basic-job.yaml) for SkyPilot
  - Start with [`slurm/hf-accelerate/`](./slurm/hf-accelerate/) for Slurm / Soperator
  - Start with [`volcano/nccl-test-pytorch/`](./volcano/nccl-test-pytorch/) for Kubernetes + Volcano

- **I want to fine-tune a model**
  - Use [`slurm/torchtune/`](./slurm/torchtune/) for Slurm-based LLM fine-tuning
  - Use [`volcano/llama-cookbook-finetuning/`](./volcano/llama-cookbook-finetuning/) for Kubernetes + Volcano

- **I want to run reinforcement learning / RLHF**
  - Use [`slurm/verl/`](./slurm/verl/)
  - See [`skypilot/examples/verl-grpo-multiturn-async.yaml`](./skypilot/examples/verl-grpo-multiturn-async.yaml) for a SkyPilot-based example

- **I want to try an agent workflow**
  - Start with [`agents/tavily/search/`](./agents/tavily/search/)
  - Use [`agents/tavily/research/`](./agents/tavily/research/) for a higher-value report-generation workflow
  - See [`agents/tavily/`](./agents/tavily/) for all five Tavily capabilities

- **I want to validate cluster communication**
  - Use [`runai/`](./runai/)
  - Use [`skypilot/examples/nccl-test.yaml`](./skypilot/examples/nccl-test.yaml)
  - Use [`volcano/nccl-test-pytorch/`](./volcano/nccl-test-pytorch/)

- **I need shared storage before training**
  - Use [`common/shared-filesystem-mount/`](./common/shared-filesystem-mount/)
  - Then use [`common/hf-downloader/`](./common/hf-downloader/)

- **I need durable Object Storage checkpoints for Slurm training**
  - Use [`slurm/object-storage-checkpointing/`](./slurm/object-storage-checkpointing/)

## License

Copyright 2025 Nebius B.V.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
