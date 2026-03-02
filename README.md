# ML Cookbook 🍳

Welcome to the **ML Cookbook** repository! This is your one-stop guide for common use cases in training and inference of machine learning models on the **Nebius.ai** cloud platform. Whether you're a seasoned data scientist or just starting your ML journey, this repository provides practical examples, best practices, and ready-to-use code snippets to help you get the most out of your ML workflows.

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