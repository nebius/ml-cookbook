# LLM finetuning with `llama-cookbook` v0.0.5

This repository contains the instructions on building a custom container image for running LLM finetuning tasks using the [`llama-cookbook` library](https://github.com/meta-llama/llama-cookbook). The `llama-cookbook` is designed to simplify the process of fine-tuning Large Language Models (LLMs) by providing a set of pre-configured scripts and tools. 

The base image for this task is the recommended PyTorch release by Nvidia (24.07 in this case), which includes **CUDA** support for **GPU** acceleration as well as necessary dependencies to enable **Infiniband** support for multi-node training.

To build and push the image, run the  following:

```bash
docker build -t pytorch-llama-cookbook:24.07-0.0.5 .
```
```bash
docker tag pytorch-llama-cookbook:24.07-0.0.5 <your-container-registry>/pytorch-llama-cookbook:24.07-0.0.5
```
```bash
docker push <your-container-registry>/pytorch-llama-cookbook:24.07-0.0.5
```
