# Working with Shared File System in Nebius

This guide provides instructions on how to work with a shared file system mounted to a managed kubernetes cluster in Nebius. A shared file system allows multiple pods and containers to access and share the same set of files.

## Prerequisites

Before you start, make sure you have the following:
- A k8s cluster provisioned with attached shared FS. Follow [this procedure](https://github.com/nebius/nebius-solution-library/tree/main/k8s-training) to provision a cluster, make sure to set `enable_filestore = true` in the `terraform.tfvars` file to attach a shared FS to cluster nodes.
- You have created a PV-PVC pair for the shared FS. Follow [this procedure](../shared-filesystem-mount).

## Overview

When working with shared FS attached to Nebius mk8s cluster, please keep in mind the following:
- The shared FS is mounted to all nodes at `/mnt/data` by default (specified in `cloud-init` during cluster creation).
- When using the CSI driver with a shared FS, it creates a separate subdirectory within the mount for each Persistent Volume (PV). To ensure consistency when copying files to a specific PV, always use the same PV-PVC pair for copying and consuming these files (use the same `claimName` in your volume definition). You will not be able to access your files from another PV-PVC pair.

## Example: Downloading models from Hugging Face to the shared FS

When training or finetuning models, you often need to download large datasets and models to some local storage in order to avoid loosing time on repeated downloads. In this example, we will download model weights from Hugging Face to the shared FS.

### Create a Persistent Volume and Persistent Volume Claim

Create a [Persistent Volume (PV) and Persistent Volume Claim (PVC) pair](../shared-filesystem-mount/README.md) to mount the shared FS.

### Create a Pod to download the data

Create a pod that mounts the PVC and uses the Hugging Face Hub CLI to download the data:
```bash
kubectl apply -f pod.yaml
```
Connect to the pod:
```bash
kubectl exec -it hf-downloader -- /bin/bash
```
The PV corresponding to shared FS is mounted at `/persistent-storage`.

Let's download base model weights (Note that the model requires permission to download from Hugging Face) and dataset:
```bash
# Llama model from Meta requires authentication to download
huggingface-cli login
mkdir -p /persistent-storage/models/hf-meta-llama3-8B
mkdir -p /persistent-storage/models/hf-meta-llama3-70B
mkdir -p /persistent-storage/dataset/samsum
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /persistent-storage/models/hf-meta-llama3-8B --exclude="original/*"
huggingface-cli download meta-llama/Llama-3.1-70B --local-dir /persistent-storage/models/hf-meta-llama3-70B --exclude="original/*"
huggingface-cli repo clone datasets/knkarthick/samsum /persistent-storage/dataset/samsum
```

### Clean up

To delete the pod, run the following command:
```bash
kubectl delete -f pod.yaml
```
