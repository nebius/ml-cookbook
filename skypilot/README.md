# SkyPilot Integration with Nebius AI Cloud

## Overview

SkyPilot is an open-source framework for running AI and batch workloads. Nebius AI Cloud offers seamless integration with SkyPilot, simplifying the process of launching and managing distributed AI workloads on powerful GPU instances.

## Examples and Solutions

- [Basic Job](examples/basic-job.yaml): A simple job to verify GPU access ([see details here](#basic-job))
- [Mount Cloud Buckets](examples/test-cloud-bucket.yaml): Mount Nebius Object Storage to the filesystem. ([see details here](#mount-cloud-buckets))
- [S3 Migration](examples/s3_migration.yaml): Migrate data from AWS S3 to Nebius Object Storage. ([see details here](#s3-migration))
- [AI Training](examples/ai-training.yaml): Train a GPT-like model using PyTorch. ([see details here](#ai-training))
- [Distributed Training](examples/distributed-training.yaml): Multi-node distributed training using PyTorch's DDP. ([see details here](#distributed-training))
- [Infiniband Test](examples/infiniband-test.yaml): Verify high-speed Infiniband connectivity between nodes. ([see details here](#infiniband-test))
- [Llama 4 Inference with SGLang](examples/llama4-sglang.yaml): Run Llama 4 inference server using SGLang. ([see details here](#llama-4-inference-with-sglang))

## Prerequisites

Before getting started, ensure you have:

- **Nebius Account and CLI**:
  - Create your Nebius account
  - Install and configure the [Nebius CLI](https://docs.nebius.com/cli)
  - Download and run the setup script:
    ```bash
    wget https://raw.githubusercontent.com/nebius/nebius-solution-library/refs/heads/main/skypilot/nebius-setup.sh
    chmod +x nebius-setup.sh 
    ./nebius-setup.sh
    ```
    - You'll be prompted to select a Nebius tenant and project ID

- **Python Requirements**:
  - Python version 3.10 or higher
  - Install SkyPilot with Nebius support:
    ```bash
    pip install "skypilot-nightly[nebius]"
    ```

## Running SkyPilot Jobs on Nebius AI Cloud

Once you have your access token and project ID configured, SkyPilot can launch and manage clusters on Nebius. Be sure to check your Nebius quotas and request increases if you are launching GPU-intensive tasks for the first time.

The `examples` directory contains several YAML configurations that demonstrate different SkyPilot capabilities on Nebius AI Cloud:

### Basic Job

Run a simple job to verify GPU access:

```bash
$ sky launch -c basic-test examples/basic-job.yaml
...
(task, pid=3791) Do we have GPUs?
(task, pid=3791) Mon Mar 24 11:57:22 2025       
(task, pid=3791) +-----------------------------------------------------------------------------------------+
(task, pid=3791) | NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4     |
(task, pid=3791) |-----------------------------------------+------------------------+----------------------+
(task, pid=3791) | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
(task, pid=3791) | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
(task, pid=3791) |                                         |                        |               MIG M. |
(task, pid=3791) |=========================================+========================+======================|
(task, pid=3791) |   0  NVIDIA H100 80GB HBM3          On  |   00000000:8A:00.0 Off |                    0 |
(task, pid=3791) | N/A   28C    P0             68W /  700W |       1MiB /  81559MiB |      0%      Default |
(task, pid=3791) |                                         |                        |             Disabled |
(task, pid=3791) +-----------------------------------------+------------------------+----------------------+
(task, pid=3791)                                                                                          
(task, pid=3791) +-----------------------------------------------------------------------------------------+
(task, pid=3791) | Processes:                                                                              |
(task, pid=3791) |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
(task, pid=3791) |        ID   ID                                                               Usage      |
(task, pid=3791) |=========================================================================================|
(task, pid=3791) |  No running processes found                                                             |
(task, pid=3791) +-----------------------------------------------------------------------------------------+
```

This example launches a single node with 8 H100 GPUs and runs `nvidia-smi` to verify GPU access.

### Mount Cloud Buckets

Run a job that mounts Nebius Object Storage to filesystem:

```bash
$ sky launch -c test-cloud-bucket examples/test-cloud-bucket.yaml
...
(task, pid=3791) total 377487364
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:21 file_1
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:21 file_2
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:22 file_3
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:22 file_4
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:23 file_5
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:23 file_6
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:24 file_7
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:24 file_8
(task, pid=3791) -rw-r--r-- 1 ubuntu ubuntu 32212254720 Mar 10 14:25 file_9
```

### S3 Migration

Run a distributed data migration job from AWS S3 to Nebius Object Storage:

```bash
export SOURCE_AWS_PROFILE=... # e.g. default
export SOURCE_ENDPOINT_URL=... # e.g. https://s3.us-east-1.amazonaws.com
export SOURCE_BUCKET= # e.g. s3://source-bucket
export TARGET_AWS_PROFILE=nebius
export TARGET_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 # change to your region
export TARGET_BUCKET= # e.g. s3://target-bucket

# First launch
sky launch -c s3-migration examples/s3_migration.yaml \
  --env SOURCE_AWS_PROFILE \
  --env SOURCE_ENDPOINT_URL \
  --env SOURCE_BUCKET \
  --env TARGET_AWS_PROFILE \
  --env TARGET_ENDPOINT_URL \
  --env TARGET_BUCKET

# Or rerun in case of failure
sky exec s3-migration examples/s3_migration.yaml \
  --env SOURCE_AWS_PROFILE \
  --env SOURCE_ENDPOINT_URL \
  --env SOURCE_BUCKET \
  --env TARGET_AWS_PROFILE \
  --env TARGET_ENDPOINT_URL \
  --env TARGET_BUCKET
```

This example launches a distributed data migration task across multiple nodes:
- SkyPilot for provisioning multiple nodes
- `s5cmd` parallel downloading
- Performs post-transfer verification
- Supports different AWS profiles for source and target buckets (by mounting `~/.aws` directory from the local machine) 

### AI Training

Run a single-node AI training job using PyTorch:

```bash
sky launch -c ai-training examples/ai-training.yaml
```

This example trains a GPT-like model (based on minGPT) on a single node with 8 H100 GPUs.

### Distributed Training

Run a multi-node distributed training job:

```bash
sky launch -c dist-training examples/distributed-training.yaml
```

This example distributes the same minGPT training across 2 nodes, each with 8 H100 GPUs, using PyTorch's Distributed Data Parallel (DDP).

### Infiniband Test

Verify high-speed Infiniband connectivity between nodes:

```bash
sky launch -c ib-test examples/infiniband-test.yaml
```

This example launches 2 nodes and tests the Infiniband bandwidth between them using the `ib_send_bw` utility.

### Llama 4 Inference with SGLang

This section explains how to deploy Meta's Llama 4 models on Nebius AI Cloud using SkyPilot and SGLang.

**Overview**

The configuration provides two deployment options:
- **Single-node deployment** using `sky launch` (development, testing)
- **Scalable service** using `sky serve` (production with multiple replicas)

#### Deployment Options

**Option 1: Single Node Deployment with `sky launch`**

Run Llama 4 on a single node (good for development and testing):

```bash
# Set your HuggingFace token
export HF_TOKEN=<your_huggingface_token>

# Launch the inference server
sky launch -c llama4 examples/llama4-sglang.yaml --env HF_TOKEN -y
```

After deployment, you can access the endpoint:

```bash
# Get endpoint
export ENDPOINT=$(sky status --endpoint 8000 llama4)

# Make API calls
curl http://$ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {
        "role": "user", 
        "content": "Tell me about Nebius AI"
      }
    ]
  }' | jq .
```

The server is compatible with the OpenAI API format, so you can also use the Python SDK:

```python
import os
import openai

ENDPOINT = os.getenv("ENDPOINT")
client = openai.Client(base_url=f"http://{ENDPOINT}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[
        {"role": "user", "content": "Tell me about Nebius AI"},
    ],
    temperature=0
)

print(response.choices[0].message.content)
```

**Option 2: Scalable Service with `sky serve`**

Deploy as a scalable service with multiple replicas, load balancing, and autoscaling:

```bash
# Set your HuggingFace token
export HF_TOKEN=<your_huggingface_token>

# Launch as a service
sky serve up -n llama4-serve examples/llama4-sglang.yaml --env HF_TOKEN
```

The service configuration includes:
- Readiness probe with appropriate delay for model loading
- Autoscaling configuration (min/max replicas, target QPS)
- Optional security features for production

To check service status:

```bash
sky serve status
```

To access the service endpoint:

```bash
ENDPOINT=$(sky serve status --endpoint llama4-serve)
```

**Performance Considerations**

- **Scout model (17B, 16 experts)**: Fits on a single node, good performance/cost ratio
- **Maverick model (17B, 128 experts)**: Requires multi-node setup, provides higher quality

For the Maverick model, modify the configuration:
- Set `num_nodes: 2`
- Change `MODEL_NAME: meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- Increase `disk_size: 1024`
- Update the `run` section with distributed settings:
  ```bash
  run: |
    export GLOO_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}')
    MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
    TOTAL_GPUS=$(($SKYPILOT_NUM_GPUS_PER_NODE * $SKYPILOT_NUM_NODES))

    python -m sglang.launch_server \
      --model-path $MODEL_NAME \
      --tp $TOTAL_GPUS \
      --dp 8 \
      --enable-dp-attention \
      --dist-init-addr ${MASTER_ADDR}:5000 \
      --nnodes ${SKYPILOT_NUM_NODES} \
      --node-rank ${SKYPILOT_NODE_RANK} \
      --trust-remote-code \
      --torch-compile-max-bs 8 \
      --host 0.0.0.0 \
      --port 8000 \
      --chat-template llama-4
  ```

### Managing Clusters

View all your clusters:

```bash
sky status
```

Terminate a specific cluster:

```bash
sky down <cluster-name>
```

Terminate a specific service:

```bash
sky serve down <service-name>
```

Terminate all clusters and services:

```bash
sky down --all
sky serve down --all
```

## Debugging

If you switch between Service Accounts using the `nebius-setup.sh`, you might see errors when provisioning new clusters.

That could be because the [SkyPilot API server](https://docs.skypilot.co/en/latest/reference/async.html#skypilot-api-server) has cached old credentials.

You can fix this by running `sky api stop; sky api start` and then retrying. 

For other useful tips go to: https://docs.skypilot.co/en/latest/reference/faq.html
