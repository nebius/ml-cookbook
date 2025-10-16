# Running Llama 3.1 8B finetuning with Volcano
This document provides a step-by-step guide to a full parameter finetuning of Llama 3.1 8B model with [`lmsys/toxic-chat` dataset](https://huggingface.co/datasets/lmsys/toxic-chat) using Volcano, a Kubernetes-based batch scheduling system.
## Prerequisites

Before you start, make sure you have the following:
- A Kubernetes cluster with GPU operator, network operator and Volcano installed.
- Created resource [`Queue` named `test-queue` in Volcano](../queue.yaml).
- Prepared the [image for running this workload](../../workload-samples/llama-cookbook/Dockerfile).
- Access to the Llama 3.1 8B model weights.

## Steps

### Examine the job manifest
Let's take a look at the job manifest file `finetuning-job.yaml`:

#### PyTorch plugin
One of the very useful features of Volcano is it's plugins ecosystem. In this example, we use the `pytorch` plugin to simplify the configuration of a PyTorch job.

Typically, when running a PyTorch job, you need to specify the master and worker nodes, and configure the communication between them. With the `pytorch` plugin, you can simply specify the `--master` and `--worker` flags with task names corresponding to nodes' roles, and the plugin will handle the rest for you.

Here is how you can specify the `pytorch` plugin in your job manifest:
```yaml
plugins:
    pytorch: [
        "--master=master",
        "--worker=worker",
        "--port=29500", 
    ]
```
#### Defining the tasks

Please note that you need to define your distributed PyTorch job as 2 tasks: `master` and `worker`. The `master` task corresponds to the master node of your job, and the `worker` task corresponds to the ramining nodes. This means that if you are running distributed training on 8 nodes, you will have 1 replica in the `master` task and 7 replicas in the  `worker` task.

The `pytorch` plugin will automatically set the `MASTER_ADDR`, `MASTER_PORT` and `RANK` environment variables for you, but it is still user's responsibility to provide the `torchrun` command to the training container:
```yaml
command: 
    - /bin/bash
    - -c
    - |
    torchrun \
        --nnodes=<number_of_nodes>  \
        --nproc_per_node=<number_of_gpus_per_node>  \
        --node_rank=${RANK} \ # This is automatically set by the plugin
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \ # These are automatically set by the plugin
        <your_training_script>.py
```

To avoid rewriting the task spec, use ancords (`&`) and aliases (`*`) in YAML to reuse the task spec for `worker` task.
### [Optional] Configure the job

If you intent to run this job more than once, you can provide a shared filesystem with model weights and for saving the results. To download the model weights to a shared FS, [follow ths procedure](../../common/hf-downloader/README.md). To mount the filesystem, [you need to create a PV-PVC pair](../../common/shared-filesystem-mount/README.md) and then define a `volume` referencing the PVC:
```yaml
volumes:
    - name: persistent-storage
        persistentVolumeClaim:
        claimName: external-storage-persistent-volumeclaim
```
and then create a mount for it in the task spec in `volumeMounts`:
```yaml
volumeMounts:
    - mountPath: /workspace/persistent-storage
        name: persistent-storage
```
If you are just testing and do not want to mount the filesystem, you can remove the corresponding volume and mount and adjust the arguments of the `finetuning.py` script accordingly to download the model form Hugging Faace (the output will not be saved to any persistent storage in this case).
```yaml
finetuning.py \
--enable_fsdp \
--fsdp_config.pure_bf16 \
--use_fast_kernels \
--model_name=meta-llama/Llama-3.1-8B \ # Download model from Hugging Face
--batch_size_training=8 \
--num_workers_dataloader=4 \
--dist_checkpoint_root_folder=/ # Save checkpoint to ephemeral storage in pod
--dataset=llamaguard_toxicchat_dataset \
```
Use `--num_epochs` to control how long the job will run.
Please see the corresponding [documentation from Meta](https://github.com/meta-llama/llama-cookbook/blob/faae2fd877995430906e1d0904131ecdaa89a604/getting-started/finetuning/README.md) for configuring additional training parameters.

### Submit the job

By default, the image used for this job is `ghcr.io/nebius/ml-cookbook/pytorch-llama-cookbook:24.07-0.0.5`. The build instructions for this image are provided [here](../../workload-samples/llama-cookbook/).

Before submitting the job, create a `secret` with your HF and W&B:
```bash
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
kubectl create secret generic finetuning-job-secret \
  --from-literal=HF_TOKEN=$HF_TOKEN \
  --from-literal=WANDB_API_KEY=$WANDB_API_KEY
```
(If you dont want to report training to Weights & Biases, you can skip the `WANDB_API_KEY` variable and remove `--use_wandb` from the job argumnents)

Submit the job using the following command:
```bash
kubectl apply -f finetuning-job.yaml
```

### Monitor the job

You can monitor the pod logs using the following command in real time:
```bash
kubectl logs -f pytorch-job-master-0
```
Alternatively, you may go to your profile on Weights & Biases to monitor the training process.

### [Optional] Check the results and cleanup

To delete the pods associated with the job, you can use the following command:
```bash
kubectl delete -f finetuning-job.yaml
```

To check the training output, you may take a look in the shared FS PV (located in `fine-tuned-$model_name` directory):
```bash
root@hf-downloader:/persistent-storage/fine-tuned-/workspace/persistent-storage/models/meta-llama--Llama-3.1-8B# ll
total 15690309
drwxr-xr-x 1 root root          0 Feb 14 16:01 ./
drwxr-xr-x 1 root root          0 Feb 14 15:40 ../
-rw-r--r-- 1 root root     854782 Feb 14 16:01 .metadata
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __0_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __10_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __11_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __12_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __13_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __14_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __15_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __1_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __2_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __3_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __4_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __5_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __6_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __7_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __8_0.distcp
-rw-r--r-- 1 root root 1004126036 Feb 14 16:01 __9_0.distcp
-rw-r--r-- 1 root root       1312 Feb 14 16:01 train_params.yaml
```

The checkpoint is written in the `torch.distributed` format, which is a common format for saving model checkpoints in distributed training scenarios. Each file (e.g., `__5_0.distcp`) corresponds to a part of the model's state dictionary, and the `train_params.yaml` file contains the training parameters used during the fine-tuning process.
