# Running multi-node BERT pretraining with HuggingFace Accelerate
This document provides a step-by-step guide to a launching a training job for the BERT model with [HF Accelerate](https://github.com/huggingface/accelerate) launcher on Slurm (Soperator) cluster. We will use example workload from [HF Accelerate examples](https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py).

## Prerequisites

Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator).

## Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Sopeartor has `/` mounted as a shared filesystem).

### Setup the environment

Execute the setup script with `source setup.sh`. It will create a Python virtual environment, install the necessary dependencies as well as download Python script and `accelerate` config for running multinode training with FSDP (`fsdp_config.yaml`).

### [Optional] Examine the `sbatch` script

The script contains a number of arguments which configure Slurm job (starting with `#SBATCH`). If you want to change the job parameters (e.g. number of nodes, GPUs, etc.), you can modify the script accordingly.

One notable difference with [reference Slurm script](https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode_fsdp.sh) is that here we do not use `module` system as described in the [Accelerate documentation](https://github.com/huggingface/accelerate/blob/main/examples/README.md#slurm-scripts). Instead, we use a Python virtual environment with all the necessary dependencies installed. This is made possible by the fact that Soperator uses shared root filesystem which allows us to consistently use the same virtual environment on all nodes, making the setup more portable and easier to manage.

**Note**: if you are using a different Slurm distribution (not Soperator), you might need to adjust the script to account for different environment or run your job in a container.

### Submit the job

To submitt the job, simply run:
```
sbatch accelerate-nlp-example.sh
```

### Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`outputs/accelerate-example-<job_id>.out`).

### Expected output

The script will run the training process on 2 nodes with 8 GPUs each (16 GPUs total) for 3 epochs. The output log  should look similar to the following:
```
START TIME: Tue Apr 15 12:01:57 PM UTC 2025
MASTER_ADDR worker-0
...
epoch 0: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 0: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 1: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 1: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 2: {'accuracy': 0.696078431372549, 'f1': 0.8171091445427728}
epoch 2: {'accuracy': 0.696078431372549, 'f1': 0.8171091445427728}
END TIME: Tue Apr 15 12:03:16 PM UTC 2025
```
