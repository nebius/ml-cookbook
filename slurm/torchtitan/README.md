# Running FLUX.1-schnell (12B) Text-To-Image Multi-Node Pretraining with TorchTitan and Slurm (Soperator)
This document provides a step-by-step guide to launching a pretraining job for [FLUX.1-schnell](https://github.com/black-forest-labs/flux/tree/main) with [TorchTitan](https://github.com/pytorch/torchtitan) on a Nebius Soperator cluster. Flux-1-Schnell (12B) is a text-to-image diffusion model. We will utilize the `pixparse/cc12m-wds` dataset that contains 12 million image-text pairs.

## Prerequisites
Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator).

## Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Soperator has `/` mounted as a shared filesystem `jail`).

### Setup the environment

`setup.sh` will create a Python virtual environment, install the necessary dependencies, and grab necessary files. The dependencies will be installed into a Pyhon `venv` on a shared filesystem, makng it accessible for every worker node.

**HF_TOKEN required** The setup script expects your Hugging Face access token to be available in the `HF_TOKEN` environment variable. 
```
export HF_TOKEN=<your-hf-access-token>
bash setup.sh
```

### Examine the `multinode_flux.sh` script and `.toml` configs

`multinode_flux.sh`: submitting this script with `sbatch` will start training on 2 nodes. To run it on larger cluster, you need to adjust `--nodes` `SBATCH` argument.

To adjust training parameters, please refer to [`torchtitan` feature list and corresponding code examples](https://github.com/pytorch/torchtitan/tree/main?tab=readme-ov-file#key-features-available).

### Flux 1 Schnell — Dataset
By default, this workload streams data points from Hugging Face. While it is handy for ease of setting up and portability, this may introduce additional network latency to the data loading process.

**_Optional:_** To improve dataloader performance, you can download a fraction of the `cc12m-wds` dataset to the shared filesystem (you may opt to download whole dataset if necessary by removing `--include` patterns):

```
hf download --repo-type dataset pixparse/cc12m-wds --include "*.json" --include "cc12m-train-000*.tar"  --local-dir ./dataset_cc12m-wds
```

To read dataset from this location instead of streaming from HF, add `--training.dataset_path "$(pwd)/dataset_cc12m-wds"` to the training command arguments in the `multinode_flux.sh`.

### Submit the job

To submit the distributed training job, simply run:
```
sbatch multinode_flux.sh
```

### Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the job script you used to submit the job (`outputs/flux-<job_id>.out`).

Alternatively, you may push job metrics and logs to W&B by providing your API key via `export WANDB_API_KEY=<your-key>` and adding `--metrics.enable_wandb` to the training command arguments in the `multinode_flux.sh`.
