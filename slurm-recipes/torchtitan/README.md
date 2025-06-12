# 🚀 Running FLUX.1-schnell (12B) Text-To-Image Multi-Node Pretraining with TorchTitan and Slurm (Soperator)
This document provides a step-by-step guide to launching a pretraining job for [FLUX.1-schnell](https://github.com/black-forest-labs/flux/tree/main) with [TorchTitan](https://github.com/pytorch/torchtitan) on a Nebius Slurm (Soperator) cluster. Flux-1-Schnell (12B) is a distilled text-to-image diffusion model capable of generating high-quality images in just 1–4 sampling steps, allowing you to turn written descriptions into realistic images. We will utilize the cc12m-wds dataset that contains 12 million image-text pairs specifically meant to be used for vision and-language pre-training.

## ✅ Prerequisites
Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator). You can also provision a new cluster following these [steps](https://github.com/nebius/nebius-solution-library/tree/main/soperator).
- Have cloned this repo into your Soperator cluster with `git clone https://github.com/nebius/ml-cookbook.git`

## 📋 Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Sopeartor has `/` mounted as a shared filesystem).

### 🔧 Setup the environment

Execute the setup script with `source setup_flux.sh`. It will create a Python virtual environment, install the necessary dependencies, and grab necessary files.

### 📄 Examine the `.slurm` script and .toml configs

`multinode_flux.slurm`: This file will kick off multinode training, its setup for 2 nodes of 8xh100s at the moment. Adjust `--nodes --ntasks --nnodes --nproc_per_node --gpus-per-task --nproc_per_node` as needed for your hardware.

One notable point is that here we use a Python virtual environment with all the necessary dependencies installed. This is made possible by the fact that Soperator uses shared root filesystem which allows us to consistently use the same virtual environment on all nodes, making the setup more portable and easier to manage.

As for the configs in the `flux_schnell_model.toml` file, these will modify main training parameters. Some noteworthy options:
- `root_dir`: update with where your ml-cookbook/slurm-recipes root dir is
- `batch_size`: keep low with memory profiling on, batch size of 16 gives high throughput on 2x8h100s
- `epochs`: set to one, feel free to change 
- `tensor_parallel_dim`: Increase / decrease amount of model parallelism, good to keep equivalent to the number of gpus per node (8) or 0 for only data paralellism
- `profiler: True`: Set to True for detailed tracking of memory at runtime for debugging, reduce batch size if turning this on, stack trace will be saved to ./profiling_outputs

TorchTune has many prebuilt [recipes](https://github.com/pytorch/torchtune/tree/main/recipes) that you can plug into this tutorial to train different types of models, you will need to adjust the config parameter in the `tune run` command in the .slurm file and link to the associated .yaml config file.

### 🔌 Plug in your own dataset
To plug in your own chat-style dataset follow these [instructions](https://docs.pytorch.org/torchtune/0.3/basics/chat_datasets.html). Other dataset styles are also supported in the documentation.

An example is as follows, you can  pass in the Hugging Face dataset repo name, select the appropriate conversation style, and specify the conversation_column:
```
dataset:
  _component_: torchtune.datasets.chat_dataset
  source: NewEden/xlam-function-calling-60k-shareGPT
  conversation_column: conversations
  conversation_style: sharegpt
  split: train
```
***IMPORTANT: The tokenizer's vocabulary and special tokens must match your model and dataset. For example, Llama 3.1 requires its exact tokenizer, and you must specify its path in the YAML***

### 🚀 Submit the job

To submitt the job, simply run:
```
sbatch full_finetune_multinode.slurm  # For full parameter 
sbatch lora_finetune_multinode.slurm  # For LORA adapters
```

### 👀 Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`slurm_out/torchtune-%j.out`).

### 📊 Expected output

The script will run the training process on 2 nodes with 8 GPUs each (16 GPUs total) for 1 epoch. The output log  should output some setup and once training kicks off it will similar to the following:
```
[titan] 2025-06-10 21:58:27,008 - root - INFO - [31mstep: 700  [32mloss:  0.7052  [33mmemory: 76.79GiB(96.97%)  [34mtps: 1,541,366  [36mtflops: 0.00  [35mmfu: 0.00%[39m
[titan] 2025-06-11 14:51:50,202 - root - INFO - [31mstep: 8200  [32mloss:  0.5149  [33mmemory: 76.90GiB(97.11%)  [34mtps: 1,567,390  [36mtflops: 0.00  [35mmfu: 0.00%[39m
[titan] 2025-06-11 15:05:19,178 - root - INFO - [31mstep: 8300  [32mloss:  0.5146  [33mmemory: 76.82GiB(97.01%)  [34mtps: 1,555,416  [36mtflops: 0.00  [35mmfu: 0.00%[39m
```

### 🧠 Monitoring & Debugging Training (TensorBoard + Nebius Console)

#### 🔧 Monitor GPU Metrics (Nebius Console)
You can monitor some of the GPU metrics by logging into the clicking the following in Nebius console: 
Compute -> GPU Clusters -> Locate your GPU cluster and select it -> Virtual Machines -> Select desired node -> Monitoring -> GPU metrics. Here there are useful metrics such as:
- `Memory Utilization` - 60%-90%
- `Power usage for the device` - Aim for 700W
- `The number of bytes of active NVLink (RX/TX)` - Check inter-gpu comms

#### 📉 Enable TensorBoard Profiling (PyTorch Profiler)
To look at more detailed memory profiling via Tensorboard, make sure you initiated training with `profiling` config set to True. You can also modify these parameters to modify how much of your run is profiled (Profiling outputs can grow very large, good to keep it to a limited number of steps):
  `wait_steps: 5`
  `warmup_steps: 3`
  `active_steps: 20`
  `num_cycles: 1 `

Once the run is complete you can go to the folder `output/profiling_outputs/iteration_{number}`. It's better to select a few ranks you want to visualize and transfer their `.pt.trace.json.gz` files to their own folder. Example:

```
mkdir vis
mv r0-2025-6-5-21-26.1749158791382875441.pt.trace.json.gz vis
tensorboard --logdir ./vis  --port 6006 --host 0.0.0.0
```

Tensorboard is now running, but you have to port forward your ip to view on your local machines web browser. On your local machine run the following:

```
ssh -N -L 6006:localhost:6006 root@{YOUR SOPERATOR IP}
```
You can now go to http://localhost:6006/#pytorch_profiler and view the Tensorboard profiling outputs.
