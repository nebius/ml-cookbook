# 🚀 Running Multi-Node LLama3.3-70B Finetuning (Full / LORA) with TorchTune and Slurm (Soperator)
This document provides a step-by-step guide to launching a finetuning job for Llama3.3 70B with [TorchTune](https://github.com/pytorch/torchtune) on a provisioned Nebius Slurm (Soperator) cluster. We will use pre-built TorchTune recipes with configuration files that can be easily modified to support various models, including Llama3, Llama4, Qwen, and Mistral. This library supports both full and LoRA-based finetuning and is tested on 2 nodes with 8×H100 GPUs each on Nebius Soperator.

## ✅ Prerequisites
Before you start, make sure you have the following:
- Access to a [Soperator cluster](https://nebius.com/services/soperator). You can also provision a new cluster following these [steps](https://github.com/nebius/nebius-solution-library/tree/main/soperator).
- Have cloned this repo into your Soperator cluster with `git clone https://github.com/nebius/ml-cookbook.git`

## 📋 Steps

For running this workload, you will need to SSH to the login node of the Soperator cluster and clone this repository to the shared filesystem (by default, Soperator has `/` mounted as a shared filesystem).

### 🔧 Setup the environment

`setup.sh` will create a Python virtual environment, install the necessary dependencies, and grab necessary dataset. For Tensorboard to work correctly you will need Python ≤ 3.12

**HF_TOKEN required** The setup script expects your Hugging Face access token to be available in the `HF_TOKEN` environment variable. 
```
export HF_TOKEN=<your-hf-access-token>
bash setup.sh
```


**_Important note:_** For the Llama models in this tutorial you will need to request access from the [HuggingFace](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) repo BEFORE running setup.sh.

### 📄 Examine the `sbatch` script and .yaml configs

The script contains a number of arguments which configure Slurm job. If you want to change the job parameters (e.g. number of nodes, GPUs, etc.), you can modify the script accordingly.

One notable point is that here we use a Python virtual environment with all the necessary dependencies installed. This is made possible by the fact that Soperator uses shared root filesystem which allows us to consistently use the same virtual environment on all nodes, making the setup more portable and easier to manage.

As for the configs in the `llama3_3_70B_full_multinode.yaml` or `llama3_3_70B_lora_multinode.yaml` file, these will modify main training parameters. Some noteworthy options:
- `root_dir`: **Important** Update with where your ml-cookbook/slurm-recipes root dir is
- `batch_size`: Keep low with memory profiling on, batch size of 16 gives high throughput on 2x8h100s
- `epochs`: Set to one, feel free to increase 
- `tensor_parallel_dim`: Increase / decrease amount of model parallelism, good to keep equivalent to the number of gpus per node (8) or 0 for only data parallelism
- `profiler: True`: Set to True for detailed tracking of memory at runtime for debugging, reduce batch size if turning this on, stack trace will be saved to ./profiling_outputs

TorchTune has many prebuilt [recipes](https://github.com/pytorch/torchtune/tree/main/recipes) that you can plug into this tutorial to train different types of models, you will need to adjust the config parameter in the `tune run` command in the `.slurm` file and link to the associated `.yaml` config file.

### 🔌 Plug in your own dataset

In our example we use `tune download` to download the model weights to our shared jail filesystem. This automatically syncs to all Soperoator nodes to be read for training.

To plug in your own chat-style dataset follow these [instructions](https://docs.pytorch.org/torchtune/0.3/basics/chat_datasets.html). Other dataset styles are also supported in the documentation.

An example is as follows, you can  pass the Hugging Face dataset repo name, select the appropriate conversation style, and specify the conversation_column:
```
dataset:
  _component_: torchtune.datasets.chat_dataset
  source: NewEden/xlam-function-calling-60k-shareGPT
  conversation_column: conversations
  conversation_style: sharegpt
  split: train
```
**IMPORTANT: The tokenizer's vocabulary and special tokens must match your model and dataset. For example, Llama 3.1 requires its exact tokenizer, and you must specify its path in the YAML**

### 🚀 Submit the job

To submit the job, simply run:
```
sbatch full_finetune_multinode.slurm  # For full parameter 
sbatch lora_finetune_multinode.slurm  # For LORA adapters
```

### 👀 Monitor the job

You can monitor the job status using `squeue` command. Once the job is running, you can check the output in the log file specified in the script (`slurm_out/torchtune-%j.out`).

### 📊 Expected output

The script will run the training process on 2 nodes with 8 GPUs each (16 GPUs total) for 1 epoch. The output log  should output some setup and once training kicks off it will similar to the following:
```
  0%|          | 1/1367 [04:24<100:30:01, 264.86s/it]
1|1|Loss: 1.849762201309204:   0%|          | 1/1367 [04:24<100:30:01, 264.86s/it]
1|1|Loss: 1.849762201309204:   0%|          | 2/1367 [05:03<49:53:19, 131.58s/it] 
1|2|Loss: 1.2548030614852905:   0%|          | 2/1367 [05:03<49:53:19, 131.58s/it]
1|2|Loss: 1.2548030614852905:   0%|          | 3/1367 [05:04<27:20:19, 72.15s/it] 
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
