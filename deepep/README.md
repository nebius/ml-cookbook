# DeepEP Setup on Nebius Clusters

[DeepEP](https://github.com/deepseek-ai/DeepEP) is a communication library for MoE
(Mixture-of-Experts) model training and inference. It ships two engines: **v2 `ElasticBuffer`**
(NCCL GIN) and **v1 `Buffer`** (NVSHMEM, legacy). Here is a setup guide on Nebius clusters.

> Verified at DeepEP [`dd758ca`](https://github.com/deepseek-ai/DeepEP/tree/dd758caf451848bd150e1046af3d0a73e5fff38d) (v2.1.0) on H200, B200, and B300.
> Hardware requirement: Minimum 1 node with 8 GPUs.

### 1. Confirm IBGDA and GDRCopy Kernels

IBGDA and GDRCopy are enabled by default on all Nebius clusters.

```bash
# IBGDA
cat /proc/driver/nvidia/params | grep -E "EnableStreamMemOPs|PeerMappingOverride"
# Should output:
# EnableStreamMemOPs: 1
# RegistryDwords: "PeerMappingOverride=1;"

# GDRCopy (~10 s; prints per-GPU copy bandwidth)
gdrcopy_copybw   # or run the full test suite with gdrcopy_sanity (~4 min)
```

### 2. Python environment

```bash
python3 -m venv ~/venvs/deepep && source ~/venvs/deepep/bin/activate
pip install "torch==2.13.*" --index-url https://download.pytorch.org/whl/cu130
pip install numpy ninja
pip install --force-reinstall --no-deps "nvidia-nccl-cu13>=2.30.4"   # torch pins 2.29.x; DeepEP needs >=2.30.4 to build
sudo apt update && sudo apt install -y python3-dev libibverbs-dev
```

### 3. Identify the GPU fabric NICs (compute network)

```bash
for d in /sys/class/infiniband/*; do echo "$(basename $d) $(cat $d/ports/1/rate) $(cat $d/ports/1/link_layer) pkey0=$(cat $d/ports/1/pkeys/0)"; done
```

H100/H200: `mlx5_0..7` · B200/B300: `mlx5_4..11` (exclude `mlx5_0..3`) · GB200/GB300: `mlx5_0..3`

### 4. Environment variables

```bash
export HCA='mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'    # H100/H200
# export HCA='mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1' # B200/B300
# export HCA='mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1'                                        # GB200/GB300
ulimit -l unlimited   # RDMA needs pinned memory
```

## Setup DeepEP

```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # 9.0 H100/H200 · 10.0 B200 · "10.3" B300
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP && git checkout dd758ca   # v2.1.0
sed -i 's/"r"(kNumBytes)/"n"(kNumBytes)/' deep_ep/include/deep_ep/common/ptx.cuh   # CUDA 13 only
```

### Install and Test V2 Engine

```bash
python setup.py install
```

After installation, run the tests to verify everything is working. One process per node
(it forks 8 local ranks); `RANK` is the node index:

```bash
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 python tests/elastic/test_barrier.py --num-allocated-qps 8
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 python tests/elastic/test_ep.py --num-allocated-qps 65   # ~8 min (includes first-use JIT compile)
# 2 nodes: same commands on both nodes, MASTER_ADDR=<node0-ip> WORLD_SIZE=2 RANK=0|1
```

### Recommended Environment Settings (v2)

```bash
export NCCL_IB_HCA="=$HCA"                       # '=' = exact match (NCCL-only syntax)
export EP_NIC_NAME=mlx5_0                        # any fabric NIC
export EP_JIT_CACHE_DIR=/tmp/deep_ep_jit-$USER   # node-local
# In code: ElasticBuffer(..., num_allocated_qps=65)   # VF QP budget; 33 at 4+ nodes
```
End of DeepEP V2 installation instructions

---

### Install and Test v1 Engine (NVSHMEM based)

v1 is the engine needed to verify the results from the PyTorch blog
([pytorch-dsv3-mxfp8](https://github.com/nebius/ml-cookbook/tree/main/pytorch-dsv3-mxfp8)).

> Note: the blog's published numbers were produced at the older DeepEP commit `29d31c0`.
> The instructions below use `dd758ca`, where both engines are verified.

Install NVSHMEM
```bash
pip install nvidia-nvshmem-cu13   # cu12 on CUDA 12
```

Install DeepEP
```bash
# NVSHMEM setup
export NVSHMEM_DIR=$(python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")
python3 setup.py install
```

After installation, run the tests to verify everything is working:

```bash
# fix upstream test bug (num_worst_tokens sub-case is unsupported internode):
perl -0pi -e 's/if with_topk:(\s*\n\s*num_worst_tokens)/if with_topk and num_nodes == 1:$1/' tests/legacy/test_internode.py

python tests/legacy/test_intranode.py     # 1 node
python tests/legacy/test_internode.py     # 2 nodes (same MASTER_ADDR/WORLD_SIZE/RANK scheme as v2)
python tests/legacy/test_low_latency.py   # 2 nodes
```

### Recommended Environment Settings (v1)

**Enable IBGDA:**

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_MAX_TEAMS=32       # low-latency mode at 16+ ranks
```

**Restrict NIC Discovery for RDMA:**

`mlx5_12` is a virtualized NIC used for VPC offloading and should not be used for RDMA
transport. Whitelist only the physical NICs to prevent it from being discovered:

```bash
export UCX_NET_DEVICES="$HCA"    # no '=' prefix — that's NCCL-only syntax
export NVSHMEM_HCA_LIST="$HCA"
```

For Kubernetes deployments, set these as environment variables in your pod spec (example for H100/H200):

```yaml
env:
  - name: UCX_NET_DEVICES
    value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
  - name: NVSHMEM_HCA_LIST
    value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
```

## Performance

Intranode (1 node, EP8), per-rank dispatch / combine, BF16 4096×7168 tokens, 256 experts, top-8:

| | v1 | v2 |
|---|---|---|
| B200 | 490 / 402 GB/s | **744 / 730 GB/s** |
| B300 | 505 / 410 GB/s | **731 / 722 GB/s** |
| H200 | 336 / 320 GB/s | 289 / 342 GB/s |
