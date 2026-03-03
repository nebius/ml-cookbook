# DeepEP Setup on Nebius Clusters

[DeepEP](https://github.com/deepseek-ai/DeepEP) is a communication library for MoE (Mixture-of-Experts) model training and inference, built on top of NVSHMEM for GPU-direct communication over Infiniband. Here is a setup guide on Nebius clusters.

## Prerequisites

### 1. Install IBGDA and GDRCopy Kernels

First, check if your cluster already has them enabled:

```bash
# Check IBGDA
cat /proc/driver/nvidia/params | grep -E "EnableStreamMemOPs|PeerMappingOverride"
# Should output:
# EnableStreamMemOPs: 1
# RegistryDwords: "PeerMappingOverride=1;"

# Check GDRCopy (wait a few minutes, should output test results)
gdrcopy_sanity
```

If either of these fail:

- **Managed Kubernetes or Soperator**: contact support to assist with installation.
- **Virtual Machine**: follow the steps below.

**Update NVIDIA Kernel to enable IBGDA:**

```bash
echo 'NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u
# Then go to the Nebius console and stop/start the VM
```

**Install GDRCopy kernel module:**

```bash
sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
git checkout v2.5.1
cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_*.deb
sudo dpkg -i libgdrapi_*.deb
sudo dpkg -i gdrcopy-tests_*.deb
sudo dpkg -i gdrcopy_*.deb
# /dev/gdrdrv should now exist, rerun gdrcopy_sanity
```

### 2. Install NVSHMEM

```bash
# For CUDA 12
pip install nvidia-nvshmem-cu12

# For CUDA 13
pip install nvidia-nvshmem-cu13
```

## Install DeepEP

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP

NVSHMEM_DIR='<path to nvshmem install>' \
TORCH_CUDA_ARCH_LIST="10.0" \  # Use 9.0 for Hopper (H100/H200)
python setup.py build && python setup.py install
```

After installation, run the tests to verify everything is working:

```bash
python tests/test_intranode.py
python tests/test_low_latency.py
```

## Recommended Environment Settings

### Enable IBGDA

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
```

### Increase Memory Lock Limits

RDMA requires pinned memory for transfers. Memlock limits must be set high to avoid transfer failures:

```bash
ulimit -l unlimited
```

### Restrict NIC Discovery for RDMA

`mlx5_12` is a virtualized NIC used for VPC offloading and should not be used for RDMA transport. Whitelist only the physical NICs to prevent it from being discovered:

```bash
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
```

For Kubernetes deployments, set these as environment variables in your pod spec:

```yaml
env:
  - name: UCX_NET_DEVICES
    value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1"
  - name: NVSHMEM_HCA_LIST
    value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1"
```
