#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="demo"
GPU_PER_NODE=8
NUM_WORKERS=2
PLATFORM="hopper"   # hopper | blackwell

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -p, --project   <name>   Project name (default: ${PROJECT_NAME})
  -g, --gpus      <int>    GPUs per node (default: ${GPU_PER_NODE})
  -n, --nodes     <int>    Number of nodes / workers (default: ${NUM_WORKERS})
      --platform  <str>    Platform: hopper | blackwell (default: ${PLATFORM})
  -h, --help                Show this help

Examples:
  $(basename "$0") --platform hopper -p demo -g 8 -n 2
  $(basename "$0") --platform blackwell -p bw -g 16 -n 4
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project)   PROJECT_NAME="${2:-}"; shift 2 ;;
    -g|--gpus)      GPU_PER_NODE="${2:-}"; shift 2 ;;
    -n|--nodes)     NUM_WORKERS="${2:-}"; shift 2 ;;
    --platform)     PLATFORM="${2:-}"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

[[ -n "${PROJECT_NAME}" ]] || die "Project name cannot be empty."
[[ "${GPU_PER_NODE}" =~ ^[0-9]+$ ]] || die "--gpus must be a positive integer."
[[ "${NUM_WORKERS}"  =~ ^[0-9]+$ ]] || die "--nodes must be a positive integer."
(( GPU_PER_NODE > 0 )) || die "--gpus must be > 0."
(( NUM_WORKERS  > 0 )) || die "--nodes must be > 0."


case "$PLATFORM" in
  hopper)    IMAGE="cr.eu-north1.nebius.cloud/nebius-benchmarks/nccl-tests:2.23.4-ubu22.04-cu12.4" ;;
  blackwell) IMAGE="cr.eu-north1.nebius.cloud/nebius-benchmarks/nccl-tests:2.26.5-ubu22.04-cu12.8" ;;
  *)         die "Unsupported --platform '${PLATFORM}'. Use 'hopper' or 'blackwell'." ;;
esac

TOTAL_RANKS=$(( NUM_WORKERS * GPU_PER_NODE ))
JOB_NAME="${PROJECT_NAME}-nccl-${PLATFORM}-${GPU_PER_NODE}g${NUM_WORKERS}n"

echo "Submitting job:"
echo "  project   : ${PROJECT_NAME}"
echo "  platform  : ${PLATFORM}"
echo "  image     : ${IMAGE}"
echo "  nodes     : ${NUM_WORKERS}"
echo "  gpus/node : ${GPU_PER_NODE}"
echo "  total ranks: ${TOTAL_RANKS}"
echo "  job name  : ${JOB_NAME}"
echo

runai mpi submit "${JOB_NAME}" -p "${PROJECT_NAME}" \
  --image "${IMAGE}" \
  --environment "NCCL_DEBUG=version" \
  --environment "NCCL_DEBUG_SUBSYS=init,env" \
  --environment "NCCL_SOCKET_IFNAME=eth0" \
  --environment "UCX_NET_DEVICES=eth0" \
  --environment "NCCL_IB_HCA=mlx5" \
  --environment "NCCL_BUFFSIZE=8388608" \
  --environment "NCCL_IB_QPS_PER_CONNECTION=2" \
  --environment "OMPI_MCA_coll_hcoll_enable=0" \
  --environment "OMPI_MCA_coll_ucc_enable=0" \
  --environment "OMPI_MCA_hwloc_base_binding_policy=none" \
  --environment "OMPI_MCA_hwloc_base_mem_bind=never" \
  --node-pools "training-pool" \
  --auto-deletion-time-after-completion "1h" \
  --large-shm \
  --capability "IPC_LOCK" \
  --workers "${NUM_WORKERS}" \
  --gpu-devices-request "${GPU_PER_NODE}" \
  --slots-per-worker "${GPU_PER_NODE}" \
  --host-path "path=/mnt/data,mount=/sfs,mount-propagation=HostToContainer" \
  --master-no-pvcs \
  --master-command "mpirun" \
  --master-args  "-np ${TOTAL_RANKS} --map-by ppr:${GPU_PER_NODE}:node --allow-run-as-root -mca coll ^hcoll bash -c '/opt/nccl_tests/build/all_reduce_perf -b 1024M -e 16G -f 2 -g 1'" \
  --command -- bash -c "/usr/sbin/sshd -De"
