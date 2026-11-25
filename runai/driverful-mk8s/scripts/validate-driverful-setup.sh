#!/usr/bin/env bash
# validate-driverful-setup.sh
#
# Checks the driverful Run:ai setup on a Nebius Managed Kubernetes cluster.
# Prints PASS / WARN / FAIL for each item and exits non-zero if any FAIL.
#
# Requirements: kubectl (with cluster-admin access), jq
# Usage: bash scripts/validate-driverful-setup.sh

set -euo pipefail

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

PASS=0
WARN=0
FAIL=0

pass() { echo "  PASS  $*"; (( PASS++ )) || true; }
warn() { echo "  WARN  $*"; (( WARN++ )) || true; }
fail() { echo "  FAIL  $*"; (( FAIL++ )) || true; }

section() { echo; echo "=== $* ==="; }

require_tool() {
    if ! command -v "$1" > /dev/null 2>&1; then
        echo "ERROR: required tool '$1' not found in PATH" >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

require_tool kubectl
require_tool jq

# --------------------------------------------------------------------------- #
# 1. Nodes
# --------------------------------------------------------------------------- #

section "Nodes"

GPU_NODES=$(kubectl get nodes -l nebius.com/gpu=true -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
GPU_NODE_COUNT=$(echo "$GPU_NODES" | wc -w)

if [[ "$GPU_NODE_COUNT" -ge 1 ]]; then
    pass "GPU node(s) found: $GPU_NODE_COUNT"
    for n in $GPU_NODES; do
        echo "    $n"
    done
else
    fail "No nodes with label nebius.com/gpu=true"
fi

CPU_NODES=$(kubectl get nodes -l '!nebius.com/gpu' --field-selector='metadata.name!=.' -o name 2>/dev/null | wc -l) || CPU_NODES=0
if [[ "$CPU_NODES" -ge 1 ]]; then
    pass "CPU node(s) found: $CPU_NODES"
else
    warn "No CPU-only nodes detected"
fi

# --------------------------------------------------------------------------- #
# 2. Driverful label
# --------------------------------------------------------------------------- #

section "Driverful node label"

DF_COUNT=$(kubectl get nodes -l 'nebius.com/node-image-features=driverful' --no-headers 2>/dev/null | wc -l) || DF_COUNT=0
if [[ "$DF_COUNT" -ge 1 ]]; then
    pass "Driverful label present on $DF_COUNT node(s)"
else
    warn "Label nebius.com/node-image-features=driverful not found (may use a different label — verify manually)"
fi

# --------------------------------------------------------------------------- #
# 3. GPU allocatable resources
# --------------------------------------------------------------------------- #

section "nvidia.com/gpu allocatable resources"

for node in $GPU_NODES; do
    gpu_cap=$(kubectl get node "$node" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    if [[ "${gpu_cap:-0}" -ge 1 ]]; then
        pass "$node: nvidia.com/gpu=$gpu_cap"
    else
        fail "$node: nvidia.com/gpu not allocatable (got '${gpu_cap:-none}')"
    fi
done

# --------------------------------------------------------------------------- #
# 4. NVIDIA device plugin
# --------------------------------------------------------------------------- #

section "NVIDIA device-plugin"

DP_PODS=$(kubectl get pods -A -l 'app=nvidia-device-plugin-daemonset' --no-headers 2>/dev/null | wc -l) || DP_PODS=0
if [[ "$DP_PODS" -ge 1 ]]; then
    pass "NVIDIA device-plugin pods: $DP_PODS"
else
    warn "No nvidia-device-plugin-daemonset pods found (Nebius may use a different label — verify)"
fi

# --------------------------------------------------------------------------- #
# 5. NFD / GFD labels on GPU nodes
# --------------------------------------------------------------------------- #

section "NFD / GFD feature labels on GPU nodes"

REQUIRED_LABELS=(
    "nvidia.com/gpu.present"
    "nvidia.com/gpu.product"
    "nvidia.com/cuda.driver.major"
)

for node in $GPU_NODES; do
    for label in "${REQUIRED_LABELS[@]}"; do
        val=$(kubectl get node "$node" -o jsonpath="{.metadata.labels.${label//\./\\.}}" 2>/dev/null || true)
        if [[ -n "$val" ]]; then
            pass "$node: $label=$val"
        else
            fail "$node: label '$label' missing (NFD/GFD may not be running)"
        fi
    done
done

# --------------------------------------------------------------------------- #
# 6. DCGM Exporter
# --------------------------------------------------------------------------- #

section "Standalone DCGM Exporter (gpu-monitoring namespace)"

DCGM_PODS=$(kubectl get pods -n gpu-monitoring -l 'app.kubernetes.io/name=dcgm-exporter' \
    --no-headers 2>/dev/null | grep -c Running) || DCGM_PODS=0
if [[ "$DCGM_PODS" -ge 1 ]]; then
    pass "DCGM Exporter Running pods: $DCGM_PODS"
else
    fail "No Running DCGM Exporter pods in gpu-monitoring namespace"
fi

DCGM_SVC=$(kubectl get svc -n gpu-monitoring -l 'app.kubernetes.io/name=dcgm-exporter' \
    --no-headers 2>/dev/null | wc -l) || DCGM_SVC=0
if [[ "$DCGM_SVC" -ge 1 ]]; then
    pass "DCGM Exporter service found"
else
    warn "DCGM Exporter service not found in gpu-monitoring"
fi

# --------------------------------------------------------------------------- #
# 7. Prometheus ServiceMonitor
# --------------------------------------------------------------------------- #

section "DCGM ServiceMonitor"

SM=$(kubectl get servicemonitor -n gpu-monitoring -l 'release=prometheus' \
    --no-headers 2>/dev/null | wc -l) || SM=0
if [[ "$SM" -ge 1 ]]; then
    pass "ServiceMonitor with label release=prometheus found"
else
    warn "No ServiceMonitor with label release=prometheus in gpu-monitoring"
fi

# --------------------------------------------------------------------------- #
# 8. DCGM compatibility PrometheusRule
# --------------------------------------------------------------------------- #

section "DCGM compatibility PrometheusRule (runai namespace)"

RULE=$(kubectl get prometheusrule dcgm-compat -n runai --no-headers 2>/dev/null | wc -l) || RULE=0
if [[ "$RULE" -ge 1 ]]; then
    pass "PrometheusRule dcgm-compat exists in runai namespace"
else
    fail "PrometheusRule dcgm-compat not found in runai namespace"
fi

# --------------------------------------------------------------------------- #
# 9. Run:ai namespaces and core pods
# --------------------------------------------------------------------------- #

section "Run:ai installation"

RUNAI_NS=$(kubectl get ns runai --no-headers 2>/dev/null | wc -l) || RUNAI_NS=0
if [[ "$RUNAI_NS" -ge 1 ]]; then
    pass "Namespace 'runai' exists"
else
    fail "Namespace 'runai' not found — Run:ai may not be installed"
fi

RUNAI_PODS=$(kubectl get pods -n runai --no-headers 2>/dev/null | grep -c Running) || RUNAI_PODS=0
if [[ "$RUNAI_PODS" -ge 1 ]]; then
    pass "Running pods in runai namespace: $RUNAI_PODS"
else
    fail "No Running pods in runai namespace"
fi

# --------------------------------------------------------------------------- #
# 10. Ingress
# --------------------------------------------------------------------------- #

section "HAProxy Ingress"

HA_PODS=$(kubectl get pods -A -l 'app.kubernetes.io/name=kubernetes-ingress' \
    --no-headers 2>/dev/null | grep -c Running) || HA_PODS=0
if [[ "$HA_PODS" -ge 1 ]]; then
    pass "HAProxy Ingress controller Running pods: $HA_PODS"
else
    warn "No Running HAProxy Ingress pods found (check the ingress namespace)"
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

section "Summary"
echo "  PASS: $PASS   WARN: $WARN   FAIL: $FAIL"
echo

if [[ "$FAIL" -gt 0 ]]; then
    echo "One or more checks FAILED. Review the output above and consult the README troubleshooting section."
    exit 1
elif [[ "$WARN" -gt 0 ]]; then
    echo "All critical checks passed. Review WARNings above — they may indicate label differences in your cluster."
    exit 0
else
    echo "All checks passed."
    exit 0
fi
