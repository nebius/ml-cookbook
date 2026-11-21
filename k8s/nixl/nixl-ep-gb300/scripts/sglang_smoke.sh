#!/usr/bin/env bash
# Functional test for cross-rack NIXL-EP: launch SGLang with the NIXL
# expert-parallel all-to-all backend, confirm it comes up, and verify
# distributed generation. Rank 0 drives the client; the other ranks are workers.
set -euo pipefail
ulimit -l unlimited 2>/dev/null || {
  echo "could not set memlock=unlimited (need IPC_LOCK + SYS_RESOURCE)" >&2
  exit 1
}

model="${MODEL_PATH:-deepseek-ai/DeepSeek-V2-Lite}"
port="${PORT:-30000}"
total_gpus=$((NNODES * ${GPUS_PER_NODE:-4}))
job_index="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}"
node_rank=$((job_index + ${RANK_OFFSET:-0}))
log_file="/tmp/sglang-${node_rank}.log"
status_dir="/tmp/sglang-status"
status_port="${STATUS_PORT:-29999}"
moe_backend="${MOE_A2A_BACKEND:-nixl}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# Expert-parallel args for the NIXL all-to-all backend.
backend_args=(
  --ep-size "${EP_SIZE:-$total_gpus}"
  --moe-a2a-backend "$moe_backend"
  --deepep-mode "${DEEPEP_MODE:-low_latency}"
)

rm -rf "$status_dir"
mkdir -p "$status_dir"
python3 -m http.server "$status_port" --bind 0.0.0.0 --directory "$status_dir" \
  >/tmp/sglang-status-server.log 2>&1 &
status_pid=$!
success_seen_file="$status_dir/rank0-success-seen"
success_watcher_pid=""
python3 -m sglang.launch_server \
  --model-path "$model" \
  --host 0.0.0.0 \
  --port "$port" \
  --tp-size "$total_gpus" \
  --dist-init-addr "$MASTER_ADDR:$DIST_PORT" \
  --nnodes "$NNODES" \
  --node-rank "$node_rank" \
  --enable-metrics \
  "${backend_args[@]}" \
  ${SGLANG_EXTRA_ARGS:-} >"$log_file" 2>&1 &
server_pid=$!
if [[ "$node_rank" != "0" ]]; then
  # Worker ranks exit when rank 0 tears the server down after the smoke test.
  # Cache rank 0's success marker so a coordinated shutdown is not counted as a
  # worker failure.
  (
    for _ in $(seq 1 1800); do
      if curl -fsS "http://${MASTER_ADDR}:${status_port}/success" >/dev/null 2>&1; then
        touch "$success_seen_file"
        exit 0
      fi
      sleep 1
    done
  ) &
  success_watcher_pid=$!
fi
cleanup() {
  rc=$?
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  if [[ -n "$success_watcher_pid" ]]; then
    kill -TERM "$success_watcher_pid" 2>/dev/null || true
    wait "$success_watcher_pid" 2>/dev/null || true
  fi
  kill -TERM "$status_pid" 2>/dev/null || true
  wait "$status_pid" 2>/dev/null || true
  if [[ "$rc" -ne 0 ]]; then
    cat "$log_file"
  else
    grep -Ei 'server is fired up|model loaded|ready to roll|max_total_num_tokens|Using NIXL EP|Nixl MoE is enabled' \
      "$log_file" | tail -n 80 || true
  fi
}
trap cleanup EXIT
for _ in $(seq 1 120); do
  if grep -Eq 'server_args=(ServerArgs|\{)' "$log_file"; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
expected_ep_size="${EP_SIZE:-1}"
grep -Eq "server_args=.*(tp_size=${total_gpus}([,)])|'tp_size':[[:space:]]*${total_gpus}([,}]))" "$log_file"
grep -Eq "server_args=.*(ep_size=${expected_ep_size}([,)])|'ep_size':[[:space:]]*${expected_ep_size}([,}]))" "$log_file"
grep -Eq "server_args=.*(moe_a2a_backend='${moe_backend}'([,)])|'moe_a2a_backend':[[:space:]]*'${moe_backend}'([,}]))" "$log_file"
echo "SGLang parallelism assertion passed: TP${total_gpus}/EP${expected_ep_size}, backend=${moe_backend}"
if [[ "$node_rank" == "0" ]]; then
  for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "SGLang rank 0 exited before becoming healthy" >&2
      exit 1
    fi
    sleep 5
  done
  response=$(curl -fsS "http://127.0.0.1:${port}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}')
  printf '%s\n' "$response"
  python3 -c 'import json,sys; data=json.load(sys.stdin); assert data.get("text"), data' <<<"$response"
  if [[ -n "${SGLANG_EXPECT_TEXT:-}" ]]; then
    SGLANG_RESPONSE="$response" python3 -c 'import json,os; data=json.loads(os.environ["SGLANG_RESPONSE"]); expected=os.environ["SGLANG_EXPECT_TEXT"]; assert expected.lower() in data["text"].lower(), data'
  fi
  echo "distributed SGLang generation passed"
  touch "$status_dir/success"
  kill -TERM "$server_pid" 2>/dev/null || true
fi
wait "$server_pid" || {
  rc=$?
  if [[ "$node_rank" != "0" ]]; then
    for _ in $(seq 1 30); do
      if [[ -f "$success_seen_file" ]] || \
        curl -fsS "http://${MASTER_ADDR}:${status_port}/success" >/dev/null 2>&1; then
        rc=0
        break
      fi
      sleep 1
    done
  fi
  [[ "$node_rank" == "0" || "$rc" == "0" ]] || exit "$rc"
}
if [[ "$moe_backend" == "nixl" ]]; then
  grep -q 'Using NIXL EP' "$log_file"
  echo "NIXL-EP GPU-initiated path: PASS"
fi
if [[ "$node_rank" == "0" ]]; then
  sleep 10
fi
