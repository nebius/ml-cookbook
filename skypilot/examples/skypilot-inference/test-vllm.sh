#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# test-vLLM.sh: simple health and inference checks for a vLLM OpenAI server.
# - Can optionally port-forward a pod or service during the test.
# - Defaults to localhost:8000 (If you already port-forwarded).
# - Simplified CLI:
#     - Interactive:      --chat   | --prompt
#     - Default tests:    --test [health|chat|prompt|all]  (no value => health)
#     - For port-forward: --port-forward --pod/--svc (no test identified, then stays open)

# Load .env if present (never commit .env)
if [ -f ./.env ]; then
  set +u
  set -a
  . ./.env
  set +a
  set -u
fi

HOST="127.0.0.1"
PORT="8000"
NAMESPACE="default"
RESOURCE_TYPE=""
RESOURCE_NAME=""
DO_PORT_FORWARD=0
TEST_KIND="health"    # default when --test provided without a value
TEST_SPECIFIED=0       # tracks if user explicitly provided --test
INTERACTIVE=""        # "chat" | "prompt" when using interactive modes

TIMEOUT=60  # Curl max time per request (seconds)
# Controls the maximum number of tokens generated in each response.
MAX_TOKENS=128
# Controls randomness/creativity of model output. Lower = more focused, higher = more diverse.
TEMPERATURE=0.7
# Built-in defaults for --test chat/prompt
DEFAULT_CHAT_MESSAGE="Write a one-line haiku about GPUs."
DEFAULT_PROMPT_TEXT="Complete: The benefits of tensor parallelism are"

usage() {
  cat <<USAGE
Usage: $0 [options]
  (no arguments)                Run a health test on localhost:8000
  -n, --namespace <ns>          Kubernetes namespace (optional)
      --pod <name>              Pod name to port-forward (use with --port-forward)
      --svc|--service <name>    Service name to port-forward (use with --port-forward)
      --port-forward            Start kubectl port-forward. If --test is omitted or set to 'none', keeps port-forward active until Ctrl+C.
  -p, --port <port>             Local port (default: 8000)
      --host <host>             Host to target (default: 127.0.0.1)
      --chat                    Start interactive chat mode (prompts with '>')
      --prompt                  Start interactive prompt/completions mode (prompts with '>')
      --test [kind]             Run default tests: health|chat|prompt|all (default: health)
  -h, --help                    Show this help

Examples:
  $0
    # (runs a health test on localhost:8000 by default)
  $0 --chat
  $0 --prompt
  $0 --test                # default: health
  $0 --test chat           # uses a built-in default message
  $0 --test prompt         # or: completion|completions
  $0 --port-forward --pod <head-pod-name> -n <ns> --test all
  $0 --port-forward --svc <service-name> -n <ns> --test health
  $0 --port-forward --pod <head-pod-name> --port <port>
    # (keeps port-forward active until Ctrl+C; no test run)
USAGE
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
log() { printf '[test] %s\n' "$*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --pod) RESOURCE_TYPE="pod"; RESOURCE_NAME="$2"; shift 2 ;;
    --svc|--service) RESOURCE_TYPE="svc"; RESOURCE_NAME="$2"; shift 2 ;;
    --port-forward) DO_PORT_FORWARD=1; shift ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --chat) INTERACTIVE="chat"; shift ;;
    --prompt) INTERACTIVE="prompt"; shift ;;
    --test)
      TEST_SPECIFIED=1
      if [ $# -ge 2 ] && [ "${2#-}" = "$2" ]; then
        TEST_KIND="$2"; shift 2
      else
        TEST_KIND="health"; shift
      fi
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

PF_PID=""
cleanup() {
  if [ -n "$PF_PID" ] && kill -0 "$PF_PID" 2>/dev/null; then
    log "Stopping port-forward (pid=$PF_PID)"
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

pf_start() {
  need_cmd kubectl
  [ -n "$RESOURCE_TYPE" ] || die "--port-forward requires --pod <name> or --service <name>"
  [ -n "$RESOURCE_NAME" ] || die "--port-forward requires a resource name"
  local ns_args=()
  [ -n "$NAMESPACE" ] && ns_args=( -n "$NAMESPACE" )
  log "Starting port-forward: ${RESOURCE_TYPE}/${RESOURCE_NAME} -> localhost:${PORT}"
  # shellcheck disable=SC2068
  kubectl ${ns_args[@]} port-forward "${RESOURCE_TYPE}/${RESOURCE_NAME}" "${PORT}:${PORT}" >/dev/null 2>&1 &
  PF_PID=$!

  # Wait for the port to respond
  local start_ts end_ts
  start_ts=$(date +%s)
  until curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$PF_PID" 2>/dev/null; then
      die "port-forward exited unexpectedly"
    fi
    end_ts=$(date +%s)
    if [ $(( end_ts - start_ts )) -ge "$TIMEOUT" ]; then
      die "Timed out waiting for port-forward to become ready"
    fi
    sleep 1
  done
}

curl_json() {
  local url="$1"; shift
  if command -v jq >/dev/null 2>&1; then
    curl -sS --max-time "$TIMEOUT" "$url" "$@" | jq
  else
    curl -sS --max-time "$TIMEOUT" "$url" "$@"
  fi
}

health_test() {
  log "GET /v1/models"
  curl_json "http://${HOST}:${PORT}/v1/models"
}

# Interactive chat mode
chat_interactive() {
  local message
  while true; do
    printf "[chat mode] Enter user message for chat endpoint (Ctrl+D to cancel):\n> "
    IFS= read -r message || { log "No message provided, exiting chat mode."; break; }
    log "POST /v1/chat/completions"
    local data
    data=$(cat <<JSON
{
  "messages": [{"role": "user", "content": "${message}"}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE}
}
JSON
    )
    curl_json "http://${HOST}:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$data"
  done
}

# Interactive completions mode
completions_interactive() {
  local prompt
  printf "[completions mode] Enter prompt for completions endpoint (Ctrl+D to cancel):\n> "
  IFS= read -r prompt || { log "No prompt provided, skipping."; return; }
  log "POST /v1/completions"
  local data
  data=$(cat <<JSON
{
  "prompt": "${prompt}",
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE}
}
JSON
  )
  curl_json "http://${HOST}:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$data"
}

# Default chat test (non-interactive)
chat_test_default() {
  local message="$DEFAULT_CHAT_MESSAGE"
  log "POST /v1/chat/completions (default message)"
  local data
  data=$(cat <<JSON
{
  "messages": [{"role": "user", "content": "${message}"}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE}
}
JSON
  )
  curl_json "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$data"
}

# Default completions test (non-interactive)
completions_test_default() {
  local prompt="$DEFAULT_PROMPT_TEXT"
  log "POST /v1/completions (default prompt)"
  local data
  data=$(cat <<JSON
{
  "prompt": "${prompt}",
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMPERATURE}
}
JSON
  )
  curl_json "http://${HOST}:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$data"
}

# Optionally start port-forward
if [ "$DO_PORT_FORWARD" -eq 1 ]; then
  pf_start
fi

# If port-forward requested and neither --test nor interactive mode was chosen, keep PF open
if [ "$DO_PORT_FORWARD" -eq 1 ] && [ "$TEST_SPECIFIED" -eq 0 ] && [ -z "$INTERACTIVE" ]; then
  log "Port-forward active. Press Ctrl+C to stop."
  while true; do sleep 3600; done
fi

# Dispatch
if [ -n "$INTERACTIVE" ]; then
  # Print selected test type header for interactive mode
  log "Test type: ${INTERACTIVE} (interactive)"
  case "$INTERACTIVE" in
    chat)   chat_interactive ;;
    prompt) completions_interactive ;;
    *) die "Unknown interactive mode: $INTERACTIVE" ;;
  esac
else
  # Map synonyms and defaults for --test kinds
  case "$TEST_KIND" in
    ""|health) TEST_KIND="health" ;;
    chat) : ;;
    prompt|completion|completions) TEST_KIND="prompt" ;;
    all) : ;;
    *) die "Unknown --test kind: $TEST_KIND" ;;
  esac

  # Print selected test type header for non-interactive mode
  if [ "$TEST_KIND" = "all" ]; then
    log "Test type: all (health, chat, prompt)"
  else
    log "Test type: ${TEST_KIND}"
  fi

  case "$TEST_KIND" in
    health)
      health_test
      ;;
    chat)
      chat_test_default
      ;;
    prompt)
      completions_test_default
      ;;
    all)
      health_test
      chat_test_default
      completions_test_default
      ;;
  esac
fi

log "Done."
