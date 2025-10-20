# vLLM Inference with SkyPilot

This repo provides a reproducible setup for exploring vLLM inference on Kubernetes using SkyPilot. It supports any Hugging Face-compatible model and includes:
- Unified YAML for setup, model download, and serving (see example YAMLs)
- Portable test script for endpoint validation (`test-vllm.sh`)

## Quickstart
1. Prepare environment:
   - Copy `.template.env` to `.env` and fill values. If using direnv, run `direnv allow`.
   - Generate Sky config: `./generate-sky-config.sh` (renders `.sky.yaml`).
2. Create shared volume (PVC):
   - If your K8s cluster lacks the `ReadWriteMany` storage class, install it using the official helm chart (skip PVC/PV creation):
     https://docs.nebius.com/kubernetes/storage/filesystem-over-csi
   - Create the volume: `sky volumes apply -y volume.sky.yaml`
3. Launch setup + serve:
   - `sky launch -c serve -y <your-model-inference-vllm.yaml>`
   - First it will download the model weights to PVC; subsequent runs reuse cache.
4. Test endpoint:
   - Run `./test-vllm.sh` (see below for usage).
---

## Environment Variables
See the example YAMLs for required environment variables and runtime flags. Key settings include:
- Model repo (e.g., `bigscience/bloom`, `Qwen/Qwen-72B`)
- Attention backend (e.g., `TRITON_ATTN`, `TORCH_SDPA`) for compatibility
- Context length, concurrency, and chat template settings
- Debug and runtime flags for stability

Secrets:
- `HF_TOKEN` (optional, via `secrets:`): Only needed for private models

---

## Testing with test-vllm.sh
Use `test-vllm.sh` to validate the OpenAI-compatible endpoint. It supports health, completions, and chat tests with a simplified interface.

Key modes:
- Interactive chat: `./test-vllm.sh --chat` (prompts with ">")
- Interactive completions: `./test-vllm.sh --prompt` (prompts with ">")
- Default tests: `./test-vllm.sh --test [health|chat|prompt|all]` (no value => health test)
- Port-forward and test: `./test-vllm.sh --port-forward --port <port> --pod <head-pod-name> -n <ns> --test all`
- Port-forward only: `./test-vllm.sh --port-forward --port <port> --pod <head-pod-name> -n <ns>` (keeps open until Ctrl+C)

Examples:
- Health check: `./test-vllm.sh --test` (defaults to health test)
- Chat (default message): `./test-vllm.sh --test chat`
- Completions (default prompt): `./test-vllm.sh --test prompt` 

---

## Deployment Notes
- PVC is used for model cache; only rank 0 downloads weights
- Set `REDOWNLOAD=true` to force fresh download
- All secrets and runtime files are git-ignored; and only templates are committed
- Highly recomended the K8s node CUDA driver version is greater/equal than container CUDA version (Cuda runtime)
