#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/pilot.yaml}
ROOT=${PREANCHOR_ROOT:?PREANCHOR_ROOT must be set}
IMAGE=vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
SEQS=${PREANCHOR_REQUEST_CONCURRENCY:-2}
[[ "$SEQS" =~ ^[123]$ ]] || { echo 'Concurrency must be 1, 2 or 3' >&2; exit 1; }
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
REVISION=1df8507178afcc1bef68cd8c393f61a886323761
HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}
[[ -r "$HF_HOME" ]] || { echo "HF_HOME is not readable: $HF_HOME" >&2; exit 1; }
if sudo -n docker ps --format '{{.Names}}' | grep -qx preanchor-vllm; then
  echo 'preanchor-vllm is already running' >&2; exit 1
fi
mkdir -p "$ROOT/artifacts"
digest=$(sudo -n docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || true)
[[ -n "$digest" ]] || { echo "Pinned vLLM image is not available: $IMAGE" >&2; exit 1; }
printf '%s\n' "$digest" > "$ROOT/artifacts/vllm-image-digest.txt"
# The host-side publish remains localhost-only; inside the container the
# service must bind all interfaces for Docker's loopback publish to work.
exec sudo -n docker run --rm --name preanchor-vllm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 -v "$HF_HOME:/root/.cache/huggingface" "$IMAGE" \
  --model "$MODEL" --revision "$REVISION" --dtype bfloat16 --max-model-len 20480 \
  --enable-prefix-caching --gpu-memory-utilization 0.92 --max-num-seqs "$SEQS" \
  --host 0.0.0.0 --port 8000 --seed 20260914
