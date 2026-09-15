#!/usr/bin/env bash
set -euo pipefail
# Usage: start_vllm.sh [configs/lambda/<profile>.yaml]
# The profile supplies only gpu_memory_utilization. max_model_len stays at the frozen
# Stage 5 context (longest prompt plus max_new_tokens 16384) whatever the profile says,
# and max_num_seqs comes from PREANCHOR_REQUEST_CONCURRENCY so that the server limit and
# the client's in-flight request count are always the same number.
PROFILE=${1:-configs/lambda/a6000_48.yaml}
ROOT=${PREANCHOR_ROOT:?PREANCHOR_ROOT must be set}
IMAGE=vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
SEQS=${PREANCHOR_REQUEST_CONCURRENCY:-2}
MAX_SEQS=16   # preanchor.labels.batching.MAX_CONCURRENCY; measured KV capacity admits fewer
[[ "$SEQS" =~ ^[0-9]+$ && "$SEQS" -ge 1 && "$SEQS" -le "$MAX_SEQS" ]] \
  || { echo "Concurrency must be an integer between 1 and $MAX_SEQS" >&2; exit 1; }
[[ -r "$PROFILE" ]] || { echo "Hardware profile not readable: $PROFILE" >&2; exit 1; }
UTIL=$(sed -n 's/^[[:space:]]*gpu_memory_utilization:[[:space:]]*\([0-9.]*\).*/\1/p' "$PROFILE")
[[ "$UTIL" =~ ^0\.[0-9]+$ ]] || { echo "No gpu_memory_utilization in $PROFILE" >&2; exit 1; }
MAX_MODEL_LEN=20480
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
  --model "$MODEL" --revision "$REVISION" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching --gpu-memory-utilization "$UTIL" --max-num-seqs "$SEQS" \
  --host 0.0.0.0 --port 8000 --seed 20260914
