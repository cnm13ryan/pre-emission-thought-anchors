#!/usr/bin/env bash
# Phase 2 (redesign): start one pinned vLLM container per GPU. Exploratory experiment only.
# Same image digest, model revision, dtype, prefix caching and server seed as
# scripts/start_vllm.sh. Differences: max_model_len 24576 (longest Phase 2 prompt is ~4.6k
# tokens, plus max_new_tokens 16384) and one container per visible GPU on port 8000+i.
# Requires the gpu:h100 and service:vllm leases (PREANCHOR_LEASE_TOKENS), as start_vllm.sh does.
# Usage: phase2_vllm.sh start <max_num_seqs> [gpu_memory_utilization] | stop
set -euo pipefail
ROOT=${PREANCHOR_ROOT:?PREANCHOR_ROOT must be set}
IMAGE=vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
REVISION=1df8507178afcc1bef68cd8c393f61a886323761
HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
case "${1:-}" in
  start)
    SEQS=${2:?max_num_seqs required}; UTIL=${3:-0.90}
    [[ "$SEQS" =~ ^[0-9]+$ && "$SEQS" -ge 1 && "$SEQS" -le 64 ]] || { echo "bad max_num_seqs" >&2; exit 1; }
    ${PREANCHOR_CMD:-uv run --frozen preanchor} lease check gpu:h100 service:vllm --root "$ROOT" >/dev/null \
      || { echo 'refused: gpu:h100 and service:vllm leases are required' >&2; exit 1; }
    digest=$(sudo -n docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || true)
    [[ -n "$digest" ]] || { echo "Pinned vLLM image is not available: $IMAGE" >&2; exit 1; }
    printf '%s\n' "$digest" > "$ROOT/artifacts/vllm-image-digest.txt"
    for ((i = 0; i < NGPU; i++)); do
      name=preanchor-vllm-p2-$i
      if sudo -n docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "$name already running" >&2; exit 1; fi
      sudo -n docker run -d --rm --name "$name" --gpus "device=$i" --ipc=host \
        -p 127.0.0.1:$((8000 + i)):8000 -v "$HF_HOME:/root/.cache/huggingface" "$IMAGE" \
        --model "$MODEL" --revision "$REVISION" --dtype bfloat16 --max-model-len 24576 \
        --enable-prefix-caching --gpu-memory-utilization "$UTIL" --max-num-seqs "$SEQS" \
        --host 0.0.0.0 --port 8000 --seed 20260914 >/dev/null
      echo "started $name on 127.0.0.1:$((8000 + i))"
    done
    for ((i = 0; i < NGPU; i++)); do
      for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$((8000 + i))/health" >/dev/null && break; sleep 5
      done
      curl -sf "http://127.0.0.1:$((8000 + i))/health" >/dev/null || { echo "endpoint $i not healthy" >&2; exit 1; }
    done
    seq -s, -f "http://127.0.0.1:%g" 8000 $((8000 + NGPU - 1))
    ;;
  stop)
    for name in $(sudo -n docker ps --format '{{.Names}}' | grep '^preanchor-vllm-p2-' || true); do
      sudo -n docker stop --time 30 "$name" >/dev/null
    done
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ;;
  *) echo "usage: $0 start <max_num_seqs> [util] | stop" >&2; exit 2 ;;
esac
