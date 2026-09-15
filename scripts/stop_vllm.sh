#!/usr/bin/env bash
set -euo pipefail
if sudo -n docker ps -q --filter name='^preanchor-vllm$' | grep -q .; then
  sudo -n docker stop --time 30 preanchor-vllm
fi
for _ in $(seq 1 30); do
  if ! sudo -n docker ps -q --filter name='^preanchor-vllm$' | grep -q .; then break; fi
  sleep 1
done
if sudo -n docker ps -q --filter name='^preanchor-vllm$' | grep -q .; then
  echo 'vLLM did not exit cleanly' >&2; exit 1
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
if [[ "${used:-0}" -gt 1024 ]]; then echo "GPU memory not released: ${used} MiB" >&2; exit 1; fi
