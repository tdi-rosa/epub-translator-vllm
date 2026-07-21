#!/usr/bin/env bash
# Serves Qwen3.5-9B-AWQ behind an OpenAI-compatible API using vLLM + Docker.
#
# Requires: Docker, NVIDIA Container Toolkit, a CUDA-capable GPU.
# Every setting below can be overridden via environment variables, e.g.:
#   HF_CACHE_DIR=/mnt/models GPU_MEM_UTIL=0.85 ./serve_vllm.sh

set -euo pipefail

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MODEL="${MODEL:-QuantTrio/Qwen3.5-9B-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

sudo docker run --rm --gpus all \
    -p "${PORT}:8000" \
    --ipc=host \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    vllm/vllm-openai:latest \
    --model "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --structured-outputs-config.backend=guidance
