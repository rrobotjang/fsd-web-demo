#!/bin/bash
# Start vLLM + Qwen2.5-VL as a background daemon on the Colab VM.
# Runs OUTSIDE the ipykernel context (nohup + real stdio) to avoid
# `io.UnsupportedOperation: fileno` from ipykernel iostream.
#
# Usage (inside colab exec, via subprocess/bash):
#   bash /content/start_vllm_daemon.sh
set -e

CU13_LIB="/usr/local/lib/python3.13/dist-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${CU13_LIB}:${LD_LIBRARY_PATH:-}"

MODEL="/content/qwen_vl_fsd"
# NOTE: serving the Qwen3-VL family requires vLLM >= 0.11.0 and
# transformers >= 4.57. Install before starting:
#   pip install -q -U "vllm>=0.11.0" "transformers>=4.57" qwen-vl-utils
PORT=8000
LOG=/tmp/vllm.log
PIDFILE=/tmp/vllm.pid

# Kill any stale server on this port
if [ -f "$PIDFILE" ]; then
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 1

# Launch daemon with REAL stdio (file), detached from the kernel
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "qwen3-vl-4b" \
  --port "$PORT" \
  --dtype half \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code \
  --allowed-local-media-path /content \
  >"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "vLLM daemon PID: $(cat "$PIDFILE")"
echo "Log: $LOG (startup takes 1-3 min)"