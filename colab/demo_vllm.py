#!/usr/bin/env python3
"""colab run demo helper: provision a T4 VM with vLLM serving Qwen3-VL-4B.

Run from the Mac (local file; executed on a fresh Colab VM):

    colab run -s lora-train --gpu T4 --keep --timeout 1800 \\
        colab/demo_vllm.py --token "$(cat ~/.cache/huggingface/token)"

--keep holds the VM after provisioning; this script also starts a cloudflared
tunnel so the daemon is reachable from the Mac. Point the backend at the
printed URL:

    COLAB_VLLM_URL=<printed-url>/v1/chat/completions \\
      backend/venv/bin/python -m uvicorn main:app --port 8000

Stop the VM when the demo is over:

    colab stop -s lora-train
"""
import base64
import os
import re
import subprocess
import sys
import time


def sh(cmd, timeout=1200, check=False):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and check:
        raise SystemExit(f"FAILED: {cmd}\n{r.stderr[-800:]}")
    return r


def main():
    token = None
    args = sys.argv[1:]
    if args and args[0] == "--token" and len(args) > 1:
        token = args[1]
    if not token:
        print("WARN: no --token arg; gated model download may fail")

    print("== 1/4 install vLLM stack ==")
    sh("python3 -m pip install -q 'vllm>=0.11.0' 'transformers>=4.57' "
       "'qwen-vl-utils==0.0.14' accelerate peft bitsandbytes", check=True)

    print("== 2/4 realign torchvision/torchaudio ==")
    r = subprocess.run(["python3", "-c", "import torch; print(torch.version.cuda)"],
                       capture_output=True, text=True)
    cuda_ver = r.stdout.strip()
    if cuda_ver:
        cu = "cu" + cuda_ver.replace(".", "")
        sh(f"python3 -m pip install -q --upgrade --index-url "
           f"https://download.pytorch.org/whl/{cu} torchvision torchaudio")
    print("torch CUDA:", cuda_ver)

    print("== 3/4 HF token ==")
    if token:
        hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(hf, exist_ok=True)
        with open(os.path.join(hf, "token"), "w") as f:
            f.write(token)
        print("written HF token")

    print("== 4/4 start vLLM daemon (background, kept alive by --keep) ==")
    daemon = r"""#!/bin/bash
set -e
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
NVIDIA_LIBS=$(python3 -c "
import glob
print(':'.join(glob.glob('/usr/local/lib/python3.*/dist-packages/nvidia/*/lib')))
" 2>/dev/null || true)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
PORT=8000
LOG=/tmp/vllm.log
PIDFILE=/tmp/vllm.pid
if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 1
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "qwen3-vl-4b" --port "$PORT" \
  --dtype half --enforce-eager --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code --allowed-local-media-path /content \
  >"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "vLLM daemon PID: $(cat "$PIDFILE")"
"""
    with open("/content/start_vllm_daemon.sh", "w") as f:
        f.write(daemon)
    sh("bash /content/start_vllm_daemon.sh", check=True)

    print("== waiting for /v1/models (download + load can take 3-10 min) ==")
    ready = False
    for i in range(60):
        r = sh("curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models", timeout=60)
        code = r.stdout.strip()
        if code == "200":
            ready = True
            print(f"DAEMON_READY after ~{i*15}s")
            break
        if i % 4 == 0:
            tail = sh("tail -3 /tmp/vllm.log").stdout.strip()
            print(f"[{(i+1)*15}s] http={code or 'n/a'} | {tail[-200:]}")
        time.sleep(15)
    if not ready:
        print("FAIL: daemon not ready; check /tmp/vllm.log")
        sys.exit(1)

    print("== 5/5 start cloudflared tunnel (daemon -> public URL) ==")
    sh("pkill -f 'cloudflared tunnel' 2>/dev/null || true")
    sh("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
       check=True)
    sh("nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate "
       ">/tmp/cloudflared.log 2>&1 &")
    tunnel_url = None
    for i in range(30):
        try:
            txt = open("/tmp/cloudflared.log").read()
        except FileNotFoundError:
            txt = ""
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
        if m:
            tunnel_url = m.group(0)
            break
        time.sleep(2)
    if tunnel_url:
        print(f"PUBLIC_VLLM_URL={tunnel_url}/v1/chat/completions")
        print("Point the Mac backend at it: COLAB_VLLM_URL=<above> "
              "backend/venv/bin/python -m uvicorn main:app --port 8000")
        print("Stop afterwards with: colab stop -s lora-train")
    else:
        print("WARN: tunnel not up yet. Mac backend cannot reach the daemon.\n"
              + open("/tmp/cloudflared.log").read()[-600:])
        sys.exit(1)


if __name__ == "__main__":
    main()