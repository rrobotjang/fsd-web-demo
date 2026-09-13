#!/bin/bash
# Repair: realign torch/torchvision/torchaudio to one consistent CUDA 12.8 build,
# then install the Qwen3-VL serving/training stack. Run detached (client timeouts
# kill exec output, but the kernel keeps running the launched process).
set -euo pipefail
trap 'echo "FATAL line $LINENO: $BASH_COMMAND" >&2' ERR

echo "=== 1/3 realign torch stack (cu128) ==="
python3 -m pip install -q --upgrade --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

echo "=== 2/3 verify torch imports sync ==="
python3 - <<'PY'
import torch, torchvision, torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
PY

echo "=== 3/3 install vllm + training deps ==="
python3 -m pip install -q --upgrade "vllm>=0.11.0" "transformers>=4.57" \
  "qwen-vl-utils==0.0.14" accelerate peft bitsandbytes

echo "=== verify imports ==="
python3 - <<'PY'
import importlib
for m in ["vllm", "transformers", "qwen_vl_utils", "peft", "accelerate", "bitsandbytes"]:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAIL:", e)
PY

echo "=== REPAIR DONE ==="