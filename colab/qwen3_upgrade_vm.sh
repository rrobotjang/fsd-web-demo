#!/bin/bash
# One-shot: Qwen3-VL-4B upgrade + train + merge + daemon start.
# vLLM install owns the torch version; torchvision/torchaudio must be realigned
# to that exact CUDA build afterwards or transformers lazy-import crashes.
set -euo pipefail
trap 'echo "FATAL line $LINENO: $BASH_COMMAND" >&2' ERR

echo "=== 1/5 deps: vLLM keeps its torch build ==="
python3 -m pip install -q "vllm>=0.11.0" "transformers>=4.57" \
  "qwen-vl-utils==0.0.14" accelerate peft bitsandbytes

echo "=== 2/5 realign torchvision/torchaudio to torch's CUDA ==="
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
CU_INDEX="cu${CUDA_VER//./}"
echo "torch CUDA=${CUDA_VER} -> index ${CU_INDEX}"
python3 -m pip install -q --upgrade \
  --index-url "https://download.pytorch.org/whl/${CU_INDEX}" \
  torchvision torchaudio

echo "=== 3/5 verify stack imports ==="
python3 - <<'PY'
import torch, torchvision, torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
print("transformers Qwen3VL import OK")
PY

echo "=== 4/5 kit_facts + frames checks ==="
test -f /content/kit_facts.json || { echo "MISSING: upload kit_facts.json to /content/"; exit 1; }
test -d /content/frames && [ "$(ls /content/frames/*.png 2>/dev/null | wc -l)" -ge 1 ] || \
  { echo "MISSING: upload 20 PNGs to /content/frames/"; exit 1; }

echo "=== QLoRA train (Qwen3-VL-4B-Instruct) ==="
python3 /content/lora_train.py train

echo "=== fp16 merge -> /content/qwen_vl_fsd ==="
python3 /content/lora_train.py merge

echo "=== start vLLM daemon ==="
bash /content/start_vllm_daemon.sh

echo "=== verify daemon ==="
for i in $(seq 1 36); do
  HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null || true)
  [ "$HTTP" = "200" ] && echo "OK: daemon ready (model list served)" && exit 0
  sleep 5
done
echo "WARN: daemon not responding after 180s; check /tmp/vllm.log"