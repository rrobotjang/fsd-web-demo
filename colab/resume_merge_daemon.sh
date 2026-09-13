#!/bin/bash
# Resume: realign torchao for PEFT compatibility, re-run merge (adapter is
# already trained at /content/qwen_lora_adapter), then start vLLM daemon.
set -euo pipefail
trap 'echo "FATAL line $LINENO: $BASH_COMMAND" >&2' ERR

echo "=== 1/4 upgrade torchao (PEFT needs >0.16.0) ==="
python3 -m pip install -q --upgrade "torchao>=0.16.0"

echo "=== 2/4 verify peft + torchao ==="
python3 - <<'PY'
import torchao, peft
print("torchao", torchao.__version__)
print("peft", peft.__version__)
PY

echo "=== 3/4 verify adapter present ==="
ls /content/qwen_lora_adapter/ | head -5

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