"""Local-side test: send a KITTI frame from macOS -> Colab T4 vLLM daemon.

Tests two transport paths:
  A) base64 data URI embedded in the chat payload (no colab-side file)
  B) upload frame bytes to /content/latest.jpg on Colab, then file:// URL

Requires env: COLAB_SESSION (default vllm-vl)
"""
import base64
import io
import json
import os
import subprocess
import sys
import time

import PIL.Image as Image

COLAB = "/opt/anaconda3/envs/venv/bin/colab"
SESSION = os.environ.get("COLAB_SESSION", "lora-train")
FRAME = sys.argv[1] if len(sys.argv) > 1 else "/Users/robotjang/fsd-web-demo/backend/data/demo_frames/um_000000.png"
MODEL = "qwen3-vl-4b"
VLLM_URL = "http://localhost:8000/v1/chat/completions"

PROMPT = (
    "You are an autonomous driving narrator. Describe this KITTI driving scene "
    "in 2 sentences: road condition, vehicles, and any hazards."
)


def frame_to_jpeg_b64(path: str, max_w=640) -> str:
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        h = round(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def colab_exec(code: str, timeout=240) -> str:
    r = subprocess.run(
        [COLAB, "exec", "-s", SESSION, "--timeout", str(timeout)],
        input=code, capture_output=True, text=True, timeout=timeout + 30,
    )
    return r.stdout


def make_payload(image_ref: dict) -> dict:
    return {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image_ref},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 96,
        "temperature": 0.7,
    }


def request_via_code(payload: dict) -> dict:
    """Wrap curl to the daemon inside a colab exec (daemon only listens on VM localhost)."""
    payload_json = json.dumps(payload)
    code = f"""import subprocess, json
payload_json = {payload_json!r}
r = subprocess.run(["bash","-c",
    "curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @-"],
    input=payload_json, capture_output=True, text=True, timeout=200)
print("RAW_BEGIN")
print(r.stdout)
print("RAW_END")"""
    out = colab_exec(code)
    begin = out.find("RAW_BEGIN")
    end = out.find("RAW_END")
    if begin < 0 or end < 0:
        return {"error": "colab exec failed. Output: " + out[-800:]}
    raw = out[begin + len("RAW_BEGIN"):end].strip()
    try:
        return json.loads(raw)
    except Exception as e:
        return {"error": f"parse fail: {e}, raw={raw[:400]}"}


def main():
    print(f"[local] frame={FRAME} session={SESSION}")
    b64 = frame_to_jpeg_b64(FRAME)
    print(f"[local] jpeg base64 len={len(b64)}")

    # ---- Path A: data URI embedded in payload ----
    print("\n=== PATH A: base64 data URI ===")
    payload_a = make_payload({"url": f"data:image/jpeg;base64,{b64}"})
    t0 = time.time()
    res_a = request_via_code(payload_a)
    print(f"[A] time={time.time()-t0:.1f}s")
    if "error" in res_a:
        print("[A] FAIL:", res_a["error"][:400])
    else:
        print("[A] RESPONSE:", res_a["choices"][0]["message"]["content"][:250])

    # ---- Path B: upload to /content/latest.jpg then file:// ----
    print("\n=== PATH B: upload + file:// ===")
    up = subprocess.run(
        [COLAB, "upload", "-s", SESSION, "/dev/stdin", "/content/latest.jpg"],
        input=b64, capture_output=True, text=True, timeout=120,
    )
    if up.returncode != 0 or "Uploaded" not in up.stdout:
        # upload from stdin may not be supported; write via exec instead
        code = f"""import base64
open("/content/latest.jpg","wb").write(base64.b64decode({json.dumps(b64)}))
print("written")"""
        wr = colab_exec(code, timeout=120)
        print("[B] write result:", wr.strip()[-100:])
    else:
        print("[B] upload ok:", up.stdout.strip()[:100])

    payload_b = make_payload({"url": "file:///content/latest.jpg"})
    t0 = time.time()
    res_b = request_via_code(payload_b)
    print(f"[B] time={time.time()-t0:.1f}s")
    if "error" in res_b:
        print("[B] FAIL:", res_b["error"][:400])
    else:
        print("[B] RESPONSE:", res_b["choices"][0]["message"]["content"][:250])


if __name__ == "__main__":
    main()