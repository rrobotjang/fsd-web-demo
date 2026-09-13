"""ColabQwenBridge: real Qwen3-VL narration via any network-reachable vLLM daemon.

Transport: plain HTTPS/HTTP to an OpenAI-compatible vLLM endpoint
(`COLAB_VLLM_URL`). The image travels inline as a base64 data URL in the
request body, so the daemon can live anywhere reachable from the Mac:
  - a Colab runtime attached via the official Google Colab VS Code extension
    (cloudflared/ngrok tunnel URL, e.g. https://xxx.trycloudflare.com),
  - a Google Cloud VM (SSH tunnel -> http://localhost:8000),
  - an always-on server (direct URL).

No colab CLI, no session naming, no file staging on the VM: this bridge only
needs an HTTP client.

Two modes:
  - streaming (`generate_narration`): records the latest scene, returns the cached
    narration immediately (never blocks the WS pipeline); a background thread calls
    vLLM at most once per `interval` seconds.
  - one-shot (`infer_once`): synchronous call for REST endpoints like /api/qwen.

Falls back wirelessly: if the daemon is down or the request fails, the last
good narration is kept.
"""
import base64
import json
import os
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

COLAB_VLLM_URL = os.getenv("COLAB_VLLM_URL", "http://localhost:8000/v1/chat/completions")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl-4b")
QWEN_INTERVAL = float(os.getenv("QWEN_INTERVAL", "4.0"))
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "60"))
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.9"))
QWEN_REROLL_TEMPERATURE = float(os.getenv("QWEN_REROLL_TEMPERATURE", "1.0"))
QWEN_FREQUENCY_PENALTY = float(os.getenv("QWEN_FREQUENCY_PENALTY", "1.0"))
QWEN_HTTP_TIMEOUT = int(os.getenv("QWEN_HTTP_TIMEOUT", "100"))

# Rotated per call to break the fixed-metalanguage monotony the model memorized
# from a homogeneous fine-tune set. Each template still enforces the 2-sentence
# narration contract, but with different phrasing so outputs diversify.
_NARRATION_TEMPLATES = (
    "You are the narration module of an autonomous driving system. {facts} "
    "Describe the current driving situation in exactly 2 short sentences: "
    "road condition, nearby vehicles/pedestrians, and any hazards. "
    "Answer in plain English, no preamble.",
    "You are the driving narration system of an autonomous vehicle. {facts} "
    "In exactly 2 short sentences, tell the driver about the road state, "
    "nearby vehicles or pedestrians, and any hazard ahead. "
    "Plain English, no preamble.",
    "You are the AI copilot of a self-driving car. {facts} "
    "Say in 2 short sentences what is happening on the road right now: "
    "the drivable surface, other traffic, and anything to watch out for. "
    "No preamble, plain English.",
    "You are an automotive safety narrator. {facts} "
    "Using the vision facts, describe this driving moment in exactly 2 short "
    "sentences: road ahead, surrounding vehicles/pedestrians, and any risk. "
    "Be brief and direct.",
)  # _NARRATION_TEMPLATES


def _image_to_jpeg_b64(image) -> str:
    """Convert ndarray (RGB) or base64-str image to a compact base64 JPEG."""
    if isinstance(image, str):
        try:
            if image.startswith("data:"):
                image = image.split(",", 1)[1]
            return image  # already base64
        except Exception:
            return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Downscale so the HTTP payload stays small (~640px wide is plenty for narration)
    h, w = arr.shape[:2]
    max_w = 640
    if w > max_w:
        arr = Image.fromarray(arr).resize((max_w, round(h * max_w / w)), Image.LANCZOS)
        arr = np.asarray(arr)
    buf = __import__("io").BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _facts_prompt(detections: List[Dict], lanes: List) -> str:
    """Summarize vision facts into the narration prompt."""
    counts: Dict[str, int] = {}
    for d in detections or []:
        cls = str(d.get("class", "object"))
        counts[cls] = counts.get(cls, 0) + 1
    parts = [f"{n} {c}(s)" for c, n in sorted(counts.items(), key=lambda x: -x[1])]
    facts = ", ".join(parts) if parts else "no objects detected"
    lane_desc = f"{len(lanes or [])} lane(s) fitted" if lanes else "no lane info"
    return f"Vision facts: {facts}. Lane state: {lane_desc}."


class ColabQwenBridge:
    def __init__(self):
        self._tpl_idx = 0
        self._lock = threading.Lock()
        self._latest_scene: Optional[Tuple[str, List, List]] = None
        self._narration = "Vision narrator: connecting to vLLM daemon..."
        self._last_success = 0.0
        self._last_attempt = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True, name="colab-qwen")
        self._thread.start()

    # ------------------------------------------------------------------ public
    def generate_narration(self, image, detections: List[Dict] = None,
                           lanes: List = None) -> str:
        """Streaming mode: stash the latest scene, return cached narration synchronously."""
        b64 = _image_to_jpeg_b64(image)
        with self._lock:
            self._latest_scene = (b64, detections or [], lanes or [])
            return self._narration

    def infer_once(self, image, detections: List[Dict] = None,
                   lanes: List = None, timeout: int = 90) -> str:
        """One-shot synchronous inference (used by REST endpoints)."""
        b64 = _image_to_jpeg_b64(image)
        narration = self._request_narration(b64, detections or [], lanes or [], timeout)
        if narration:
            with self._lock:
                self._narration = narration
            return narration
        with self._lock:
            return self._narration

    def health_check(self) -> bool:
        """Daemon is up when the OpenAI-compatible /models endpoint answers 200."""
        models_url = COLAB_VLLM_URL.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ worker
    def _worker(self):
        while not self._stop:
            time.sleep(0.5)
            now = time.time()
            with self._lock:
                scene = self._latest_scene
                if scene is None:
                    continue
                if now - self._last_attempt < QWEN_INTERVAL:
                    continue
                b64, detections, lanes = scene
                self._last_attempt = now
            with self._lock:
                prev = self._narration
            narration = self._request_narration(b64, detections, lanes, timeout=120)
            if narration and narration == prev:
                narration = self._request_narration(
                    b64, detections, lanes, timeout=120, rephrase=True)
            if narration:
                with self._lock:
                    self._narration = narration
                    self._last_success = time.time()

    def _request_narration(self, image_b64: str, detections: List[Dict],
                           lanes: List, timeout: int,
                           rephrase: bool = False) -> Optional[str]:
        with self._lock:
            self._tpl_idx = (self._tpl_idx + 1) % len(_NARRATION_TEMPLATES)
            prev = self._narration
        template = _NARRATION_TEMPLATES[self._tpl_idx]
        if rephrase:
            instruction = (
                "Now describe the situation with completely different wording. "
                "Do NOT reuse any sentence from your previous narration, do NOT "
                "restate earlier sentences. Write exactly 2 new short sentences "
                "about what is on the road right now."
            )
        elif prev and not prev.startswith("Vision narrator:"):
            instruction = (
                "Describe the CURRENT moment in exactly 2 new short sentences. "
                "Do NOT repeat or reword your previous narration - every sentence "
                "must be fresh. Say what is happening in front of the car right now."
            )
        else:
            instruction = None
        user_text = (
            template.format(facts=_facts_prompt(detections, lanes))
            + (f" {instruction}" if instruction else "")
        )
        if any(d.get("class") in ("person", "pedestrian")
               for d in (detections or [])):
            user_text += (
                " A pedestrian is present - the narration must explicitly mention"
                " the pedestrian's location and that caution is needed."
            )
        payload = {
            "model": QWEN_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": user_text},
                ],
            }],
            "max_tokens": QWEN_MAX_TOKENS,
            "temperature": QWEN_REROLL_TEMPERATURE if rephrase else QWEN_TEMPERATURE,
            "frequency_penalty": QWEN_FREQUENCY_PENALTY,
        }
        try:
            req = urllib.request.Request(
                COLAB_VLLM_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=min(timeout, QWEN_HTTP_TIMEOUT)) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


bridge = None


def get_bridge() -> Optional[ColabQwenBridge]:
    """Lazy singleton so importing models never crashes when colab is absent."""
    global bridge
    if not os.getenv("COLAB_VLLM_URL"):
        return None
    if bridge is None:
        bridge = ColabQwenBridge()
    return bridge