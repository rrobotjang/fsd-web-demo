"""Generate kit_facts.json: real ONNX detection + lane facts for the 20 KITTI
demo frames, in the exact shape lora_train.py consumes.

Usage (local Mac, backend venv):
    MODEL_DIR=/Users/robotjang/fsd-web-demo/models \
    FRAMES_DIR=/Users/robotjang/fsd-web-demo/backend/data/demo_frames \
    OUT=/content/kit_facts.json \
    ./venv/bin/python ../../colab/gen_kit_facts.py

Output records:
    {"frame": "um_000000.png", "counts": {"person": 1}, "lane_count": 1}
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from models.onnx_models import ONNXDetector, ONNXLaneDetector  # noqa: E402

FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "/content/frames"))
OUT = Path(os.getenv("OUT", "/content/kit_facts.json"))
DETECTOR = ONNXDetector()
LANE = ONNXLaneDetector()
CONF = 0.5


def facts_for(frame_path: Path) -> dict:
    img = cv2.imread(str(frame_path))
    dets = DETECTOR.predict(img)
    counts = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    polygons = LANE.predict(img, threshold=CONF)
    lane_count = 1 if polygons else 0
    return {"frame": frame_path.name, "counts": counts, "lane_count": lane_count}


def main() -> int:
    if not DETECTOR.available:
        print("FATAL: detection ONNX unavailable", file=sys.stderr)
        return 1
    if not LANE.available:
        print("FATAL: lane ONNX unavailable", file=sys.stderr)
        return 1
    frames = sorted(FRAMES_DIR.glob("*.png"))
    if not frames:
        print(f"FATAL: no frames under {FRAMES_DIR}", file=sys.stderr)
        return 1
    records = [facts_for(fp) for fp in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2))
    classes = {c for r in records for c in r["counts"]}
    lanned = sum(1 for r in records if r["lane_count"])
    print(f"OK: {len(records)} records -> {OUT}")
    print(f"    classes seen: {sorted(classes)}")
    print(f"    lane detected on {lanned}/{len(records)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())