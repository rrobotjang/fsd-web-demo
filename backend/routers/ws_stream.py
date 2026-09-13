import asyncio
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEMO_FRAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_frames"
CANVAS_W, CANVAS_H = 640, 480
STREAM_INTERVAL = 0.25


def _letterbox(image):
    h, w = image.shape[:2]
    r = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    dx, dy = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


class StreamingPipeline:
    def __init__(self):
        try:
            from models.onnx_models import ONNXLaneDetector, ONNXDetector
            self.lane_detector = ONNXLaneDetector()
            self.detector = ONNXDetector()
            if not (self.detector.available and self.lane_detector.available):
                from models.mock_models import MockDetector, MockLaneDetector
                if not self.detector.available:
                    self.detector = MockDetector()
                if not self.lane_detector.available:
                    self.lane_detector = MockLaneDetector()
        except Exception:
            from models.mock_models import MockDetector, MockLaneDetector
            self.detector = MockDetector()
            self.lane_detector = MockLaneDetector()
        from models.mock_models import MockQwenInference
        from models.colab_qwen_bridge import get_bridge
        self.qwen = get_bridge() or MockQwenInference()
        self.frames = sorted(DEMO_FRAMES_DIR.glob("*.png")) if DEMO_FRAMES_DIR.is_dir() else []
        self.frame_idx = 0

    @property
    def interval(self):
        return STREAM_INTERVAL

    def next_frame_path(self):
        if not self.frames:
            return None
        path = self.frames[self.frame_idx % len(self.frames)]
        self.frame_idx += 1
        return path

    def process_frame(self, frame_data):
        image_b64 = frame_data.get("image", "")
        if image_b64:
            image = np.array(Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB"))
            return self._process(image)
        path = self.next_frame_path()
        if path is None:
            return {}
        return self.process_path(path)

    def process_path(self, path):
        image = np.array(Image.open(path).convert("RGB"))
        return self._process(image)

    def _process(self, image):
        canvas = _letterbox(image)
        detections = self.detector.predict(canvas)
        lanes = self.lane_detector.predict(canvas)
        narration = self.qwen.generate_narration(canvas, detections, lanes)
        situation, confidence = self._situation(detections)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return {
            "frame": base64.b64encode(buf.tobytes()).decode() if ok else "",
            "frame_width": CANVAS_W,
            "frame_height": CANVAS_H,
            "detections": detections,
            "lanes": [{"points": [list(p) for p in lane]} for lane in lanes],
            "narration": narration,
            "situation": situation,
            "confidence": confidence,
            "timestamp": asyncio.get_event_loop().time(),
        }

    @staticmethod
    def _situation(detections):
        if not detections:
            return "normal_driving", 0.0
        confidence = round(max(d["confidence"] for d in detections), 4)
        classes = {d["class"] for d in detections}
        if classes & {"person"}:
            return "pedestrian_warning", confidence
        if classes & {"traffic light", "stop sign"}:
            return "traffic_sign_detected", confidence
        return "normal_driving", confidence


pipeline = StreamingPipeline()