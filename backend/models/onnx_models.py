"""Real ONNX inference models (lane + detection).

Replaces mock models when ONNX files are present in MODEL_DIR.
Gracefully falls back to mock implementations when files are missing.
"""

import os
from pathlib import Path
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models")).resolve()
LANE_ONNX = MODEL_DIR / "lane.onnx"
LANE_ONNX_INT8 = MODEL_DIR / "lane_int8_qdq.onnx"
DET_ONNX = MODEL_DIR / "yolov8n.onnx"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO 80 class names (index = class id), subset we care about for FSD demo
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

FSD_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person",
               "traffic light", "stop sign"}


def _letterbox(img, size=384):
    """Resize preserving aspect ratio, pad to square."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


class ONNXLaneDetector:
    """Road/lane-area segmentation via ONNX U-Net trained on KITTI Road."""

    def __init__(self, model_path: os.PathLike = None):
        # Prefer int8-quantized model when an explicit path is not given
        self.model_path = Path(model_path) if model_path else LANE_ONNX_INT8
        self.session = None
        self._load()
        if self.session is None and not model_path and LANE_ONNX.exists():
            self.model_path = LANE_ONNX
            self._load()

    def _load(self):
        if not HAS_ORT or not self.model_path.exists():
            return
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"])
        except Exception:
            self.session = None

    @property
    def available(self) -> bool:
        return self.session is not None

    def predict(self, image: np.ndarray, threshold: float = 0.5
                ) -> list[list[tuple[int, int]]]:
        """Return lane/road polygon as list of point lists (image coords)."""
        if self.session is None:
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        inp = cv2.resize(rgb, (384, 288)).astype(np.float32) / 255.0
        inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
        x = inp.transpose(2, 0, 1)[None].astype(np.float32)

        out = self.session.run(None, {"input": x})[0]  # (1,1,288,384)
        mask = (out[0, 0] > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return []

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Extract road-region contour as a polygon
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        main = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main) < w * h * 0.005:
            return []

        epsilon = 0.002 * cv2.arcLength(main, True)
        approx = cv2.approxPolyDP(main, epsilon, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        # Close polygon -> return as single lane polygon (drivable area)
        return [pts]


class ONNXDetector:
    """Object detection via YOLOv8n ONNX (COCO 80 classes)."""

    def __init__(self, model_path: os.PathLike = None,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self.model_path = Path(model_path) if model_path else DET_ONNX
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self._load()

    def _load(self):
        if not HAS_ORT or not self.model_path.exists():
            return
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"])
        except Exception:
            self.session = None

    @property
    def available(self) -> bool:
        return self.session is not None

    def predict(self, image: np.ndarray) -> list[dict]:
        """Run YOLOv8n detection. Returns [{'class','confidence','bbox'}...]"""
        if self.session is None:
            return []

        h, w = image.shape[:2]
        img, r, dx, dy = _letterbox(image, size=384)
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]

        out = self.session.run(None, {"images": x})[0]  # (1, 84, 3024)
        pred = out[0].T  # (3024, 84)

        boxes_xywh = pred[:, :4]  # in letterbox coords
        scores = pred[:, 4:]       # class scores (already sigmoid)

        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)

        keep = confs >= self.conf_threshold
        if not keep.any():
            return []

        boxes, cls_ids, conf_vals = boxes_xywh[keep], class_ids[keep], confs[keep]

        # xywh (center) -> xyxy in letterbox coords
        cx, cy, bw_, bh_ = boxes.T
        x1 = cx - bw_ / 2
        y1 = cy - bh_ / 2
        x2 = cx + bw_ / 2
        y2 = cy + bh_ / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        nms_idx = cv2.dnn.NMSBoxes(
            xyxy.tolist(), conf_vals.tolist(), self.conf_threshold,
            self.iou_threshold)
        if isinstance(nms_idx, tuple) or nms_idx is None:
            nms_idx = np.array([], dtype=int)
        nms_idx = np.asarray(nms_idx).flatten()

        results = []
        for i in nms_idx:
            cid = int(cls_ids[i])
            name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else "unknown"
            if name not in FSD_CLASSES:
                continue
            # Map from letterbox coords back to original image
            ox1 = int((xyxy[i][0] - dx) / r)
            oy1 = int((xyxy[i][1] - dy) / r)
            ox2 = int((xyxy[i][2] - dx) / r)
            oy2 = int((xyxy[i][3] - dy) / r)
            ox1 = max(0, min(w, ox1))
            ox2 = max(0, min(w, ox2))
            oy1 = max(0, min(h, oy1))
            oy2 = max(0, min(h, oy2))
            results.append({
                "class": name,
                "confidence": round(float(conf_vals[i]), 4),
                "bbox": [ox1, oy1, ox2, oy2],
            })
        return results