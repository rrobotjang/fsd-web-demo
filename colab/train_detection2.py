#!/usr/bin/env python3
"""Reorganize KITTI det data to canonical ultralytics layout and train YOLOv8n."""

import sys, os, logging, subprocess, shutil
from pathlib import Path

LOG = Path("/content/detect_train.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("det")


def step(msg):
    log.info(f"[STEP] {msg}")


def main():
    step("YOLOv8n re-train (canonical layout)")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ultralytics", "onnx", "onnxruntime", "opencv-python-headless"], check=True)

    root = Path("/content/kitti_det")
    # Canonical layout: images/{train,val}, labels/{train,val}
    for split in ("train", "val"):
        dst_i = root / "images" / split
        dst_l = root / "labels" / split
        dst_i.mkdir(parents=True, exist_ok=True)
        dst_l.mkdir(parents=True, exist_ok=True)
        src_i = root / f"{split}_images"
        src_l = root / f"{split}_labels"
        if src_i.exists():
            for p in src_i.glob("*.png"):
                shutil.move(str(p), str(dst_i / p.name))
        if src_l.exists():
            for p in src_l.glob("*.txt"):
                shutil.move(str(p), str(dst_l / p.name))

    n_train = len(list((root / "images/train").glob("*.png")))
    n_val = len(list((root / "images/val").glob("*.png")))
    n_tr_lbl = len(list((root / "labels/train").glob("*.txt")))
    n_val_lbl = len(list((root / "labels/val").glob("*.txt")))
    log.info(f"images: train={n_train} val={n_val} | labels: train={n_tr_lbl} val={n_val_lbl}")

    yaml_path = "/content/kitti_det_yolo.yaml"
    Path(yaml_path).write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['car']\n"
    )
    log.info(f"yaml:\n{Path(yaml_path).read_text()}")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data=yaml_path, epochs=40, imgsz=384, batch=16, patience=8,
                project="/content/yolo_out", name="fsd_det", verbose=False)

    best = Path("/content/yolo_out/fsd_det/weights/best.pt")
    step(f"Best weights: {best}")
    log.info(f"results: {Path('/content/yolo_out/fsd_det/results.csv').read_text()[:500]}")

    m = YOLO(str(best))
    m.export(format="onnx", imgsz=384, opset=11, simplify=True)
    exported = best.with_suffix(".onnx")
    step(f"Exported: {exported}")

    import onnxruntime as ort
    import numpy as np
    s = ort.InferenceSession(str(exported))
    x = np.random.randn(1, 3, 384, 384).astype(np.float32)
    out = s.run(None, {s.get_inputs()[0].name: x})
    log.info(f"ONNX outputs: {[o.shape for o in out]}")

    step("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("DETECTION FAILED")
        print(f"DETECTION_FAILED: {e}")
        raise