#!/usr/bin/env python3
"""One-shot: kill stuck train, verify lane model quality vs GT, export pretrained YOLOv8n to ONNX."""

import sys, subprocess, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
log = logging.getLogger("one")

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.final(d1))


def main():
    # 1. Kill lingering training processes
    subprocess.run(["pkill", "-f", "train_detection"], capture_output=True)
    subprocess.run(["pkill", "-f", "ultralytics"], capture_output=True)
    log.info("killed lingering training")

    import numpy as np
    import cv2

    # 2. Verify lane model output vs GT on training images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ckpt = Path("/content/unet_final.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log.info(f"lane ckpt loaded: {ckpt.name}")

    img_dir = Path("/content/data_road/training/image_2")
    gt_dir = Path("/content/data_road/training/gt_image_2")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for name in ["um_000000", "um_000014", "uu_000000", "um_000100"]:
        imgp = img_dir / f"{name}.png"
        gtp = gt_dir / f"{name.replace('um_', 'um_lane_').replace('uu_', 'uu_lane_').replace('umm_', 'umm_lane_')}.png"
        if not imgp.exists():
            log.info(f"skip {name}: no image")
            continue
        if not gtp.exists():
            log.info(f"skip {name}: no gt ({gtp.name})")
            continue
        img = cv2.imread(str(imgp))
        h, w = img.shape[:2]
        inp = cv2.resize(img, (384, 288)).astype(np.float32) / 255.0
        inp = (inp - mean) / std
        x = inp.transpose(2, 0, 1)[None]
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()
        # GT road mask: magenta (255,0,255) road region
        gt = cv2.imread(str(gtp))
        road = (gt[:, :, 0] > 200) & (gt[:, :, 2] > 200) & (gt[:, :, 1] < 150)
        road = cv2.resize(road.astype(np.uint8), (384, 288), interpolation=cv2.INTER_NEAREST) > 0
        pm = pred > 0.5
        inter = (pm & road).sum(); union = (pm | road).sum()
        iou = inter / max(union, 1)
        log.info(f"{name}: pred max={pred.max():.3f} mean={pred.mean():.3f} px>0.5={pm.sum()} | GT road px={road.sum()} | IoU={iou:.3f}")

    # 3. Export pretrained YOLOv8n (COCO) to ONNX - no training needed
    log.info("exporting pretrained YOLOv8n -> ONNX")
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    m.export(format="onnx", imgsz=384, opset=11, simplify=True)
    exported = Path("/content/yolov8n.onnx")
    src = Path("/content/yolov8n.pt").with_suffix(".onnx")
    if src.exists() and not exported.exists():
        src.rename(exported)
    sz = exported.stat().st_size / 1e6
    log.info(f"yolov8n.onnx exported: {sz:.1f} MB")

    yolo = YOLO("/content/yolov8n.pt", task="detect")
    res = yolo("/content/data_road/training/image_2/um_000000.png", verbose=False)[0]
    n_car = sum(1 for c in res.boxes.cls.tolist())
    log.info(f"pretrained YOLO on um_000000: {len(res.boxes)} boxes, classes={res.boxes.cls.tolist()}, names check car={res.names[0]}")

    log.info("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("FAILED")
        raise