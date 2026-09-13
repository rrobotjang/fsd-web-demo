#!/usr/bin/env python3
"""Verify lane model vs GT + test yolov8n.onnx inference."""

import sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
log = logging.getLogger("v")

import torch
import torch.nn as nn
import numpy as np
import cv2


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load("/content/unet_final.pth", map_location=device))
    model.eval()

    img_dir = Path("/content/data_road/training/image_2")
    gt_dir = Path("/content/data_road/training/gt_image_2")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print("=== LANE MODEL vs GT (training images) ===")
    for name in ["um_000000", "um_000014", "uu_000000", "um_000100", "uu_000014"]:
        imgp = img_dir / f"{name}.png"
        prefix = name.split("_")[0]
        gtp = gt_dir / f"{prefix}_lane_{name.split('_')[1]}.png"
        if not imgp.exists():
            print(f"skip {name}: no image"); continue
        if not gtp.exists():
            print(f"skip {name}: no gt {gtp.name}"); continue
        img = cv2.imread(str(imgp))
        inp = cv2.resize(img, (384, 288)).astype(np.float32) / 255.0
        inp = (inp - mean) / std
        x = inp.transpose(2, 0, 1)[None]
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()
        gt = cv2.imread(str(gtp))
        if gt is None:
            print(f"skip {name}: gt unreadable"); continue
        road = (gt[:, :, 0] > 200) & (gt[:, :, 2] > 200) & (gt[:, :, 1] < 150)
        road = cv2.resize(road.astype(np.uint8), (384, 288), interpolation=cv2.INTER_NEAREST) > 0
        pm = pred > 0.5
        inter = (pm & road).sum(); union = (pm | road).sum()
        iou = inter / max(union, 1)
        print(f"{name}: pred max={pred.max():.3f} mean={pred.mean():.3f} px>0.5={pm.sum()} "
              f"| GT road px={road.sum()} | IoU={iou:.3f}")

    print("=== yolov8n.onnx inference test ===")
    import onnxruntime as ort
    s = ort.InferenceSession("/content/yolov8n.onnx")
    inp_name = s.get_inputs()[0].name
    print(f"input: {inp_name} {s.get_inputs()[0].shape}")
    for name in ["um_000000", "um_000014"]:
        img = cv2.imread(str(img_dir / f"{name}.png"))
        r = cv2.resize(img, (384, 384))
        xx = r.astype(np.float32) / 255.0
        xx = xx.transpose(2, 0, 1)[None]
        out = s.run(None, {inp_name: xx})
        print(f"{name}: {len(out)} outputs, shapes={[o.shape for o in out]}")

    print("ALL_VALIDATED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("FAILED")
        raise