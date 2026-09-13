#!/usr/bin/env python3
"""U-Net Lane Detection Training - runs on Colab T4 via colab CLI.

Logs progress to /content/train.log and saves checkpoints + final lane.onnx.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

LOG = Path("/content/train.log")
LOG.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("lane")


def step(msg: str):
    log.info(f"[STEP] {msg}")


def main():
    step("Starting U-Net lane detection training")

    # 1. Install deps
    step("Installing dependencies")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch", "torchvision", "opencv-python-headless",
         "onnx", "onnxruntime", "albumentations", "numpy"],
        check=True,
    )
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import cv2
    step(f"PyTorch {torch.__version__} ready")

    # 2. Download KITTI Road data
    step("Downloading KITTI Road dataset")
    data_zip = Path("/content/data_road.zip")
    if not data_zip.exists():
        subprocess.run(
            ["wget", "-q",
             "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"],
            cwd="/content", check=True,
        )
        subprocess.run(["unzip", "-q", "-o", str(data_zip), "-d", "/content"], check=True)
    step("KITTI Road data ready")

    # 3. Build dataset
    step("Building dataset")
    data_root = Path("/content/data_road/training")
    image_dir = data_root / "image_2"
    gt_dir = data_root / "gt_image_2"
    images = sorted(image_dir.glob("um_*.png"))
    log.info(f"Found {len(images)} training images")
    if len(images) > 300:
        images = images[:300]

    # 4. Define U-Net (compact)
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 5. Tensor dataset (preload, small enough)
    step("Preloading dataset tensors")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    resize = A.Compose([
        A.Resize(288, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    xs, ys = [], []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_name = f"um_lane_{img_path.name.removeprefix('um_')}"
        gt_path = gt_dir / gt_name
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = (gt > 127).astype(np.float32)
        else:
            gt = np.zeros(img.shape[:2], dtype=np.float32)
        a = resize(image=img, mask=gt)
        xs.append(a["image"])
        ys.append(a["mask"].unsqueeze(0))

    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    log.info(f"Tensors: X={tuple(X.shape)} Y={tuple(Y.shape)}")

    # 6. Train
    step("Starting training")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 40
    batch_size = 8
    n = X.size(0)

    ckpt_dir = Path("/content/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n)
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        log.info(f"Epoch {epoch}/{n_epochs} loss={avg:.4f}")
        if epoch % 5 == 0 or epoch == n_epochs:
            torch.save(model.state_dict(), ckpt_dir / f"unet_epoch_{epoch}.pth")
            log.info(f"  checkpoint saved epoch_{epoch}")

    torch.save(model.state_dict(), "/content/unet_final.pth")
    step("Training complete")

    # 7. Export ONNX
    step("Exporting ONNX")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 288, 384).to(device)
        torch.onnx.export(
            model, dummy, "/content/lane.onnx",
            export_params=True, opset_version=11,
            do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                          "output": {0: "batch", 2: "height", 3: "width"}},
        )
        import onnxruntime as ort
        s = ort.InferenceSession("/content/lane.onnx")
        out = s.run(None, {"input": dummy.cpu().numpy()})
        log.info(f"ONNX verified output shape: {out[0].shape}")

    step("DONE - /content/lane.onnx ready")
    print("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("TRAINING FAILED")
        print(f"TRAINING_FAILED: {e}")
        raise