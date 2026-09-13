#!/usr/bin/env python3
"""Retrain lane U-Net with CORRECT KITTI GT (magenta road mask) and single-file ONNX export."""

import sys, logging, subprocess
from pathlib import Path

LOG = Path("/content/lane_retrain.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("lane")


def step(msg):
    log.info(f"[STEP] {msg}")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "albumentations", "onnx", "onnxruntime", "opencv-python-headless"], check=True)

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    img_dir = Path("/content/data_road/training/image_2")
    gt_dir = Path("/content/data_road/training/gt_image_2")
    images = sorted(img_dir.glob("um_*.png"))
    log.info(f"{len(images)} um images")

    resize_xy = Compose([
        Resize(288, 384),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Verify GT extraction on first image
    g0 = cv2.imread(str(gt_dir / "um_lane_000000.png"))
    m0 = (g0[:, :, 0] > 200) & (g0[:, :, 1] < 150) & (g0[:, :, 2] > 200)
    log.info(f"um_000000 magenta road: {m0.sum()} / {m0.size} = {m0.mean()*100:.1f}%")

    xs, ys = [], []
    skipped = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1; continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_name = f"um_lane_{img_path.name.removeprefix('um_')}"
        gt_path = gt_dir / gt_name
        if gt_path.exists():
            gt = cv2.imread(str(gt_path))
            mask = (gt[:, :, 0] > 200) & (gt[:, :, 1] < 150) & (gt[:, :, 2] > 200)
            mask = mask.astype(np.float32)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        a = resize_xy(image=img, mask=mask)
        xs.append(a["image"]); ys.append(a["mask"].unsqueeze(0))

    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    mask_ratio = Y.mean().item()
    log.info(f"X={tuple(X.shape)} Y={tuple(Y.shape)} avg mask ratio={mask_ratio:.3f} skipped={skipped}")

    torch.save({"X": X.cpu(), "Y": Y.cpu()}, "/content/kitti_road_tensors.pt")

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 40
    batch_size = 8
    n = X.size(0)
    ckpt_dir = Path("/content/checkpoints2")
    ckpt_dir.mkdir(exist_ok=True)

    step("Starting training")
    best_loss = float("inf")
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
            total_loss += loss.item() * xb.size(0)
            n_batches += 1
        avg = total_loss / n
        log.info(f"Epoch {epoch}/{n_epochs} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"unet_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "/content/unet_kitti.pth")
    log.info(f"Final loss={best_loss:.4f} saved /content/unet_kitti.pth")

    # Export ONNX (single file)
    step("Exporting ONNX (single file)")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 288, 384).to(device)
        torch.onnx.export(
            model, dummy, "/content/lane_v2_raw.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                          "output": {0: "batch", 2: "height", 3: "width"}},
        )

    # Inline external data (if split) -> single self-contained file
    import onnx
    m = onnx.load("/content/lane_v2_raw.onnx")
    onnx.save_model(m, "/content/lane.onnx", save_as_external_data=False)
    sz = Path("/content/lane.onnx").stat().st_size / 1e6
    log.info(f"lane.onnx single file: {sz:.2f} MB")

    # Verify inference + sanity check on KITTI image
    import onnxruntime as ort
    s = ort.InferenceSession("/content/lane.onnx")
    img = cv2.imread(str(img_dir / "um_000000.png"))
    inp = cv2.resize(img, (384, 288)).astype(np.float32) / 255.0
    inp = (inp - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    inp = inp.transpose(2, 0, 1)[None]
    out = s.run(None, {"input": inp})[0]
    road_px = (out[0, 0] > 0.5).sum()
    log.info(f"Inference on um_000000: out={out.shape} max={out.max():.4f} mean={out.mean():.4f} road_px>0.5={road_px}")
    step("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("TRAINING FAILED")
        print(f"TRAINING_FAILED: {e}")
        raise