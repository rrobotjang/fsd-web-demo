#!/usr/bin/env python3
"""Load trained U-Net checkpoint and export to ONNX."""

import sys, logging
from pathlib import Path

LOG = Path("/content/train.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("export")

def step(msg):
    log.info(f"[STEP] {msg}")

def main():
    step("Installing onnxscript")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxscript"], check=True)

    import torch
    import torch.nn as nn

    # Same UNet architecture as training script
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
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

    ckpt = Path("/content/unet_final.pth")
    if not ckpt.exists():
        ckpt = sorted(Path("/content/checkpoints").glob("unet_epoch_*.pth"))[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log.info(f"Loaded checkpoint: {ckpt.name}")

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

    size = Path("/content/lane.onnx").stat().st_size / 1e6
    log.info(f"lane.onnx saved ({size:.1f} MB)")
    step("ALL_DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("EXPORT FAILED")
        print(f"EXPORT_FAILED: {e}")
        raise