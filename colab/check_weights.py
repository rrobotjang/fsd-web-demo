import torch
from pathlib import Path

for name, p in [("unet_final.pth", Path("/content/unet_final.pth")),
                ("epoch_40.pth", Path("/content/checkpoints/unet_epoch_40.pth"))]:
    sd = torch.load(p, map_location="cpu")
    total = sum(v.numel() for v in sd.values())
    first_keys = list(sd.keys())[:3]
    print(f"{name}: {p.stat().st_size/1e6:.2f} MB, params={total:,}")
    print(f"  first keys: {first_keys}")