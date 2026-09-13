import cv2
import numpy as np
from pathlib import Path

gt_dir = Path("/content/data_road/training/gt_image_2")
emb = np.zeros((1, 1, 3), dtype=np.uint8)  # placeholder

print("=== GT unique colors (top 8) ===")
for name in ["um_lane_000000", "um_lane_000014", "mm_lane_000000", "uu_road_000000"]:
    p = gt_dir / f"{name}.png"
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    bgr = cv2.imread(str(p))
    if bgr is None:
        print(f"{name}: unreadable")
        continue
    px = bgr.reshape(-1, 3)
    colors, counts = np.unique(px, axis=0, return_counts=True)
    order = np.argsort(-counts)[:8]
    print(f"{name}: H={bgr.shape[0]} W={bgr.shape[1]} total={len(px)}")
    for i in order:
        c = colors[i]
        print(f"  BGR=({c[0]},{c[1]},{c[2]}) count={counts[i]} ({counts[i]/len(px)*100:.1f}%)")

# Check road as magenta in BGR (255,0,255)
print("\n=== magenta road ratio ===")
for name in ["um_lane_000000", "um_lane_000014"]:
    p = gt_dir / f"{name}.png"
    if not p.exists():
        continue
    bgr = cv2.imread(str(p))
    magenta = (bgr[:, :, 2] > 200) & (bgr[:, :, 0] < 100) & (bgr[:, :, 1] < 150)
    print(f"{name}: magenta(R) px={magenta.sum()} = {magenta.sum()/bgr.shape[0]/bgr.shape[1]*100:.1f}%")
    # BGR interpretation check: magenta in BGR should be (255,0,255) => b=255,g=0,r=255
    magenta_bgr = (bgr[:, :, 0] > 200) & (bgr[:, :, 2] > 200) & (bgr[:, :, 1] < 150)
    print(f"  magenta(BGR) px={magenta_bgr.sum()} = {magenta_bgr.sum()/bgr.shape[0]/bgr.shape[1]*100:.1f}%")