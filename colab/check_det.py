import sys
from pathlib import Path

root = Path("/content/kitti_det")
imgs = sorted((root / "train_images").glob("*.png"))
labels = sorted((root / "train_labels").glob("*.txt"))
print(f"train_images: {len(imgs)}")
print(f"train_labels txt: {len(labels)}")
if imgs and labels:
    print(f"sample img: {imgs[0].name}, sample label: {labels[0].name}")
    print(labels[0].read_text())
v_imgs = sorted((root / "val_images").glob("*.png"))
v_labels = sorted((root / "val_labels").glob("*.txt"))
print(f"val_images: {len(v_imgs)}, val_labels: {len(v_labels)}")
print(Path("/content/kitti_det_yolo.yaml").read_text())
# cross check naming
img_stems = {p.stem for p in imgs}
lbl_stems = {p.stem for p in labels}
print(f"img-lbl mismatch: {len(img_stems - lbl_stems)} images without label")