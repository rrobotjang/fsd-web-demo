import onnx
import numpy as np

m = onnx.load("/content/lane.onnx")
print(f"Nodes: {len(m.graph.node)}, Initializers: {len(m.graph.initializer)}")

total_vals = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
print(f"Total initializer values: {total_vals:,}")

import onnxruntime as ort, cv2
s = ort.InferenceSession("/content/lane.onnx")
img = cv2.imread("/content/data_road/training/image_2/um_000000.png")
img = cv2.resize(img, (384, 288))
x = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
x = x.transpose(2, 0, 1)[None]
out = s.run(None, {"input": x})[0]
print(f"Output: shape={out.shape} min={out.min():.4f} max={out.max():.4f} mean={out.mean():.4f}")
print(f"Lane pixels (>0.5): {(out[0,0]>0.5).sum()} / {out[0,0].size}")