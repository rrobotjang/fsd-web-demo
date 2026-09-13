import onnx
import numpy as np

m = onnx.load("/content/lane.onnx")
print(f"Graph: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers")

total_initializer_bytes = sum(
    np.prod(t.dims, dtype=int) * (4 if t.data_type == 1 else 1)
    for t in m.graph.initializer
)
print(f"Initializer values total: {total_initializer_bytes:,}")

# Check a few weights for sanity (not constant)
for init in m.graph.initializer[:5]:
    arr = onnx.numpy_helper.to_array(init)
    print(f"  {init.name}: shape={arr.shape} std={arr.std():.4f}")

# Run inference with a real KITTI image
import onnxruntime as ort, cv2
s = ort.InferenceSession("/content/lane.onnx")
img = cv2.imread("/content/data_road/training/image_2/um_000000.png")
img = cv2.resize(img, (384, 288))
x = img.astype(np.float32) / 255.0
x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
x = x.transpose(2, 0, 1)[None]
out = s.run(None, {"input": x})[0]
print(f"Inference output: shape={out.shape} min={out.min():.4f} max={out.max():.4f} mean={out.mean():.4f}")
print(f"Lane pixels (out>0.5): {(out[0,0]>0.5).sum()}")