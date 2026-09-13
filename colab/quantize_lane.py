"""Quantize lane U-Net ONNX to int8 (static) using KITTI calibration frames."""
import glob
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_H, IN_W = 384, 288


class KittiReader(CalibrationDataReader):
    def __init__(self, cal_files, in_size=(IN_H, IN_W)):
        self.h, self.w = in_size
        self.data = []
        for f in sorted(cal_files):
            img = cv2.imread(f)  # BGR
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
            x = (x - MEAN) / STD
            self.data.append({"input": x.transpose(2, 0, 1)[None].astype(np.float32)})
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d

    def rewind(self):
        self.idx = 0


cal = glob.glob("/content/cal_*.png")
print("calibration frames:", len(cal))

reader = KittiReader(cal)
quantize_static(
    "/content/lane.onnx",
    "/content/lane_int8.onnx",
    reader,
    quant_format=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
print("int8 quantization done")

m = onnx.load("/content/lane_int8.onnx")
print("int8 model size MB:", round(len(m.SerializeToString()) / 1e6, 2))