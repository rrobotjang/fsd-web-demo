#!/usr/bin/env python3
"""Pull trained models from Google Drive and set up for local inference."""

import os
import shutil
from pathlib import Path

GDRIVE_BASE = "/content/drive/MyDrive/fsd_models"
LOCAL_MODELS = "models"

def setup_models():
    """Pull models from GDrive to local models directory."""
    
    os.makedirs(LOCAL_MODELS, exist_ok=True)
    
    # Check if running in Colab
    if os.path.exists("/content"):
        print("Running in Colab - using GDrive paths")
        source_base = GDRIVE_BASE
    else:
        print("Running locally - expecting models in local directory")
        return
    
    # Copy lane model
    lane_src = f"{source_base}/lane.onnx"
    if os.path.exists(lane_src):
        shutil.copy(lane_src, f"{LOCAL_MODELS}/lane.onnx")
        print(f"✓ Copied lane.onnx")
    
    # Copy detection model
    detect_src = f"{source_base}/fsd_detection/weights/best.onnx"
    if os.path.exists(detect_src):
        shutil.copy(detect_src, f"{LOCAL_MODELS}/detection.onnx")
        print(f"✓ Copied detection.onnx")
    
    # Copy Qwen adapter
    qwen_src = f"{source_base}/qwen_lora/final"
    if os.path.exists(qwen_src):
        shutil.copytree(qwen_src, f"{LOCAL_MODELS}/qwen_adapter", dirs_exist_ok=True)
        print(f"✓ Copied qwen_adapter")
    
    print("\nModels ready in models/ directory")

if __name__ == "__main__":
    setup_models()
