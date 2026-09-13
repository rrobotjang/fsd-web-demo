#!/usr/bin/env python3
"""ONNX Model Conversion Utility for FSD Web Demo.

Usage:
    pip install torch onnxruntime
    python convert_models.py

Converts:
    - Downloads/segmentation/seg_model_unet.pth -> models/lane.onnx
    - Downloads/best_model.pt -> models/detection.onnx
"""

import os
import sys
import torch
import torch.onnx
import numpy as np


def convert_lane_model():
    """Convert U-Net segmentation model to ONNX."""
    print("Converting lane detection model...")
    
    model_path = "/Users/robotjang/Downloads/segmentation/seg_model_unet.pth"
    output_path = "models/lane.onnx"
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to extract model from checkpoint
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Need model architecture - create a simple U-Net stub
            print("  Checkpoint contains state_dict only, creating stub model...")
            model = create_unet_stub()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint
    else:
        model = checkpoint
    
    model.eval()
    
    # Create dummy input (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 480, 640)
    
    # Export to ONNX
    os.makedirs("models", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def convert_detection_model():
    """Convert object detection model to ONNX."""
    print("Converting detection model...")
    
    model_path = "/Users/robotjang/Downloads/best_model.pt"
    output_path = "models/detection.onnx"
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to extract model
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            print("  Checkpoint contains state_dict, creating stub...")
            model = create_detection_stub()
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint
    else:
        model = checkpoint
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Export to ONNX
    os.makedirs("models", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"  Saved to {output_path}")
    return output_path


def create_unet_stub():
    """Create a simple U-Net stub for conversion."""
    import torch.nn as nn
    
    class UNetStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    return UNetStub()


def create_detection_stub():
    """Create a simple detection stub for conversion."""
    import torch.nn as nn
    
    class DetectionStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(32, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return DetectionStub()


def verify_onnx_model(model_path):
    """Verify the ONNX model loads correctly."""
    import onnxruntime as ort
    
    print(f"Verifying {model_path}...")
    session = ort.InferenceSession(model_path)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in input_shape]).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})
    
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print("  OK!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("FSD Web Demo - ONNX Model Conversion")
    print("=" * 50)
    
    try:
        lane_path = convert_lane_model()
        detect_path = convert_detection_model()
        
        print("\nVerifying converted models...")
        verify_onnx_model(lane_path)
        verify_onnx_model(detect_path)
        
        print("\n" + "=" * 50)
        print("Conversion complete!")
        print("Models saved in models/ directory")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure torch is installed: pip install torch onnxruntime")
        sys.exit(1)
