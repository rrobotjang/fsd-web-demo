import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import base64
import io
from PIL import Image

router = APIRouter()

class DetectionRequest(BaseModel):
    image: str  # base64 encoded image

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]

@router.post("/detect")
async def detect(request: DetectionRequest):
    """Run object detection on image."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Use real ONNX model when available, fall back to mock
        from models.onnx_models import ONNXDetector
        from models.mock_models import MockDetector
        detector = ONNXDetector()
        if not detector.available:
            detector = MockDetector()
        detections = detector.predict(image_np)
        
        # Format results
        results = [
            {
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox": d["bbox"]
            }
            for d in detections
        ]
        
        return {
            "detections": results,
            "count": len(results),
            "message": "Detection completed"
        }
    except Exception as e:
        return {
            "detections": [],
            "count": 0,
            "message": f"Detection error: {str(e)}"
        }
