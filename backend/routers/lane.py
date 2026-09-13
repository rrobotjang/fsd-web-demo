import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Tuple
import base64
import io
from PIL import Image

router = APIRouter()

class LaneRequest(BaseModel):
    image: str  # base64 encoded image

@router.post("/lane")
async def detect_lanes(request: LaneRequest):
    """Run lane detection on image."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Use real ONNX model when available, fall back to mock
        from models.onnx_models import ONNXLaneDetector
        from models.mock_models import MockLaneDetector
        detector = ONNXLaneDetector()
        if not detector.available:
            detector = MockLaneDetector()
        lanes = detector.predict(image_np)
        
        # Format results
        results = [
            {"points": lane}
            for lane in lanes
        ]
        
        return {
            "lanes": results,
            "count": len(results),
            "message": "Lane detection completed"
        }
    except Exception as e:
        return {
            "lanes": [],
            "count": 0,
            "message": f"Lane detection error: {str(e)}"
        }
