from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class QwenRequest(BaseModel):
    image: str  # base64 encoded image
    detections: List[Dict]  # detection results from object detection

@router.post("/qwen")
async def qwen_inference(request: QwenRequest):
    """Run Qwen 2.5 multimodal inference."""
    try:
        from models.colab_qwen_bridge import get_bridge
        bridge = get_bridge()
        if bridge is not None:
            narration = bridge.infer_once(request.image, request.detections, [])
        else:
            from models.mock_models import MockQwenInference
            narration = MockQwenInference().generate_narration(
                request.image, request.detections, [])
        
        # Determine scenario based on detections
        has_pedestrian = any(d.get("class") in ("pedestrian", "person") for d in request.detections)
        has_sign = any(d.get("class") == "traffic_sign" for d in request.detections)
        
        if has_pedestrian:
            situation = "pedestrian_warning"
        elif has_sign:
            situation = "traffic_sign_detected"
        else:
            situation = "normal_driving"
        
        return {
            "narration": narration,
            "situation": situation,
            "confidence": 0.85,
            "message": "Qwen inference completed"
        }
    except Exception as e:
        return {
            "narration": "Unable to generate narration",
            "situation": "unknown",
            "confidence": 0.0,
            "message": f"Qwen inference error: {str(e)}"
        }
