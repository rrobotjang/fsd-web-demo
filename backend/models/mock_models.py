import numpy as np
from typing import List, Dict, Tuple
import os

class MockDetector:
    """Mock object detection for demo purposes."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.confidence_threshold = 0.5
        
    def predict(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image frame."""
        # Mock detection results based on image content
        detections = []
        
        # Simulate random vehicles
        np.random.seed(42)
        num_vehicles = np.random.randint(2, 5)
        for i in range(num_vehicles):
            x = np.random.randint(50, 500)
            y = np.random.randint(200, 400)
            w = np.random.randint(60, 120)
            h = np.random.randint(40, 80)
            conf = np.random.uniform(0.7, 0.99)
            detections.append({
                "class": "car",
                "confidence": float(conf),
                "bbox": [int(x), int(y), int(x + w), int(y + h)]
            })
        
        # Simulate pedestrians
        num_pedestrians = np.random.randint(0, 3)
        for i in range(num_pedestrians):
            x = np.random.randint(100, 550)
            y = np.random.randint(250, 420)
            w = np.random.randint(20, 40)
            h = np.random.randint(50, 80)
            conf = np.random.uniform(0.6, 0.95)
            detections.append({
                "class": "pedestrian",
                "confidence": float(conf),
                "bbox": [int(x), int(y), int(x + w), int(y + h)]
            })
        
        # Simulate traffic signs
        if np.random.random() > 0.5:
            x = np.random.randint(400, 600)
            y = np.random.randint(100, 200)
            detections.append({
                "class": "traffic_sign",
                "confidence": 0.92,
                "bbox": [int(x), int(y), int(x + 40), int(y + 60)]
            })
        
        return detections


class MockLaneDetector:
    """Mock lane detection for demo purposes."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        
    def predict(self, image: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Run lane detection on image frame."""
        height, width = image.shape[:2]
        
        # Generate realistic lane curves
        lanes = []
        
        # Left lane (curves slightly left)
        left_lane = []
        for y in range(height // 2, height, 15):
            x = int(width * 0.35 + (y - height // 2) * 0.05)
            left_lane.append((x, y))
        lanes.append(left_lane)
        
        # Center lane
        center_lane = []
        for y in range(height // 2, height, 15):
            x = int(width * 0.5 + (y - height // 2) * 0.02)
            center_lane.append((x, y))
        lanes.append(center_lane)
        
        # Right lane (curves slightly right)
        right_lane = []
        for y in range(height // 2, height, 15):
            x = int(width * 0.65 + (y - height // 2) * 0.05)
            right_lane.append((x, y))
        lanes.append(right_lane)
        
        return lanes


class MockQwenInference:
    """Mock Qwen 2.5 multimodal inference for demo purposes."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.narration_templates = {
            "normal": [
                "Road ahead is clear. Detected {num_cars} vehicle(s), maintaining a safe distance.",
                "Good road conditions, {num_cars} vehicle(s) ahead, speed is stable.",
                "Normal driving, {num_cars} vehicle(s) in view."
            ],
            "pedestrian": [
                "Caution! Pedestrian detected ahead, please slow down.",
                "Pedestrian crossing risk detected, recommend early braking.",
                "Pedestrian activity ahead, stay alert."
            ],
            "traffic_sign": [
                "Traffic sign detected, please obey traffic regulations.",
                "Traffic sign ahead, observe and comply."
            ]
        }
    
    def generate_narration(self, image=None, detections: List[Dict] = None,
                           lanes: List = None) -> str:
        """Generate natural language narration based on detections."""
        detections = detections or []
        num_cars = sum(1 for d in detections if d["class"] == "car")
        num_pedestrians = sum(1 for d in detections if d["class"] == "pedestrian")
        has_sign = any(d["class"] == "traffic_sign" for d in detections)
        
        import random
        
        if num_pedestrians > 0:
            template = random.choice(self.narration_templates["pedestrian"])
            return template
        elif has_sign:
            template = random.choice(self.narration_templates["traffic_sign"])
            return template
        else:
            template = random.choice(self.narration_templates["normal"])
            return template.format(num_cars=num_cars)
