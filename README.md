# FSD Web Demo

90-second autonomous driving web demo featuring real-time object detection, lane detection, and AI-powered driving narration.

## Features

- **Object Detection**: YOLOv8-based vehicle/pedestrian/sign detection
- **Lane Detection**: U-Net segmentation for lane recognition
- **Qwen 2.5 Multimodal**: AI driving narration and situation analysis
- **In-car Payment Simulation**: Tollgate, parking, and fuel payment scenarios

## Prerequisites

- Python 3.10+
- Node.js 18+
- ONNX Runtime compatible CPU/GPU

## Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Running

### Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Project Structure

```
fsd-web-demo/
├── backend/           # FastAPI backend
│   ├── routers/       # API endpoints
│   └── main.py        # Application entry
├── frontend/          # React + Vite frontend
│   └── src/
│       ├── components/  # UI components
│       └── hooks/       # React hooks
├── colab/             # Colab training notebooks
│   ├── 01_lane_detection.ipynb
│   ├── 02_object_detection.ipynb
│   └── 03_qwen_lora.ipynb
├── scripts/           # Utility scripts
│   ├── convert_models.py
│   └── pull_models.py
├── models/            # ONNX model files
├── data/              # KITTI data/simulation
└── .env.example       # Environment template
```

## Colab Training (600 CU/month)

### 학습 순서
1. `colab/01_lane_detection.ipynb` - U-Net 세그멘테이션 (40 CU)
2. `colab/02_object_detection.ipynb` - YOLOv8 파인튜닝 (30 CU)
3. `colab/03_qwen_lora.ipynb` - Qwen 2.5 LoRA (100 CU)

### 모델 가져오기
```bash
# Colab에서 학습 후:
python scripts/pull_models.py
```

## 90-Second Demo Scenario

| Time | Event |
|------|-------|
| 0-30s | Normal driving with vehicle/pedestrian detection |
| 30-45s | Tollgate approach → Payment simulation |
| 45-60s | Parking lot approach → Parking payment |
| 60-75s | Gas station approach → Fuel payment |
| 75-90s | Destination arrival → AI summary |

## License

Demo purposes only. Not for production use.
