import config  # loads .env (MODEL_DIR etc.) before any model imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import detection, lane, payment, ws, qwen

app = FastAPI(title="FSD Web Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api")
app.include_router(lane.router, prefix="/api")
app.include_router(payment.router, prefix="/api")
app.include_router(qwen.router, prefix="/api")
app.include_router(ws.router)

@app.get("/health")
async def health():
    return {"status": "ok"}
