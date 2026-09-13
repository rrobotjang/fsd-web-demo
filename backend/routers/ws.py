import asyncio

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    from routers.ws_stream import pipeline
    await websocket.accept()
    try:
        frame_id = 0
        auto = False
        while True:
            if auto:
                result = pipeline.process_frame({})
                if not result:
                    break
                result["frame_id"] = frame_id
                frame_id += 1
                await websocket.send_json(result)
                await asyncio.sleep(pipeline.interval)
            else:
                data = await websocket.receive_json()
                if data.get("start"):
                    auto = True
                    continue
                if data.get("image"):
                    result = pipeline.process_frame(data)
                    result["frame_id"] = frame_id
                    frame_id += 1
                    await websocket.send_json(result)
    except Exception:
        pass
    finally:
        await websocket.close()