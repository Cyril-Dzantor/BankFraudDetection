from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ....core.manager import manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket: Handshake initiating...")
    await manager.connect(websocket)
    print(f"WebSocket: Connection established. Total active: {len(manager.active_connections)}")
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # Handle incoming messages if needed, otherwise just heartbeats
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"WebSocket: Connection closed. Remaining active: {len(manager.active_connections)}")
