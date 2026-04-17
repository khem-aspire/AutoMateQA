"""
WebSocket router for real-time test execution updates.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.services.executor import execution_manager

router = APIRouter()


@router.websocket("/runs/{run_id}")
async def ws_run_updates(websocket: WebSocket, run_id: str):
    """Stream real-time step execution updates for a specific run."""
    await websocket.accept()

    queue = execution_manager.subscribe(run_id)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(message)
                # If run is complete, close the connection
                if message.get("type") in ("run_completed", "run_error"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        execution_manager.unsubscribe(run_id, queue)
