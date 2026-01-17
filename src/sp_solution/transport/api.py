from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..agent.graph import AgentRunner
from ..config import load_settings
from ..models import ClientMessage, Event, ServerMessage
from ..session import SessionManager


logging.basicConfig(level=logging.INFO)
settings = load_settings()
app = FastAPI()
manager = SessionManager()


async def _ws_sender(websocket: WebSocket, session_id: str, queue: asyncio.Queue[ServerMessage]) -> None:
    while True:
        message = await queue.get()
        await websocket.send_text(message.model_dump_json())


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = manager.get_or_create(session_id)
    if session.runner is None:
        session.runner = AgentRunner(session, settings)
    sender_task = asyncio.create_task(_ws_sender(websocket, session_id, session.messages))
    await session.send_message(ServerMessage(type="status", payload={"value": session.status}))
    try:
        while True:
            raw = await websocket.receive_text()
            message = ClientMessage.model_validate_json(raw)
            try:
                if message.type == "user_message":
                    text = message.payload.get("text", "")
                    async with session.lock:
                        await session.runner.handle_user_message(text)
                elif message.type == "user_confirm":
                    confirmed = bool(message.payload.get("confirmed"))
                    reference = str(message.payload.get("reference", ""))
                    async with session.lock:
                        await session.runner.handle_user_confirm(reference, confirmed)
                elif message.type == "control":
                    command = message.payload.get("command")
                    await session.send_message(
                        ServerMessage(type="agent_message", payload={"text": f"Control '{command}' received."})
                    )
                else:
                    await session.send_message(
                        ServerMessage(type="agent_message", payload={"text": "Unknown message type."})
                    )
            except Exception as exc:
                await session.publish_event(
                    Event.create(
                        "error",
                        session.session_id,
                        {"stage": "ws_message", "message": str(exc)},
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task


@app.get("/events/{session_id}")
async def sse_endpoint(session_id: str) -> StreamingResponse:
    session = manager.get_or_create(session_id)
    queue = session.events.subscribe()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            while True:
                event = await queue.get()
                data = json.dumps(event.model_dump())
                yield f"data: {data}\n\n"
        finally:
            session.events.unsubscribe(queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def main() -> None:
    import uvicorn

    uvicorn.run("sp_solution.transport.api:app", host="0.0.0.0", port=8000, reload=False)
