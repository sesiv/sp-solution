from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib.parse import urlparse

import httpx
import typer
import websockets
from prompt_toolkit import PromptSession
from rich.console import Console


app = typer.Typer(add_completion=False)
console = Console()


def _ws_url(server: str, session_id: str) -> str:
    parsed = urlparse(server)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc}/ws/{session_id}"


def _sse_url(server: str, session_id: str) -> str:
    return f"{server.rstrip('/')}/events/{session_id}"


async def _sse_listener(url: str, pending_ref: dict) -> None:
    headers = {"Accept": "text/event-stream"}
    while True:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get("type")
                        payload = event.get("payload", {})
                        if event_type == "tool_call":
                            console.print(f"[tool_call] {payload.get('name')} {payload.get('args')}")
                        elif event_type == "tool_result":
                            console.print(f"[tool_result] {payload.get('name')} {payload.get('result')}")
                        elif event_type == "observation":
                            console.print(f"[observation] {payload.get('summary')}")
                        elif event_type == "policy_request":
                            pending_ref["value"] = payload.get("reference")
                            console.print(
                                f"[policy_request] {payload.get('reason')} reference={payload.get('reference')}"
                            )
                        elif event_type == "error":
                            console.print(f"[error] {payload.get('stage')}: {payload.get('message')}")
                        elif event_type == "final":
                            console.print(f"[final] {payload.get('result')}")
        except Exception as exc:
            console.print(f"[sse] connection failed: {exc}. retrying...")
            await asyncio.sleep(1.5)


async def _ws_handler(url: str, outgoing: asyncio.Queue[dict], pending_ref: dict) -> None:
    while True:
        try:
            async with websockets.connect(url) as ws:
                async def sender() -> None:
                    while True:
                        message = await outgoing.get()
                        await ws.send(json.dumps(message))

                async def receiver() -> None:
                    while True:
                        raw = await ws.recv()
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        msg_type = message.get("type")
                        payload = message.get("payload", {})
                        if msg_type == "agent_message":
                            console.print(f"[agent] {payload.get('text')}")
                        elif msg_type == "agent_question":
                            pending_ref["value"] = payload.get("reference")
                            console.print(f"[question] {payload.get('text')}")
                        elif msg_type == "status":
                            console.print(f"[status] {payload.get('value')}")

                await asyncio.gather(sender(), receiver())
        except Exception as exc:
            console.print(f"[ws] connection failed: {exc}. retrying...")
            await asyncio.sleep(1.5)


async def _prompt_loop(outgoing: asyncio.Queue[dict], pending_ref: dict) -> None:
    session = PromptSession()
    while True:
        text = await session.prompt_async("sp> ")
        text = text.strip()
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            break
        if text in {"/yes", "/no"}:
            reference = pending_ref.get("value")
            if not reference:
                console.print("No pending confirmation.")
                continue
            outgoing.put_nowait(
                {
                    "type": "user_confirm",
                    "payload": {"confirmed": text == "/yes", "reference": reference},
                }
            )
            continue
        outgoing.put_nowait({"type": "user_message", "payload": {"text": text}})


@app.command()
def repl(
    server: str = typer.Option("http://127.0.0.1:8000", help="Server base URL"),
    session: str = typer.Option("demo", help="Session id"),
) -> None:
    """Run the CLI REPL."""
    ws_url = _ws_url(server, session)
    sse_url = _sse_url(server, session)
    pending_ref: dict[str, Optional[str]] = {"value": None}
    outgoing: asyncio.Queue[dict] = asyncio.Queue()

    async def runner() -> None:
        await asyncio.gather(
            _sse_listener(sse_url, pending_ref),
            _ws_handler(ws_url, outgoing, pending_ref),
            _prompt_loop(outgoing, pending_ref),
        )

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        console.print("Exiting.")
