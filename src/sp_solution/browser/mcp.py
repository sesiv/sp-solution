from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import websockets

from ..config import Settings


class MCPError(RuntimeError):
    pass


MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPToolNames:
    launch: str
    observe: str
    click: str
    type: str
    scroll: str
    wait: str
    screenshot: str


class MCPTransport:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        parsed = urlparse(endpoint)
        if parsed.scheme in {"http", "https"}:
            self._impl: MCPTransportImpl = MCPHttpTransport(endpoint)
        elif parsed.scheme in {"ws", "wss"}:
            self._impl = MCPWebsocketTransport(endpoint)
        else:
            raise MCPError(f"Unsupported MCP endpoint scheme: {parsed.scheme}")

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self._impl.call(name, arguments)


class MCPTransportImpl:
    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MCPWebsocketTransport(MCPTransportImpl):
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = _build_request("tools/call", {"name": name, "arguments": arguments})
        async with websockets.connect(self.endpoint) as ws:
            await ws.send(json.dumps(payload))
            raw = await ws.recv()
            response = json.loads(raw)
        return _extract_result(response)


class MCPHttpTransport(MCPTransportImpl):
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(timeout=30.0)
        self._sse_client: httpx.AsyncClient | None = None
        self._sse_task: asyncio.Task | None = None
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._session_id: str | None = None
        self._initialized = False

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_initialized()
        payload = _build_request("tools/call", {"name": name, "arguments": arguments})
        try:
            response = await self._request(payload)
        except MCPError as exc:
            if "Session not found" not in str(exc):
                raise
            await self._reset_session()
            await self._ensure_initialized()
            response = await self._request(payload)
        return _extract_result(response)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        payload = _build_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "sp-solution", "version": "0.1.0"},
            },
        )
        response = await self._request(payload, include_session=False)
        _extract_result(response)
        if not self._session_id:
            raise MCPError("MCP session id missing from initialize response")
        await self._start_sse_listener()
        self._initialized = True

    async def _request(self, payload: Dict[str, Any], include_session: bool = True) -> Dict[str, Any]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": MCP_PROTOCOL_VERSION,
        }
        if include_session and self._session_id:
            headers["mcp-session-id"] = self._session_id

        async with self._client.stream("POST", self.endpoint, json=payload, headers=headers) as response:
            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id

            if response.status_code >= 400:
                raise MCPError(f"MCP HTTP error {response.status_code}: {await response.aread()}")

            if response.status_code == 202:
                request_id = str(payload.get("id"))
                await self._start_sse_listener()
                return await self._await_pending(request_id)

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                raw = await response.aread()
                return json.loads(raw)
            if "text/event-stream" in content_type:
                request_id = str(payload.get("id"))
                return await self._read_sse_response(response, request_id)
        raise MCPError("MCP HTTP request did not return a response")

    async def _start_sse_listener(self) -> None:
        if not self._session_id:
            return
        if self._sse_task and not self._sse_task.done():
            return
        if not self._sse_client:
            self._sse_client = httpx.AsyncClient(timeout=None)
        headers = {
            "accept": "text/event-stream",
            "mcp-protocol-version": MCP_PROTOCOL_VERSION,
            "mcp-session-id": self._session_id,
        }
        request = self._sse_client.build_request("GET", self.endpoint, headers=headers)
        response = await self._sse_client.send(request, stream=True)
        if response.status_code >= 400:
            await response.aclose()
            raise MCPError(f"MCP SSE error {response.status_code}: {response.text}")
        self._sse_task = asyncio.create_task(self._listen_sse(response))

    async def _reset_session(self) -> None:
        self._session_id = None
        self._initialized = False
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
        self._sse_task = None
        if self._sse_client:
            await self._sse_client.aclose()
            self._sse_client = None

    async def _listen_sse(self, response: httpx.Response) -> None:
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                await self._handle_server_message(message)
        except Exception:
            pass
        finally:
            await response.aclose()
            if self._sse_client:
                await self._sse_client.aclose()
                self._sse_client = None

    async def _handle_server_message(self, message: Dict[str, Any]) -> None:
        msg_id = message.get("id")
        if msg_id is not None:
            pending = self._pending.pop(str(msg_id), None)
            if pending and not pending.done():
                pending.set_result(message)
                return
        method = message.get("method")
        if method == "ping" and msg_id is not None:
            await self._send_response({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            return
        if method and msg_id is not None:
            await self._send_response(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not supported: {method}"},
                }
            )

    async def _send_response(self, payload: Dict[str, Any]) -> None:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": MCP_PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        response = await self._client.post(self.endpoint, json=payload, headers=headers)
        if response.status_code >= 400:
            raise MCPError(f"MCP HTTP error {response.status_code}: {response.text}")

    async def _await_pending(self, request_id: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        finally:
            self._pending.pop(request_id, None)

    async def _read_sse_response(self, response: httpx.Response, request_id: str) -> Dict[str, Any]:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                continue
            await self._handle_server_message(message)
            if str(message.get("id")) == request_id:
                return message
        raise MCPError("MCP SSE response did not include expected message")


class MCPBrowserClient:
    def __init__(self, session_id: str, settings: Settings) -> None:
        if not settings.mcp_endpoint:
            raise MCPError("MCP_ENDPOINT is not configured")
        self._session_id = session_id
        self._settings = settings
        self._tools = MCPToolNames(
            launch=settings.mcp_tool_launch,
            observe=settings.mcp_tool_observe,
            click=settings.mcp_tool_click,
            type=settings.mcp_tool_type,
            scroll=settings.mcp_tool_scroll,
            wait=settings.mcp_tool_wait,
            screenshot=settings.mcp_tool_screenshot,
        )
        self._transport = MCPTransport(settings.mcp_endpoint)
        self._started = False

    async def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True

    async def observe(self, mode: str = "main", target: str | None = None) -> Dict[str, Any]:
        await self.ensure_started()
        return await self._transport.call(self._tools.observe, {})

    async def click(self, eid: str, element: str | None = None) -> Dict[str, Any]:
        await self.ensure_started()
        payload = {"ref": eid, "element": element or eid}
        return await self._transport.call(self._tools.click, payload)

    async def type(self, eid: str, value: str, element: str | None = None) -> Dict[str, Any]:
        await self.ensure_started()
        payload = {"ref": eid, "element": element or eid, "text": value}
        return await self._transport.call(self._tools.type, payload)

    async def scroll(self, direction: str = "down", amount: int = 600) -> Dict[str, Any]:
        await self.ensure_started()
        key = "PageDown" if direction.lower() != "up" else "PageUp"
        return await self._transport.call(self._tools.scroll, {"key": key})

    async def wait(self, timeout_ms: int = 1000) -> Dict[str, Any]:
        await self.ensure_started()
        return await self._transport.call(self._tools.wait, {"time": max(0.1, timeout_ms / 1000)})

    async def screenshot(self) -> Dict[str, Any]:
        await self.ensure_started()
        return await self._transport.call(self._tools.screenshot, {})


def _build_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": method,
        "params": params,
    }


def _parse_sse_messages(raw_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            messages.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return messages


def _match_response(messages: List[Dict[str, Any]], request_id: str | None) -> Dict[str, Any] | None:
    if not request_id:
        return None
    for message in messages:
        if message.get("id") == request_id:
            return message
    return None


def _extract_result(response: Dict[str, Any]) -> Dict[str, Any]:
    if response.get("error"):
        raise MCPError(str(response["error"]))
    result = response.get("result", {})
    if isinstance(result, dict) and result.get("isError"):
        message = _result_text(result)
        raise MCPError(message or "MCP tool returned error")
    return result


def _result_text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)
