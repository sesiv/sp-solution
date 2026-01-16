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
        try:
            response = await self._request(payload, include_session=False)
            _extract_result(response)
            if self._session_id:
                self._initialized = True
                return
        except MCPError as exc:
            if "Session not found" not in str(exc):
                raise

        await self._ensure_session()
        response = await self._request(payload)
        _extract_result(response)
        self._initialized = True

    async def _request(self, payload: Dict[str, Any], include_session: bool = True) -> Dict[str, Any]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": MCP_PROTOCOL_VERSION,
        }
        if include_session and self._session_id:
            headers["mcp-session-id"] = self._session_id

        response = await self._client.post(self.endpoint, json=payload, headers=headers)
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        if response.status_code >= 400:
            raise MCPError(f"MCP HTTP error {response.status_code}: {response.text}")

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        if "text/event-stream" in content_type:
            messages = _parse_sse_messages(response.text)
            if not messages:
                raise MCPError("MCP SSE response did not include any messages")
            response_msg = _match_response(messages, payload.get("id"))
            if not response_msg:
                response_msg = messages[0]
            return response_msg
        raise MCPError(f"Unexpected MCP response content-type: {content_type}")

    async def _ensure_session(self) -> None:
        if self._session_id:
            return
        if self._sse_task and not self._sse_task.done():
            return
        headers = {
            "accept": "text/event-stream",
            "mcp-protocol-version": MCP_PROTOCOL_VERSION,
        }
        self._sse_client = httpx.AsyncClient(timeout=None)
        request = self._sse_client.build_request("GET", self.endpoint, headers=headers)
        response = await self._sse_client.send(request, stream=True)
        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            await response.aclose()
            raise MCPError("MCP HTTP session id missing from SSE response")
        self._session_id = session_id
        # Keep the SSE stream open to maintain the session.
        self._sse_task = asyncio.create_task(self._drain_sse(response))

    async def _reset_session(self) -> None:
        self._session_id = None
        self._initialized = False
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
        self._sse_task = None
        if self._sse_client:
            await self._sse_client.aclose()
            self._sse_client = None

    async def _drain_sse(self, response: httpx.Response) -> None:
        try:
            async for _line in response.aiter_lines():
                pass
        except Exception:
            pass
        finally:
            await response.aclose()
            if self._sse_client:
                await self._sse_client.aclose()
                self._sse_client = None


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
