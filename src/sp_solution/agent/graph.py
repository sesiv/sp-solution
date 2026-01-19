from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.types import Command

from ..config import Settings
from ..models import ServerMessage
from ..session import Session
from .langgraph_agent import LangGraphAgent


class AgentRunner:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._agent = LangGraphAgent(session, settings)
        self._mode: str = "cmd"
        self._pending_interrupt: dict[str, Optional[str]] | None = None
        self._logger = logging.getLogger(__name__)

    async def handle_user_message(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if self._pending_interrupt:
            await self._handle_pending_interrupt(cleaned)
            return
        if cleaned.startswith("/"):
            await self._handle_command(cleaned)
            return
        if self._mode != "chat":
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Use /chat to enter chat mode."})
            )
            await self._set_status("waiting_user")
            return
        if not self._agent.planner_available():
            await self._session.send_message(
                ServerMessage(
                    type="agent_message",
                    payload={
                        "text": "LLM is not configured. Set OPENROUTER_API_KEY to use chat mode."
                    },
                )
            )
            await self._set_status("waiting_user")
            return
        await self._invoke_graph({"user_message": cleaned})

    async def handle_user_confirm(self, reference: str, confirmed: bool) -> None:
        if not self._pending_interrupt or self._pending_interrupt.get("kind") != "confirm":
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "No pending action found."})
            )
            await self._set_status("waiting_user")
            return
        if reference != self._pending_interrupt.get("reference"):
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "No pending action found."})
            )
            await self._set_status("waiting_user")
            return
        self._pending_interrupt = None
        await self._invoke_graph(Command(resume={"reference": reference, "confirmed": confirmed}))

    async def _handle_pending_interrupt(self, cleaned: str) -> None:
        kind = self._pending_interrupt.get("kind") if self._pending_interrupt else None
        if kind == "confirm":
            if cleaned in {"/yes", "/no"}:
                reference = self._pending_interrupt.get("reference") or ""
                self._pending_interrupt = None
                await self._invoke_graph(
                    Command(resume={"reference": reference, "confirmed": cleaned == "/yes"})
                )
                return
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Reply /yes or /no to confirm."})
            )
            await self._set_status("waiting_user")
            return
        if kind == "manual":
            self._pending_interrupt = None
            await self._invoke_graph(Command(resume={"done": True, "message": cleaned}))
            return
        self._pending_interrupt = None

    async def _handle_command(self, text: str) -> None:
        command = text.split()[0].lstrip("/").lower()
        if command == "chat":
            self._mode = "chat"
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Chat mode enabled. Tell me what to do."})
            )
            await self._set_status("waiting_user")
            return
        if command == "exit":
            self._mode = "cmd"
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "Chat mode disabled."})
            )
            await self._set_status("waiting_user")
            return
        if command in {"yes", "no"}:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": "No pending confirmation."})
            )
            await self._set_status("waiting_user")
            return
        if self._mode == "chat":
            self._mode = "cmd"
        await self._invoke_graph({"user_message": text})

    async def _invoke_graph(self, payload: object) -> None:
        await self._set_status("running")
        try:
            result = await self._agent.ainvoke(payload)
        except Exception as exc:
            await self._session.send_message(
                ServerMessage(type="agent_message", payload={"text": f"Tool failed: {exc}"})
            )
            await self._set_status("waiting_user")
            return
        await self._handle_graph_result(result)
        await self._set_status("waiting_user")

    async def _handle_graph_result(self, result: Any) -> None:
        if isinstance(result, dict) and "__interrupt__" in result:
            payload = result.get("__interrupt__")
            if isinstance(payload, dict):
                await self._handle_interrupt(payload)
            return
        self._pending_interrupt = None

    async def _handle_interrupt(self, payload: dict[str, Any]) -> None:
        kind = payload.get("kind")
        reference = payload.get("reference")
        text = payload.get("text")
        if kind == "confirm":
            if not text:
                text = f"Confirm action (reference {reference}). Reply /yes or /no."
            self._pending_interrupt = {"kind": "confirm", "reference": reference}
            await self._session.send_message(
                ServerMessage(
                    type="agent_question",
                    payload={"text": text, "reference": reference},
                )
            )
            return
        if kind == "manual":
            text = text or "Manual action required. Please handle it and reply when done."
            self._pending_interrupt = {"kind": "manual", "reference": None}
            await self._session.send_message(
                ServerMessage(
                    type="agent_question",
                    payload={"text": text, "reference": None},
                )
            )
            return
        self._pending_interrupt = None

    async def _set_status(self, status: str) -> None:
        self._session.status = status
        await self._session.send_message(ServerMessage(type="status", payload={"value": status}))
